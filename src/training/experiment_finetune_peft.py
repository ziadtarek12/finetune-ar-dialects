from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from datasets import load_from_disk, concatenate_datasets
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, PeftModel, PeftConfig
import argparse
import torch
import os
from transformers import EarlyStoppingCallback

torch.multiprocessing.set_sharing_strategy("file_system")
from src.whisper_utils import (  # noqa: E402
    DataCollatorSpeechSeq2SeqWithPadding,
    compute_metrics,
    TimingCallback,
)


class SavePeftModelCallback(TrainerCallback):
    """Callback to save only PEFT adapter weights and remove base model weights."""
    
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control


def setup_peft_model(model, use_peft=True, lora_config=None):
    """Setup model for PEFT training with LoRA adapters."""
    if not use_peft:
        return model
    
    print("Setting up PEFT model with LoRA adapters...")
    
    # Prepare model for 8-bit training
    model = prepare_model_for_int8_training(model, output_embedding_layer_name="proj_out")
    
    # Make inputs require grad for convolutional layers
    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)

    model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)
    
    # Configure LoRA if not provided
    if lora_config is None:
        lora_config = LoraConfig(
            r=32,  # Rank
            lora_alpha=64,  # Alpha parameter for LoRA scaling
            target_modules=["q_proj", "v_proj"],  # Target modules for LoRA
            lora_dropout=0.05,  # Dropout for LoRA layers
            bias="none",  # Bias type
        )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model


def load_peft_model_for_inference(adapter_path, base_model_name="openai/whisper-small"):
    """Load PEFT model for inference."""
    # Load PEFT config
    peft_config = PeftConfig.from_pretrained(adapter_path)
    
    # Load base model
    base_model = WhisperForConditionalGeneration.from_pretrained(
        base_model_name, torch_dtype=torch.float16, device_map="auto"
    )
    
    # Load PEFT model
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    return model


if __name__ == "__main__":
    root = ""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dialect",
        required=True,
        help="all, egyptian, gulf, iraqi, levantine, maghrebi",
    )
    parser.add_argument(
        "--use_peft",
        action="store_true",
        help="Use PEFT (LoRA) for parameter-efficient fine-tuning"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=32,
        help="LoRA rank (default: 32)"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help="LoRA alpha parameter (default: 64)"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (default: 0.05)"
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8-bit for memory efficiency (recommended with PEFT)"
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 84, 168],
        help="Random seeds for multiple runs (default: 42 84 168)"
    )
    
    args = parser.parse_args()
    
    # Early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3, early_stopping_threshold=0.001
    )

    # Load dataset
    if args.dialect == "all":
        dialect_dataset = load_from_disk(os.path.join(root, "egyptian_train/"))
        for d in ["gulf", "iraqi", "levantine", "maghrebi"]:
            train_d = load_from_disk(os.path.join(root, f"{d}_train/"))
            dialect_dataset = concatenate_datasets([train_d, dialect_dataset])
    else:
        dialect_dataset = load_from_disk(os.path.join(root, f"{args.dialect}_train/"))
    
    # Set cache directories
    experiment_type = "peft" if args.use_peft else "finetune"
    os.environ["TRANSFORMERS_CACHE"] = f"model_cache_{args.dialect}_{experiment_type}"
    os.environ["HF_HOME"] = f"hf_cache_{args.dialect}_{experiment_type}"

    print(f"Training on {args.dialect} dialect, loaded from {dialect_dataset}")
    print(f"Using {'PEFT (LoRA)' if args.use_peft else 'full fine-tuning'}")
    
    # Initialize processor
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", language="Arabic", task="transcribe"
    )
    
    # Run experiments for each seed
    for seed in args.seeds:
        print(f"\n{'='*50}")
        print(f"Running experiment with seed: {seed}")
        print(f"{'='*50}")
        
        # Set random seeds
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Load model
        if args.use_peft and args.load_in_8bit:
            model = WhisperForConditionalGeneration.from_pretrained(
                "openai/whisper-small",
                load_in_8bit=True,
                device_map="auto"
            )
        else:
            model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        
        # Set forced decoder ids
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        
        # Setup PEFT if requested
        if args.use_peft:
            lora_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=args.lora_dropout,
                bias="none",
            )
            model = setup_peft_model(model, use_peft=True, lora_config=lora_config)
        
        # Data collator
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
        
        # Training arguments
        output_dir = f"./whisper-small-{experiment_type}-{args.dialect}_seed{seed}"
        
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=16 if args.use_peft else 8,  # Larger batch size with PEFT
            gradient_accumulation_steps=1 if args.use_peft else 2,
            learning_rate=1e-3 if args.use_peft else 1e-5,  # Higher LR for LoRA
            warmup_steps=50 if args.use_peft else 500,
            max_steps=2000 if args.use_peft else 4000,  # Fewer steps needed with PEFT
            gradient_checkpointing=True,
            fp16=True,
            evaluation_strategy="steps",
            per_device_eval_batch_size=8,
            predict_with_generate=not args.use_peft,  # Disabled for PEFT due to 8-bit constraints
            generation_max_length=225,
            save_steps=500,
            eval_steps=500,
            logging_steps=25,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            save_total_limit=2,
            push_to_hub=False,
            # PEFT specific settings
            remove_unused_columns=False if args.use_peft else True,
            label_names=["labels"] if args.use_peft else None,
        )
        
        # Initialize trainer
        callbacks = [TimingCallback(args.dialect, experiment_type, seed)]
        if args.use_peft:
            callbacks.append(SavePeftModelCallback())
        callbacks.append(early_stopping_callback)
        
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=dialect_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics if not args.use_peft else None,  # Disabled for PEFT
            tokenizer=processor.feature_extractor,
            callbacks=callbacks,
        )
        
        # Disable cache for training
        model.config.use_cache = False
        
        # Start training
        print(f"Starting {experiment_type} training for {args.dialect} with seed {seed}...")
        trainer.train()
        
        # Save final model
        final_model_path = f"{output_dir}_final"
        if args.use_peft:
            trainer.model.save_pretrained(final_model_path)
            processor.save_pretrained(final_model_path)
            print(f"PEFT adapter saved to: {final_model_path}")
        else:
            trainer.save_model(final_model_path)
            processor.save_pretrained(final_model_path)
            print(f"Full model saved to: {final_model_path}")
        
        # Clean up for next iteration
        del model
        del trainer
        torch.cuda.empty_cache()
        
        print(f"Completed experiment with seed {seed}")
    
    print(f"\nAll experiments completed for {args.dialect} dialect using {experiment_type}!")
