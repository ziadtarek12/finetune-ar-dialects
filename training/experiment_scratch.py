from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import load_from_disk, concatenate_datasets
import argparse
import torch
import os
from transformers import EarlyStoppingCallback

torch.multiprocessing.set_sharing_strategy("file_system")
from whisper_utils import (  # noqa: E402
    DataCollatorSpeechSeq2SeqWithPadding,
    compute_metrics,
    TimingCallback,
)

if __name__ == "__main__":
    root = "/Users/otoz/Documents/MSc Voice Technology/Thesis VT/whisper_fine_tune"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dialect",
        required=True,
        help="all, egyptian, gulf, iraqi, levantine, maghrebi",
    )
    args = parser.parse_args()
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3, early_stopping_threshold=0.001
    )

    if args.dialect == "all":
        dialect_dataset = load_from_disk(os.path.join(root, "egyptian_train/"))
        for d in ["gulf", "iraqi", "levantine", "maghrebi"]:
            train_d = load_from_disk(os.path.join(root, f"{d}_train/"))
            dialect_dataset["train"] = concatenate_datasets(
                [train_d["train"], dialect_dataset["train"]]
            )
            dialect_dataset["test"] = concatenate_datasets(
                [train_d["test"], dialect_dataset["test"]]
            )
    else:
        dialect_dataset = load_from_disk(os.path.join(root, f"{args.dialect}_train/"))

    print(f"Training on {args.dialect} dialect, loaded from {dialect_dataset}")
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", language="Arabic", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.language = "ar"
    model.config.max_length = 512

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./whisper-small-dialect_{args.dialect}",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=5000,
        gradient_checkpointing=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=250,
        eval_steps=250,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )

    for seed in [42, 84, 168]:
        print(f"Training with seed {seed}")
        train_test = dialect_dataset.train_test_split(test_size=0.2, seed=seed)
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=train_test["train"],
            eval_dataset=train_test["test"],
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=processor.feature_extractor,
            callbacks=[early_stopping_callback, TimingCallback()],
        )
        training_args.output_dir = f"./whisper-small-dialect_{args.dialect}_seed{seed}"
        trainer.train()
