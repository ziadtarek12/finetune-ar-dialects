"""
PEFT utilities for Whisper fine-tuning on Arabic dialects.
This module provides functions for setting up, training, and loading PEFT models.
"""

import torch
import os
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, PeftModel, PeftConfig


def create_lora_config(
    rank=32,
    alpha=64,
    target_modules=None,
    dropout=0.05,
    bias="none"
):
    """
    Create LoRA configuration for Whisper fine-tuning.
    
    Args:
        rank (int): LoRA rank (default: 32)
        alpha (int): LoRA alpha parameter (default: 64)
        target_modules (list): Target modules for LoRA (default: ["q_proj", "v_proj"])
        dropout (float): LoRA dropout (default: 0.05)
        bias (str): Bias type (default: "none")
    
    Returns:
        LoraConfig: Configured LoRA configuration
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]
    
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias=bias,
    )


def setup_peft_model(model, lora_config=None, load_in_8bit=True):
    """
    Setup model for PEFT training with LoRA adapters.
    
    Args:
        model: Base Whisper model
        lora_config (LoraConfig): LoRA configuration (optional)
        load_in_8bit (bool): Whether the model was loaded in 8-bit
    
    Returns:
        PeftModel: Model with LoRA adapters applied
    """
    print("Setting up PEFT model with LoRA adapters...")
    
    if load_in_8bit:
        # Prepare model for 8-bit training
        model = prepare_model_for_int8_training(model, output_embedding_layer_name="proj_out")
        
        # Make inputs require grad for convolutional layers
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)
    
    # Configure LoRA if not provided
    if lora_config is None:
        lora_config = create_lora_config()
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model


def load_whisper_for_peft(
    model_name="openai/whisper-small",
    load_in_8bit=True,
    device_map="auto"
):
    """
    Load Whisper model prepared for PEFT training.
    
    Args:
        model_name (str): Whisper model name or path
        load_in_8bit (bool): Load model in 8-bit for memory efficiency
        device_map (str): Device mapping strategy
    
    Returns:
        WhisperForConditionalGeneration: Loaded model
    """
    if load_in_8bit:
        model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map=device_map
        )
    else:
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    # Set forced decoder ids for Arabic
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    return model


def load_peft_model_for_inference(adapter_path, base_model_name="openai/whisper-small"):
    """
    Load PEFT model for inference.
    
    Args:
        adapter_path (str): Path to saved PEFT adapter
        base_model_name (str): Base model name
    
    Returns:
        PeftModel: Model with LoRA adapters loaded
    """
    print(f"Loading PEFT model from {adapter_path}...")
    
    # Load PEFT config
    peft_config = PeftConfig.from_pretrained(adapter_path)
    
    # Load base model
    base_model = WhisperForConditionalGeneration.from_pretrained(
        base_model_name, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    # Load PEFT model
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Enable cache for inference
    model.config.use_cache = True
    
    print("PEFT model loaded successfully for inference!")
    return model


def get_peft_training_args(
    output_dir,
    per_device_train_batch_size=16,
    learning_rate=1e-3,
    num_train_epochs=3,
    max_steps=None,
    warmup_steps=50,
    save_steps=500,
    eval_steps=500,
    logging_steps=25,
    **kwargs
):
    """
    Get recommended training arguments for PEFT fine-tuning.
    
    Args:
        output_dir (str): Output directory
        per_device_train_batch_size (int): Batch size per device
        learning_rate (float): Learning rate
        num_train_epochs (int): Number of training epochs
        max_steps (int): Maximum training steps (optional)
        warmup_steps (int): Warmup steps
        save_steps (int): Save every N steps
        eval_steps (int): Evaluate every N steps
        logging_steps (int): Log every N steps
        **kwargs: Additional training arguments
    
    Returns:
        dict: Training arguments dictionary
    """
    training_args = {
        "output_dir": output_dir,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": 1,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "num_train_epochs": num_train_epochs,
        "evaluation_strategy": "steps",
        "fp16": True,
        "per_device_eval_batch_size": per_device_train_batch_size,
        "generation_max_length": 128,
        "logging_steps": logging_steps,
        "save_steps": save_steps,
        "eval_steps": eval_steps,
        "report_to": ["tensorboard"],
        "load_best_model_at_end": True,
        "metric_for_best_model": "wer",
        "greater_is_better": False,
        "save_total_limit": 2,
        # PEFT specific settings
        "remove_unused_columns": False,
        "label_names": ["labels"],
        "push_to_hub": False,
    }
    
    if max_steps is not None:
        training_args["max_steps"] = max_steps
        training_args.pop("num_train_epochs")
    
    # Update with any additional arguments
    training_args.update(kwargs)
    
    return training_args


def test_peft_model(model, processor, test_sample):
    """
    Test PEFT model on a sample audio.
    
    Args:
        model: PEFT model
        processor: Whisper processor
        test_sample: Test sample with audio data
    
    Returns:
        str: Transcription
    """
    # Prepare input
    input_features = processor(
        test_sample["audio"]["array"], 
        sampling_rate=test_sample["audio"]["sampling_rate"], 
        return_tensors="pt"
    ).input_features
    
    # Move to device
    if torch.cuda.is_available():
        input_features = input_features.cuda()
    
    # Generate prediction
    with torch.no_grad():
        predicted_ids = model.generate(input_features, max_length=128)
    
    # Decode prediction
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    return transcription


def compare_model_sizes(base_model_path, adapter_path):
    """
    Compare sizes of base model and PEFT adapter.
    
    Args:
        base_model_path (str): Path to base model
        adapter_path (str): Path to PEFT adapter
    
    Returns:
        dict: Size comparison information
    """
    def get_dir_size(path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size
    
    base_size = get_dir_size(base_model_path) if os.path.exists(base_model_path) else 0
    adapter_size = get_dir_size(adapter_path) if os.path.exists(adapter_path) else 0
    
    return {
        "base_model_size_mb": base_size / (1024 * 1024),
        "adapter_size_mb": adapter_size / (1024 * 1024),
        "size_reduction_ratio": base_size / adapter_size if adapter_size > 0 else 0,
        "adapter_percentage": (adapter_size / base_size * 100) if base_size > 0 else 0
    }


def print_model_info(model):
    """
    Print information about the model including parameter counts.
    
    Args:
        model: Model to analyze
    """
    if hasattr(model, 'print_trainable_parameters'):
        print("=== PEFT Model Information ===")
        model.print_trainable_parameters()
    else:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"=== Model Information ===")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
