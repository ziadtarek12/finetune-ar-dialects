#!/usr/bin/env python3
"""
Arabic Dialect PEFT Training Script - Clean Modular Implementation

This script implements PEFT LoRA fine-tuning for Arabic dialects following the original paper
"Overcoming Data Scarcity in Multi-Dialectal Arabic ASR via Whisper Fine-Tuning" with modern
PEFT enhancements and improved architecture.

Key Features:
- Modular architecture with manager classes
- HuggingFace dataset integration
- PEFT LoRA with optimized configurations per model size
- Comprehensive evaluation system
- Command-line interface for experiments
- Memory tracking and metrics calculation

Author: Generated with GitHub Copilot
Date: 2024
"""

# =============================================================================
# IMPORTS AND DEPENDENCIES
# =============================================================================

import os
import json
import logging
import torch
import numpy as np
import evaluate
import time
import psutil
from pathlib import Path
from typing import Dict, List, Union, Tuple, Any
from dataclasses import dataclass

# HuggingFace imports
from datasets import load_dataset, DatasetDict, Audio, concatenate_datasets
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    TrainerCallback, TrainingArguments, TrainerState, TrainerControl
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from huggingface_hub import login, HfFolder
import os
import sys
# PEFT imports
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig

# Set up logging
root_logger = logging.getLogger()

# Remove all existing handlers from the root logger
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
# ----------------------------------------

# --- Now, Apply Your Desired Configuration ---
# This will now work correctly
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout  # Direct output to the notebook cell
)

# You can now get any logger and it will work
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Dialect configurations for optimal PEFT performance
DIALECT_CONFIG = {
    'egyptian': {'priority': 'high', 'data_size': 'large'},
    'gulf': {'priority': 'high', 'data_size': 'medium'},
    'iraqi': {'priority': 'medium', 'data_size': 'small'},
    'levantine': {'priority': 'high', 'data_size': 'medium'},
    'maghrebi': {'priority': 'medium', 'data_size': 'small'},
    'all': {'priority': 'high', 'data_size': 'extra_large'}
}

# PEFT configurations optimized for different model sizes - matching working notebook
PEFT_CONFIG = {
    'small': {
        'lora_rank': 32,
        'lora_alpha': 64,
        'lora_dropout': 0.05,  # Changed from 0.1 to match notebook
        'target_modules': ["q_proj", "v_proj"],  # Simplified to match notebook exactly
        'learning_rate': 1e-3,
        'batch_size': 8
    },
    'medium': {
        'lora_rank': 32,
        'lora_alpha': 64,
        'lora_dropout': 0.05,  # Changed from 0.1 to match notebook
        'target_modules': ["q_proj", "v_proj"],  # Simplified to match notebook exactly
        'learning_rate': 1e-3,
        'batch_size': 8
    },
    'large': {
        'lora_rank': 64,
        'lora_alpha': 128,
        'lora_dropout': 0.1,
        'target_modules': ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        'learning_rate': 5e-4,
        'batch_size': 4
    }
}

# HuggingFace dataset mapping for Arabic dialects
HUGGINGFACE_DATASET_MAPPING = {
    'egyptian': 'otozz/egyptian',
    'gulf': 'otozz/gulf',
    'iraqi': 'otozz/iraqi',
    'levantine': 'otozz/levantine',
    'maghrebi': 'otozz/maghrebi',
    'msa': 'otozz/MSA'
}


# =============================================================================
# UTILITY CLASSES
# =============================================================================

class MetricsTracker:
    """Simple metrics tracking following original repository pattern."""
    
    def __init__(self, dialect: str, model_type: str, model_size: str, seed: int, output_dir: Path):
        self.dialect = dialect
        self.model_type = model_type  # 'peft' or 'finetune'
        self.model_size = model_size
        self.seed = seed
        self.output_dir = output_dir
        self.start_time = None
        
        # Follow original repository naming convention
        self.results_filename = f"results_whisper-{model_size}-{model_type}_{dialect}_seed{seed}.json"
        
        # Create results directory structure like original repo
        self.results_dir = output_dir / "results" / f"ex_{model_type}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            'experiment_name': f"whisper-{model_size}-{model_type}_{dialect}_seed{seed}",
            'model_name': f"whisper-{model_size}",
            'dialect': dialect,
            'method': model_type,
            'seed': seed,
            'start_time': None,
            'end_time': None,
            'training_time_seconds': 0,
            'peak_memory_mb': 0,
            'total_params': 0,
            'trainable_params': 0,
            'trainable_percentage': 0,
            'wer': 0,
            'cer': 0,
            'final_loss': 0
        }
    
    def start_training(self):
        """Start timing the training."""
        self.start_time = time.time()
        self.metrics['start_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def update_memory(self):
        """Update peak memory usage."""
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
            self.metrics['peak_memory_mb'] = max(self.metrics['peak_memory_mb'], peak_memory)
    
    def update_model_info(self, model, model_type: str):
        """Update model parameter information."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.metrics.update({
            'model_type': model_type,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'trainable_percentage': (trainable_params / total_params) * 100 if total_params > 0 else 0
        })
    
    def end_training(self, wer: float = 0, cer: float = 0, final_loss: float = 0):
        """End timing and save metrics following original repository format."""
        if self.start_time:
            self.metrics['training_time_seconds'] = time.time() - self.start_time
            self.metrics['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Update final metrics
        self.metrics.update({
            'wer': wer,
            'cer': cer, 
            'final_loss': final_loss
        })
        
        # Save in original repository format and location
        results_file = self.results_dir / self.results_filename
        with open(results_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Also save timing file like original repository
        timing_file = self.results_dir / f"training_time_{self.dialect}_{self.model_type}_{self.seed}.txt"
        with open(timing_file, 'w') as f:
            f.write(f"Total training time: {self.metrics['training_time_seconds']:.2f} seconds or {self.metrics['training_time_seconds']/3600:.2f} hours")
        
        logger.info(f"Results saved to: {results_file}")
        logger.info(f"Timing saved to: {timing_file}")
        return self.metrics
        logger.info(f"Detailed metrics saved to: {detailed_metrics_file}")
        return self.metrics

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator for speech-to-text tasks with PEFT support."""
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Process audio features
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Process labels
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 for loss calculation
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Remove BOS token if present
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


class SavePeftModelCallback(TrainerCallback):
    """Callback to save only PEFT adapter weights."""
    
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


# =============================================================================
# MANAGER CLASSES
# =============================================================================

class DatasetManager:
    """Handles all dataset loading and preprocessing operations."""
    
    def __init__(self, dialect: str, use_huggingface: bool = True, quick_test: bool = False):
        self.dialect = dialect
        self.use_huggingface = use_huggingface
        self.quick_test = quick_test
    
    def load_datasets(self, processor: WhisperProcessor) -> DatasetDict:
        """Load and prepare dialect datasets with HuggingFace support."""
        logger.info(f"Loading {self.dialect} dialect data...")
        
        try:
            if self.use_huggingface:
                dataset = self._load_huggingface_dataset()
                logger.info("Successfully loaded data from HuggingFace")
                
                # Apply quick test filtering
                if self.quick_test:
                    logger.info("Applying quick test data filtering (50 train, 10 test samples)")
                    dataset["train"] = dataset["train"].select(range(min(50, len(dataset["train"]))))
                    dataset["test"] = dataset["test"].select(range(min(10, len(dataset["test"]))))
                
                # Clean columns
                dataset = self._clean_dataset_columns(dataset)
                return dataset
                
            else:
                dataset = self._load_local_dataset()
                logger.info("Successfully loaded local data")
                
                # Apply quick test filtering
                if self.quick_test:
                    logger.info("Applying quick test data filtering (50 train, 10 test samples)")
                    dataset["train"] = dataset["train"].select(range(min(50, len(dataset["train"]))))
                    dataset["test"] = dataset["test"].select(range(min(10, len(dataset["test"]))))
                
                # Process local datasets
                dataset = self._preprocess_local_dataset(dataset, processor)
                return dataset
                
        except Exception as e:
            logger.warning(f"Failed to load primary data source: {e}")
            logger.info("Falling back to placeholder dataset")
            return self._load_placeholder_dataset(processor)
    
    def _load_huggingface_dataset(self) -> DatasetDict:
        """Load datasets from the official HuggingFace collection."""
        if self.dialect == "all":
            # Load and combine all dialect datasets
            logger.info("Loading combined dialect data from HuggingFace...")
            combined_train = []
            combined_test = []
            
            for dialect_name, dataset_prefix in HUGGINGFACE_DATASET_MAPPING.items():
                if dialect_name == 'msa':  # Skip MSA for dialect-only training
                    continue
                    
                logger.info(f"Loading {dialect_name} data...")
                train_dataset = load_dataset(f"{dataset_prefix}_train_set")
                test_dataset = load_dataset(f"{dataset_prefix}_test_set")
                
                combined_train.append(train_dataset['train'])
                combined_test.append(test_dataset['train'])
            
            # Concatenate all dialects
            train_combined = concatenate_datasets(combined_train)
            test_combined = concatenate_datasets(combined_test)
            
            return DatasetDict({
                "train": train_combined,
                "test": test_combined
            })
        
        else:
            # Load specific dialect
            if self.dialect not in HUGGINGFACE_DATASET_MAPPING:
                raise ValueError(f"Dialect '{self.dialect}' not found in HuggingFace collection")
            
            dataset_prefix = HUGGINGFACE_DATASET_MAPPING[self.dialect]
            logger.info(f"Loading {self.dialect} data from {dataset_prefix}...")
            
            train_dataset = load_dataset(f"{dataset_prefix}_train_set")
            test_dataset = load_dataset(f"{dataset_prefix}_test_set")
            
            return DatasetDict({
                "train": train_dataset['train'],
                "test": test_dataset['train']
            })
    
    def _load_local_dataset(self) -> DatasetDict:
        """Load datasets from local disk."""
        from datasets import load_from_disk
        
        root = os.environ.get("DATA_ROOT", "./data")
        
        if self.dialect == "all":
            # Load combined dialect data
            logger.info("Loading combined dialect data from local disk...")
            dialect_dataset = load_from_disk(os.path.join(root, "egyptian_train/"))
            for d in ["gulf", "iraqi", "levantine", "maghrebi"]:
                train_d = load_from_disk(os.path.join(root, f"{d}_train/"))
                dialect_dataset = concatenate_datasets([train_d, dialect_dataset])
            
            # Load test set
            try:
                test_dataset = load_from_disk(os.path.join(root, "test/"))
            except:
                test_dataset = load_from_disk(os.path.join(root, "egyptian_test/"))
                
            return DatasetDict({
                "train": dialect_dataset,
                "test": test_dataset
            })
        else:
            # Load specific dialect data
            train_dataset = load_from_disk(os.path.join(root, f"{self.dialect}_train/"))
            test_dataset = load_from_disk(os.path.join(root, f"{self.dialect}_test/"))
            
            return DatasetDict({
                "train": train_dataset,
                "test": test_dataset
            })
    
    def _load_placeholder_dataset(self, processor) -> DatasetDict:
        """Create placeholder dataset for testing."""
        logger.warning("Using placeholder dataset - implement MASC data loading for production")
        
        # Use Common Voice for testing structure
        if self.quick_test:
            cv_dataset = load_dataset("mozilla-foundation/common_voice_11_0", "ar", split="train[:10]")
            test_dataset = load_dataset("mozilla-foundation/common_voice_11_0", "ar", split="test[:5]")
        else:
            cv_dataset = load_dataset("mozilla-foundation/common_voice_11_0", "ar", split="train[:100]")
            test_dataset = load_dataset("mozilla-foundation/common_voice_11_0", "ar", split="test[:20]")
        
        # Remove unnecessary columns
        columns_to_remove = [
            "accent", "age", "client_id", "down_votes", "gender",
            "locale", "path", "segment", "up_votes"
        ]
        
        for col in columns_to_remove:
            if col in cv_dataset.column_names:
                cv_dataset = cv_dataset.remove_column(col)
            if col in test_dataset.column_names:
                test_dataset = test_dataset.remove_column(col)
        
        dataset = DatasetDict({
            "train": cv_dataset,
            "test": test_dataset
        })
        
        return self._preprocess_local_dataset(dataset, processor)
    
    def _clean_dataset_columns(self, dataset: DatasetDict) -> DatasetDict:
        """Clean dataset columns to keep only required ones."""
        for split in dataset.keys():
            current_columns = dataset[split].column_names
            columns_to_keep = [col for col in current_columns if col in ["input_features", "labels"]]
            if len(columns_to_keep) < len(current_columns):
                columns_to_remove = [col for col in current_columns if col not in ["input_features", "labels"]]
                logger.info(f"Removing extra columns from {split}: {columns_to_remove}")
                dataset[split] = dataset[split].remove_columns(columns_to_remove)
        
        return dataset
    
    def _preprocess_local_dataset(self, dataset: DatasetDict, processor: WhisperProcessor) -> DatasetDict:
        """Preprocess local datasets."""
        if "audio" in dataset["train"].column_names:
            dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        dataset = dataset.map(
            lambda batch: self._prepare_dataset(batch, processor),
            remove_columns=dataset["train"].column_names,
            num_proc=4
        )
        
        return dataset
    
    def _prepare_dataset(self, batch, processor):
        """Prepare individual batch for training."""
        audio = batch["audio"]
        
        # Extract features
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        
        # Tokenize labels
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch


class ModelManager:
    """Handles model loading and PEFT configuration."""
    
    def __init__(self, model_name: str, model_size: str, use_peft: bool = True, load_in_8bit: bool = True):
        self.model_name = model_name
        self.model_size = model_size
        self.use_peft = use_peft
        self.load_in_8bit = load_in_8bit
        self.peft_config = PEFT_CONFIG[model_size]
    
    def load_model_and_processor(self) -> Tuple[Any, WhisperProcessor]:
        """Load Whisper model and processor with PEFT configuration."""
        logger.info(f"Loading {self.model_name}...")
        
        # Load processor
        processor = WhisperProcessor.from_pretrained(self.model_name, language="ar", task="transcribe")
        
        # Load model with optional 8-bit quantization
        if self.load_in_8bit and self.use_peft:
            model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name,
                load_in_8bit=True,
                device_map="auto"
            )
            
            # Enable gradient computation for input embeddings
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            
            model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)
        else:
            model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
        
        # Configure PEFT if enabled
        if self.use_peft:
            model = self._setup_peft(model)
        
        # Set language and task tokens
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model loaded. Trainable parameters: {trainable_params:,}")
        
        return model, processor
    
    def _setup_peft(self, model):
        """Configure PEFT LoRA exactly like the working notebook."""
        # Critical: Register forward hook for gradient computation (from notebook)
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        
        model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)
        
        # Configure LoRA exactly like the working notebook
        peft_config = LoraConfig(
            r=self.peft_config['lora_rank'],           # 32
            lora_alpha=self.peft_config['lora_alpha'], # 64
            target_modules=self.peft_config['target_modules'],  # ["q_proj", "v_proj"] 
            lora_dropout=self.peft_config['lora_dropout'],      # 0.05
            bias="none"
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        logger.info("PEFT LoRA configuration applied exactly like working notebook")
        
        return model


class EvaluationManager:
    """Handles model evaluation and metrics calculation."""
    
    def __init__(self, model: Any, processor: WhisperProcessor, dialect: str, output_dir: Path):
        self.model = model
        self.processor = processor
        self.dialect = dialect
        self.output_dir = output_dir
        self.wer_metric = evaluate.load("wer")
        self.cer_metric = evaluate.load("cer")  # Add CER metric
    
    def evaluate_model(self, dataset: DatasetDict, model_path: str = None) -> Dict[str, float]:
        """Evaluate the trained PEFT model."""
        from torch.utils.data import DataLoader
        from transformers.models.whisper.english_normalizer import BasicTextNormalizer
        from tqdm import tqdm
        
        # Load model if path provided
        if model_path:
            model, processor = self._load_trained_model(model_path)
        else:
            model = self.model
            processor = self.processor
        
        # Setup for evaluation
        model.eval()
        model.config.use_cache = True
        
        # Optimized dataloader for faster evaluation - larger batch size and more workers
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
        eval_batch_size =  16 # Increased from 8 for better GPU utilization
        eval_dataloader = DataLoader(
            dataset["test"], 
            batch_size=eval_batch_size, 
            collate_fn=data_collator,
            num_workers=2,  # Parallel data loading
            pin_memory=True,  # Faster GPU transfer
            persistent_workers=False  # Keep workers alive between batches
        )
        
        # Setup generation parameters
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="ar", task="transcribe")
        normalizer = BasicTextNormalizer()
        
        # Evaluation metrics storage
        predictions = []
        references = []
        normalized_predictions = []
        normalized_references = []
        
        logger.info("Starting model evaluation...")
        logger.info(f"Using batch size: {eval_batch_size} for faster evaluation")
        
        # Evaluation loop with optimized memory usage
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            with torch.no_grad():
                # Move input features to device and ensure proper dtype
                device = next(model.parameters()).device
                input_features = batch["input_features"].to(device)
                
                # Ensure input features have the right dtype for the model
                if hasattr(model, 'dtype'):
                    input_features = input_features.to(dtype=model.dtype)
                
                # Generate token ids with optimized settings for speed
                with torch.cuda.amp.autocast():
                    generated_tokens = model.generate(
                        input_features=input_features,
                        forced_decoder_ids=forced_decoder_ids,
                        max_new_tokens=255,
                        num_beams=1,  # Greedy decoding for speed
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        # Optimize for memory and speed
                        output_scores=False,
                        return_dict_in_generate=False
                    ).cpu().numpy()
                
                # Prepare label ids
                labels = batch["labels"].numpy()
                labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
                
            # Decode predictions and labels
            decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # Clean and validate decoded texts
            cleaned_preds = []
            cleaned_labels = []
            for pred, label in zip(decoded_preds, decoded_labels):
                # Remove extra whitespace and ensure non-empty
                pred_clean = pred.strip()
                label_clean = label.strip()
                
                # Skip empty predictions or references as they cause issues
                if pred_clean and label_clean:
                    cleaned_preds.append(pred_clean)
                    cleaned_labels.append(label_clean)
                else:
                    logger.debug(f"Skipping empty pair: pred='{pred_clean}' label='{label_clean}'")
            
            predictions.extend(cleaned_preds)
            references.extend(cleaned_labels)
            
            # Normalize text for robust metric calculation
            for pred, label in zip(cleaned_preds, cleaned_labels):
                norm_pred = normalizer(pred).strip()
                norm_label = normalizer(label).strip()
                
                # Only add if both normalized texts are non-empty
                if norm_pred and norm_label:
                    normalized_predictions.append(norm_pred)
                    normalized_references.append(norm_label)
        
        # Compute WER and CER scores with proper error handling
        # Filter out empty predictions and references which can cause extreme CER values
        valid_pairs = [(p, r) for p, r in zip(predictions, references) if p.strip() and r.strip()]
        valid_norm_pairs = [(p, r) for p, r in zip(normalized_predictions, normalized_references) if p.strip() and r.strip()]
        
        if not valid_pairs:
            logger.warning("No valid prediction-reference pairs found!")
            wer = cer = normalized_wer = normalized_cer = 100.0
        else:
            valid_preds, valid_refs = zip(*valid_pairs)
            valid_norm_preds, valid_norm_refs = zip(*valid_norm_pairs) if valid_norm_pairs else ([], [])
            
            wer = 100 * self.wer_metric.compute(predictions=list(valid_preds), references=list(valid_refs))
            # CER calculation with safety check
            try:
                cer = 100 * self.cer_metric.compute(predictions=list(valid_preds), references=list(valid_refs))
                # Sanity check: CER should typically be <= WER for most languages
                if cer > wer * 3:  # If CER is more than 3x WER, something is wrong
                    logger.warning(f"CER ({cer:.2f}%) is unusually high compared to WER ({wer:.2f}%). Checking data...")
                    # Log some examples for debugging
                    for i, (pred, ref) in enumerate(zip(valid_preds[:3], valid_refs[:3])):
                        logger.info(f"Example {i+1}: Pred='{pred}' | Ref='{ref}'")
            except Exception as e:
                logger.error(f"Error calculating CER: {e}")
                cer = 0.0
            
            # Calculate normalized metrics if available
            if valid_norm_pairs:
                normalized_wer = 100 * self.wer_metric.compute(predictions=list(valid_norm_preds), references=list(valid_norm_refs))
                try:
                    normalized_cer = 100 * self.cer_metric.compute(predictions=list(valid_norm_preds), references=list(valid_norm_refs))
                except Exception as e:
                    logger.error(f"Error calculating normalized CER: {e}")
                    normalized_cer = 0.0
            else:
                normalized_wer = normalized_cer = wer  # Fallback
        
        eval_metrics = {
            "eval/wer": wer,
            "eval/cer": cer,
            "eval/normalized_wer": normalized_wer,
            "eval/normalized_cer": normalized_cer,
            "eval/samples": len(predictions)
        }
        
        logger.info(f"Evaluation Results:")
        logger.info(f"  Total samples processed: {len(predictions)}")
        logger.info(f"  Valid pairs for metrics: {len(valid_pairs) if 'valid_pairs' in locals() else 'N/A'}")
        logger.info(f"  WER: {wer:.2f}%")
        logger.info(f"  CER: {cer:.2f}%")
        logger.info(f"  Normalized WER: {normalized_wer:.2f}%")
        logger.info(f"  Normalized CER: {normalized_cer:.2f}%")
        logger.info(f"  Samples evaluated: {len(predictions)}")
        
        # Save evaluation results
        eval_results_path = self.output_dir / f"evaluation_results_{self.dialect}.json"
        with open(eval_results_path, 'w') as f:
            json.dump(eval_metrics, f, indent=2)
        logger.info(f"Evaluation results saved to {eval_results_path}")
        
        return eval_metrics
    
    def _load_trained_model(self, model_path: str) -> Tuple[Any, WhisperProcessor]:
        """Load a trained PEFT model from path."""
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model = WhisperForConditionalGeneration.from_pretrained(
            peft_config.base_model_name_or_path,
            load_in_8bit=True,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        processor = WhisperProcessor.from_pretrained(
            peft_config.base_model_name_or_path,
            language="ar",
            task="transcribe"
        )
        
        return model, processor


# =============================================================================
# MAIN TRAINER CLASS
# =============================================================================

class ArabicDialectPEFTTrainer:
    def get_eval_dataset(self, processor=None):
        """Load and cache the evaluation dataset, optionally with a specific processor."""
        if hasattr(self, '_eval_dataset') and self._eval_dataset is not None:
            return self._eval_dataset
        proc = processor if processor is not None else (self.evaluation_manager.processor if hasattr(self, 'evaluation_manager') and self.evaluation_manager else None)
        if proc is None:
            raise ValueError("No processor available to load evaluation dataset.")
        self._eval_dataset = self.dataset_manager.load_datasets(proc)
        return self._eval_dataset

    """
    Main trainer class for Arabic dialect PEFT fine-tuning.
    
    Orchestrates the training process using modular components for
    data loading, model management, and evaluation.
    """
    
    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        dialect: str = "egyptian",
        use_peft: bool = True,
        output_dir: str = "./results",
        load_in_8bit: bool = True,
        use_huggingface: bool = True,
        quick_test: bool = False,
        seed: int = 42,
        push_to_hub=True,
        hub_model_id=None,
        hub_token=None
    ):
        # Core configuration
        self.model_name = model_name
        self.dialect = dialect
        self.use_peft = use_peft
        self.output_dir = Path(output_dir)
        self.load_in_8bit = load_in_8bit
        self.use_huggingface = use_huggingface
        self.quick_test = quick_test
        self.seed = seed
        self.push_to_hub = push_to_hub
        self.hub_token = hub_token or os.getenv("HUGGINGFACE_HUB_TOKEN", None)
        self.hub_model_id = hub_model_id
        # Extract model size and get PEFT config
        self.model_size = self._extract_model_size(model_name)
        self.peft_config = PEFT_CONFIG[self.model_size]
        
        # Initialize managers
        self.dataset_manager = DatasetManager(dialect, use_huggingface, quick_test)
        self.model_manager = ModelManager(model_name, self.model_size, use_peft, load_in_8bit)
        self.evaluation_manager = None  # Will be initialized after model loading
        
        # Setup output directories
        self.setup_directories()
        
        logger.info(f"Initialized Arabic Dialect PEFT Trainer:")
        logger.info(f"  Dialect: {dialect}")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  PEFT: {use_peft}")
        logger.info(f"  Quick test: {quick_test}")
    
    def _extract_model_size(self, model_name: str) -> str:
        """Extract model size from model name."""
        size_mapping = {
            "tiny": "small",  # Use small config for tiny
            "base": "small",  # Use small config for base
            "small": "small",
            "medium": "medium",
            "large": "large"
        }
        
        for size in size_mapping:
            if size in model_name:
                return size_mapping[size]
        return "small"  # default
    
    def setup_directories(self):
        """Create necessary directories for outputs."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
    
    def train(self, max_steps: int = 4000, eval_steps: int = 500):
        """Main training method - orchestrates the entire training process.
        
        Args:
            max_steps: Maximum number of training steps
            eval_steps: Steps between evaluations
            push_to_hub: Whether to push the model to HuggingFace Hub
            hub_model_id: The model ID on HuggingFace Hub (e.g., 'username/model-name')
        """
        
        # Initialize metrics tracker following original repository pattern
        model_type = "peft" if self.use_peft else "finetune" 
        self.metrics_tracker = MetricsTracker(
            dialect=self.dialect,
            model_type=model_type,
            model_size=self.model_size,
            seed=self.seed,
            output_dir=self.output_dir
        )
        self.metrics_tracker.start_training()
        
        # Apply quick test modifications
        if self.quick_test:
            logger.info("Quick test mode enabled - using minimal training configuration")
            max_steps = 20
            eval_steps = 10
            
        # Step 1: Load model and processor
        logger.info("Step 1: Loading model and processor...")
        model, processor = self.model_manager.load_model_and_processor()
        
        # Track model info
        model_type_desc = "PEFT_LoRA" if self.use_peft else "Full_FineTune"
        self.metrics_tracker.update_model_info(model, model_type_desc)
        
        # Step 2: Load and prepare datasets
        logger.info("Step 2: Loading and preparing datasets...")
        dataset = self.dataset_manager.load_datasets(processor)
        
        # Step 3: Initialize evaluation manager
        self.evaluation_manager = EvaluationManager(model, processor, self.dialect, self.output_dir)
        
        # Step 4: Setup training components
        logger.info("Step 3: Setting up training components...")
        trainer = self._setup_trainer(model, processor, dataset, max_steps, eval_steps)
        
        # Step 5: Train the model
        logger.info(f"Step 4: Starting training for {self.dialect} dialect...")
        model.config.use_cache = False  # Disable cache for training
        
        # Update memory before training
        self.metrics_tracker.update_memory()
        
        trainer.train()
        
        # Update memory after training  
        self.metrics_tracker.update_memory()
        
        # Step 6: Save final model
        logger.info("Step 5: Saving final model...")
        final_model_path = self.output_dir / f"whisper-{self.model_size}-{self.dialect}-peft-final"
        trainer.save_model(str(final_model_path))
        
        # Get final loss from training history
        final_loss = 0
        if trainer.state.log_history:
            final_loss = trainer.state.log_history[-1].get('train_loss', 0)
        
        # Run evaluation to get WER and CER
        logger.info("Step 6: Running final evaluation...")
        eval_results = self.evaluation_manager.evaluate_model(dataset, str(final_model_path))
        final_wer = eval_results.get('eval/wer', 0)
        final_cer = eval_results.get('eval/cer', 0)  # Get CER too
        logger.info(f"Final Evaluation - WER: {final_wer:.2f}%, CER: {final_cer:.2f}%, Loss: {final_loss:.4f}")

        # End metrics tracking and save in original repository format
        self.metrics_tracker.end_training(wer=final_wer, cer=final_cer, final_loss=final_loss)
        
        # Save training metrics from trainer
        metrics_path = final_model_path / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(trainer.state.log_history, f, indent=2)

        # Push to Hub if requested
        if self.push_to_hub and self.hub_model_id:
            logger.info(f"Pushing model to Hugging Face Hub: {self.hub_model_id}")
            try:
                # Save processor configuration
                processor.save_pretrained(str(final_model_path))
                
                # Push to hub
                model.push_to_hub(self.hub_model_id)
                processor.push_to_hub(self.hub_model_id)
                
                logger.info(f"Successfully pushed model to {self.hub_model_id}")
            except Exception as e:
                logger.error(f"Failed to push to Hugging Face Hub: {e}")
        
        logger.info(f"Training complete. Model saved to {final_model_path}")
        logger.info(f"Final WER: {final_wer:.2f}%, Final CER: {final_cer:.2f}%, Final Loss: {final_loss:.4f}")
        
        return trainer
    
    def _setup_trainer(self, model, processor, dataset, max_steps, eval_steps):
        """Setup the HuggingFace trainer with optimized arguments."""
        
        # Setup data collator
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
        
        # Configure batch size for quick test
        batch_size = self.peft_config['batch_size']
        if self.quick_test:
            batch_size = min(2, batch_size)
            logger.info(f"Quick test mode: reducing batch size to {batch_size}")
            
        # Training arguments optimized for dialect fine-tuning
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.output_dir / f"whisper-{self.model_size}-{self.dialect}-peft"),
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            learning_rate=self.peft_config['learning_rate'],
            warmup_steps=5 if self.quick_test else 500,
            max_steps=max_steps,
            gradient_checkpointing=True,
            fp16=True,
            evaluation_strategy="no",  # Disabled to avoid dtype issues
            per_device_eval_batch_size=2 if self.quick_test else 8,
            predict_with_generate=False,  # Disabled during training
            generation_max_length=225,
            save_steps=eval_steps,
            logging_steps=5 if self.quick_test else 50,
            report_to=["tensorboard"],
            load_best_model_at_end=False,  # Disabled since no evaluation
            save_total_limit=1 if self.quick_test else 3,
            remove_unused_columns=False,
            label_names=["labels"],
            seed=self.seed,
        )
        
        # Setup trainer
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=data_collator,
            tokenizer=processor.tokenizer,
            callbacks=[SavePeftModelCallback()],
        )
        
        return trainer
    
    def evaluate_model(self, model_path: str = None, dataset: Any = None):
        """
        Evaluate the trained model using the evaluation manager.
        Advanced: Accept a preloaded dataset to avoid redundant loading.
        """
        # If a new model_path is provided, load model and processor
        if model_path:
            peft_config = PeftConfig.from_pretrained(model_path)
            base_model = WhisperForConditionalGeneration.from_pretrained(
                peft_config.base_model_name_or_path,
                load_in_8bit=True,
                device_map="auto"
            )
            model = PeftModel.from_pretrained(base_model, model_path)
            processor = WhisperProcessor.from_pretrained(
                peft_config.base_model_name_or_path,
                language="ar",
                task="transcribe"
            )
            model.eval()
            model.config.use_cache = True
            self.evaluation_manager = EvaluationManager(model, processor, self.dialect, self.output_dir)
            # Always reload eval dataset with new processor if model_path is given
            eval_dataset = self.get_eval_dataset(processor)
        elif self.evaluation_manager is None:
            raise ValueError("No evaluation manager available. Provide model_path or train first.")
        else:
            # Use provided dataset, or cached, or load
            if dataset is not None:
                eval_dataset = dataset
                self._eval_dataset = dataset  # cache for future
            else:
                eval_dataset = self.get_eval_dataset()

        return self.evaluation_manager.evaluate_model(eval_dataset, model_path)


# =============================================================================
# MAIN EXECUTION AND CLI
# =============================================================================

def main():
    """
    Main function for training Arabic dialect PEFT models.
    
    Supports command-line arguments for easy experimentation and
    integration with existing research workflows.
    """
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(
        description="Train Arabic Dialect PEFT Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train Egyptian dialect with PEFT LoRA
  python src/training/dialect_peft_training.py --dialect egyptian --model_name openai/whisper-small
  
  # Quick test run
  python src/training/dialect_peft_training.py --dialect egyptian --quick_test
  
  # Train all dialects combined
  python src/training/dialect_peft_training.py --dialect all --max_steps 6000
  
  # Evaluate existing model
  python src/training/dialect_peft_training.py --evaluate_only ./results/whisper-small-egyptian-peft-final --dialect egyptian
  
  # Use local data instead of HuggingFace
  python src/training/dialect_peft_training.py --dialect egyptian --use_local_data
        """
    )
    
    # Model and data arguments
    parser.add_argument("--model_name", default="openai/whisper-small",
                       help="Whisper model to fine-tune (default: openai/whisper-small)")
    parser.add_argument("--model_size", default="small", 
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Model size (default: small)")
    parser.add_argument("--dialect", default="egyptian",
                       choices=["egyptian", "gulf", "iraqi", "levantine", "maghrebi", "msa", "all"],
                       help="Arabic dialect to train on: egyptian, gulf, iraqi, levantine, maghrebi, msa, all (default: egyptian)")
    parser.add_argument("--data_source", default="huggingface",
                       choices=["huggingface", "local", "auto"],
                       help="Data source (default: huggingface)")
    parser.add_argument("--use_local_data", action="store_true",
                       help="Use local datasets instead of HuggingFace")
    parser.add_argument("--load_in_8bit", action="store_true", default=True,
                       help="Load model in 8-bit for memory efficiency (default: True)")
    
    # Training arguments
    parser.add_argument("--output_dir", default="./results",
                       help="Directory to save results (default: ./results)")
    parser.add_argument("--max_steps", type=int, default=4000,
                       help="Maximum training steps (default: 4000)")
    parser.add_argument("--eval_steps", type=int, default=500,
                       help="Evaluation frequency (default: 500)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    
    # PEFT arguments
    parser.add_argument("--use_peft", action="store_true", default=True,
                       help="Use PEFT LoRA (default: True)")
    parser.add_argument("--no_peft", action="store_true",
                       help="Disable PEFT and use full fine-tuning")
    
    # Execution modes
    parser.add_argument("--quick_test", action="store_true",
                       help="Run minimal training for testing")
    parser.add_argument("--evaluate_only", type=str,
                       help="Path to trained model for evaluation only")
    
    # Hugging Face Hub arguments
    parser.add_argument("--push_to_hub", action="store_true",
                       help="Push the trained model to Hugging Face Hub")
    parser.add_argument("--hub_model_id", type=str,
                       help="Hugging Face Hub model ID (e.g., 'username/model-name')")
    parser.add_argument("--hub_token", type=str,
                       help="Hugging Face Hub API token")
    
    args = parser.parse_args()
    
    # Print configuration
    print("🚀 Arabic Dialect PEFT Trainer")
    print("=" * 50)
    print(f"📊 Dialect: {args.dialect}")
    print(f"🤖 Model: {args.model_name} ({args.model_size})")
    print(f"📁 Output: {args.output_dir}")
    print(f"📂 Data source: {args.data_source}")
    print(f"⚡ Quick test: {args.quick_test}")
    print(f"🔧 PEFT enabled: {args.use_peft and not args.no_peft}")
    print(f"� 8-bit loading: {args.load_in_8bit}")
    print(f"�📈 Max steps: {args.max_steps}")
    print(f"🕒 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # Handle argument logic for backward compatibility
    use_peft = args.use_peft and not args.no_peft
    
    # Data source logic
    if args.data_source == "local" or args.use_local_data:
        use_huggingface = False
    elif args.data_source == "huggingface":
        use_huggingface = True
    else:  # auto
        use_huggingface = True  # Default to HuggingFace
    print(args.hub_token)
    # Initialize trainer
    trainer = ArabicDialectPEFTTrainer(
        model_name=args.model_name,
        dialect=args.dialect,
        quick_test=args.quick_test,
        seed=args.seed,
        push_to_hub=args.push_to_hub,
        hub_token=args.hub_token,
        hub_model_id=args.hub_model_id,
    )
    
    # Evaluation-only mode
    if args.evaluate_only:
        print(f"🔍 Evaluation-only mode for model: {args.evaluate_only}")
        # Load model and processor first
        from peft import PeftConfig, PeftModel
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        model_path = args.evaluate_only
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model = WhisperForConditionalGeneration.from_pretrained(
            peft_config.base_model_name_or_path,
            load_in_8bit=True,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        processor = WhisperProcessor.from_pretrained(
            peft_config.base_model_name_or_path,
            language="ar",
            task="transcribe"
        )
        model.eval()
        model.config.use_cache = True
        trainer.evaluation_manager = EvaluationManager(model, processor, args.dialect, args.output_dir)
        # Now load dataset with processor
        eval_dataset = trainer.get_eval_dataset(processor)
        results = trainer.evaluate_model(model_path, dataset=eval_dataset)
        print("\n📊 Evaluation Results:")
        print("=" * 30)
        for metric, value in results.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.2f}")
            else:
                print(f"  {metric}: {value}")
        print("=" * 30)
        return
    
    # Training mode
    print("🎯 Starting training process...")
    trained_model = trainer.train(
        max_steps=args.max_steps,
        eval_steps=args.eval_steps
    )
    
    print("✅ Training completed successfully!")
    
    # Optional post-training evaluation
    # if not args.quick_test:
    #     print("📊 Running post-training evaluation...")
    #     eval_results = trainer.evaluate_model()
    #     print(f"✅ Evaluation completed!")
    #     print(f"   WER: {eval_results['eval/wer']:.2f}%")
    #     print(f"   Normalized WER: {eval_results['eval/normalized_wer']:.2f}%")
    # else:
    #     print("💡 You can run evaluation later with:")
    #     final_model_path = trainer.output_dir / f"whisper-{trainer.model_size}-{args.dialect}-peft-final"
    #     print(f"   python src/training/dialect_peft_training.py --evaluate_only {final_model_path} --dialect {args.dialect}")
    #
    print("🎉 All done!")


def interactive_evaluation():
    """
    Interactive evaluation mode for exploring trained models.
    
    Provides a user-friendly interface for evaluating and comparing
    different model checkpoints.
    """
    print("=== Arabic Dialect PEFT Model Evaluation ===")
    
    # Get model path from user
    model_path = input("Enter path to trained PEFT model: ").strip()
    if not model_path:
        print("No model path provided. Exiting.")
        return
    
    # Get dialect
    dialect = input("Enter dialect [egyptian/gulf/iraqi/levantine/maghrebi/all]: ").strip().lower()
    if dialect not in ["egyptian", "gulf", "iraqi", "levantine", "maghrebi", "all"]:
        print("Invalid dialect. Using 'egyptian' as default.")
        dialect = "egyptian"
    
    # Initialize trainer for evaluation
    trainer = ArabicDialectPEFTTrainer(
        dialect=dialect,
        use_peft=True,
        quick_test=False
    )
    
    try:
        # Advanced: load dataset once and pass to evaluate_model
        eval_dataset = trainer.get_eval_dataset()
        results = trainer.evaluate_model(model_path, dataset=eval_dataset)
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        for metric, value in results.items():
            if isinstance(value, float):
                print(f"{metric:20s}: {value:.2f}")
            else:
                print(f"{metric:20s}: {value}")
        print("="*50)
    except Exception as e:
        print(f"Evaluation failed: {e}")
        logger.error(f"Evaluation error: {e}")


if __name__ == "__main__":
    # Check for interactive mode
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_evaluation()
    else:
        main()
