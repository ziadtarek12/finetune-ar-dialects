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

# Default test split size for custom datasets without train/test splits
DEFAULT_TEST_SPLIT_SIZE = 0.2


# =============================================================================
# UTILITY CLASSES
# =============================================================================

class MetricsTracker:
    """Enhanced metrics tracking with comprehensive LORA-specific analysis."""
    
    def __init__(self, dialect: str, model_type: str, model_size: str, seed: int, output_dir: Path):
        self.dialect = dialect
        self.model_type = model_type  # 'peft' or 'finetune'
        self.model_size = model_size
        self.seed = seed
        self.output_dir = output_dir
        self.start_time = None
        self.step_metrics = []  # Store per-step metrics
        self.layer_metrics = {}  # Store layer-wise metrics
        self.gradient_history = []  # Store gradient statistics
        
        # Follow original repository naming convention
        self.results_filename = f"results_whisper-{model_size}-{model_type}_{dialect}_seed{seed}.json"
        self.detailed_filename = f"detailed_metrics_whisper-{model_size}-{model_type}_{dialect}_seed{seed}.json"
        
        # Create results directory structure like original repo
        self.results_dir = output_dir / "results" / f"ex_{model_type}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create detailed metrics subdirectory
        self.detailed_dir = self.results_dir / "detailed"
        self.detailed_dir.mkdir(parents=True, exist_ok=True)
        
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
            'final_loss': 0,
            
            # Enhanced LORA-specific metrics
            'lora_config': {},
            'parameter_efficiency_ratio': 0,
            'memory_efficiency_ratio': 0,
            'training_efficiency_score': 0,
            'convergence_step': 0,
            'gradient_norm_stats': {},
            'layer_adaptation_stats': {},
            'performance_per_param': 0,
            'lora_rank': 0,
            'lora_alpha': 0,
            'lora_dropout': 0,
            'target_modules_count': 0,
            'adapter_weights_norm': 0,
            'base_model_frozen_params': 0,
            'effective_rank': 0,
            'adaptation_magnitude': 0
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
    
    def update_model_info(self, model, model_type: str, lora_config=None):
        """Update comprehensive model parameter information including LORA specifics."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Basic metrics
        self.metrics.update({
            'model_type': model_type,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'trainable_percentage': (trainable_params / total_params) * 100 if total_params > 0 else 0
        })
        
        # LORA-specific metrics
        if lora_config and model_type == 'peft':
            self.metrics.update({
                'lora_config': {
                    'r': getattr(lora_config, 'r', 0),
                    'lora_alpha': getattr(lora_config, 'lora_alpha', 0),
                    'lora_dropout': getattr(lora_config, 'lora_dropout', 0),
                    'target_modules': getattr(lora_config, 'target_modules', [])
                },
                'lora_rank': getattr(lora_config, 'r', 0),
                'lora_alpha': getattr(lora_config, 'lora_alpha', 0),
                'lora_dropout': getattr(lora_config, 'lora_dropout', 0),
                'target_modules_count': len(getattr(lora_config, 'target_modules', [])),
                'parameter_efficiency_ratio': trainable_params / total_params,
                'base_model_frozen_params': total_params - trainable_params
            })
            
            # Calculate adapter-specific metrics
            adapter_params = 0
            adapter_norm = 0.0
            
            for name, param in model.named_parameters():
                if 'lora_' in name and param.requires_grad:
                    adapter_params += param.numel()
                    adapter_norm += param.data.norm().item()
            
            self.metrics.update({
                'adapter_params': adapter_params,
                'adapter_weights_norm': adapter_norm,
                'effective_rank': self._calculate_effective_rank(model)
            })
    
    def _calculate_effective_rank(self, model):
        """Calculate effective rank of LORA adapters."""
        try:
            effective_ranks = []
            for name, module in model.named_modules():
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    # Calculate effective rank as approximation
                    A = module.lora_A.default.weight.data
                    B = module.lora_B.default.weight.data
                    combined = torch.mm(B, A)
                    s = torch.linalg.svdvals(combined)
                    # Effective rank using 90% energy criterion
                    cumsum = torch.cumsum(s**2, dim=0)
                    total_energy = cumsum[-1]
                    eff_rank = torch.sum(cumsum < 0.9 * total_energy).item() + 1
                    effective_ranks.append(eff_rank)
            
            return np.mean(effective_ranks) if effective_ranks else 0
        except:
            return 0
    
    def record_step_metrics(self, step: int, loss: float, learning_rate: float, 
                          gradient_norm: float = None, memory_usage: float = None):
        """Record per-step training metrics for detailed analysis."""
        step_data = {
            'step': step,
            'loss': loss,
            'learning_rate': learning_rate,
            'timestamp': time.time() - self.start_time if self.start_time else 0
        }
        
        if gradient_norm is not None:
            step_data['gradient_norm'] = gradient_norm
            
        if memory_usage is not None:
            step_data['memory_usage_mb'] = memory_usage
            
        self.step_metrics.append(step_data)
        
        # Update gradient statistics
        if gradient_norm is not None:
            self.gradient_history.append(gradient_norm)
            if len(self.gradient_history) >= 10:  # Keep rolling statistics
                recent_grads = self.gradient_history[-100:]  # Last 100 steps
                self.metrics['gradient_norm_stats'] = {
                    'mean': np.mean(recent_grads),
                    'std': np.std(recent_grads),
                    'min': np.min(recent_grads),
                    'max': np.max(recent_grads),
                    'current': gradient_norm
                }
    
    def analyze_layer_adaptation(self, model, step: int):
        """Analyze how different layers are adapting during LORA training."""
        if self.model_type != 'peft':
            return
            
        layer_stats = {}
        
        for name, param in model.named_parameters():
            if 'lora_' in name and param.requires_grad:
                layer_key = name.split('.lora_')[0]  # Get base layer name
                
                if layer_key not in layer_stats:
                    layer_stats[layer_key] = {
                        'param_count': 0,
                        'total_norm': 0,
                        'grad_norm': 0,
                        'updates': []
                    }
                
                layer_stats[layer_key]['param_count'] += param.numel()
                layer_stats[layer_key]['total_norm'] += param.data.norm().item()
                
                if param.grad is not None:
                    layer_stats[layer_key]['grad_norm'] += param.grad.data.norm().item()
        
        # Store layer statistics
        self.layer_metrics[step] = layer_stats
        
        # Update running layer adaptation statistics
        if layer_stats:
            self.metrics['layer_adaptation_stats'] = {
                'active_layers': len(layer_stats),
                'total_adaptation_norm': sum(stats['total_norm'] for stats in layer_stats.values()),
                'adaptation_distribution': {k: v['total_norm'] for k, v in layer_stats.items()}
            }
    
    def end_training(self, wer: float = 0, cer: float = 0, final_loss: float = 0, 
                    convergence_step: int = 0):
        """End timing and save comprehensive metrics including efficiency analysis."""
        if self.start_time:
            training_time = time.time() - self.start_time
            self.metrics['training_time_seconds'] = training_time
            self.metrics['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Update final metrics
        self.metrics.update({
            'wer': wer,
            'cer': cer, 
            'final_loss': final_loss,
            'convergence_step': convergence_step
        })
        
        # Calculate efficiency metrics
        self._calculate_efficiency_metrics()
        
        # Save standard results file (original format)
        results_file = self.results_dir / self.results_filename
        with open(results_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save detailed metrics file
        detailed_metrics = {
            'experiment_metadata': self.metrics,
            'step_by_step_metrics': self.step_metrics,
            'layer_adaptation_history': self.layer_metrics,
            'gradient_evolution': self.gradient_history,
            'training_summary': self._generate_training_summary()
        }
        
        detailed_file = self.detailed_dir / self.detailed_filename
        with open(detailed_file, 'w') as f:
            json.dump(detailed_metrics, f, indent=2)
        
        # Save timing file like original repository
        timing_file = self.results_dir / f"training_time_{self.dialect}_{self.model_type}_{self.seed}.txt"
        with open(timing_file, 'w') as f:
            f.write(f"Total training time: {self.metrics['training_time_seconds']:.2f} seconds "
                   f"or {self.metrics['training_time_seconds']/3600:.2f} hours\n")
            f.write(f"Peak memory usage: {self.metrics['peak_memory_mb']:.2f} MB\n")
            f.write(f"Training efficiency score: {self.metrics.get('training_efficiency_score', 0):.4f}\n")
        
        # Generate efficiency report
        self._save_efficiency_report()
        
        logger.info(f"Results saved to: {results_file}")
        logger.info(f"Detailed metrics saved to: {detailed_file}")
        logger.info(f"Timing saved to: {timing_file}")
        return self.metrics
    
    def _calculate_efficiency_metrics(self):
        """Calculate various efficiency metrics for LORA analysis."""
        # Memory efficiency (performance per MB of memory)
        if self.metrics['peak_memory_mb'] > 0:
            self.metrics['memory_efficiency_ratio'] = (100 - self.metrics['wer']) / self.metrics['peak_memory_mb']
        
        # Parameter efficiency (performance per trainable parameter)
        if self.metrics['trainable_params'] > 0:
            self.metrics['performance_per_param'] = (100 - self.metrics['wer']) / (self.metrics['trainable_params'] / 1e6)
        
        # Training efficiency (performance per training time)
        if self.metrics['training_time_seconds'] > 0:
            time_hours = self.metrics['training_time_seconds'] / 3600
            self.metrics['training_efficiency_score'] = (100 - self.metrics['wer']) / time_hours
        
        # LORA-specific efficiency metrics
        if self.model_type == 'peft':
            # Adaptation magnitude relative to base model
            if self.metrics.get('adapter_weights_norm', 0) > 0:
                self.metrics['adaptation_magnitude'] = (
                    self.metrics['adapter_weights_norm'] / self.metrics['base_model_frozen_params']
                ) * 1e6  # Scale for readability
    
    def _generate_training_summary(self):
        """Generate a comprehensive training summary."""
        summary = {
            'experiment_type': f"{self.model_type.upper()} training on {self.dialect} dialect",
            'model_efficiency': {
                'trainable_params_percentage': self.metrics['trainable_percentage'],
                'memory_efficiency': self.metrics.get('memory_efficiency_ratio', 0),
                'parameter_efficiency': self.metrics.get('performance_per_param', 0)
            },
            'training_dynamics': {
                'total_steps': len(self.step_metrics),
                'convergence_analysis': self._analyze_convergence(),
                'gradient_stability': self._analyze_gradient_stability()
            },
            'performance_metrics': {
                'final_wer': self.metrics['wer'],
                'final_cer': self.metrics['cer'],
                'loss_reduction': self._calculate_loss_reduction()
            }
        }
        
        if self.model_type == 'peft':
            summary['lora_analysis'] = {
                'effective_rank': self.metrics.get('effective_rank', 0),
                'adaptation_strength': self.metrics.get('adaptation_magnitude', 0),
                'layer_utilization': self._analyze_layer_utilization()
            }
        
        return summary
    
    def _analyze_convergence(self):
        """Analyze convergence patterns from step metrics."""
        if len(self.step_metrics) < 10:
            return {'status': 'insufficient_data'}
        
        losses = [step['loss'] for step in self.step_metrics]
        
        # Simple convergence detection
        recent_losses = losses[-10:]
        early_losses = losses[:10]
        
        improvement = np.mean(early_losses) - np.mean(recent_losses)
        stability = np.std(recent_losses)
        
        return {
            'loss_improvement': improvement,
            'recent_stability': stability,
            'converged': stability < 0.01 and improvement > 0.1
        }
    
    def _analyze_gradient_stability(self):
        """Analyze gradient norm stability."""
        if len(self.gradient_history) < 10:
            return {'status': 'insufficient_data'}
        
        grads = np.array(self.gradient_history)
        return {
            'mean_gradient_norm': np.mean(grads),
            'gradient_variance': np.var(grads),
            'gradient_trend': np.polyfit(range(len(grads)), grads, 1)[0],  # Slope
            'exploding_gradients': np.any(grads > 10.0),
            'vanishing_gradients': np.any(grads < 1e-6)
        }
    
    def _calculate_loss_reduction(self):
        """Calculate total loss reduction during training."""
        if len(self.step_metrics) < 2:
            return 0
        
        initial_loss = self.step_metrics[0]['loss']
        final_loss = self.step_metrics[-1]['loss']
        return initial_loss - final_loss
    
    def _analyze_layer_utilization(self):
        """Analyze which layers are being utilized most in LORA training."""
        if not self.layer_metrics:
            return {}
        
        # Get final layer statistics
        final_step = max(self.layer_metrics.keys())
        final_stats = self.layer_metrics[final_step]
        
        total_norm = sum(stats['total_norm'] for stats in final_stats.values())
        
        utilization = {}
        for layer, stats in final_stats.items():
            utilization[layer] = stats['total_norm'] / total_norm if total_norm > 0 else 0
        
        return utilization
    
    def _save_efficiency_report(self):
        """Save a human-readable efficiency report."""
        report_file = self.detailed_dir / f"efficiency_report_{self.dialect}_{self.model_type}_{self.seed}.txt"
        
        with open(report_file, 'w') as f:
            f.write(f"EFFICIENCY ANALYSIS REPORT\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Experiment: {self.metrics['experiment_name']}\n")
            f.write(f"Method: {self.model_type.upper()}\n")
            f.write(f"Dialect: {self.dialect}\n")
            f.write(f"Seed: {self.seed}\n\n")
            
            f.write(f"PERFORMANCE METRICS:\n")
            f.write(f"- Word Error Rate (WER): {self.metrics['wer']:.2f}%\n")
            f.write(f"- Character Error Rate (CER): {self.metrics['cer']:.2f}%\n")
            f.write(f"- Final Loss: {self.metrics['final_loss']:.4f}\n\n")
            
            f.write(f"EFFICIENCY METRICS:\n")
            f.write(f"- Trainable Parameters: {self.metrics['trainable_params']:,} "
                   f"({self.metrics['trainable_percentage']:.2f}% of total)\n")
            f.write(f"- Peak Memory Usage: {self.metrics['peak_memory_mb']:.2f} MB\n")
            f.write(f"- Training Time: {self.metrics['training_time_seconds']:.2f} seconds "
                   f"({self.metrics['training_time_seconds']/3600:.2f} hours)\n")
            f.write(f"- Memory Efficiency: {self.metrics.get('memory_efficiency_ratio', 0):.4f} points/MB\n")
            f.write(f"- Parameter Efficiency: {self.metrics.get('performance_per_param', 0):.4f} points/M-params\n")
            f.write(f"- Training Efficiency: {self.metrics.get('training_efficiency_score', 0):.4f} points/hour\n\n")
            
            if self.model_type == 'peft':
                f.write(f"LORA-SPECIFIC METRICS:\n")
                f.write(f"- LoRA Rank: {self.metrics.get('lora_rank', 0)}\n")
                f.write(f"- LoRA Alpha: {self.metrics.get('lora_alpha', 0)}\n")
                f.write(f"- LoRA Dropout: {self.metrics.get('lora_dropout', 0):.3f}\n")
                f.write(f"- Target Modules: {self.metrics.get('target_modules_count', 0)}\n")
                f.write(f"- Effective Rank: {self.metrics.get('effective_rank', 0):.2f}\n")
                f.write(f"- Adaptation Magnitude: {self.metrics.get('adaptation_magnitude', 0):.6f}\n")
        
        logger.info(f"Efficiency report saved to: {report_file}")

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


class EnhancedMetricsCallback(TrainerCallback):
    """Advanced callback for comprehensive LORA training monitoring."""
    
    def __init__(self, metrics_tracker: MetricsTracker, log_every_n_steps: int = 50):
        self.metrics_tracker = metrics_tracker
        self.log_every_n_steps = log_every_n_steps
        self.step_count = 0
        
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Enhanced logging with detailed metrics collection."""
        if logs is None:
            return
            
        self.step_count += 1
        current_step = state.global_step
        
        # Extract standard metrics
        loss = logs.get('train_loss', logs.get('loss', 0))
        learning_rate = logs.get('learning_rate', 0)
        
        # Calculate gradient norm if available
        gradient_norm = None
        if hasattr(model, 'parameters'):
            total_norm = 0
            param_count = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            if param_count > 0:
                gradient_norm = total_norm ** (1. / 2)
        
        # Update memory tracking
        self.metrics_tracker.update_memory()
        memory_usage = self.metrics_tracker.metrics['peak_memory_mb']
        
        # Record step metrics
        self.metrics_tracker.record_step_metrics(
            step=current_step,
            loss=loss,
            learning_rate=learning_rate,
            gradient_norm=gradient_norm,
            memory_usage=memory_usage
        )
        
        # Analyze layer adaptation periodically
        if self.step_count % self.log_every_n_steps == 0:
            self.metrics_tracker.analyze_layer_adaptation(model, current_step)
            
            # Log detailed progress
            logger.info(f"Step {current_step}: Loss={loss:.4f}, "
                       f"LR={learning_rate:.2e}, "
                       f"Grad_norm={gradient_norm:.4f if gradient_norm else 'N/A'}, "
                       f"Memory={memory_usage:.1f}MB")


class LoRAAnalysisCallback(TrainerCallback):
    """Specialized callback for LoRA adapter analysis during training."""
    
    def __init__(self, analysis_frequency: int = 100):
        self.analysis_frequency = analysis_frequency
        self.adapter_evolution = []
        
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Analyze LoRA adapter evolution."""
        if state.global_step % self.analysis_frequency != 0:
            return
            
        if not hasattr(model, 'peft_config'):
            return
            
        # Analyze adapter matrices
        adapter_analysis = self._analyze_adapters(model, state.global_step)
        self.adapter_evolution.append(adapter_analysis)
        
    def _analyze_adapters(self, model, step):
        """Comprehensive analysis of LoRA adapter matrices."""
        analysis = {
            'step': step,
            'adapters': {},
            'global_stats': {
                'total_adapter_norm': 0,
                'max_singular_value': 0,
                'min_singular_value': float('inf'),
                'rank_utilization': []
            }
        }
        
        total_norm = 0
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # Get adapter matrices
                lora_A = module.lora_A.default.weight.data
                lora_B = module.lora_B.default.weight.data
                
                # Calculate combined matrix
                combined = torch.mm(lora_B, lora_A)
                
                # Singular value decomposition
                try:
                    U, S, V = torch.linalg.svd(combined)
                    singular_values = S.cpu().numpy()
                    
                    # Calculate statistics
                    adapter_norm = torch.norm(combined).item()
                    total_norm += adapter_norm
                    
                    rank_threshold = 0.01 * singular_values[0] if len(singular_values) > 0 else 0
                    effective_rank = np.sum(singular_values > rank_threshold)
                    
                    analysis['adapters'][name] = {
                        'norm': adapter_norm,
                        'singular_values': singular_values.tolist()[:10],  # Top 10
                        'effective_rank': int(effective_rank),
                        'condition_number': float(singular_values[0] / singular_values[-1]) if len(singular_values) > 1 else 1.0
                    }
                    
                    # Update global stats
                    analysis['global_stats']['max_singular_value'] = max(
                        analysis['global_stats']['max_singular_value'], 
                        float(singular_values[0])
                    )
                    analysis['global_stats']['min_singular_value'] = min(
                        analysis['global_stats']['min_singular_value'],
                        float(singular_values[-1]) if len(singular_values) > 0 else float('inf')
                    )
                    analysis['global_stats']['rank_utilization'].append(effective_rank)
                    
                except Exception as e:
                    logger.warning(f"SVD analysis failed for {name}: {e}")
        
        analysis['global_stats']['total_adapter_norm'] = total_norm
        analysis['global_stats']['avg_rank_utilization'] = np.mean(analysis['global_stats']['rank_utilization']) if analysis['global_stats']['rank_utilization'] else 0
        
        return analysis
    
    def get_evolution_summary(self):
        """Get summary of adapter evolution throughout training."""
        if not self.adapter_evolution:
            return {}
            
        steps = [data['step'] for data in self.adapter_evolution]
        total_norms = [data['global_stats']['total_adapter_norm'] for data in self.adapter_evolution]
        rank_utilizations = [data['global_stats']['avg_rank_utilization'] for data in self.adapter_evolution]
        
        return {
            'training_steps': steps,
            'adapter_norm_evolution': total_norms,
            'rank_utilization_evolution': rank_utilizations,
            'final_adapter_count': len(self.adapter_evolution[-1]['adapters']) if self.adapter_evolution else 0,
            'convergence_analysis': self._analyze_convergence(total_norms)
        }
    
    def _analyze_convergence(self, values):
        """Analyze convergence of adapter norms."""
        if len(values) < 10:
            return {'status': 'insufficient_data'}
            
        # Simple trend analysis
        recent_values = values[-10:]
        early_values = values[:10]
        
        recent_trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
        overall_change = values[-1] - values[0]
        
        return {
            'overall_change': overall_change,
            'recent_trend': recent_trend,
            'is_converging': abs(recent_trend) < 0.01,
            'final_value': values[-1]
        }


class AttentionAnalysisCallback(TrainerCallback):
    """Callback to analyze attention patterns during LoRA training."""
    
    def __init__(self, analysis_frequency: int = 200):
        self.analysis_frequency = analysis_frequency
        self.attention_stats = []
        
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Analyze attention pattern changes."""
        if state.global_step % self.analysis_frequency != 0:
            return
            
        # This is a placeholder for attention analysis
        # In practice, you'd need to hook into the model's attention layers
        # and collect attention weights during a forward pass
        
        attention_analysis = self._analyze_attention_patterns(model, state.global_step)
        if attention_analysis:
            self.attention_stats.append(attention_analysis)
    
    def _analyze_attention_patterns(self, model, step):
        """Analyze attention patterns (placeholder implementation)."""
        # This would require more complex implementation to hook into
        # the model's attention mechanisms during inference
        
        return {
            'step': step,
            'attention_entropy': 0,  # Placeholder
            'attention_sparsity': 0,  # Placeholder
            'head_utilization': {},  # Placeholder
        }


# =============================================================================
# MANAGER CLASSES
# =============================================================================

class DatasetManager:
    """Handles all dataset loading and preprocessing operations."""
    
    def __init__(self, dialect: str, use_huggingface: bool = True, quick_test: bool = False, custom_dataset: str = None):
        self.dialect = dialect
        self.use_huggingface = use_huggingface
        self.quick_test = quick_test
        self.custom_dataset = custom_dataset
    
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
        """Load datasets from the official HuggingFace collection or a custom dataset."""
        # If a custom dataset is provided, load it directly
        if self.custom_dataset:
            logger.info(f"Loading custom dataset from HuggingFace: {self.custom_dataset}")
            try:
                dataset = load_dataset(self.custom_dataset)
            except FileNotFoundError:
                raise ValueError(
                    f"Dataset '{self.custom_dataset}' not found on HuggingFace Hub. "
                    "Please check the dataset name and ensure it exists."
                )
            except ConnectionError as e:
                raise ValueError(
                    f"Failed to connect to HuggingFace Hub while loading '{self.custom_dataset}'. "
                    f"Please check your internet connection. Error: {e}"
                )
            except Exception as e:
                raise ValueError(f"Failed to load custom dataset '{self.custom_dataset}': {e}")
            
            # Check if it has train/test splits
            if "train" in dataset and "test" in dataset:
                return DatasetDict({
                    "train": dataset["train"],
                    "test": dataset["test"]
                })
            elif "train" in dataset:
                # If only train split, create a test split from it
                logger.info(f"Custom dataset has only 'train' split, creating {int((1-DEFAULT_TEST_SPLIT_SIZE)*100)}/{int(DEFAULT_TEST_SPLIT_SIZE*100)} train/test split")
                split_dataset = dataset["train"].train_test_split(test_size=DEFAULT_TEST_SPLIT_SIZE)
                return DatasetDict({
                    "train": split_dataset["train"],
                    "test": split_dataset["test"]
                })
            else:
                # Use the first available split
                available_splits = list(dataset.keys())
                logger.info(f"Available splits in custom dataset: {available_splits}")
                if len(available_splits) >= 2:
                    return DatasetDict({
                        "train": dataset[available_splits[0]],
                        "test": dataset[available_splits[1]]
                    })
                else:
                    # Create train/test split from the single split
                    split_dataset = dataset[available_splits[0]].train_test_split(test_size=DEFAULT_TEST_SPLIT_SIZE)
                    return DatasetDict({
                        "train": split_dataset["train"],
                        "test": split_dataset["test"]
                    })
        
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
    
    def __init__(self, model: Any, processor: WhisperProcessor, dialect: str, output_dir: Path, eval_output_filename: str = None):
        self.model = model
        self.processor = processor
        self.dialect = dialect
        self.output_dir = output_dir
        self.eval_output_filename = eval_output_filename or f"evaluation_results_{self.dialect}.json"
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
        eval_batch_size = 32  # Increased from 8 for better GPU utilization
        eval_dataloader = DataLoader(
            dataset["test"], 
            batch_size=eval_batch_size, 
            collate_fn=data_collator,
            num_workers=4,  # Parallel data loading
            pin_memory=True,  # Faster GPU transfer
            persistent_workers=True  # Keep workers alive between batches
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
        eval_results_path = self.output_dir / self.eval_output_filename
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
        hub_token=None,
        eval_output_filename: str = None,
        custom_dataset: str = None
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
        self.eval_output_filename = eval_output_filename or f"evaluation_results_{self.dialect}.json"
        self.custom_dataset = custom_dataset
        # Extract model size and get PEFT config
        self.model_size = self._extract_model_size(model_name)
        self.peft_config = PEFT_CONFIG[self.model_size]
        
        # Initialize managers
        self.dataset_manager = DatasetManager(dialect, use_huggingface, quick_test, custom_dataset)
        self.model_manager = ModelManager(model_name, self.model_size, use_peft, load_in_8bit)
        self.evaluation_manager = None  # Will be initialized after model loading
        
        # Setup output directories
        self.setup_directories()
        
        logger.info(f"Initialized Arabic Dialect PEFT Trainer:")
        logger.info(f"  Dialect: {dialect}")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  PEFT: {use_peft}")
        logger.info(f"  Quick test: {quick_test}")
        if custom_dataset:
            logger.info(f"  Custom dataset: {custom_dataset}")
    
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
        self.evaluation_manager = EvaluationManager(model, processor, self.dialect, self.output_dir, self.eval_output_filename)
        
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
    
    def evaluate_model(self, model_path: str = None):
        """Evaluate the trained model using the evaluation manager."""
        if model_path:
            # Load model from path for evaluation
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
            
            # Prepare model for evaluation
            model.eval()
            model.config.use_cache = True
            
            # Create evaluation manager with loaded model
            self.evaluation_manager = EvaluationManager(model, processor, self.dialect, self.output_dir, self.eval_output_filename)
        
        elif self.evaluation_manager is None:
            raise ValueError("No evaluation manager available. Provide model_path or train first.")
        
        # Load dataset for evaluation
        dataset = self.dataset_manager.load_datasets(self.evaluation_manager.processor)
        
        return self.evaluation_manager.evaluate_model(dataset, model_path)


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
  
  # Use a custom HuggingFace dataset
  python src/training/dialect_peft_training.py --custom_dataset username/my-arabic-dataset
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
    parser.add_argument("--custom_dataset", type=str,
                       help="Custom HuggingFace dataset path (e.g., 'username/dataset-name'). "
                            "When specified, this dataset is used instead of the default dialect datasets.")
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
    parser.add_argument("--eval_output_filename", type=str,
                       help="Custom filename for evaluation results (default: evaluation_results_{dialect}.json)")
    
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
    if args.custom_dataset:
        print(f"📦 Custom dataset: {args.custom_dataset}")
    print(f"⚡ Quick test: {args.quick_test}")
    print(f"🔧 PEFT enabled: {args.use_peft and not args.no_peft}")
    print(f"💾 8-bit loading: {args.load_in_8bit}")
    print(f"📈 Max steps: {args.max_steps}")
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
    
    # Initialize trainer
    trainer = ArabicDialectPEFTTrainer(
        model_name=args.model_name,
        dialect=args.dialect,
        quick_test=args.quick_test,
        seed=args.seed,
        push_to_hub=args.push_to_hub,
        hub_token=args.hub_token,
        hub_model_id=args.hub_model_id,
        eval_output_filename=args.eval_output_filename,
        custom_dataset=args.custom_dataset
    )
    
    # Evaluation-only mode
    if args.evaluate_only:
        print(f"🔍 Evaluation-only mode for model: {args.evaluate_only}")
        results = trainer.evaluate_model(args.evaluate_only)
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
        results = trainer.evaluate_model(model_path)
        
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
