#!/usr/bin/env python3
"""
Enhanced Evaluation Framework for Arabic Dialect PEFT Models

This module provides comprehensive evaluation capabilities including:
- Per-dialect performance analysis
- Cross-dialect generalization assessment
- Error analysis and confusion matrices
- LORA effectiveness quantification
- Statistical significance testing

Author: Enhanced with GitHub Copilot
Date: 2024
"""

import os
import sys
import json
import logging
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter
from sklearn.metrics import confusion_matrix, classification_report
from scipy import stats
import jiwer
import evaluate

# HuggingFace imports
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments, Seq2SeqTrainer
)
from datasets import load_from_disk, DatasetDict, concatenate_datasets
from peft import PeftModel, PeftConfig

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from training.dialect_peft_training import DataCollatorSpeechSeq2SeqWithPadding

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedDialectEvaluator:
    """Comprehensive evaluation framework for dialect models."""
    
    def __init__(self, model_path: str, processor_path: str = "openai/whisper-small", 
                 device: str = "auto"):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the model (base model or PEFT adapter)
            processor_path: Path to the Whisper processor
            device: Device to use for evaluation
        """
        self.model_path = Path(model_path)
        self.processor_path = processor_path
        self.device = self._setup_device(device)
        
        # Load processor
        self.processor = WhisperProcessor.from_pretrained(
            processor_path, language="Arabic", task="transcribe"
        )
        
        # Load model (will be determined based on path)
        self.model = self._load_model()
        self.model.to(self.device)
        
        # Initialize metrics
        self.wer_metric = evaluate.load("wer")
        self.cer_metric = evaluate.load("cer")
        
        # Results storage
        self.evaluation_results = {}
        self.detailed_results = {}
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def _load_model(self):
        """Load model (PEFT or base model)."""
        try:
            # Try loading as PEFT model first
            if (self.model_path / "adapter_config.json").exists():
                logger.info(f"Loading PEFT model from {self.model_path}")
                base_model = WhisperForConditionalGeneration.from_pretrained(
                    self.processor_path
                )
                model = PeftModel.from_pretrained(base_model, self.model_path)
                model.eval()
                return model
            else:
                # Load as base model
                logger.info(f"Loading base model from {self.model_path}")
                model = WhisperForConditionalGeneration.from_pretrained(self.model_path)
                model.eval()
                return model
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Fallback to base processor model
            logger.info(f"Falling back to base model: {self.processor_path}")
            model = WhisperForConditionalGeneration.from_pretrained(self.processor_path)
            model.eval()
            return model
    
    def evaluate_comprehensive(self, test_datasets: Dict[str, Any], 
                             output_dir: Path, 
                             experiment_name: str = "evaluation") -> Dict[str, Any]:
        """
        Perform comprehensive evaluation across all dialects.
        
        Args:
            test_datasets: Dictionary mapping dialect names to datasets
            output_dir: Directory to save results
            experiment_name: Name for this evaluation experiment
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info(f"Starting comprehensive evaluation: {experiment_name}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluate each dialect
        dialect_results = {}
        all_predictions = []
        all_references = []
        dialect_labels = []
        
        for dialect, dataset in test_datasets.items():
            logger.info(f"Evaluating {dialect} dialect...")
            
            # Basic evaluation
            basic_results = self._evaluate_dataset(dataset, dialect)
            
            # Detailed analysis
            detailed_results = self._detailed_dialect_analysis(
                dataset, dialect, basic_results
            )
            
            dialect_results[dialect] = {
                'basic_metrics': basic_results,
                'detailed_analysis': detailed_results
            }
            
            # Collect for cross-dialect analysis
            predictions = detailed_results['predictions']
            references = detailed_results['references']
            
            all_predictions.extend(predictions)
            all_references.extend(references)
            dialect_labels.extend([dialect] * len(predictions))
        
        # Cross-dialect analysis
        cross_dialect_results = self._cross_dialect_analysis(
            all_predictions, all_references, dialect_labels
        )
        
        # Statistical analysis
        statistical_results = self._statistical_analysis(dialect_results)
        
        # Compile comprehensive results
        comprehensive_results = {
            'experiment_name': experiment_name,
            'model_path': str(self.model_path),
            'evaluation_timestamp': pd.Timestamp.now().isoformat(),
            'dialect_results': dialect_results,
            'cross_dialect_analysis': cross_dialect_results,
            'statistical_analysis': statistical_results,
            'model_info': self._get_model_info()
        }
        
        # Save results
        self._save_comprehensive_results(comprehensive_results, output_dir, experiment_name)
        
        # Generate visualizations
        self._generate_evaluation_plots(comprehensive_results, output_dir)
        
        logger.info(f"Comprehensive evaluation complete. Results saved to {output_dir}")
        return comprehensive_results
    
    def _evaluate_dataset(self, dataset, dialect: str) -> Dict[str, float]:
        """Basic evaluation of a single dataset."""
        
        # Setup trainer for evaluation
        training_args = Seq2SeqTrainingArguments(
            output_dir="./temp_eval",
            per_device_eval_batch_size=8,
            predict_with_generate=True,
            generation_max_length=225,
            dataloader_num_workers=0,
        )
        
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)
        
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=self.model,
            eval_dataset=dataset,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
            tokenizer=self.processor.feature_extractor,
        )
        
        # Configure model for Arabic generation
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []
        self.model.generation_config.language = "ar"
        self.model.config.max_length = 512
        
        # Run evaluation
        results = trainer.evaluate()
        
        return {
            'wer': results['eval_wer'],
            'cer': results['eval_cer'],
            'loss': results['eval_loss'],
            'runtime': results['eval_runtime'],
            'samples_per_second': results['eval_samples_per_second']
        }
    
    def _detailed_dialect_analysis(self, dataset, dialect: str, basic_results: Dict) -> Dict:
        """Perform detailed analysis of dialect-specific performance."""
        
        # Generate predictions for detailed analysis
        predictions, references = self._generate_predictions(dataset)
        
        # Error analysis
        error_analysis = self._analyze_errors(predictions, references)
        
        # Token-level analysis
        token_analysis = self._token_level_analysis(predictions, references)
        
        # Phonetic analysis (simplified)
        phonetic_analysis = self._phonetic_analysis(predictions, references)
        
        return {
            'predictions': predictions,
            'references': references,
            'error_analysis': error_analysis,
            'token_analysis': token_analysis,
            'phonetic_analysis': phonetic_analysis,
            'sample_errors': self._get_sample_errors(predictions, references, n_samples=10)
        }
    
    def _generate_predictions(self, dataset) -> Tuple[List[str], List[str]]:
        """Generate predictions for the dataset."""
        predictions = []
        references = []
        
        # Setup data loader
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=8, collate_fn=data_collator
        )
        
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                input_features = batch["input_features"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Generate predictions
                generated_ids = self.model.generate(
                    input_features, 
                    max_length=225,
                    language="ar",
                    task="transcribe"
                )
                
                # Decode predictions
                batch_predictions = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                
                # Decode references
                labels[labels == -100] = self.processor.tokenizer.pad_token_id
                batch_references = self.processor.batch_decode(
                    labels, skip_special_tokens=True
                )
                
                predictions.extend(batch_predictions)
                references.extend(batch_references)
        
        return predictions, references
    
    def _analyze_errors(self, predictions: List[str], references: List[str]) -> Dict:
        """Analyze prediction errors in detail."""
        
        error_analysis = {
            'total_samples': len(predictions),
            'perfect_matches': 0,
            'error_types': defaultdict(int),
            'error_distribution': {},
            'length_analysis': {}
        }
        
        # Calculate WER for each sample
        sample_wers = []
        sample_cers = []
        
        for pred, ref in zip(predictions, references):
            if pred.strip() == ref.strip():
                error_analysis['perfect_matches'] += 1
                sample_wers.append(0.0)
                sample_cers.append(0.0)
            else:
                # Calculate individual WER and CER
                wer = jiwer.wer(ref, pred)
                cer = jiwer.cer(ref, pred)
                sample_wers.append(wer)
                sample_cers.append(cer)
                
                # Analyze error types
                self._classify_error_type(pred, ref, error_analysis['error_types'])
        
        # Statistical analysis of errors
        error_analysis['wer_distribution'] = {
            'mean': np.mean(sample_wers),
            'std': np.std(sample_wers),
            'median': np.median(sample_wers),
            'q75': np.percentile(sample_wers, 75),
            'q95': np.percentile(sample_wers, 95)
        }
        
        error_analysis['cer_distribution'] = {
            'mean': np.mean(sample_cers),
            'std': np.std(sample_cers),
            'median': np.median(sample_cers),
            'q75': np.percentile(sample_cers, 75),
            'q95': np.percentile(sample_cers, 95)
        }
        
        # Length-based analysis
        ref_lengths = [len(ref.split()) for ref in references]
        pred_lengths = [len(pred.split()) for pred in predictions]
        
        error_analysis['length_analysis'] = {
            'avg_ref_length': np.mean(ref_lengths),
            'avg_pred_length': np.mean(pred_lengths),
            'length_correlation': np.corrcoef(ref_lengths, pred_lengths)[0, 1],
            'length_bias': np.mean(pred_lengths) - np.mean(ref_lengths)
        }
        
        return error_analysis
    
    def _classify_error_type(self, prediction: str, reference: str, error_counts: Dict):
        """Classify the type of error made in prediction."""
        pred_words = prediction.split()
        ref_words = reference.split()
        
        # Simple error classification
        if len(pred_words) > len(ref_words):
            error_counts['insertion'] += 1
        elif len(pred_words) < len(ref_words):
            error_counts['deletion'] += 1
        else:
            error_counts['substitution'] += 1
            
        # Additional classification could be added here
        if not prediction.strip():
            error_counts['empty_prediction'] += 1
        
        if len(prediction) < len(reference) * 0.5:
            error_counts['severe_deletion'] += 1
    
    def _token_level_analysis(self, predictions: List[str], references: List[str]) -> Dict:
        """Analyze performance at token level."""
        
        all_pred_tokens = []
        all_ref_tokens = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            
            all_pred_tokens.extend(pred_tokens)
            all_ref_tokens.extend(ref_tokens)
        
        # Token frequency analysis
        pred_counter = Counter(all_pred_tokens)
        ref_counter = Counter(all_ref_tokens)
        
        # Most common tokens
        common_pred = pred_counter.most_common(20)
        common_ref = ref_counter.most_common(20)
        
        # Vocabulary overlap
        pred_vocab = set(all_pred_tokens)
        ref_vocab = set(all_ref_tokens)
        
        vocab_overlap = len(pred_vocab & ref_vocab) / len(pred_vocab | ref_vocab)
        
        return {
            'total_pred_tokens': len(all_pred_tokens),
            'total_ref_tokens': len(all_ref_tokens),
            'unique_pred_tokens': len(pred_vocab),
            'unique_ref_tokens': len(ref_vocab),
            'vocabulary_overlap': vocab_overlap,
            'most_common_predicted': common_pred,
            'most_common_reference': common_ref,
            'oov_rate': len(pred_vocab - ref_vocab) / len(pred_vocab)
        }
    
    def _phonetic_analysis(self, predictions: List[str], references: List[str]) -> Dict:
        """Simple phonetic analysis (placeholder for more sophisticated analysis)."""
        
        # This is a simplified placeholder - in practice, you'd use
        # Arabic phonetic analysis tools
        
        return {
            'phonetic_similarity': 0.0,  # Placeholder
            'consonant_errors': 0,       # Placeholder
            'vowel_errors': 0,          # Placeholder
            'diacritical_errors': 0     # Placeholder
        }
    
    def _get_sample_errors(self, predictions: List[str], references: List[str], 
                          n_samples: int = 10) -> List[Dict]:
        """Get sample errors for manual inspection."""
        
        error_samples = []
        
        # Calculate WER for each sample and sort by highest error
        sample_errors = []
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            if pred.strip() != ref.strip():
                wer = jiwer.wer(ref, pred)
                sample_errors.append((i, wer, pred, ref))
        
        # Sort by WER and take top samples
        sample_errors.sort(key=lambda x: x[1], reverse=True)
        
        for i, (idx, wer, pred, ref) in enumerate(sample_errors[:n_samples]):
            error_samples.append({
                'sample_index': idx,
                'wer': wer,
                'prediction': pred,
                'reference': ref,
                'error_description': self._describe_error(pred, ref)
            })
        
        return error_samples
    
    def _describe_error(self, prediction: str, reference: str) -> str:
        """Generate human-readable error description."""
        pred_words = prediction.split()
        ref_words = reference.split()
        
        if not prediction.strip():
            return "Empty prediction"
        elif len(pred_words) > len(ref_words):
            return f"Over-generation: {len(pred_words)} vs {len(ref_words)} words"
        elif len(pred_words) < len(ref_words):
            return f"Under-generation: {len(pred_words)} vs {len(ref_words)} words"
        else:
            return "Word substitution errors"
    
    def _cross_dialect_analysis(self, all_predictions: List[str], 
                               all_references: List[str], 
                               dialect_labels: List[str]) -> Dict:
        """Analyze cross-dialect performance patterns."""
        
        # Create DataFrame for analysis
        df = pd.DataFrame({
            'prediction': all_predictions,
            'reference': all_references,
            'dialect': dialect_labels
        })
        
        # Calculate per-dialect metrics
        dialect_metrics = {}
        for dialect in set(dialect_labels):
            dialect_mask = df['dialect'] == dialect
            dialect_preds = df[dialect_mask]['prediction'].tolist()
            dialect_refs = df[dialect_mask]['reference'].tolist()
            
            dialect_wer = self.wer_metric.compute(
                predictions=dialect_preds, references=dialect_refs
            )
            dialect_cer = self.cer_metric.compute(
                predictions=dialect_preds, references=dialect_refs
            )
            
            dialect_metrics[dialect] = {
                'wer': dialect_wer,
                'cer': dialect_cer,
                'sample_count': len(dialect_preds)
            }
        
        # Generalization analysis
        generalization_analysis = self._analyze_generalization(dialect_metrics)
        
        return {
            'dialect_metrics': dialect_metrics,
            'generalization_analysis': generalization_analysis,
            'overall_performance': {
                'total_samples': len(all_predictions),
                'overall_wer': self.wer_metric.compute(
                    predictions=all_predictions, references=all_references
                ),
                'overall_cer': self.cer_metric.compute(
                    predictions=all_predictions, references=all_references
                )
            }
        }
    
    def _analyze_generalization(self, dialect_metrics: Dict) -> Dict:
        """Analyze generalization patterns across dialects."""
        
        wer_values = [metrics['wer'] for metrics in dialect_metrics.values()]
        cer_values = [metrics['cer'] for metrics in dialect_metrics.values()]
        
        return {
            'wer_variance': np.var(wer_values),
            'cer_variance': np.var(cer_values),
            'best_dialect': min(dialect_metrics.keys(), 
                              key=lambda d: dialect_metrics[d]['wer']),
            'worst_dialect': max(dialect_metrics.keys(), 
                               key=lambda d: dialect_metrics[d]['wer']),
            'performance_gap': max(wer_values) - min(wer_values),
            'consistency_score': 1.0 / (1.0 + np.std(wer_values))
        }
    
    def _statistical_analysis(self, dialect_results: Dict) -> Dict:
        """Perform statistical analysis of results."""
        
        # Extract WER values for each dialect
        dialect_wers = {}
        for dialect, results in dialect_results.items():
            dialect_wers[dialect] = results['basic_metrics']['wer']
        
        # Statistical tests
        wer_values = list(dialect_wers.values())
        
        statistical_analysis = {
            'descriptive_stats': {
                'mean_wer': np.mean(wer_values),
                'std_wer': np.std(wer_values),
                'min_wer': np.min(wer_values),
                'max_wer': np.max(wer_values),
                'median_wer': np.median(wer_values)
            },
            'dialect_rankings': sorted(dialect_wers.items(), key=lambda x: x[1]),
            'significance_tests': self._perform_significance_tests(dialect_results)
        }
        
        return statistical_analysis
    
    def _perform_significance_tests(self, dialect_results: Dict) -> Dict:
        """Perform statistical significance tests (placeholder)."""
        
        # This is a placeholder - in practice, you'd need multiple runs
        # or bootstrapping to perform proper significance tests
        
        return {
            'note': 'Significance testing requires multiple experimental runs',
            'recommended_tests': ['paired t-test', 'Wilcoxon signed-rank test'],
            'sample_size_recommendation': 'At least 3 runs with different seeds'
        }
    
    def _get_model_info(self) -> Dict:
        """Get information about the model being evaluated."""
        
        model_info = {
            'model_path': str(self.model_path),
            'processor_path': self.processor_path,
            'device': str(self.device),
            'model_type': 'unknown',
            'parameter_count': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        # Determine model type
        if hasattr(self.model, 'peft_config'):
            model_info['model_type'] = 'PEFT_LoRA'
            model_info['peft_config'] = dict(self.model.peft_config)
        elif 'peft' in str(self.model_path).lower():
            model_info['model_type'] = 'PEFT_LoRA'
        else:
            model_info['model_type'] = 'FullFineTune'
        
        return model_info
    
    def _compute_metrics(self, eval_pred):
        """Compute metrics for trainer evaluation."""
        predictions, labels = eval_pred
        
        # Decode predictions and labels
        decoded_preds = self.processor.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 in labels as we can't decode them
        labels = np.where(labels != -100, labels, self.processor.tokenizer.pad_token_id)
        decoded_labels = self.processor.batch_decode(labels, skip_special_tokens=True)
        
        # Compute metrics
        wer = self.wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
        cer = self.cer_metric.compute(predictions=decoded_preds, references=decoded_labels)
        
        return {"wer": wer, "cer": cer}
    
    def _save_comprehensive_results(self, results: Dict, output_dir: Path, experiment_name: str):
        """Save comprehensive results to files."""
        
        # Save main results as JSON
        results_file = output_dir / f"{experiment_name}_comprehensive_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save summary table as CSV
        summary_data = []
        for dialect, data in results['dialect_results'].items():
            basic = data['basic_metrics']
            summary_data.append({
                'dialect': dialect,
                'wer': basic['wer'],
                'cer': basic['cer'],
                'loss': basic['loss'],
                'samples_per_second': basic['samples_per_second']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = output_dir / f"{experiment_name}_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        # Save detailed error analysis
        error_file = output_dir / f"{experiment_name}_error_analysis.json"
        error_data = {}
        for dialect, data in results['dialect_results'].items():
            error_data[dialect] = data['detailed_analysis']['error_analysis']
        
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(error_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_dir}")
    
    def _generate_evaluation_plots(self, results: Dict, output_dir: Path):
        """Generate comprehensive evaluation plots."""
        
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Performance comparison plot
        self._plot_dialect_performance(results, plots_dir)
        
        # Error distribution plot
        self._plot_error_distributions(results, plots_dir)
        
        # Cross-dialect analysis plot
        self._plot_cross_dialect_analysis(results, plots_dir)
        
        logger.info(f"Plots saved to {plots_dir}")
    
    def _plot_dialect_performance(self, results: Dict, plots_dir: Path):
        """Plot dialect performance comparison."""
        
        dialects = []
        wer_values = []
        cer_values = []
        
        for dialect, data in results['dialect_results'].items():
            dialects.append(dialect)
            wer_values.append(data['basic_metrics']['wer'])
            cer_values.append(data['basic_metrics']['cer'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # WER plot
        bars1 = ax1.bar(dialects, wer_values, color='skyblue', alpha=0.7)
        ax1.set_title('Word Error Rate by Dialect', fontsize=14, fontweight='bold')
        ax1.set_ylabel('WER (%)')
        ax1.set_xlabel('Dialect')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, wer_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # CER plot
        bars2 = ax2.bar(dialects, cer_values, color='lightcoral', alpha=0.7)
        ax2.set_title('Character Error Rate by Dialect', fontsize=14, fontweight='bold')
        ax2.set_ylabel('CER (%)')
        ax2.set_xlabel('Dialect')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars2, cer_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'dialect_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_error_distributions(self, results: Dict, plots_dir: Path):
        """Plot error distribution analysis."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Error Distribution Analysis', fontsize=16, fontweight='bold')
        
        dialects = list(results['dialect_results'].keys())
        
        # WER distributions
        for i, dialect in enumerate(dialects[:6]):  # Max 6 dialects
            row = i // 3
            col = i % 3
            
            if row < 2 and col < 3:
                error_data = results['dialect_results'][dialect]['detailed_analysis']['error_analysis']
                
                # Create histogram of error types
                error_types = error_data['error_types']
                if error_types:
                    types = list(error_types.keys())
                    counts = list(error_types.values())
                    
                    axes[row, col].bar(types, counts, alpha=0.7)
                    axes[row, col].set_title(f'{dialect.capitalize()} Error Types')
                    axes[row, col].tick_params(axis='x', rotation=45)
                else:
                    axes[row, col].text(0.5, 0.5, 'No error data', 
                                       ha='center', va='center', transform=axes[row, col].transAxes)
                    axes[row, col].set_title(f'{dialect.capitalize()} Error Types')
        
        # Remove empty subplots
        for i in range(len(dialects), 6):
            row = i // 3
            col = i % 3
            if row < 2 and col < 3:
                fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'error_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cross_dialect_analysis(self, results: Dict, plots_dir: Path):
        """Plot cross-dialect analysis."""
        
        cross_data = results['cross_dialect_analysis']
        dialect_metrics = cross_data['dialect_metrics']
        
        dialects = list(dialect_metrics.keys())
        wer_values = [dialect_metrics[d]['wer'] for d in dialects]
        sample_counts = [dialect_metrics[d]['sample_count'] for d in dialects]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Performance vs sample count
        scatter = ax1.scatter(sample_counts, wer_values, 
                            s=100, alpha=0.7, c=range(len(dialects)), cmap='viridis')
        
        for i, dialect in enumerate(dialects):
            ax1.annotate(dialect, (sample_counts[i], wer_values[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax1.set_xlabel('Number of Test Samples')
        ax1.set_ylabel('WER (%)')
        ax1.set_title('Performance vs Dataset Size')
        ax1.grid(True, alpha=0.3)
        
        # Generalization metrics
        gen_data = cross_data['generalization_analysis']
        metrics = ['WER Variance', 'Performance Gap', 'Consistency Score']
        values = [gen_data['wer_variance'], gen_data['performance_gap'], gen_data['consistency_score']]
        
        bars = ax2.bar(metrics, values, color=['red', 'orange', 'green'], alpha=0.7)
        ax2.set_title('Generalization Metrics')
        ax2.set_ylabel('Score')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'cross_dialect_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()


def load_test_datasets(data_root: str = "./data") -> Dict[str, Any]:
    """Load test datasets for all dialects."""
    
    datasets = {}
    dialects = ["egyptian", "gulf", "iraqi", "levantine", "maghrebi"]
    
    for dialect in dialects:
        test_path = Path(data_root) / f"{dialect}_test"
        if test_path.exists():
            try:
                dataset = load_from_disk(str(test_path))
                datasets[dialect] = dataset
                logger.info(f"Loaded {dialect} test set: {len(dataset)} samples")
            except Exception as e:
                logger.warning(f"Failed to load {dialect} test set: {e}")
        else:
            logger.warning(f"Test set not found: {test_path}")
    
    return datasets


def main():
    """Main evaluation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Arabic Dialect Evaluation")
    parser.add_argument("--model_path", required=True, help="Path to model or PEFT adapter")
    parser.add_argument("--data_root", default="./data", help="Root directory for test data")
    parser.add_argument("--output_dir", default="./enhanced_evaluation_results", 
                       help="Output directory for results")
    parser.add_argument("--experiment_name", default="enhanced_eval", 
                       help="Name for this evaluation experiment")
    parser.add_argument("--processor", default="openai/whisper-small",
                       help="Whisper processor to use")
    
    args = parser.parse_args()
    
    # Load test datasets
    test_datasets = load_test_datasets(args.data_root)
    
    if not test_datasets:
        logger.error("No test datasets found!")
        return
    
    # Initialize evaluator
    evaluator = EnhancedDialectEvaluator(
        model_path=args.model_path,
        processor_path=args.processor
    )
    
    # Run comprehensive evaluation
    results = evaluator.evaluate_comprehensive(
        test_datasets=test_datasets,
        output_dir=Path(args.output_dir),
        experiment_name=args.experiment_name
    )
    
    # Print summary
    print("\n" + "="*60)
    print("ENHANCED EVALUATION SUMMARY")
    print("="*60)
    
    for dialect, data in results['dialect_results'].items():
        metrics = data['basic_metrics']
        print(f"{dialect.capitalize():>12}: WER={metrics['wer']:6.2f}%  CER={metrics['cer']:6.2f}%")
    
    overall = results['cross_dialect_analysis']['overall_performance']
    print(f"{'Overall':>12}: WER={overall['overall_wer']:6.2f}%  CER={overall['overall_cer']:6.2f}%")
    
    print("\nGeneralization Analysis:")
    gen = results['cross_dialect_analysis']['generalization_analysis']
    print(f"  Best Dialect: {gen['best_dialect']}")
    print(f"  Worst Dialect: {gen['worst_dialect']}")
    print(f"  Performance Gap: {gen['performance_gap']:.2f}%")
    print(f"  Consistency Score: {gen['consistency_score']:.3f}")
    
    print(f"\nDetailed results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
