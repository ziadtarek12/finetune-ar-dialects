#!/usr/bin/env python3
"""
LORA Analysis Tools for Arabic Dialect Fine-tuning

This module provides specialized tools for analyzing LORA adapter effectiveness:
- Adapter weight visualization and analysis
- Layer-wise adaptation patterns
- Rank utilization analysis
- Adapter matrix decomposition
- Effectiveness quantification

Author: Enhanced with GitHub Copilot
Date: 2024
"""

import os
import sys
import json
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# PEFT imports
from peft import PeftModel, PeftConfig, LoraConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LoRAAnalyzer:
    """Comprehensive analyzer for LoRA adapters."""
    
    def __init__(self, model_path: str, base_model_path: str = "openai/whisper-small"):
        """
        Initialize LoRA analyzer.
        
        Args:
            model_path: Path to PEFT model or adapter
            base_model_path: Path to base model
        """
        self.model_path = Path(model_path)
        self.base_model_path = base_model_path
        
        # Load models
        self.base_model = self._load_base_model()
        self.peft_model = self._load_peft_model()
        
        # Analysis results
        self.adapter_analysis = {}
        self.layer_analysis = {}
        self.effectiveness_metrics = {}
        
    def _load_base_model(self):
        """Load the base model."""
        from transformers import WhisperForConditionalGeneration
        return WhisperForConditionalGeneration.from_pretrained(self.base_model_path)
    
    def _load_peft_model(self):
        """Load the PEFT model."""
        if (self.model_path / "adapter_config.json").exists():
            base_model = self._load_base_model()
            return PeftModel.from_pretrained(base_model, self.model_path)
        else:
            raise ValueError(f"No PEFT adapter found at {self.model_path}")
    
    def analyze_comprehensive(self, output_dir: Path, experiment_name: str = "lora_analysis"):
        """
        Perform comprehensive LoRA analysis.
        
        Args:
            output_dir: Directory to save results
            experiment_name: Name for this analysis
        """
        logger.info(f"Starting comprehensive LoRA analysis: {experiment_name}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Adapter matrix analysis
        self.adapter_analysis = self._analyze_adapter_matrices()
        
        # 2. Layer-wise adaptation analysis
        self.layer_analysis = self._analyze_layer_adaptation()
        
        # 3. Rank utilization analysis
        rank_analysis = self._analyze_rank_utilization()
        
        # 4. Effectiveness quantification
        self.effectiveness_metrics = self._quantify_effectiveness()
        
        # 5. Parameter sensitivity analysis
        sensitivity_analysis = self._analyze_parameter_sensitivity()
        
        # Compile comprehensive results
        comprehensive_results = {
            'experiment_name': experiment_name,
            'model_path': str(self.model_path),
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'adapter_analysis': self.adapter_analysis,
            'layer_analysis': self.layer_analysis,
            'rank_analysis': rank_analysis,
            'effectiveness_metrics': self.effectiveness_metrics,
            'sensitivity_analysis': sensitivity_analysis,
            'model_info': self._get_model_info()
        }
        
        # Save results
        self._save_analysis_results(comprehensive_results, output_dir, experiment_name)
        
        # Generate visualizations
        self._generate_analysis_plots(comprehensive_results, output_dir)
        
        # Generate interactive dashboard
        self._generate_interactive_dashboard(comprehensive_results, output_dir)
        
        logger.info(f"LoRA analysis complete. Results saved to {output_dir}")
        return comprehensive_results
    
    def _analyze_adapter_matrices(self) -> Dict:
        """Analyze LoRA adapter matrices in detail."""
        
        adapter_analysis = {
            'adapters': {},
            'global_statistics': {
                'total_adapters': 0,
                'total_parameters': 0,
                'parameter_distribution': {},
                'norm_statistics': {},
                'singular_value_statistics': {}
            }
        }
        
        total_norm = 0
        all_singular_values = []
        parameter_counts = []
        
        for name, module in self.peft_model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # Extract adapter matrices
                lora_A = module.lora_A.default.weight.data  # [r, in_features]
                lora_B = module.lora_B.default.weight.data  # [out_features, r]
                
                # Calculate combined adaptation matrix
                combined = torch.mm(lora_B, lora_A)  # [out_features, in_features]
                
                # Basic statistics
                adapter_norm = torch.norm(combined).item()
                total_norm += adapter_norm
                
                param_count = lora_A.numel() + lora_B.numel()
                parameter_counts.append(param_count)
                
                # Singular value decomposition
                try:
                    U, S, V = torch.linalg.svd(combined)
                    singular_values = S.cpu().numpy()
                    all_singular_values.extend(singular_values)
                    
                    # Effective rank calculation
                    energy_threshold = 0.99
                    cumulative_energy = np.cumsum(singular_values**2)
                    total_energy = cumulative_energy[-1]
                    effective_rank = np.sum(cumulative_energy < energy_threshold * total_energy) + 1
                    
                    # Condition number
                    condition_number = float(singular_values[0] / singular_values[-1]) if len(singular_values) > 1 else 1.0
                    
                    # Spectral properties
                    spectral_properties = {
                        'rank': len(singular_values),
                        'effective_rank': int(effective_rank),
                        'condition_number': condition_number,
                        'spectral_norm': float(singular_values[0]),
                        'nuclear_norm': float(np.sum(singular_values)),
                        'entropy': self._calculate_spectral_entropy(singular_values)
                    }
                    
                except Exception as e:
                    logger.warning(f"SVD failed for {name}: {e}")
                    spectral_properties = {'error': str(e)}
                
                # Weight distribution analysis
                weight_stats = {
                    'A_matrix': {
                        'shape': list(lora_A.shape),
                        'norm': torch.norm(lora_A).item(),
                        'mean': torch.mean(lora_A).item(),
                        'std': torch.std(lora_A).item(),
                        'sparsity': (lora_A == 0).float().mean().item()
                    },
                    'B_matrix': {
                        'shape': list(lora_B.shape),
                        'norm': torch.norm(lora_B).item(),
                        'mean': torch.mean(lora_B).item(),
                        'std': torch.std(lora_B).item(),
                        'sparsity': (lora_B == 0).float().mean().item()
                    },
                    'combined': {
                        'shape': list(combined.shape),
                        'norm': adapter_norm,
                        'mean': torch.mean(combined).item(),
                        'std': torch.std(combined).item(),
                        'sparsity': (combined.abs() < 1e-6).float().mean().item()
                    }
                }
                
                adapter_analysis['adapters'][name] = {
                    'parameter_count': param_count,
                    'spectral_properties': spectral_properties,
                    'weight_statistics': weight_stats,
                    'adaptation_strength': adapter_norm,
                    'layer_type': self._classify_layer_type(name)
                }
                
                adapter_analysis['global_statistics']['total_adapters'] += 1
                adapter_analysis['global_statistics']['total_parameters'] += param_count
        
        # Global statistics
        if parameter_counts:
            adapter_analysis['global_statistics'].update({
                'total_norm': total_norm,
                'average_norm': total_norm / len(parameter_counts),
                'parameter_distribution': {
                    'mean': np.mean(parameter_counts),
                    'std': np.std(parameter_counts),
                    'min': np.min(parameter_counts),
                    'max': np.max(parameter_counts)
                },
                'singular_value_statistics': {
                    'mean': np.mean(all_singular_values),
                    'std': np.std(all_singular_values),
                    'range': [np.min(all_singular_values), np.max(all_singular_values)],
                    'percentiles': {
                        '25': np.percentile(all_singular_values, 25),
                        '50': np.percentile(all_singular_values, 50),
                        '75': np.percentile(all_singular_values, 75),
                        '95': np.percentile(all_singular_values, 95)
                    }
                }
            })
        
        return adapter_analysis
    
    def _calculate_spectral_entropy(self, singular_values: np.ndarray) -> float:
        """Calculate spectral entropy of singular values."""
        # Normalize to probabilities
        normalized = singular_values**2 / np.sum(singular_values**2)
        # Calculate entropy
        entropy = -np.sum(normalized * np.log(normalized + 1e-10))
        return float(entropy)
    
    def _classify_layer_type(self, layer_name: str) -> str:
        """Classify the type of layer based on its name."""
        if 'attention' in layer_name.lower():
            if 'q_proj' in layer_name:
                return 'attention_query'
            elif 'k_proj' in layer_name:
                return 'attention_key'
            elif 'v_proj' in layer_name:
                return 'attention_value'
            elif 'out_proj' in layer_name:
                return 'attention_output'
            else:
                return 'attention_other'
        elif 'mlp' in layer_name.lower() or 'fc' in layer_name:
            return 'feedforward'
        elif 'embed' in layer_name.lower():
            return 'embedding'
        else:
            return 'other'
    
    def _analyze_layer_adaptation(self) -> Dict:
        """Analyze adaptation patterns across different layers."""
        
        layer_analysis = {
            'layer_groups': defaultdict(list),
            'adaptation_patterns': {},
            'layer_rankings': {},
            'interaction_analysis': {}
        }
        
        # Group adapters by layer type and position
        for adapter_name, adapter_data in self.adapter_analysis['adapters'].items():
            layer_type = adapter_data['layer_type']
            adaptation_strength = adapter_data['adaptation_strength']
            
            layer_analysis['layer_groups'][layer_type].append({
                'name': adapter_name,
                'strength': adaptation_strength,
                'parameters': adapter_data['parameter_count'],
                'effective_rank': adapter_data['spectral_properties'].get('effective_rank', 0)
            })
        
        # Analyze patterns within each layer group
        for layer_type, adapters in layer_analysis['layer_groups'].items():
            if adapters:
                strengths = [a['strength'] for a in adapters]
                layer_analysis['adaptation_patterns'][layer_type] = {
                    'count': len(adapters),
                    'total_strength': sum(strengths),
                    'average_strength': np.mean(strengths),
                    'strength_variance': np.var(strengths),
                    'strongest_adapter': max(adapters, key=lambda x: x['strength'])['name'],
                    'adaptation_distribution': self._analyze_adaptation_distribution(adapters)
                }
        
        # Rank layers by adaptation strength
        layer_analysis['layer_rankings'] = self._rank_layers_by_adaptation()
        
        # Analyze cross-layer interactions (simplified)
        layer_analysis['interaction_analysis'] = self._analyze_layer_interactions()
        
        return layer_analysis
    
    def _analyze_adaptation_distribution(self, adapters: List[Dict]) -> Dict:
        """Analyze the distribution of adaptation within a layer group."""
        
        strengths = [a['strength'] for a in adapters]
        ranks = [a['effective_rank'] for a in adapters]
        
        return {
            'strength_distribution': {
                'mean': np.mean(strengths),
                'std': np.std(strengths),
                'skewness': float(pd.Series(strengths).skew()),
                'kurtosis': float(pd.Series(strengths).kurtosis())
            },
            'rank_distribution': {
                'mean': np.mean(ranks),
                'std': np.std(ranks),
                'range': [min(ranks), max(ranks)]
            },
            'uniformity_score': 1.0 / (1.0 + np.std(strengths) / np.mean(strengths)) if np.mean(strengths) > 0 else 0
        }
    
    def _rank_layers_by_adaptation(self) -> Dict:
        """Rank layers by their adaptation strength."""
        
        layer_strengths = {}
        for layer_type, pattern in self.layer_analysis.get('adaptation_patterns', {}).items():
            layer_strengths[layer_type] = pattern['total_strength']
        
        # Sort by strength
        ranked_layers = sorted(layer_strengths.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'ranking': ranked_layers,
            'most_adapted': ranked_layers[0][0] if ranked_layers else None,
            'least_adapted': ranked_layers[-1][0] if ranked_layers else None,
            'adaptation_concentration': self._calculate_adaptation_concentration(layer_strengths)
        }
    
    def _calculate_adaptation_concentration(self, layer_strengths: Dict) -> float:
        """Calculate how concentrated the adaptation is across layers."""
        if not layer_strengths:
            return 0.0
        
        total_strength = sum(layer_strengths.values())
        if total_strength == 0:
            return 0.0
        
        # Calculate Gini coefficient as measure of concentration
        values = list(layer_strengths.values())
        values.sort()
        n = len(values)
        cumsum = np.cumsum(values)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        
        return float(gini)
    
    def _analyze_layer_interactions(self) -> Dict:
        """Analyze potential interactions between adapted layers."""
        
        # This is a simplified analysis - in practice, you might analyze
        # correlation between adapter weights, gradient flow, etc.
        
        layer_types = list(self.layer_analysis.get('adaptation_patterns', {}).keys())
        
        interaction_matrix = np.eye(len(layer_types))  # Identity as placeholder
        
        return {
            'layer_types': layer_types,
            'interaction_matrix': interaction_matrix.tolist(),
            'strong_interactions': [],  # Placeholder
            'interaction_strength': 0.0  # Placeholder
        }
    
    def _analyze_rank_utilization(self) -> Dict:
        """Analyze how effectively the LoRA rank is being utilized."""
        
        rank_analysis = {
            'rank_statistics': {},
            'utilization_efficiency': {},
            'rank_recommendations': {}
        }
        
        # Extract rank information
        config_path = self.model_path / "adapter_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                configured_rank = config.get('r', 0)
        else:
            configured_rank = 0
        
        # Analyze effective ranks
        effective_ranks = []
        spectral_norms = []
        
        for adapter_data in self.adapter_analysis['adapters'].values():
            spectral_props = adapter_data['spectral_properties']
            if 'effective_rank' in spectral_props:
                effective_ranks.append(spectral_props['effective_rank'])
            if 'spectral_norm' in spectral_props:
                spectral_norms.append(spectral_props['spectral_norm'])
        
        if effective_ranks:
            rank_analysis['rank_statistics'] = {
                'configured_rank': configured_rank,
                'effective_ranks': {
                    'mean': np.mean(effective_ranks),
                    'std': np.std(effective_ranks),
                    'min': np.min(effective_ranks),
                    'max': np.max(effective_ranks),
                    'distribution': effective_ranks
                },
                'rank_utilization_ratio': np.mean(effective_ranks) / configured_rank if configured_rank > 0 else 0
            }
            
            # Efficiency analysis
            utilization_ratio = np.mean(effective_ranks) / configured_rank if configured_rank > 0 else 0
            
            rank_analysis['utilization_efficiency'] = {
                'efficiency_score': utilization_ratio,
                'over_parameterized': utilization_ratio < 0.7,
                'under_parameterized': utilization_ratio > 0.95,
                'optimal_range': 0.7 <= utilization_ratio <= 0.95
            }
            
            # Recommendations
            if utilization_ratio < 0.5:
                recommendation = f"Consider reducing rank to {int(configured_rank * 0.7)}"
            elif utilization_ratio > 0.95:
                recommendation = f"Consider increasing rank to {int(configured_rank * 1.3)}"
            else:
                recommendation = "Rank appears well-sized"
            
            rank_analysis['rank_recommendations'] = {
                'recommendation': recommendation,
                'confidence': 'high' if abs(utilization_ratio - 0.8) > 0.2 else 'medium'
            }
        
        return rank_analysis
    
    def _quantify_effectiveness(self) -> Dict:
        """Quantify the overall effectiveness of LoRA adaptation."""
        
        effectiveness = {
            'parameter_efficiency': 0,
            'adaptation_strength': 0,
            'rank_efficiency': 0,
            'layer_coverage': 0,
            'overall_score': 0
        }
        
        # Parameter efficiency
        total_base_params = sum(p.numel() for p in self.base_model.parameters())
        total_adapter_params = self.adapter_analysis['global_statistics']['total_parameters']
        
        effectiveness['parameter_efficiency'] = total_adapter_params / total_base_params
        
        # Adaptation strength (normalized)
        total_norm = self.adapter_analysis['global_statistics']['total_norm']
        adapter_count = self.adapter_analysis['global_statistics']['total_adapters']
        
        effectiveness['adaptation_strength'] = total_norm / adapter_count if adapter_count > 0 else 0
        
        # Rank efficiency
        rank_stats = getattr(self, '_analyze_rank_utilization', lambda: {})()
        if 'utilization_efficiency' in rank_stats:
            effectiveness['rank_efficiency'] = rank_stats['utilization_efficiency']['efficiency_score']
        
        # Layer coverage
        total_layers = sum(1 for _ in self.base_model.named_modules())
        adapted_layers = adapter_count
        
        effectiveness['layer_coverage'] = adapted_layers / total_layers
        
        # Overall effectiveness score (weighted combination)
        weights = {'parameter_efficiency': 0.3, 'adaptation_strength': 0.3, 
                  'rank_efficiency': 0.2, 'layer_coverage': 0.2}
        
        # Normalize metrics to 0-1 range for combination
        normalized_metrics = {
            'parameter_efficiency': min(effectiveness['parameter_efficiency'] * 100, 1.0),  # Scale to reasonable range
            'adaptation_strength': min(effectiveness['adaptation_strength'] / 10, 1.0),      # Scale adaptation strength
            'rank_efficiency': effectiveness['rank_efficiency'],                              # Already 0-1
            'layer_coverage': effectiveness['layer_coverage']                                 # Already 0-1
        }
        
        effectiveness['overall_score'] = sum(
            weights[metric] * normalized_metrics[metric] 
            for metric in weights.keys()
        )
        
        return effectiveness
    
    def _analyze_parameter_sensitivity(self) -> Dict:
        """Analyze sensitivity to LoRA hyperparameters."""
        
        # This is a placeholder for parameter sensitivity analysis
        # In practice, you'd need to run multiple experiments with different hyperparameters
        
        return {
            'rank_sensitivity': {
                'analysis': 'Requires multiple experiments with different ranks',
                'recommendation': 'Run grid search over ranks [8, 16, 32, 64]'
            },
            'alpha_sensitivity': {
                'analysis': 'Requires multiple experiments with different alpha values',
                'recommendation': 'Test alpha values in proportion to rank'
            },
            'dropout_sensitivity': {
                'analysis': 'Requires experiments with different dropout rates',
                'recommendation': 'Test dropout rates [0.0, 0.05, 0.1, 0.2]'
            }
        }
    
    def _get_model_info(self) -> Dict:
        """Get comprehensive model information."""
        
        # Load PEFT config
        config_path = self.model_path / "adapter_config.json"
        peft_config = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                peft_config = json.load(f)
        
        return {
            'model_path': str(self.model_path),
            'base_model_path': self.base_model_path,
            'peft_config': peft_config,
            'base_model_parameters': sum(p.numel() for p in self.base_model.parameters()),
            'adapter_parameters': self.adapter_analysis['global_statistics']['total_parameters'],
            'parameter_reduction_ratio': self.adapter_analysis['global_statistics']['total_parameters'] / 
                                       sum(p.numel() for p in self.base_model.parameters())
        }
    
    def _save_analysis_results(self, results: Dict, output_dir: Path, experiment_name: str):
        """Save comprehensive analysis results."""
        
        # Save main results as JSON
        results_file = output_dir / f"{experiment_name}_lora_analysis.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save summary statistics as CSV
        self._save_summary_csv(results, output_dir, experiment_name)
        
        # Save detailed adapter analysis
        adapter_file = output_dir / f"{experiment_name}_adapter_details.json"
        with open(adapter_file, 'w', encoding='utf-8') as f:
            json.dump(results['adapter_analysis'], f, indent=2)
        
        logger.info(f"Analysis results saved to {output_dir}")
    
    def _save_summary_csv(self, results: Dict, output_dir: Path, experiment_name: str):
        """Save summary statistics as CSV."""
        
        # Create summary dataframe
        summary_data = []
        
        for adapter_name, adapter_data in results['adapter_analysis']['adapters'].items():
            summary_data.append({
                'adapter_name': adapter_name,
                'layer_type': adapter_data['layer_type'],
                'parameter_count': adapter_data['parameter_count'],
                'adaptation_strength': adapter_data['adaptation_strength'],
                'effective_rank': adapter_data['spectral_properties'].get('effective_rank', 0),
                'condition_number': adapter_data['spectral_properties'].get('condition_number', 0),
                'spectral_norm': adapter_data['spectral_properties'].get('spectral_norm', 0)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = output_dir / f"{experiment_name}_adapter_summary.csv"
        summary_df.to_csv(summary_file, index=False)
    
    def _generate_analysis_plots(self, results: Dict, output_dir: Path):
        """Generate comprehensive analysis plots."""
        
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # 1. Adapter strength distribution
        self._plot_adapter_strength(results, plots_dir)
        
        # 2. Layer-wise adaptation patterns
        self._plot_layer_adaptation(results, plots_dir)
        
        # 3. Rank utilization analysis
        self._plot_rank_utilization(results, plots_dir)
        
        # 4. Spectral analysis
        self._plot_spectral_analysis(results, plots_dir)
        
        # 5. Effectiveness radar chart
        self._plot_effectiveness_radar(results, plots_dir)
        
        logger.info(f"Analysis plots saved to {plots_dir}")
    
    def _plot_adapter_strength(self, results: Dict, plots_dir: Path):
        """Plot adapter strength distribution."""
        
        adapters = results['adapter_analysis']['adapters']
        
        adapter_names = list(adapters.keys())
        strengths = [adapters[name]['adaptation_strength'] for name in adapter_names]
        layer_types = [adapters[name]['layer_type'] for name in adapter_names]
        
        # Create color map for layer types
        unique_types = list(set(layer_types))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
        color_map = dict(zip(unique_types, colors))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar plot of adapter strengths
        bar_colors = [color_map[layer_type] for layer_type in layer_types]
        bars = ax1.bar(range(len(adapter_names)), strengths, color=bar_colors, alpha=0.7)
        
        ax1.set_title('Adapter Strength by Layer', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Adapter Index')
        ax1.set_ylabel('Adaptation Strength')
        ax1.tick_params(axis='x', rotation=90)
        
        # Create legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color_map[t], alpha=0.7) for t in unique_types]
        ax1.legend(legend_elements, unique_types, title='Layer Types', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Box plot by layer type
        type_strengths = defaultdict(list)
        for name, data in adapters.items():
            type_strengths[data['layer_type']].append(data['adaptation_strength'])
        
        ax2.boxplot([type_strengths[t] for t in unique_types], labels=unique_types)
        ax2.set_title('Adaptation Strength Distribution by Layer Type', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Adaptation Strength')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'adapter_strength_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_layer_adaptation(self, results: Dict, plots_dir: Path):
        """Plot layer adaptation patterns."""
        
        layer_patterns = results['layer_analysis']['adaptation_patterns']
        
        if not layer_patterns:
            return
        
        layer_types = list(layer_patterns.keys())
        total_strengths = [layer_patterns[t]['total_strength'] for t in layer_types]
        avg_strengths = [layer_patterns[t]['average_strength'] for t in layer_types]
        counts = [layer_patterns[t]['count'] for t in layer_types]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Layer Adaptation Analysis', fontsize=16, fontweight='bold')
        
        # Total adaptation by layer type
        bars1 = ax1.bar(layer_types, total_strengths, alpha=0.7, color='skyblue')
        ax1.set_title('Total Adaptation Strength by Layer Type')
        ax1.set_ylabel('Total Strength')
        ax1.tick_params(axis='x', rotation=45)
        
        # Average adaptation by layer type
        bars2 = ax2.bar(layer_types, avg_strengths, alpha=0.7, color='lightcoral')
        ax2.set_title('Average Adaptation Strength by Layer Type')
        ax2.set_ylabel('Average Strength')
        ax2.tick_params(axis='x', rotation=45)
        
        # Adapter count by layer type
        bars3 = ax3.bar(layer_types, counts, alpha=0.7, color='lightgreen')
        ax3.set_title('Number of Adapters by Layer Type')
        ax3.set_ylabel('Count')
        ax3.tick_params(axis='x', rotation=45)
        
        # Pie chart of adaptation distribution
        ax4.pie(total_strengths, labels=layer_types, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Adaptation Distribution')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'layer_adaptation_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_rank_utilization(self, results: Dict, plots_dir: Path):
        """Plot rank utilization analysis."""
        
        rank_analysis = results['rank_analysis']
        
        if 'rank_statistics' not in rank_analysis:
            return
        
        rank_stats = rank_analysis['rank_statistics']
        configured_rank = rank_stats.get('configured_rank', 0)
        effective_ranks = rank_stats.get('effective_ranks', {}).get('distribution', [])
        
        if not effective_ranks:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram of effective ranks
        ax1.hist(effective_ranks, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax1.axvline(configured_rank, color='red', linestyle='--', linewidth=2, label=f'Configured Rank: {configured_rank}')
        ax1.axvline(np.mean(effective_ranks), color='green', linestyle='--', linewidth=2, label=f'Mean Effective Rank: {np.mean(effective_ranks):.1f}')
        ax1.set_title('Distribution of Effective Ranks', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Effective Rank')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Rank utilization ratio
        utilization_ratios = [r / configured_rank for r in effective_ranks] if configured_rank > 0 else []
        
        if utilization_ratios:
            ax2.hist(utilization_ratios, bins=20, alpha=0.7, color='orange', edgecolor='black')
            ax2.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Perfect Utilization')
            ax2.axvline(np.mean(utilization_ratios), color='green', linestyle='--', linewidth=2, 
                       label=f'Mean Utilization: {np.mean(utilization_ratios):.2f}')
            ax2.set_title('Rank Utilization Ratio Distribution', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Utilization Ratio (Effective/Configured)')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'rank_utilization_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_spectral_analysis(self, results: Dict, plots_dir: Path):
        """Plot spectral analysis of adapter matrices."""
        
        # Collect all spectral data
        spectral_norms = []
        condition_numbers = []
        nuclear_norms = []
        entropies = []
        
        for adapter_data in results['adapter_analysis']['adapters'].values():
            spectral_props = adapter_data['spectral_properties']
            if 'spectral_norm' in spectral_props:
                spectral_norms.append(spectral_props['spectral_norm'])
            if 'condition_number' in spectral_props:
                condition_numbers.append(spectral_props['condition_number'])
            if 'nuclear_norm' in spectral_props:
                nuclear_norms.append(spectral_props['nuclear_norm'])
            if 'entropy' in spectral_props:
                entropies.append(spectral_props['entropy'])
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Spectral Analysis of Adapter Matrices', fontsize=16, fontweight='bold')
        
        # Spectral norms
        if spectral_norms:
            ax1.hist(spectral_norms, bins=20, alpha=0.7, color='blue', edgecolor='black')
            ax1.set_title('Distribution of Spectral Norms')
            ax1.set_xlabel('Spectral Norm')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)
        
        # Condition numbers
        if condition_numbers:
            ax2.hist(np.log10(condition_numbers), bins=20, alpha=0.7, color='red', edgecolor='black')
            ax2.set_title('Distribution of Condition Numbers (log scale)')
            ax2.set_xlabel('log10(Condition Number)')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
        
        # Nuclear norms
        if nuclear_norms:
            ax3.hist(nuclear_norms, bins=20, alpha=0.7, color='green', edgecolor='black')
            ax3.set_title('Distribution of Nuclear Norms')
            ax3.set_xlabel('Nuclear Norm')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)
        
        # Spectral entropies
        if entropies:
            ax4.hist(entropies, bins=20, alpha=0.7, color='purple', edgecolor='black')
            ax4.set_title('Distribution of Spectral Entropies')
            ax4.set_xlabel('Spectral Entropy')
            ax4.set_ylabel('Frequency')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'spectral_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_effectiveness_radar(self, results: Dict, plots_dir: Path):
        """Plot effectiveness metrics as radar chart."""
        
        effectiveness = results['effectiveness_metrics']
        
        metrics = ['Parameter\nEfficiency', 'Adaptation\nStrength', 'Rank\nEfficiency', 
                  'Layer\nCoverage', 'Overall\nScore']
        
        # Normalize values for radar chart (0-1 scale)
        values = [
            min(effectiveness['parameter_efficiency'] * 100, 1.0),  # Scale parameter efficiency
            min(effectiveness['adaptation_strength'] / 10, 1.0),    # Scale adaptation strength
            effectiveness['rank_efficiency'],                        # Already 0-1
            effectiveness['layer_coverage'],                         # Already 0-1
            effectiveness['overall_score']                           # Already 0-1
        ]
        
        # Number of variables
        N = len(metrics)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Add values to complete the circle
        values += values[:1]
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Plot the values
        ax.plot(angles, values, 'o-', linewidth=2, label='LoRA Effectiveness', color='blue')
        ax.fill(angles, values, alpha=0.25, color='blue')
        
        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        
        # Add title
        plt.title('LoRA Effectiveness Analysis', size=16, fontweight='bold', pad=20)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'effectiveness_radar.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_interactive_dashboard(self, results: Dict, output_dir: Path):
        """Generate interactive dashboard using Plotly."""
        
        # This would create an interactive HTML dashboard
        # For brevity, I'll create a simple example
        
        dashboard_dir = output_dir / "interactive"
        dashboard_dir.mkdir(exist_ok=True)
        
        # Create interactive adapter strength plot
        adapters = results['adapter_analysis']['adapters']
        
        adapter_names = list(adapters.keys())
        strengths = [adapters[name]['adaptation_strength'] for name in adapter_names]
        layer_types = [adapters[name]['layer_type'] for name in adapter_names]
        
        fig = px.bar(
            x=adapter_names, 
            y=strengths, 
            color=layer_types,
            title="Interactive Adapter Strength Analysis",
            labels={'x': 'Adapter', 'y': 'Adaptation Strength', 'color': 'Layer Type'}
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=600,
            showlegend=True
        )
        
        # Save as HTML
        fig.write_html(dashboard_dir / "adapter_strength_interactive.html")
        
        logger.info(f"Interactive dashboard saved to {dashboard_dir}")


def compare_lora_experiments(experiment_paths: List[str], output_dir: Path, 
                           comparison_name: str = "lora_comparison"):
    """Compare multiple LoRA experiments."""
    
    logger.info(f"Comparing {len(experiment_paths)} LoRA experiments")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze each experiment
    experiment_results = {}
    for i, path in enumerate(experiment_paths):
        logger.info(f"Analyzing experiment {i+1}/{len(experiment_paths)}: {path}")
        
        try:
            analyzer = LoRAAnalyzer(path)
            results = analyzer.analyze_comprehensive(
                output_dir / f"exp_{i}", 
                experiment_name=f"exp_{i}"
            )
            experiment_results[f"experiment_{i}"] = results
            
        except Exception as e:
            logger.error(f"Failed to analyze {path}: {e}")
    
    # Generate comparison plots
    _generate_comparison_plots(experiment_results, output_dir)
    
    # Save comparison results
    comparison_file = output_dir / f"{comparison_name}_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(experiment_results, f, indent=2)
    
    logger.info(f"LoRA comparison complete. Results saved to {output_dir}")


def _generate_comparison_plots(experiment_results: Dict, output_dir: Path):
    """Generate comparison plots for multiple experiments."""
    
    plots_dir = output_dir / "comparison_plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Extract effectiveness metrics for comparison
    effectiveness_data = []
    for exp_name, results in experiment_results.items():
        effectiveness = results['effectiveness_metrics']
        effectiveness_data.append({
            'experiment': exp_name,
            'parameter_efficiency': effectiveness['parameter_efficiency'],
            'adaptation_strength': effectiveness['adaptation_strength'],
            'rank_efficiency': effectiveness['rank_efficiency'],
            'layer_coverage': effectiveness['layer_coverage'],
            'overall_score': effectiveness['overall_score']
        })
    
    if not effectiveness_data:
        return
    
    df = pd.DataFrame(effectiveness_data)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('LoRA Experiments Comparison', fontsize=16, fontweight='bold')
    
    metrics = ['parameter_efficiency', 'adaptation_strength', 'rank_efficiency', 
              'layer_coverage', 'overall_score']
    
    for i, metric in enumerate(metrics):
        row = i // 3
        col = i % 3
        
        if row < 2 and col < 3:
            axes[row, col].bar(df['experiment'], df[metric], alpha=0.7)
            axes[row, col].set_title(metric.replace('_', ' ').title())
            axes[row, col].tick_params(axis='x', rotation=45)
            axes[row, col].grid(True, alpha=0.3)
    
    # Remove empty subplot
    fig.delaxes(axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'experiments_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main analysis script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LoRA Analysis Tools")
    parser.add_argument("--model_path", required=True, help="Path to PEFT model/adapter")
    parser.add_argument("--base_model", default="openai/whisper-small", help="Base model path")
    parser.add_argument("--output_dir", default="./lora_analysis_results", help="Output directory")
    parser.add_argument("--experiment_name", default="lora_analysis", help="Experiment name")
    parser.add_argument("--compare", nargs='+', help="Multiple model paths for comparison")
    
    args = parser.parse_args()
    
    if args.compare:
        # Comparison mode
        compare_lora_experiments(
            experiment_paths=args.compare,
            output_dir=Path(args.output_dir),
            comparison_name=args.experiment_name
        )
    else:
        # Single analysis mode
        analyzer = LoRAAnalyzer(args.model_path, args.base_model)
        results = analyzer.analyze_comprehensive(
            output_dir=Path(args.output_dir),
            experiment_name=args.experiment_name
        )
        
        # Print summary
        print("\n" + "="*60)
        print("LORA ANALYSIS SUMMARY")
        print("="*60)
        
        effectiveness = results['effectiveness_metrics']
        print(f"Parameter Efficiency: {effectiveness['parameter_efficiency']:.6f}")
        print(f"Adaptation Strength: {effectiveness['adaptation_strength']:.4f}")
        print(f"Rank Efficiency: {effectiveness['rank_efficiency']:.4f}")
        print(f"Layer Coverage: {effectiveness['layer_coverage']:.4f}")
        print(f"Overall Score: {effectiveness['overall_score']:.4f}")
        
        print(f"\nDetailed analysis saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
