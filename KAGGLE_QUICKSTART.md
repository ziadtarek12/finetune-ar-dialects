# 🚀 Kaggle Quick Start Guide: PEFT LoRA Arabic Dialects

This guide helps you run the Arabic dialect PEFT experiments on Kaggle efficiently with **automatic HuggingFace data loading**.

## ✨ Zero Data Setup Required!

**The scripts automatically download data from the official paper's HuggingFace collection:**
- Collection: [`otozz/overcoming-data-scarcity-in-multi-dialectal-arabic-asr`](https://huggingface.co/collections/otozz/overcoming-data-scarcity-in-multi-dialectal-arabic-asr-67949653e522c7de7fdddc7a)
- **No manual data preparation needed!**
- Over 75k total samples across 5 Arabic dialects

## 🔧 Setup (5 minutes)

### 1. Kaggle Environment Setup
```python
# Check GPU availability
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'None'}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "No GPU")
```

### 2. Install PEFT Dependencies
```python
# Install required packages (only needed if not in requirements.txt)
!pip install peft>=0.7.0 bitsandbytes>=0.41.0 accelerate>=0.20.0
!pip install evaluate jiwer librosa soundfile
```

## 🚀 Quick Start Commands

### 🧪 Step 0: Pipeline Validation (RECOMMENDED FIRST)
```python
# Before running full experiments, validate your setup works!
!python dialect_peft_training.py 
    --dialect egyptian 
    --model_size small 
    --quick_test 
    --use_peft 
    --data_source huggingface

# This completes in 5-10 minutes and validates:
# ✅ Data loading from HuggingFace
# ✅ Model setup and PEFT configuration  
# ✅ Training pipeline functionality
# ✅ Evaluation and metrics computation
```

### Option 1: Single Dialect Training
```python
# Egyptian dialect - fastest training, most data (16.1k samples auto-downloaded)
!python src/training/dialect_peft_training.py \
    --dialect egyptian \
    --model_size small \
    --use_peft True \
    --use_huggingface True \
    --load_in_8bit True \
    --seed 42 \
    --output_dir ./results/egyptian \
    --max_epochs 5 \
    --batch_size 16

# Expected: ✅ Auto-downloads otozz/egyptian_train_set & otozz/egyptian_test_set
```

### Option 2: All Dialects (2-3 hours total)
```python
# Run each dialect sequentially with auto-download
dialects = ['egyptian', 'gulf', 'iraqi', 'levantine', 'maghrebi']

for dialect in dialects:
    print(f"🚀 Training {dialect} dialect...")
    !python dialect_peft_training.py \
        --dialect {dialect} \
        --use_peft \
        --load_in_8bit \
        --seed 42 \
        --output_dir ./results/{dialect}
```

### Option 3: Statistical Significance (Multiple Seeds)
```python
# Run with multiple seeds for robust evaluation
seeds = [42, 84, 168]

for seed in seeds:
    print(f"🎯 Training Egyptian dialect with seed {seed}...")
    !python dialect_peft_training.py \
        --dialect egyptian \
        --use_peft \
        --load_in_8bit \
        --seed {seed} \
        --output_dir ./results/egyptian_seed{seed}
```

## 📊 Generate Publication Results
```python
# After experiments complete, generate analysis
!python generate_publication_results.py \
    --results_dir ./results \
    --output_dir ./publication

# View the generated files
import os
print("📁 Generated files:")
for root, dirs, files in os.walk("./publication"):
    for file in files:
        print(f"  {os.path.join(root, file)}")
```

## 🎯 Expected Training Times on Kaggle T4

| Mode | Dialect | Data Size | Training Time | Memory Usage |
|------|---------|-----------|---------------|--------------|
| **Quick Test** | Any | 50 samples | 5-10 min | ~2GB |
| Egyptian | 20h | 30-45 min | ~4GB |
| Gulf | 20h | 30-45 min | ~4GB |
| Iraqi | 13h | 20-30 min | ~4GB |
| Levantine | 20h | 30-45 min | ~4GB |
| Maghrebi | 17h | 25-35 min | ~4GB |

## 🧪 Interactive Notebook
For step-by-step experimentation, use:
```python
# Open the enhanced notebook
%run arabicfintunewhisper-peft.ipynb
```

## 🔍 Monitor Progress
```python
# Check training progress
import json
import glob

# Find result files
result_files = glob.glob("./results/*/results*.json")
for file in result_files:
    with open(file, 'r') as f:
        data = json.load(f)
        print(f"File: {file}")
        print(f"  WER: {data.get('wer', 'N/A'):.2f}%")
        print(f"  CER: {data.get('cer', 'N/A'):.2f}%")
        print()
```

## 📈 Quick Results Visualization
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load and visualize results
results = []
for file in glob.glob("./results/*/results*.json"):
    with open(file, 'r') as f:
        data = json.load(f)
        results.append(data)

df = pd.DataFrame(results)

# Plot WER comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='dialect', y='wer')
plt.title('PEFT LoRA WER Results by Dialect')
plt.ylabel('Word Error Rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## 💡 Kaggle-Specific Tips

1. **Start Small**: Test with Egyptian dialect first (fastest)
2. **Save Regularly**: Kaggle can timeout, save checkpoints frequently
3. **Memory Management**: Use `load_in_8bit=True` for T4 GPUs
4. **Time Limits**: Single dialect takes 30-45 minutes max
5. **Storage**: PEFT adapters are small (~60MB), save multiple experiments

## 🚨 Troubleshooting

### Out of Memory
```python
# Reduce batch size
--batch_size 8  # instead of 16

# Use gradient accumulation
--gradient_accumulation_steps 2
```

### Slow Training
```python
# Reduce epochs for testing
--max_epochs 3

# Use smaller model
--model_size tiny  # instead of small
```

### Dataset Issues
```python
# Check dataset loading
from datasets import load_dataset
dataset = load_dataset("mozilla-foundation/common_voice_16_1", "ar")
print(f"Dataset loaded: {len(dataset['train'])} samples")
```

## 📊 Publication-Ready Outputs

After running experiments, you'll have:
- `publication/tables/performance_comparison.csv` - Main results
- `publication/figures/efficiency_analysis.png` - Key plots  
- `publication/PEFT_Analysis_Report.md` - Complete analysis
- `publication/latex/` - LaTeX tables for papers

Perfect for academic publication! 🎓
