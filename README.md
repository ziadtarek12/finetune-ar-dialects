[![DOI](https://zenodo.org/badge/803227691.svg)](https://doi.org/10.5281/zenodo.15553149)

# Overcoming Data Scarcity in Multi-Dialectal Arabic ASR via Whisper Fine-Tuning

This repository contains the code for training the models evaluated in the paper as well as the results and plotting. The repository now supports both traditional fine-tuning and **Parameter-Efficient Fine-Tuning (PEFT)** using LoRA adapters.

## 🚀 New: PEFT Support with LoRA

We've added support for **Parameter-Efficient Fine-Tuning (PEFT)** using Low-Rank Adaptation (LoRA):

### Benefits of PEFT:
- **Memory Efficient**: Train with ~4GB GPU memory instead of ~16GB
- **Parameter Efficient**: Only train 1% of model parameters (~2M vs 240M)
- **Storage Efficient**: Model adapters are ~60MB vs ~1.5GB full models  
- **Faster Training**: Higher batch sizes and faster convergence
- **Better Generalization**: Less prone to catastrophic forgetting

### Quick Start with PEFT:
```shell
# Train Egyptian dialect with PEFT
python src/training/experiment_finetune_peft.py --dialect egyptian --use_peft --load_in_8bit

# Run comprehensive experiments (both traditional and PEFT)
./run_experiments.sh --type both --dialect egyptian
```

## Setting up environment

To get started, please install the requirements on a Python 3.10 environment. An example using `conda`:

```shell
conda create -n venv python=3.10
conda activate venv
pip install -r requirements.txt
```

The requirements now include PEFT dependencies (`peft>=0.7.0`, `bitsandbytes>=0.41.0`).

## Instructions for usage

### Experiments

#### Traditional Fine-tuning
The `training/experiment_*.py` files expect datasets to be available. Please check out the files before trying to run them. Example uses are displayed below.

```shell
usage: experiment_scratch.py [-h] -d DIALECT

options:
  -h, --help            show this help message and exit
  -d DIALECT, --dialect DIALECT
                        all, egyptian, gulf, iraqi, levantine, maghrebi
```

```shell
usage: experiment_finetune.py [-h] -d DIALECT

options:
  -h, --help            show this help message and exit
  -d DIALECT, --dialect DIALECT
                        all, egyptian, gulf, iraqi, levantine, maghrebi
```

#### PEFT Fine-tuning (New!)
Use the new PEFT training script for parameter-efficient fine-tuning:

```shell
usage: experiment_finetune_peft.py [-h] -d DIALECT [--use_peft] [--lora_rank LORA_RANK] 
                                   [--lora_alpha LORA_ALPHA] [--lora_dropout LORA_DROPOUT] 
                                   [--load_in_8bit] [--seeds SEEDS [SEEDS ...]]

options:
  -h, --help            show this help message and exit
  -d DIALECT, --dialect DIALECT
                        all, egyptian, gulf, iraqi, levantine, maghrebi
  --use_peft            Use PEFT (LoRA) for parameter-efficient fine-tuning
  --lora_rank LORA_RANK LoRA rank (default: 32)
  --lora_alpha LORA_ALPHA LoRA alpha parameter (default: 64)
  --lora_dropout LORA_DROPOUT LoRA dropout (default: 0.05)
  --load_in_8bit        Load model in 8-bit for memory efficiency
  --seeds SEEDS [SEEDS ...] Random seeds for multiple runs (default: 42 84 168)
```

#### Automated Experiment Runner
Use the experiment runner script for comprehensive experiments:

```shell
# Run both traditional and PEFT experiments for all dialects
./run_experiments.sh

# Run only PEFT experiments for a specific dialect
./run_experiments.sh --type peft --dialect egyptian

# Run traditional experiments only
./run_experiments.sh --type traditional

# Get help
./run_experiments.sh --help
```

#### Training Size Experiments
```shell
usage: experiment_trainsize.py [-h] -t TRAIN_SIZE

options:
  -h, --help            show this help message and exit
  -t TRAIN_SIZE, --train_size TRAIN_SIZE
                        Train size between 0 and 1
```

## 📓 Jupyter Notebooks

### Interactive Training Notebooks
- **`ArabicFintuneWhisper.ipynb`**: Traditional two-stage fine-tuning (MSA → Dialect)
- **`ArabicFintuneWhisper_PEFT.ipynb`**: **NEW!** PEFT fine-tuning with LoRA adapters
- **`Whisper_w_PEFT(1).ipynb`**: Reference PEFT implementation for Hindi (used as template)

### Analysis Notebooks
- **`results.ipynb`**: Results analysis and plotting for paper recreation

### Getting Started with Notebooks
```python
# Traditional approach
jupyter notebook ArabicFintuneWhisper.ipynb

# PEFT approach (recommended for limited GPU memory)
jupyter notebook ArabicFintuneWhisper_PEFT.ipynb
```

## 🔧 PEFT Utilities

The repository includes utility modules for PEFT operations:

- **`src/peft_utils.py`**: PEFT model setup, loading, and inference utilities
- **`src/whisper_utils.py`**: Updated with PEFT callbacks and data collators

### Example PEFT Usage
```python
from src.peft_utils import load_peft_model_for_inference, test_peft_model
from transformers import WhisperProcessor

# Load PEFT model for inference
model = load_peft_model_for_inference("./whisper-small-peft-egyptian_final")
processor = WhisperProcessor.from_pretrained("./whisper-small-peft-egyptian_final")

# Test on audio sample
transcription = test_peft_model(model, processor, test_sample)
```

### Evaluation

Evaluation can be done with the `evaluation/evaluate_dialects.py` file for all dialects and the MSA test set, the `evaluation/evaluate_notraining.py` file for the Whisper checkpoints, and the `evaluation/evaluate_trainsize.py` for the training size experiments. Example uses are displayed below.

```shell
usage: evaluate_dialects.py [-h] -c CHECKPOINT

options:
  -h, --help            show this help message and exit
  -c CHECKPOINT, --checkpoint CHECKPOINT
```
```shell
usage: evaluate_notraining.py [-h] -m MODEL_NAME

options:
  -h, --help            show this help message and exit
  -m MODEL_NAME, --model_name MODEL_NAME
```

```shell
usage: evaluate_trainsize.py [-h] -t TRAIN_SIZE

options:
  -h, --help            show this help message and exit
  -t TRAIN_SIZE, --train_size TRAIN_SIZE
```

### Results

The results can be found in `results/` as well as the Jupyter notebook `results.ipynb` required for recreation of the plots in the paper. The final plots can also be found in `results/plots/`.
