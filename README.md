[![DOI](https://zenodo.org/badge/803227691.svg)](https://doi.org/10.5281/zenodo.15553149)

# Overcoming Data Scarcity in Multi-Dialectal Arabic ASR via Whisper Fine-Tuning

This repository contains the code for training the models evaluated in the paper as well as the results and plotting.

## Setting up environment

To get started, please install the requirements on a Python 3.10 environment. An example using `conda`:

```shell
conda create -n venv python=3.10
conda activate venv
pip install -r requirements.txt
```

## Instructions for usage

### Experiments
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

```shell
usage: experiment_trainsize.py [-h] -t TRAIN_SIZE

options:
  -h, --help            show this help message and exit
  -t TRAIN_SIZE, --train_size TRAIN_SIZE
                        Train size between 0 and 1
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
