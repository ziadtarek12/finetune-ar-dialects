# Overcoming Data Scarcity in Multi-Dialectal Arabic ASR via Whisper Fine-Tuning
[![DOI](https://zenodo.org/badge/803227691.svg)](https://doi.org/10.5281/zenodo.15553149)
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
usage: experiment_dialect.py [-h] -d DIALECT

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
usage: experiment_msa.py [-h] -t TRAIN_SIZE

options:
  -h, --help            show this help message and exit
  -t TRAIN_SIZE, --train_size TRAIN_SIZE
                        Train size between 0 and 1
```

### Evaluation

Evaluation can be done with both the `training/evaluate_all.py` and `training/evaluate_whisper*.py` files, with the latter being a manual input of the model checkpoint and only evaluating on MSA. `training/evaluate_all.py` evaluates on all test sets:

```shell
usage: evaluate_all.py [-h] -c CHECKPOINT

options:
  -h, --help            show this help message and exit
  -c CHECKPOINT, --checkpoint CHECKPOINT
```

### Results

The results can be found in `results/` as well as the Jupyter notebooks required for recreation of the plots in the thesis. `results/training_plots.ipynb` plots the training processes, while `results/results.ipynb` plots the final results. The plots can also be found in `results/plots/`
