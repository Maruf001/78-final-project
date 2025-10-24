# Generalized Mixup Model 

Implementation of **Generalized Mixup** – an extension of the classic *mixup* data-augmentation technique that adaptively re-weights the pairing distribution using the model’s confusion matrix.  The codebase supports experiments on **CIFAR-10** and **CIFAR-100** with *standard*, *weighted*, and *ERM* (no mixup) baselines.

The method is introduced in our paper:

> [*From Confusion to Clarity – Generalized Mixup*](https://github.com/Maruf001/generalized_mixup_model/blob/main/From%20Confusion%20to%20Clarity%20%E2%80%93%20Generalized%20Mixup.pdf)  
> COSC278 – Deep Learning, Dartmouth College, Spring 2024

The report is included (`From Confusion to Clarity – Generalized Mixup.pdf`).

---

## Table of Contents
1. [Key Ideas](#key-ideas)  
2. [Repository Layout](#repository-layout)  
3. [Requirements](#requirements)  
4. [Installation](#installation)  
5. [Quick Start](#quick-start)  
6. [Script Arguments](#script-arguments)  
7. [Logging & Artifacts](#logging--artifacts)  
8. [Reproducing Paper Results](#reproducing-paper-results)  
9. [Citation](#citation)  
10. [License](#license)

---

## Key Ideas
* **Stratified mini-batches** – every batch contains the same number of samples per class ensuring a well-conditioned confusion matrix.
* **Weighted mixup** – two samples *(x₁, y₁)* and *(x₂, y₂)* are drawn according to a probability *p(y₁→y₂)* proportional to the model’s current confusion matrix **C** raised to a power *γ*.  A higher *γ* focuses the model on frequent confusions.
* **Adaptive regularisation** – *γ* is multiplied by `gamma_growth_factor` every `step_size` epochs, gradually sharpening the distribution as training progresses.

## Repository Layout
```
├── train.py                 # CIFAR-10 training loop
├── trainCIFAR100.py         # CIFAR-100 training loop
├── StratifiedSampler.py     # balanced batch sampler
├── models/                  # ResNet & VGG architectures (CIFAR-10)
├── cifar100models/          # ResNet & VGG (100-class output)
├── utils.py                 # misc helpers + progress bar
├── results/                 # CIFAR-10 logs & checkpoints
├── cifar100results/         # CIFAR-100 logs & checkpoints
└── *.ipynb                  # analysis / visualisation notebooks
```

## Requirements
* Python ≥ 3.8
* PyTorch ≥ 2.0 (CUDA strongly recommended)
* torchvision
* numpy
* (optional) jupyter, matplotlib, seaborn for notebooks

The exact versions used for the paper were:
```
Python          3.10.13
PyTorch         2.3.0
Torchvision     0.18.0
CUDA            12.1
```

## Installation
```bash
# 1) create an isolated environment (conda, venv, poetry, etc.)
conda create -n mixup python=3.10 -y
conda activate mixup

# 2) install dependencies
pip install torch torchvision numpy
```
> Substitute the matching CUDA wheel if using a GPU.

## Quick Start
### CIFAR-10
```python
from train import train_cifar10

# Standard mixup baseline
train_cifar10(mixup="standard", alpha=1.0, name="std_mixup")

# Generalised (confusion-weighted) mixup
train_cifar10(mixup="weighted", gamma=0.125, step_size=20, name="weighted_mixup")
```

### CIFAR-100
```python
from trainCIFAR100 import train_cifar100
train_cifar100(mixup="weighted", batch_size=1000, gamma=1.0, name="cifar100_wmixup")
```
Training automatically creates `results/` or `cifar100results/` and checkpoints after every epoch.

## Script Arguments
Both `train_cifar10` and `train_cifar100` expose the same signature:

| Argument | Default | Description |
|---|---|---|
| `lr` | `0.1` | SGD learning-rate (decays ×0.1 @ 50 & 75 epochs) |
| `resume` | `False` | Load checkpoint from `checkpoint/` |
| `model` | `"ResNet18"` | Any key in `models.__dict__` or `cifar100models.__dict__` |
| `batch_size` | `250` (`1000` for CIFAR-100) | Must be divisible by number of classes |
| `n_epochs` | `100` | Total training epochs |
| `mixup` | `"standard"` | `standard`, `weighted`, or `erm` (no mixup) |
| `alpha` | `1.0` | Beta distribution parameter controlling interpolation |
| `gamma` | `0.125` | Regularising exponent for *weighted* mixup |
| `gamma_growth_factor` | `1.5` | Multiplier applied to *γ* every `step_size` epochs |
| `mu` | `0.9` | EMA momentum for confusion-matrix update |
| `augment` | `True` | Use random-crop & flip augmentation |
| `live` | `False` | Print tqdm-style progress bar |
| `seed` | `0` | Reproducibility seed |

Call the function with keyword arguments, e.g. `train_cifar10(gamma=0.25, step_size=10)`.

## Logging & Artifacts
* **CSV logs** – one row per epoch containing train/test metrics (`results/log_<run>.csv`).
* **Checkpoints** – full model & rng state every epoch (`checkpoint/ckpt.t7_<run>_<epoch>`).
* **Confusion Matrices** – tensor of shape *[C × C × epochs]* saved after training (`results/cm_<run>.pt`).

Notebooks in the repo demonstrate how to visualise these artifacts.

## Reproducing Paper Results
1. Download CIFAR-10/100 to `~/data` (torchvision will fetch automatically on first run if `download=True`).
2. Execute the hyper-parameters listed in *Table 2* of the report using a single NVIDIA A100-80 GB (or similar).
3. Plot training curves with `analysis.ipynb`.

## Citation
If you use this code, please cite the course paper:
```text
@report{generalized_mixup_2025,
  title     = {From Confusion to Clarity – Generalized Mixup},
  author    = {T. Simpson, H. Stropkay, A. Maruf, D. Mason, A. Johnson},
  year      = {2024},
  institution = {Dartmouth College},
  note      = {COSC278 – Deep Learning}
}
```

## License
This repository re-implements and extends [facebookresearch/mixup-cifar10](https://github.com/facebookresearch/mixup-cifar10) (BSD-style license).  Unless stated otherwise the new code is released under the same terms.  See the original repository for full details.
