# PACI

Geometry-matched relational learning for imbalanced clustering.

## Installation

Use Python 3.10–3.12 and an NVIDIA GPU for training. Install matching PyTorch and torchvision builds for your CUDA runtime. For CUDA 12.1:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e .
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1`.

## Data

CIFAR-10, CIFAR-100 and STL-10 download automatically through torchvision. CIFAR-20 uses the coarse labels of CIFAR-100. Extract Tiny-ImageNet manually:

```text
data/tiny-imagenet-200/train/<class-id>/images/*.JPEG
```

CIFAR and Tiny-ImageNet use their training splits. STL-10 uses the combined labeled train and test splits for transductive clustering. Labels are used to construct long-tailed subsets and compute evaluation metrics; training uses unlabeled images.

## Training

Run from the repository root:

```bash
python -m paci train --config configs/cifar10_if10.json --data-root ./data --output ./runs/cifar10_if10_s4801
```

| Dataset | Configuration | Imbalance factor |
|---|---|---:|
| CIFAR-10 | `configs/cifar10_if10.json` | 10 |
| CIFAR-20 | `configs/cifar20_if10.json` | 10 |
| STL-10 | `configs/stl10_if10.json` | 10 |
| CIFAR-100 | `configs/cifar100_if100.json` | 100 |
| Tiny-ImageNet | `configs/tinyimagenet_if100.json` | 100 |

Each configuration trains for 800 epochs, including 200 epochs of BYOL warm-up. Relation learning uses weight 0.05, temperature 0.40, a 100-epoch ramp and a 25-epoch refresh interval.

The default seed is 4801. Use `--seed 4802` or `--seed 4803` with a separate output directory for each run. `--workers` sets the number of data-loader workers, `--no-download` requires existing datasets, and `--dry-run` prints the resolved configuration without loading data.

## Resume and evaluation

```bash
python -m paci train --config configs/cifar10_if10.json --data-root ./data --output ./runs/cifar10_if10_s4801 --resume
python -m paci evaluate --run ./runs/cifar10_if10_s4801
```

Resume with the same configuration, seed and worker count. To stop after warm-up, pass `--stop-after 200`; resume without this option to finish training. Checkpoints retain the model, optimizer, scaler, relation bank and main-process random states. Data-loader worker random states are not saved, so resumed multiworker runs may differ from uninterrupted runs.

Completed runs contain `features.npz`, `predictions.npy` and `metrics.json`. Evaluation applies spherical K-means to EMA target-projector features. Metrics are stored as fractions; multiply by 100 for percentages. Cluster IDs in `predictions.npy` are unaligned; Hungarian alignment is used to compute accuracy.

To regenerate features from a completed checkpoint:

```bash
python -m paci extract --run ./runs/cifar10_if10_s4801 --data-root ./data
```
