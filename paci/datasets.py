from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pickle
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, STL10, ImageFolder

@dataclass
class DatasetBundle:
    dataset: Dataset
    labels: np.ndarray
    name: str
    num_classes: int

class IndexedSubset(Dataset):
    def __init__(self, base: Dataset, indices: np.ndarray):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        return self.base[int(self.indices[i])]

class CIFAR20Coarse(Dataset):
    """CIFAR-100 train images with official 20 coarse labels."""
    def __init__(self, root: str, train: bool = True, download: bool = True, transform=None):
        self.base = CIFAR100(root=root, train=train, download=download, transform=None)
        self.transform = transform
        raw_dir = Path(root) / "cifar-100-python"
        file = raw_dir / ("train" if train else "test")
        with open(file, "rb") as f:
            obj = pickle.load(f, encoding="latin1")
        self.targets = list(map(int, obj["coarse_labels"]))
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        img, _ = self.base[i]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[i]



def _tinyimagenet_train_dir(root: str) -> Path:
    base = Path(root)
    candidates = [
        base / "tiny-imagenet-200" / "train",
        base / "tinyimagenet" / "train",
        base / "tiny-imagenet" / "train",
        base / "Tiny-ImageNet" / "train",
        base / "train",
    ]
    for c in candidates:
        if c.is_dir() and any(x.is_dir() for x in c.iterdir()):
            return c
    raise FileNotFoundError(
        "TinyImageNet train split not found. Expected e.g. "
        f"{base / 'tiny-imagenet-200' / 'train'}. Extract tiny-imagenet-200 under the dataset root."
    )

class STLCombined(Dataset):
    def __init__(self, root: str, download: bool = True, transform=None):
        self.train = STL10(root=root, split="train", download=download, transform=None)
        self.test = STL10(root=root, split="test", download=download, transform=None)
        self.transform = transform
        self.targets = np.concatenate([np.asarray(self.train.labels), np.asarray(self.test.labels)]).astype(int).tolist()
    def __len__(self):
        return len(self.train) + len(self.test)
    def __getitem__(self, i):
        if i < len(self.train): img, y = self.train[i]
        else: img, y = self.test[i - len(self.train)]
        if self.transform is not None:
            img = self.transform(img)
        return img, int(y)

def load_dataset(name: str, root: str, transform=None, download: bool = True) -> DatasetBundle:
    name = name.lower()
    if name == "cifar10":
        ds = CIFAR10(root=root, train=True, download=download, transform=transform)
        labels = np.asarray(ds.targets, dtype=np.int64)
        return DatasetBundle(ds, labels, name, 10)
    if name == "cifar20":
        ds = CIFAR20Coarse(root=root, train=True, download=download, transform=transform)
        labels = np.asarray(ds.targets, dtype=np.int64)
        return DatasetBundle(ds, labels, name, 20)
    if name == "cifar100":
        ds = CIFAR100(root=root, train=True, download=download, transform=transform)
        labels = np.asarray(ds.targets, dtype=np.int64)
        return DatasetBundle(ds, labels, name, 100)
    if name == "tinyimagenet":
        train_dir = _tinyimagenet_train_dir(root)
        ds = ImageFolder(root=str(train_dir), transform=transform)
        labels = np.asarray(ds.targets, dtype=np.int64)
        return DatasetBundle(ds, labels, name, len(ds.classes))
    if name == "stl10":
        ds = STLCombined(root=root, download=download, transform=transform)
        labels = np.asarray(ds.targets, dtype=np.int64)
        return DatasetBundle(ds, labels, name, 10)
    raise ValueError(f"unknown dataset {name}")

def long_tail_counts(labels: np.ndarray, k: int, imbalance_factor: float) -> np.ndarray:
    if not np.isfinite(imbalance_factor) or imbalance_factor < 1:
        raise ValueError("imbalance_factor must be finite and >= 1")
    counts0 = np.bincount(labels, minlength=k)
    nmax = int(counts0.min())
    factor = 1.0 / float(imbalance_factor)
    counts = np.array([int(round(nmax * (factor ** (c / max(k - 1, 1))))) for c in range(k)], dtype=np.int64)
    counts = np.maximum(counts, 2)
    counts = np.minimum(counts, counts0)
    return counts

def make_long_tail_indices(labels: np.ndarray, k: int, imbalance_factor: float, seed: int,
                           permute_frequency_order: bool = False) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    desired = long_tail_counts(labels, k, imbalance_factor)
    class_order = np.arange(k)
    if permute_frequency_order:
        class_order = rng.permutation(k)
    desired_by_class = np.empty(k, dtype=np.int64)
    for rank, cls in enumerate(class_order):
        desired_by_class[cls] = desired[rank]
    chosen = []
    for cls in range(k):
        ids = np.flatnonzero(labels == cls)
        ids = rng.permutation(ids)
        chosen.append(ids[: int(desired_by_class[cls])])
    indices = np.concatenate(chosen)
    indices = rng.permutation(indices)
    return indices.astype(np.int64), desired_by_class
