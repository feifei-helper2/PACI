from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from .datasets import load_dataset, make_long_tail_indices

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class TwoCropTransform:
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, x):
        return self.transform(x), self.transform(x)

class SelectedDataset(Dataset):
    def __init__(self, base: Dataset, indices: np.ndarray, transform=None):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        base_idx = int(self.indices[i])
        img, y = self.base[base_idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, int(y), base_idx

@dataclass
class DatasetBundle:
    train: SelectedDataset
    eval: SelectedDataset
    labels: np.ndarray
    indices: np.ndarray
    desired_counts: np.ndarray
    num_classes: int
    image_size: int


def normalization(name: str):
    if name == "cifar10":
        return transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    if name in {"cifar20", "cifar100"}:
        return transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)


def byol_train_transform(name: str, crop_scale_min: float = 0.08,
                         gaussian_blur: bool = False):
    size = 96 if name == "stl10" else (64 if name == "tinyimagenet" else 32)
    ops = [
        transforms.RandomResizedCrop(size=size, scale=(crop_scale_min, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
    ]
    if gaussian_blur:
        kernel = 23 if size >= 96 else 3
        ops.append(transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=kernel, sigma=(0.1, 2.0))
        ], p=0.5))
    ops += [transforms.ToTensor(), normalization(name)]
    return TwoCropTransform(transforms.Compose(ops))


def byol_eval_transform(name: str):
    return transforms.Compose([transforms.ToTensor(), normalization(name)])


def build_datasets(name: str, root: str, imbalance_factor: float, seed: int,
                       crop_scale_min: float = 0.08, gaussian_blur: bool = False,
                       download: bool = True) -> DatasetBundle:
    bundle = load_dataset(name, root, transform=None, download=download)
    idx, desired = make_long_tail_indices(bundle.labels, bundle.num_classes,
                                          imbalance_factor, seed,
                                          permute_frequency_order=False)
    labels = bundle.labels[idx].astype(np.int64)
    train = SelectedDataset(bundle.dataset, idx,
                            byol_train_transform(name, crop_scale_min, gaussian_blur))
    ev = SelectedDataset(bundle.dataset, idx, byol_eval_transform(name))
    return DatasetBundle(train, ev, labels, idx, desired.astype(np.int64),
                            bundle.num_classes, 96 if name == "stl10" else (64 if name == "tinyimagenet" else 32))
