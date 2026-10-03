from __future__ import annotations
import numpy as np
import torch

def gpu_pca_reduce(x: np.ndarray, dim: int, seed: int, max_dim: int | None = None) -> np.ndarray:
    """Center features, apply randomized PCA and L2-normalize each row."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
    xt = xt - xt.mean(dim=0, keepdim=True)
    q = int(max(dim + 16, dim if max_dim is None else max_dim))
    q = min(q, xt.shape[1], xt.shape[0] - 1)
    torch.manual_seed(seed)
    _, _, v = torch.pca_lowrank(xt, q=q, center=False, niter=2)
    z = xt @ v[:, :dim]
    z = torch.nn.functional.normalize(z, dim=1)
    return z.cpu().numpy().astype(np.float32)


def preprocess_features(raw: np.ndarray, frozen_dim: int, seed: int) -> tuple[np.ndarray, int]:
    d = int(min(frozen_dim, raw.shape[1]))
    if d == raw.shape[1]:
        x = raw.astype(np.float32)
        x = x - x.mean(axis=0, keepdims=True)
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return (x / np.clip(n, 1e-12, None)).astype(np.float32), d
    return gpu_pca_reduce(raw, d, seed), d

