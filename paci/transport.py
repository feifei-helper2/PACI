from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import torch
import torch.nn.functional as F

@dataclass(frozen=True)
class SinkhornDiag:
    converged: bool
    iterations: int
    source_residual: float
    target_residual: float


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensor(x: np.ndarray | torch.Tensor, device=None) -> torch.Tensor:
    if device is None: device = _device()
    if isinstance(x, torch.Tensor): return x.to(device)
    return torch.from_numpy(np.asarray(x)).to(device)


def row_l2_t(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=1)


@torch.inference_mode()
def gpu_spherical_kmeans(x: torch.Tensor, k: int, seed: int, n_init: int = 12,
                         max_iter: int = 100, tol: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor, float]:
    x = row_l2_t(x)
    n, d = x.shape
    if not 1 <= k <= n or n_init < 1 or max_iter < 1:
        raise ValueError("Require 1 <= k <= n and positive iteration counts")
    best_obj = -1e30
    best_labels = best_centers = None
    for r in range(n_init):
        g = torch.Generator(device=x.device); g.manual_seed(seed * 1009 + r * 7919 + 17)
        centers = []
        first = int(torch.randint(n, (1,), generator=g, device=x.device))
        centers.append(x[first])
        min_dist = 1.0 - x @ centers[0]
        for _ in range(1, k):
            prob = torch.clamp(min_dist, min=1e-8)
            prob = prob / prob.sum()
            idx = int(torch.multinomial(prob, 1, generator=g))
            c = x[idx]
            centers.append(c)
            min_dist = torch.minimum(min_dist, 1.0 - x @ c)
        c = torch.stack(centers, dim=0)
        prev_obj = None
        for _ in range(max_iter):
            sim = x @ c.T
            labels = sim.argmax(dim=1)
            newc = torch.zeros((k, d), device=x.device, dtype=torch.float32)
            newc.index_add_(0, labels, x)
            counts = torch.bincount(labels, minlength=k).float()
            empty = counts == 0
            if empty.any():
                repl = torch.randperm(n, generator=g, device=x.device)[: int(empty.sum())]
                newc[empty] = x[repl]
                counts[empty] = 1.0
            newc = row_l2_t(newc / counts[:, None])
            obj = float((x * newc[labels]).sum().item())
            if prev_obj is not None and abs(obj - prev_obj) <= tol * max(abs(prev_obj), 1.0):
                c = newc; break
            c = newc; prev_obj = obj
        sim = x @ c.T
        labels = sim.argmax(dim=1)
        obj = float(sim.max(dim=1).values.sum().item())
        if obj > best_obj:
            best_obj = obj; best_labels = labels.clone(); best_centers = c.clone()
    return best_labels, best_centers, best_obj


@torch.inference_mode()
def sinkhorn_log(cost: torch.Tensor, q: torch.Tensor, epsilon: float,
                 max_iterations: int = 2000, tolerance: float = 1e-6,
                 check_interval: int = 20) -> tuple[torch.Tensor, SinkhornDiag]:
    if epsilon <= 0 or max_iterations < 1 or tolerance <= 0:
        raise ValueError("Sinkhorn parameters must be positive")
    if q.ndim != 1 or cost.ndim != 2 or cost.shape[1] != len(q):
        raise ValueError("Incompatible cost and capacity shapes")
    if not torch.isfinite(cost).all() or not torch.isfinite(q).all() or (q < 0).any() or q.sum() <= 0:
        raise ValueError("Expected finite costs and nonnegative, nonzero capacities")
    cost = cost.float(); q = q.float(); q = q / q.sum()
    n, k = cost.shape
    a = torch.full((n,), 1.0 / n, device=cost.device, dtype=torch.float32)
    la, lq = torch.log(a), torch.log(q)
    f = torch.zeros(n, device=cost.device); g = torch.zeros(k, device=cost.device)
    plan = None; src = tgt = math.inf; converged = False
    for it in range(1, max_iterations + 1):
        f = epsilon * (la - torch.logsumexp((g[None, :] - cost) / epsilon, dim=1))
        g = epsilon * (lq - torch.logsumexp((f[:, None] - cost) / epsilon, dim=0))
        if it == 1 or it % check_interval == 0 or it == max_iterations:
            plan = torch.exp((f[:, None] + g[None, :] - cost) / epsilon)
            src = float((plan.sum(1) - a).abs().max().item())
            tgt = float((plan.sum(0) - q).abs().max().item())
            if max(src, tgt) <= tolerance:
                converged = True; break
    return plan, SinkhornDiag(converged, it, src, tgt)


class LowRankRelationTorch:
    def __init__(self, factor: torch.Tensor, eps: float = 1e-12):
        self.f = factor.float()
        self.diag_h = (self.f * self.f).sum(1)
        rowsum_h0 = self.f @ self.f.sum(0) - self.diag_h
        self.inv_rowsum = 1.0 / rowsum_h0.clamp_min(eps)
    def h0_matmat(self, v: torch.Tensor) -> torch.Tensor:
        return self.f @ (self.f.T @ v) - self.diag_h[:, None] * v
    def matmat(self, v: torch.Tensor) -> torch.Tensor:
        left = self.inv_rowsum[:, None] * self.h0_matmat(v)
        right = self.h0_matmat(self.inv_rowsum[:, None] * v)
        return 0.5 * (left + right)
    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        return self.matmat(v[:, None])[:, 0]
    def degree(self) -> torch.Tensor:
        return self.matvec(torch.ones(self.f.shape[0], device=self.f.device))


@torch.inference_mode()
def relation_from_prototypes(x: torch.Tensor, prototypes: torch.Tensor, q: torch.Tensor,
                             epsilon: float, max_iterations: int, tolerance: float):
    x = row_l2_t(x); prototypes = row_l2_t(prototypes); q = q.float(); q = q / q.sum()
    cost = 1.0 - x @ prototypes.T
    plan, diag = sinkhorn_log(cost, q, epsilon, max_iterations, tolerance)
    factor = math.sqrt(float(len(x))) * plan / torch.sqrt(q.clamp_min(1e-30))[None, :]
    return LowRankRelationTorch(factor), diag

