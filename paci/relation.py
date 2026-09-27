from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .preprocessing import preprocess_features
from .transport import to_tensor, row_l2_t, gpu_spherical_kmeans, relation_from_prototypes


@dataclass
class RelationBank:
    factor: torch.Tensor
    inv_rowsum: torch.Tensor
    q: torch.Tensor
    diagnostics: dict

    @property
    def n(self) -> int:
        return int(self.factor.shape[0])

    @property
    def k(self) -> int:
        return int(self.factor.shape[1])

    def batch_distribution(self, positions: torch.Tensor) -> torch.Tensor:
        pos = positions.long().to(self.factor.device)
        f = self.factor[pos].float()
        inv = self.inv_rowsum[pos].float()
        h = f @ f.T
        b = h.shape[0]
        eye = torch.eye(b, device=h.device, dtype=torch.bool)
        h = h.masked_fill(eye, 0.0)
        r = 0.5 * (inv[:, None] + inv[None, :]) * h
        r = r.clamp_min(0.0).masked_fill(eye, 0.0)
        row = r.sum(1, keepdim=True)
        fallback = (~eye).float() / max(b - 1, 1)
        p = torch.where(row > 1e-20, r / row.clamp_min(1e-20), fallback)
        p = p.masked_fill(eye, 0.0)
        p = p / p.sum(1, keepdim=True).clamp_min(1e-20)
        return p.detach()


@torch.inference_mode()
def collect_target_projector_features(model, eval_dataset, byol_cfg) -> np.ndarray:
    if not torch.cuda.is_available():
        raise RuntimeError("PACI relation-bank refresh requires CUDA")
    device = torch.device("cuda")
    was_training = model.training
    model.eval()
    loader = DataLoader(eval_dataset, batch_size=byol_cfg.eval_batch_size,
                        shuffle=False, num_workers=byol_cfg.num_workers,
                        pin_memory=True, persistent_workers=byol_cfg.num_workers > 0)
    feats = []
    for x, _, _ in loader:
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=byol_cfg.amp):
            z = model.target_features(x)
        feats.append(z.float().cpu())
    if was_training:
        model.train()
    return torch.cat(feats, dim=0).numpy().astype(np.float32)


@torch.inference_mode()
def build_relation_bank(model, eval_dataset, byol_cfg, base_cfg, k: int,
                        seed: int) -> RelationBank:
    raw = collect_target_projector_features(model, eval_dataset, byol_cfg)
    x_np, pca_eff = preprocess_features(raw, base_cfg.pca_dim, seed)
    device = torch.device("cuda")
    x = row_l2_t(to_tensor(x_np, device).float())
    gps, proto, gps_obj = gpu_spherical_kmeans(
        x, k, seed=seed, n_init=base_cfg.gps_n_init, max_iter=base_cfg.gps_max_iter)
    counts = torch.bincount(gps, minlength=k).float()
    q = counts / counts.sum()
    rel, diag = relation_from_prototypes(
        x, proto, q, base_cfg.epsilon,
        base_cfg.sinkhorn_max_iterations, base_cfg.sinkhorn_tolerance)
    diagnostics = {
        "n": int(len(x)),
        "k": int(k),
        "feature_dim": int(raw.shape[1]),
        "pca_dim_effective": int(pca_eff),
        "gps_obj": float(gps_obj),
        "q_min": float(q.min().item()),
        "q_max": float(q.max().item()),
        "q_entropy": float((-(q * torch.log(q.clamp_min(1e-12))).sum()).item()),
        "sinkhorn_converged": bool(diag.converged),
        "sinkhorn_iterations": int(diag.iterations),
        "sinkhorn_source_residual": float(diag.source_residual),
        "sinkhorn_target_residual": float(diag.target_residual),
    }
    return RelationBank(rel.f.detach(), rel.inv_rowsum.detach(), q.detach(), diagnostics)


def relation_distillation_loss(online_projected: torch.Tensor,
                               target_projected: torch.Tensor,
                               target_distribution: torch.Tensor,
                               temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = F.normalize(online_projected.float(), dim=1)
    t = F.normalize(target_projected.float().detach(), dim=1)
    b = z.shape[0]
    if b < 2:
        return z.sum() * 0.0
    logits = (z @ t.T) / float(temperature)
    eye = torch.eye(b, device=logits.device, dtype=torch.bool)
    logits = logits.masked_fill(eye, -1e4)
    logp = F.log_softmax(logits, dim=1)
    ce = -(target_distribution.detach().float() * logp).sum(1).mean()
    return ce / math.log(max(b - 1, 2))
