from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import os
import time
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from .config import BYOLConfig
from .utils import setup_runtime, seed_all, write_json, capture_rng, restore_rng


def build_resnet18_backbone(dataset: str) -> nn.Module:
    model = resnet18(weights=None)
    if dataset in {"cifar10", "cifar20", "cifar100"}:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    model.fc = nn.Identity()
    return model


def projector_dim(dataset: str, cfg: BYOLConfig) -> int:
    return cfg.stl_projector_dim if dataset in {"stl10", "tinyimagenet"} else cfg.cifar_projector_dim


def mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )


class BYOLModel(nn.Module):
    def __init__(self, dataset: str, cfg: BYOLConfig):
        super().__init__()
        self.dataset = dataset
        self.ema_momentum = float(cfg.ema_momentum)
        out_dim = projector_dim(dataset, cfg)
        self.online_encoder = build_resnet18_backbone(dataset)
        self.online_projector = mlp(512, cfg.projector_hidden_dim, out_dim)
        self.predictor = mlp(out_dim, cfg.projector_hidden_dim, out_dim)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        for p in self.target_projector.parameters():
            p.requires_grad_(False)

    def _online(self, x):
        z = self.online_projector(self.online_encoder(x))
        return self.predictor(z)

    @torch.no_grad()
    def _target(self, x):
        z = self.target_projector(self.target_encoder(x))
        return F.normalize(z.float(), dim=1)

    def forward(self, x1, x2):
        q1 = self._online(x1)
        q2 = self._online(x2)
        with torch.no_grad():
            t1 = self._target(x1)
            t2 = self._target(x2)
        q1 = F.normalize(q1.float(), dim=1)
        q2 = F.normalize(q2.float(), dim=1)
        loss12 = 2.0 - 2.0 * (q1 * t2).sum(dim=1)
        loss21 = 2.0 - 2.0 * (q2 * t1).sum(dim=1)
        return 0.5 * (loss12.mean() + loss21.mean())

    @torch.no_grad()
    def ema_update(self):
        m = self.ema_momentum
        online = list(self.online_encoder.parameters()) + list(self.online_projector.parameters())
        target = list(self.target_encoder.parameters()) + list(self.target_projector.parameters())
        for po, pt in zip(online, target):
            pt.data.mul_(m).add_(po.data, alpha=1.0 - m)

    @torch.no_grad()
    def target_features(self, x):
        return self._target(x)


def _atomic_torch_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _lr_at_step(global_step: int, steps_per_epoch: int, cfg: BYOLConfig) -> float:
    epoch = global_step / max(steps_per_epoch, 1) + 1.0
    if epoch < cfg.warmup_epochs:
        return cfg.learning_rate * epoch / cfg.warmup_epochs
    progress = (epoch - cfg.warmup_epochs) / max(cfg.epochs - cfg.warmup_epochs, 1)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * cfg.learning_rate * (1.0 + math.cos(math.pi * progress))


def _make_optimizer(model: BYOLModel, cfg: BYOLConfig):
    base = list(model.online_encoder.parameters()) + list(model.online_projector.parameters())
    pred = list(model.predictor.parameters())
    return torch.optim.SGD([
        {"params": base, "lr": cfg.learning_rate, "lr_mult": 1.0},
        {"params": pred, "lr": cfg.learning_rate * cfg.predictor_lr_multiplier,
         "lr_mult": cfg.predictor_lr_multiplier},
    ], lr=cfg.learning_rate, momentum=cfg.momentum, weight_decay=cfg.weight_decay)


def train_warmup(dataset: str, train_dataset, cfg: BYOLConfig, seed: int,
               run_dir: str | Path, resume: bool = True,
               epochs_override: int | None = None) -> tuple[BYOLModel, dict]:
    setup_runtime(); seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("PACI warm-up requires CUDA")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "checkpoint_last.pt"
    model = BYOLModel(dataset, cfg).to(device)
    model.online_encoder.to(memory_format=torch.channels_last)
    model.target_encoder.to(memory_format=torch.channels_last)
    optimizer = _make_optimizer(model, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
    loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
                        persistent_workers=cfg.num_workers > 0)
    if len(loader) == 0:
        raise RuntimeError("training subset is smaller than batch size")
    start_epoch = 1; global_step = 0
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt["epoch"]) + 1
        global_step = int(ckpt.get("global_step", (start_epoch - 1) * len(loader)))
        if 'rng' in ckpt:
            restore_rng(ckpt['rng'])
        del ckpt
    final_epoch = int(cfg.epochs if epochs_override is None else epochs_override)
    log_path = run_dir / "train_log.jsonl"
    t_train = time.time()
    for epoch in range(start_epoch, final_epoch + 1):
        model.train()
        loss_sum = 0.0; n_batches = 0
        t0 = time.time()
        for views, _, _ in loader:
            x1, x2 = views
            x1 = x1.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            x2 = x2.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            base_lr = _lr_at_step(global_step, len(loader), cfg)
            for pg in optimizer.param_groups:
                pg["lr"] = base_lr * float(pg.get("lr_mult", 1.0))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.amp):
                loss = model(x1, x2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.ema_update()
            loss_sum += float(loss.detach().item()); n_batches += 1; global_step += 1
        rec = {
            "epoch": epoch,
            "loss": loss_sum / max(n_batches, 1),
            "base_lr": _lr_at_step(global_step, len(loader), cfg),
            "epoch_seconds": time.time() - t0,
            "global_step": global_step,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            import json; f.write(json.dumps(rec) + "\n")
        if epoch % cfg.checkpoint_every == 0 or epoch == final_epoch:
            _atomic_torch_save({
                "epoch": epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "dataset": dataset,
                "seed": seed,
                "config": cfg.__dict__,
                "rng": capture_rng(),
            }, ckpt_path)
    stats = {
        "train_seconds": time.time() - t_train,
        "epochs_completed": final_epoch,
        "steps_per_epoch": len(loader),
        "global_step": global_step,
        "checkpoint": str(ckpt_path),
    }
    write_json(run_dir / "train_stats.json", stats)
    return model, stats


@torch.inference_mode()
def extract_target_projector_features(model: BYOLModel, eval_dataset, cfg: BYOLConfig,
                                      out_path: str | Path, metadata: dict) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("feature extraction requires CUDA")
    device = torch.device("cuda")
    model.eval()
    loader = DataLoader(eval_dataset, batch_size=cfg.eval_batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True,
                        persistent_workers=cfg.num_workers > 0)
    feats = []; labels = []; indices = []
    t0 = time.time()
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.amp):
            z = model.target_features(x)
        feats.append(z.float().cpu())
        labels.append(y.long().cpu())
        indices.append(idx.long().cpu())
    f = torch.cat(feats).numpy().astype("float16")
    y = torch.cat(labels).numpy().astype("int64")
    ix = torch.cat(indices).numpy().astype("int64")
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np
    np.savez_compressed(out_path, features=f, labels=y, indices=ix,
                        metadata=np.array(json.dumps(metadata)))
    return {"feature_seconds": time.time() - t0, "feature_dim": int(f.shape[1]), "n": int(f.shape[0])}
