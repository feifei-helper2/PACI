from __future__ import annotations
from pathlib import Path
import json
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from .model import BYOLModel, _make_optimizer, _lr_at_step, _atomic_torch_save
from .config import RelationConfig
from .relation import RelationBank, build_relation_bank, relation_distillation_loss
from .utils import setup_runtime, seed_all, write_json, capture_rng, restore_rng


class LocalPositionDataset(Dataset):
    """Expose the subset row index, aligning minibatches with the relation bank."""
    def __init__(self, base):
        self.base = base
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        views, y, _ = self.base[i]
        return views, y, int(i)


def relation_beta(epoch: int, cfg: RelationConfig) -> float:
    if epoch <= cfg.relation_start_epoch:
        return 0.0
    if cfg.relation_ramp_epochs <= 0:
        return float(cfg.relation_beta_max)
    p = (epoch - cfg.relation_start_epoch) / float(cfg.relation_ramp_epochs)
    return float(cfg.relation_beta_max) * min(max(p, 0.0), 1.0)


def should_refresh_bank(epoch: int, first_epoch: int, cfg: RelationConfig) -> bool:
    if epoch == first_epoch:
        return True
    return epoch > cfg.relation_start_epoch and epoch % cfg.relation_refresh_epochs == 0


def _load_training_state(model, optimizer, scaler, path: Path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt["epoch"]) + 1, int(ckpt.get("global_step", 0)), ckpt


def _forward_components(model: BYOLModel, x1: torch.Tensor, x2: torch.Tensor):
    z1_raw = model.online_projector(model.online_encoder(x1))
    z2_raw = model.online_projector(model.online_encoder(x2))
    q1 = F.normalize(model.predictor(z1_raw).float(), dim=1)
    q2 = F.normalize(model.predictor(z2_raw).float(), dim=1)
    with torch.no_grad():
        t1 = model._target(x1)
        t2 = model._target(x2)
    loss12 = 2.0 - 2.0 * (q1 * t2).sum(dim=1)
    loss21 = 2.0 - 2.0 * (q2 * t1).sum(dim=1)
    byol_loss = 0.5 * (loss12.mean() + loss21.mean())
    return byol_loss, z1_raw, z2_raw, t1, t2


def train_paci(dataset: str, train_dataset, eval_dataset, k: int,
                     byol_cfg, relation_cfg: RelationConfig, base_cfg, seed: int,
                     warmup_checkpoint: str | Path,
                     run_dir: str | Path, resume: bool = True,
                     final_epoch_override: int | None = None):
    setup_runtime(); seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("PACI training requires CUDA")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    branch_ckpt = run_dir / "checkpoint_last.pt"
    warmup_checkpoint = Path(warmup_checkpoint)
    if not warmup_checkpoint.exists():
        raise FileNotFoundError(warmup_checkpoint)
    model = BYOLModel(dataset, byol_cfg).to(device)
    model.online_encoder.to(memory_format=torch.channels_last)
    model.target_encoder.to(memory_format=torch.channels_last)
    optimizer = _make_optimizer(model, byol_cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=byol_cfg.amp)
    source_ckpt = branch_ckpt if resume and branch_ckpt.exists() else warmup_checkpoint
    start_epoch, global_step, saved = _load_training_state(model, optimizer, scaler, source_ckpt, device)
    expected = relation_cfg.shared_warmup_epochs + 1
    if source_ckpt == warmup_checkpoint and start_epoch != expected:
        raise RuntimeError(f"warmup checkpoint must end at epoch {relation_cfg.shared_warmup_epochs}; got next epoch {start_epoch}")
    final_epoch = int(relation_cfg.total_epochs if final_epoch_override is None else final_epoch_override)
    if start_epoch > final_epoch:
        return model, {"epochs_completed": final_epoch, "checkpoint": str(branch_ckpt if branch_ckpt.exists() else source_ckpt), "resumed_complete": True}
    train_generator = torch.Generator()
    train_generator.manual_seed(int(seed) + 700001)
    loader = DataLoader(LocalPositionDataset(train_dataset), batch_size=byol_cfg.batch_size,
                        shuffle=True, num_workers=byol_cfg.num_workers, pin_memory=True,
                        drop_last=True, persistent_workers=byol_cfg.num_workers > 0,
                        generator=train_generator)
    if len(loader) == 0:
        raise RuntimeError("training subset is smaller than batch size")
    bank = None
    if source_ckpt == branch_ckpt:
        state = saved.get('relation_bank')
        if state is not None:
            bank = RelationBank(**state)
        if 'train_generator' in saved:
            train_generator.set_state(saved['train_generator'].cpu())
        if 'rng' in saved:
            restore_rng(saved['rng'])
    del saved
    bank_refresh_count = 0
    log_path = run_dir / "train_log.jsonl"
    bank_log_path = run_dir / "relation_bank_log.jsonl"
    t_train = time.time()
    for epoch in range(start_epoch, final_epoch + 1):
        if bank is None or should_refresh_bank(epoch, expected, relation_cfg):
            bank = build_relation_bank(model, eval_dataset, byol_cfg, base_cfg, k, seed)
            rec_bank = {"epoch": epoch, **bank.diagnostics}
            with open(bank_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec_bank) + "\n")
            bank_refresh_count += 1
        model.train()
        sum_total = sum_byol = sum_rel = 0.0
        n_batches = 0
        t0 = time.time()
        beta = relation_beta(epoch, relation_cfg)
        for views, _, pos in loader:
            x1, x2 = views
            x1 = x1.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            x2 = x2.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            pos = pos.to(device, non_blocking=True).long()
            base_lr = _lr_at_step(global_step, len(loader), byol_cfg)
            for pg in optimizer.param_groups:
                pg["lr"] = base_lr * float(pg.get("lr_mult", 1.0))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=byol_cfg.amp):
                byol_loss, z1, z2, t1, t2 = _forward_components(model, x1, x2)
            rel_loss = byol_loss.new_zeros(())
            if beta > 0.0:
                if bank is None:
                    raise RuntimeError("relation bank missing")
                # Compute relation probabilities and logits in FP32.
                with torch.autocast(device_type="cuda", enabled=False):
                    target = bank.batch_distribution(pos)
                    r12 = relation_distillation_loss(z1, t2, target, relation_cfg.relation_temperature)
                    r21 = relation_distillation_loss(z2, t1, target, relation_cfg.relation_temperature)
                    rel_loss = 0.5 * (r12 + r21)
            loss = byol_loss + float(beta) * rel_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.ema_update()
            sum_total += float(loss.detach().item())
            sum_byol += float(byol_loss.detach().item())
            sum_rel += float(rel_loss.detach().item())
            n_batches += 1; global_step += 1
        rec = {
            "epoch": epoch,
            "method": "paci",
            "total_loss": sum_total / max(n_batches, 1),
            "byol_loss": sum_byol / max(n_batches, 1),
            "relation_loss": sum_rel / max(n_batches, 1),
            "relation_beta": float(beta),
            "base_lr": _lr_at_step(global_step, len(loader), byol_cfg),
            "epoch_seconds": time.time() - t0,
            "global_step": global_step,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        if epoch % byol_cfg.checkpoint_every == 0 or epoch == final_epoch:
            _atomic_torch_save({
                "epoch": epoch, "global_step": global_step,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "dataset": dataset, "seed": seed,
                "method": "paci", "byol_config": byol_cfg.__dict__,
                "relation_config": relation_cfg.__dict__,
                "rng": capture_rng(),
                "train_generator": train_generator.get_state(),
                "relation_bank": {'factor': bank.factor, 'inv_rowsum': bank.inv_rowsum,
                                  'q': bank.q, 'diagnostics': bank.diagnostics},
            }, branch_ckpt)
    stats = {
        "method": "paci",
        "train_seconds": time.time() - t_train,
        "epochs_completed": final_epoch,
        "branch_start_epoch": start_epoch,
        "global_step": global_step,
        "relation_bank_refreshes": int(bank_refresh_count),
        "checkpoint": str(branch_ckpt),
    }
    write_json(run_dir / "train_stats.json", stats)
    return model, stats
