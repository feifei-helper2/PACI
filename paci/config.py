from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class BYOLConfig:
    epochs: int = 800
    batch_size: int = 256
    eval_batch_size: int = 1024
    num_workers: int = 8
    learning_rate: float = 0.05
    predictor_lr_multiplier: float = 10.0
    momentum: float = 0.9
    weight_decay: float = 5e-4
    warmup_epochs: int = 10
    ema_momentum: float = 0.996
    projector_hidden_dim: int = 4096
    cifar_projector_dim: int = 512
    stl_projector_dim: int = 256
    crop_scale_min: float = 0.08
    gaussian_blur: bool = False
    amp: bool = True
    checkpoint_every: int = 25
    byol_kmeans_n_init: int = 10
    byol_kmeans_max_iter: int = 300


@dataclass(frozen=True)
class RelationConfig:
    total_epochs: int = 800
    shared_warmup_epochs: int = 200
    relation_start_epoch: int = 200
    relation_ramp_epochs: int = 100
    relation_beta_max: float = 0.05
    relation_temperature: float = 0.40
    relation_refresh_epochs: int = 25


@dataclass(frozen=True)
class GeometryConfig:
    pca_dim: int = 384
    epsilon: float = 0.12
    sinkhorn_max_iterations: int = 2000
    sinkhorn_tolerance: float = 1e-6
    gps_n_init: int = 12
    gps_max_iter: int = 100
