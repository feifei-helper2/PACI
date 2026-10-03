"""Spherical K-means evaluation on target-projector features."""
from pathlib import Path
import numpy as np
from .transport import row_l2_t, to_tensor, gpu_spherical_kmeans
from .metrics import evaluate


def evaluate_features(feature_path: str | Path, num_classes: int,
                      n_init: int = 10, max_iter: int = 300) -> dict:
    with np.load(feature_path, allow_pickle=False) as data:
        raw = data['features'].astype(np.float32)
        y = data['labels'].astype(np.int64)
    if len(raw) != len(y) or not np.isfinite(raw).all():
        raise ValueError('Invalid feature cache')
    if (y < 0).any() or (y >= num_classes).any():
        raise ValueError('Labels are outside the configured class range')
    x = row_l2_t(to_tensor(raw))
    pred, _, obj = gpu_spherical_kmeans(x, num_classes, seed=0,
                                       n_init=n_init, max_iter=max_iter)
    pred = pred.cpu().numpy()
    result = evaluate(y, pred, np.bincount(y, minlength=num_classes), num_classes)
    np.save(Path(feature_path).with_name('predictions.npy'), pred)
    return {'method': 'paci', 'evaluator': 'target_projector_spherical_kmeans',
            'metric_scale': 'fraction', 'n': len(y), 'k': num_classes,
            'feature_dim': raw.shape[1], 'kmeans_objective': obj, **result}
