from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, confusion_matrix

def hungarian_map(y_true: np.ndarray, y_pred: np.ndarray, k: int):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(k))
    r, c = linear_sum_assignment(cm.max() - cm)
    pred_to_true = {int(pc): int(tr) for tr, pc in zip(r, c)}
    mapped = np.array([pred_to_true.get(int(z), -1) for z in y_pred], dtype=np.int64)
    return mapped, cm, pred_to_true

def hmt_groups(class_counts: np.ndarray):
    order = np.argsort(-np.asarray(class_counts))
    groups = np.array_split(order, 3)
    return {"head": groups[0], "medium": groups[1], "tail": groups[2]}

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, class_counts: np.ndarray, k: int) -> dict:
    mapped, _, _ = hungarian_map(y_true, y_pred, k)
    correct = mapped == y_true
    acc = float(correct.mean())
    class_acc = np.array([correct[y_true == c].mean() if np.any(y_true == c) else np.nan for c in range(k)])
    caa = float(np.nanmean(class_acc))
    groups = hmt_groups(class_counts)
    out = {
        "acc": acc,
        "caa": caa,
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "head_acc": float(np.nanmean(class_acc[groups["head"]])),
        "medium_acc": float(np.nanmean(class_acc[groups["medium"]])),
        "tail_acc": float(np.nanmean(class_acc[groups["tail"]])),
    }
    return out
