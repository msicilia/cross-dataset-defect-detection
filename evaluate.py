from __future__ import annotations
"""Evaluation metrics for anomaly detection and defect classification."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalResult:
    image_auroc: float
    image_ap: float
    pixel_auroc: Optional[float] = None
    pixel_ap: Optional[float] = None
    threshold: Optional[float] = None
    f1_at_threshold: Optional[float] = None


def image_level_metrics(scores: np.ndarray, labels: np.ndarray) -> EvalResult:
    """Compute image-level AUROC and AP.

    Args:
        scores: anomaly scores, higher = more anomalous, shape (N,)
        labels: binary labels, 1 = defective, 0 = normal, shape (N,)
    """
    auroc = roc_auc_score(labels, scores)
    ap    = average_precision_score(labels, scores)

    # Optimal threshold via Youden's J statistic
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j        = tpr - fpr
    best_idx = int(np.argmax(j))
    thresh   = float(thresholds[best_idx])

    preds = (scores >= thresh).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return EvalResult(
        image_auroc=auroc,
        image_ap=ap,
        threshold=thresh,
        f1_at_threshold=f1,
    )


def pixel_level_metrics(
    score_maps: list[np.ndarray],
    mask_paths: list[Optional[str]],
) -> tuple[Optional[float], Optional[float]]:
    """Compute pixel-level AUROC and AP where ground-truth masks exist.

    Args:
        score_maps: list of 2-D anomaly score arrays, one per image
        mask_paths: list of mask file paths (None if no mask available)

    Returns:
        (pixel_auroc, pixel_ap) or (None, None) if no masks present
    """
    from PIL import Image

    all_scores, all_labels = [], []
    for smap, mpath in zip(score_maps, mask_paths):
        if mpath is None:
            continue
        mask = np.array(Image.open(mpath).convert("L")) > 0
        if smap.shape != mask.shape:
            from PIL import Image as PILImage
            import PIL
            smap_img = PILImage.fromarray(smap.astype(np.float32))
            smap = np.array(smap_img.resize(mask.shape[::-1], PIL.Image.BILINEAR))
        all_scores.append(smap.ravel())
        all_labels.append(mask.ravel().astype(np.uint8))

    if not all_scores:
        return None, None

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    if labels.sum() == 0:
        return None, None

    return (
        float(roc_auc_score(labels, scores)),
        float(average_precision_score(labels, scores)),
    )


def generalisation_gap(
    results: dict[str, dict[str, EvalResult]],
) -> dict[str, float]:
    """Compute per-method generalisation gap Δ = mean_diag - mean_off_diag.

    Args:
        results: results[method][source__target] = EvalResult
    """
    datasets = sorted({k.split("__")[0] for k in next(iter(results.values()))})
    gaps = {}
    for method, res_dict in results.items():
        diag, off = [], []
        for src in datasets:
            for tgt in datasets:
                key = f"{src}__{tgt}"
                if key not in res_dict:
                    continue
                auroc = res_dict[key].image_auroc
                if src == tgt:
                    diag.append(auroc)
                else:
                    off.append(auroc)
        if diag and off:
            gaps[method] = float(np.mean(diag) - np.mean(off))
    return gaps
