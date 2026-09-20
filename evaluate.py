"""Image-level evaluation metrics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass
class ImageMetrics:
    image_auroc: float
    image_ap: float


def image_level_metrics(scores, labels) -> ImageMetrics:
    """AUROC and average precision of anomaly scores (higher = more anomalous)
    against binary labels (1 = defective)."""
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    return ImageMetrics(image_auroc=float(roc_auc_score(labels, scores)),
                        image_ap=float(average_precision_score(labels, scores)))
