"""Interface shared by all anomaly detectors."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class AnomalyMethod(ABC):
    name: str

    @abstractmethod
    def fit(self, image_paths: list[Path], seed: int = 0) -> None:
        """Build the normality model from defect-free reference images."""

    @abstractmethod
    def score(self, image_paths: list[Path]) -> np.ndarray:
        """One anomaly score per image; higher means more anomalous."""

    def score_maps(self, image_paths: list[Path]) -> list[np.ndarray]:
        """Per-pixel anomaly maps, for detectors that provide them."""
        raise NotImplementedError(f"{self.name} does not produce anomaly maps")
