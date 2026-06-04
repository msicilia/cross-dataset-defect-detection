from __future__ import annotations
"""Abstract base class for all anomaly detection methods."""
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np


class AnomalyMethod(ABC):
    """Interface shared by all methods in the benchmark."""

    name: str  # set by subclass

    @abstractmethod
    def fit(self, image_paths: list[Path], seed: int = 0) -> None:
        """Build the anomaly model from defect-free training images."""

    @abstractmethod
    def score(self, image_paths: list[Path]) -> np.ndarray:
        """Return an image-level anomaly score for each image (higher = more anomalous)."""

    def score_maps(self, image_paths: list[Path]) -> list[np.ndarray]:
        """Return pixel-level anomaly score maps (optional; default raises NotImplementedError)."""
        raise NotImplementedError(f"{self.name} does not support pixel-level scoring")
