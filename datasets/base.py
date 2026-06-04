from __future__ import annotations
"""Abstract base class for all defect datasets."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DatasetSplit:
    """Unified representation of a dataset split."""
    image_paths: list[Path]
    labels: list[int]           # 0 = normal, 1 = defective
    mask_paths: list[Optional[Path]]  # None where pixel masks are unavailable


class DefectDataset(ABC):
    """Common interface for all datasets used in the benchmark."""

    name: str  # set by subclass

    @abstractmethod
    def normal_train(self) -> DatasetSplit:
        """Defect-free training images (used to build anomaly model)."""

    @abstractmethod
    def test(self) -> DatasetSplit:
        """Test split with labels and optional pixel masks."""

    def __repr__(self) -> str:
        tr = self.normal_train()
        te = self.test()
        n_def = sum(te.labels)
        return (
            f"{self.name}: {len(tr.image_paths)} normal train | "
            f"{len(te.image_paths)} test ({n_def} defective)"
        )
