"""Split representation, dataset interface and manifest reader."""
from __future__ import annotations

import json
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
        """Defect-free reference images used to build the normality model."""

    @abstractmethod
    def test(self) -> DatasetSplit:
        """Test split with labels and optional pixel masks."""


def read_manifest(root: Path, name: str) -> list[dict]:
    """Records of <root>/manifest_<name>.json, in file order."""
    p = Path(root) / f"manifest_{name}.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run preprocess.py first")
    with open(p) as f:
        return json.load(f)
