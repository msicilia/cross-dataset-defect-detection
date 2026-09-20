"""MVTec AD loader, official split.

Reads manifest_train.json / manifest_test.json written by preprocess.py.
Reference images are train/good; the test split mixes test/good (label 0) with
the defective test images (label 1), whose masks are in ground_truth/. Records
are restricted to the loader's categories and keep their manifest order.

Directory layout:
    <root>/<category>/
        train/good/
        test/good/, test/<defect_type>/
        ground_truth/<defect_type>/<image>_mask.png
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import DatasetSplit, DefectDataset, read_manifest


class MVTecDataset(DefectDataset):
    name = "mvtec"

    def __init__(self, root: Path, categories: Optional[list[str]] = None):
        self.root = Path(root)
        available = sorted(p.name for p in self.root.iterdir()
                           if p.is_dir() and not p.name.startswith("."))
        self.categories = categories if categories is not None else available

    def _from_manifest(self, name: str) -> DatasetSplit:
        wanted = set(self.categories)
        records = [r for r in read_manifest(self.root, name) if r["category"] in wanted]
        imgs = [self.root / r["path"] for r in records]
        labels = [r["label"] for r in records]
        masks = [(self.root / r["mask"] if r.get("mask") else None) for r in records]
        return DatasetSplit(imgs, labels, masks)

    def normal_train(self) -> DatasetSplit:
        return self._from_manifest("train")

    def test(self) -> DatasetSplit:
        return self._from_manifest("test")
