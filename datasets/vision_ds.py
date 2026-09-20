"""VISION loader (metallic subsets).

Reads manifest_train.json / manifest_test.json written by preprocess.py, which
draws the split. Reference images are 80% of inference/ per category; the test
split is the remaining 20% of inference/ (label 0) plus all of train/ and val/
(label 1). A record's category is the first component of its path; records are
restricted to the loader's categories and keep their manifest order. Pixel
masks are rasterised from the COCO polygons by vision_masks.py.

Directory layout (Hugging Face release):
    <root>/<Category>/
        train/, val/     annotated images, each carrying at least one defect
        inference/       unannotated images, treated as defect-free
Each split directory holds _annotations.coco.json and its images.

Reference:
    Bai et al., "VISION Datasets", arXiv:2306.07890, 2023.
    https://huggingface.co/datasets/VISION-Workshop/VISION-Datasets
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import DatasetSplit, DefectDataset, read_manifest


class VISIONDataset(DefectDataset):
    name = "vision"

    def __init__(self, root: Path, categories: Optional[list[str]] = None):
        self.root = Path(root)
        available = sorted(p.name for p in self.root.iterdir()
                           if p.is_dir() and not p.name.startswith("."))
        self.categories = categories if categories is not None else available

    def _from_manifest(self, name: str) -> DatasetSplit:
        wanted = set(self.categories)
        records = [r for r in read_manifest(self.root, name)
                   if Path(r["path"]).parts[0] in wanted]
        imgs = [self.root / r["path"] for r in records]
        labels = [r["label"] for r in records]
        return DatasetSplit(imgs, labels, [None] * len(records))

    def normal_train(self) -> DatasetSplit:
        return self._from_manifest("train")

    def test(self) -> DatasetSplit:
        return self._from_manifest("test")
