"""SDNET2018 loader.

Reads manifest_train.json / manifest_test.json written by preprocess.py. The
split pools the three subsets (decks D, pavements P, walls W) and partitions the
uncracked and the cracked images 80/20 each, once, with seed 0; every run uses
this same split. Reference images are the uncracked 80%; the test split is the
uncracked and cracked 20%.

Directory layout:
    <root>/D/CD, D/UD, P/CP, P/UP, W/CW, W/UW   (C = cracked, U = uncracked)

Reference:
    Dorafshan et al., SDNET2018, Utah State University, 2018.
    https://digitalcommons.usu.edu/all_datasets/48/
"""
from __future__ import annotations

from pathlib import Path

from .base import DatasetSplit, DefectDataset, read_manifest


class SDNETDataset(DefectDataset):
    name = "sdnet"

    def __init__(self, root: Path):
        self.root = Path(root)

    def _from_manifest(self, name: str) -> DatasetSplit:
        records = read_manifest(self.root, name)
        imgs = [self.root / r["path"] for r in records]
        labels = [r["label"] for r in records]
        return DatasetSplit(imgs, labels, [None] * len(records))

    def normal_train(self) -> DatasetSplit:
        return self._from_manifest("train")

    def test(self) -> DatasetSplit:
        return self._from_manifest("test")
