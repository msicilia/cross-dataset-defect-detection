from __future__ import annotations
"""SDNET2018 dataset loader.

Uses manifest_train.json / manifest_test.json produced by preprocess.py when
available (fast path). Falls back to directory scanning otherwise.

Expected directory layout (original SDNET2018 structure):
    <root>/
      D/CD/  D/UD/    ← Bridge Decks (cracked / uncracked)
      P/CP/  P/UP/    ← Pavements
      W/CW/  W/UW/    ← Walls  ← default subset

Reference:
    Dorafshan et al., SDNET2018, Utah State University, 2018.
    https://digitalcommons.usu.edu/all_datasets/48/
"""
import json
import random
from pathlib import Path

from .base import DefectDataset, DatasetSplit

_SUBSET_MAP = {
    "walls":     ("W",  "CW",  "UW"),
    "decks":     ("D",  "CD",  "UD"),
    "pavements": ("P",  "CP",  "UP"),
}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMG_EXTS)


class SDNETDataset(DefectDataset):
    name = "sdnet"

    def __init__(
        self,
        root: Path,
        subsets: list[str] | None = None,
        train_ratio: float = 0.8,
        seed: int = 0,
    ):
        self.root = Path(root)
        self.subsets = subsets or ["walls"]
        self.train_ratio = train_ratio
        self.seed = seed

    # ── manifest fast path ────────────────────────────────────────────────────

    def _from_manifest(self, name: str) -> DatasetSplit | None:
        p = self.root / f"manifest_{name}.json"
        if not p.exists():
            return None
        with open(p) as f:
            records = json.load(f)
        imgs   = [self.root / r["path"] for r in records]
        labels = [r["label"] for r in records]
        masks  = [
            (self.root / r["mask"] if r.get("mask") else None)
            for r in records
        ]
        return DatasetSplit(imgs, labels, masks)

    # ── directory scan fallback ───────────────────────────────────────────────

    def _load_all(self) -> tuple[list[Path], list[Path]]:
        cracked, normal = [], []
        for subset in self.subsets:
            folder, c_dir, u_dir = _SUBSET_MAP[subset]
            cracked += _images(self.root / folder / c_dir)
            normal  += _images(self.root / folder / u_dir)
        return normal, cracked

    def _split(self, paths: list[Path]) -> tuple[list[Path], list[Path]]:
        rng = random.Random(self.seed)
        shuffled = paths[:]
        rng.shuffle(shuffled)
        cut = int(len(shuffled) * self.train_ratio)
        return shuffled[:cut], shuffled[cut:]

    # ── public interface ──────────────────────────────────────────────────────

    def normal_train(self) -> DatasetSplit:
        cached = self._from_manifest("train")
        if cached is not None:
            return cached
        normal, _ = self._load_all()
        train_normal, _ = self._split(normal)
        return DatasetSplit(train_normal, [0] * len(train_normal), [None] * len(train_normal))

    def test(self) -> DatasetSplit:
        cached = self._from_manifest("test")
        if cached is not None:
            return cached
        normal, cracked = self._load_all()
        _, test_normal   = self._split(normal)
        _, test_cracked  = self._split(cracked)
        imgs   = test_normal + test_cracked
        labels = [0] * len(test_normal) + [1] * len(test_cracked)
        return DatasetSplit(imgs, labels, [None] * len(imgs))
