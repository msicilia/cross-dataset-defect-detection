from __future__ import annotations
"""MVTec Anomaly Detection dataset loader.

Uses manifest_train.json / manifest_test.json produced by preprocess.py when
available (fast path). Falls back to directory scanning otherwise.

Expected directory layout:
    <root>/
      <category>/
        train/good/          ← defect-free training images
        test/good/           ← defect-free test images
        test/<defect_type>/  ← defective test images
        ground_truth/<defect_type>/  ← binary mask images (.png)
"""
import json
from pathlib import Path
from typing import Optional

from .base import DefectDataset, DatasetSplit

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _images(d: Path) -> list[Path]:
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXTS)


class MVTecDataset(DefectDataset):
    name = "mvtec"

    def __init__(self, root: Path, categories: Optional[list[str]] = None):
        self.root = Path(root)
        available = sorted(p.name for p in self.root.iterdir() if p.is_dir()
                           if not p.name.startswith("."))
        self.categories = categories if categories is not None else available

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

    def _scan_category(self, cat: str):
        cat_dir  = self.root / cat
        test_dir = cat_dir / "test"
        gt_dir   = cat_dir / "ground_truth"
        train_imgs, test_imgs, test_labels, masks = [], [], [], []

        good_train = cat_dir / "train" / "good"
        train_imgs = _images(good_train) if good_train.exists() else []

        for split_dir in sorted(test_dir.iterdir()):
            is_normal = (split_dir.name == "good")
            gt_sub = gt_dir / split_dir.name if not is_normal else None
            for p in _images(split_dir):
                mask = None
                if gt_sub:
                    m = gt_sub / (p.stem + "_mask.png")
                    mask = m if m.exists() else None
                test_imgs.append(p)
                test_labels.append(0 if is_normal else 1)
                masks.append(mask)

        return train_imgs, test_imgs, test_labels, masks

    # ── public interface ──────────────────────────────────────────────────────

    def normal_train(self) -> DatasetSplit:
        cached = self._from_manifest("train")
        if cached is not None:
            return cached
        imgs = []
        for cat in self.categories:
            train_good = self.root / cat / "train" / "good"
            if train_good.exists():
                imgs += _images(train_good)
        return DatasetSplit(imgs, [0] * len(imgs), [None] * len(imgs))

    def test(self) -> DatasetSplit:
        cached = self._from_manifest("test")
        if cached is not None:
            return cached
        all_imgs, all_labels, all_masks = [], [], []
        for cat in self.categories:
            _, t_imgs, t_labels, t_masks = self._scan_category(cat)
            all_imgs   += t_imgs
            all_labels += t_labels
            all_masks  += t_masks
        return DatasetSplit(all_imgs, all_labels, all_masks)
