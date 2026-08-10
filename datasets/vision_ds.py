from __future__ import annotations
"""VISION industrial inspection dataset loader.

Uses manifest_train.json / manifest_test.json produced by preprocess_vision.py
when available (fast path). Falls back to COCO JSON parsing otherwise.

Actual directory layout (after HuggingFace download):
    <root>/
      <Category>/
        train/          ← ALL defective images (annotated)
          _annotations.coco.json
          *.jpg
        val/            ← ALL defective images (annotated)
          _annotations.coco.json
          *.jpg
        inference/      ← ALL normal / defect-free images (zero annotations)
          _annotations.coco.json
          *.jpg

Normal training split: 80% of inference/ images (per-category, seeded).
Test split: 20% of inference/ (label=0) + all train/ + val/ (label=1).
No pixel-level masks; evaluation is image-level only.

Categories used (metallic, as corrosion proxy):
    Casting, Ring, Screw, Cylinder

Reference:
    Bai et al., "VISION Datasets", arXiv:2306.07890, 2023.
    https://huggingface.co/datasets/VISION-Workshop/VISION-Datasets
"""
import json
from pathlib import Path
from typing import Optional

from .base import DefectDataset, DatasetSplit


def _load_coco_images(split_dir: Path) -> list[Path]:
    """Return all image paths listed in the COCO JSON for a split directory."""
    ann_file = split_dir / "_annotations.coco.json"
    if not ann_file.exists():
        return sorted(p for p in split_dir.glob("*.jpg"))
    with open(ann_file) as f:
        coco = json.load(f)
    return [split_dir / img["file_name"] for img in coco["images"]
            if (split_dir / img["file_name"]).exists()]


class VISIONDataset(DefectDataset):
    name = "vision"

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
        masks  = [None] * len(records)   # VISION has no pixel masks
        return DatasetSplit(imgs, labels, masks)

    # ── public interface ──────────────────────────────────────────────────────

    def normal_train(self) -> DatasetSplit:
        cached = self._from_manifest("train")
        if cached is not None:
            return cached
        import numpy as np
        rng = np.random.default_rng(0)
        imgs = []
        for cat in self.categories:
            inf_dir = self.root / cat / "inference"
            if not inf_dir.exists():
                continue
            all_normal = _load_coco_images(inf_dir)
            n_train = int(0.8 * len(all_normal))
            idx = rng.choice(len(all_normal), n_train, replace=False)
            imgs += [all_normal[i] for i in sorted(idx)]
        return DatasetSplit(imgs, [0] * len(imgs), [None] * len(imgs))

    def test(self) -> DatasetSplit:
        cached = self._from_manifest("test")
        if cached is not None:
            return cached
        import numpy as np
        rng = np.random.default_rng(0)
        all_imgs, all_labels = [], []
        for cat in self.categories:
            # Normal test: 20% of inference images
            inf_dir = self.root / cat / "inference"
            if inf_dir.exists():
                all_normal = _load_coco_images(inf_dir)
                n_train = int(0.8 * len(all_normal))
                train_idx = set(rng.choice(len(all_normal), n_train, replace=False))
                test_normal = [p for i, p in enumerate(all_normal) if i not in train_idx]
                all_imgs   += test_normal
                all_labels += [0] * len(test_normal)
            # Defective test: all train/ and val/ images
            for split in ("train", "val"):
                split_dir = self.root / cat / split
                if split_dir.exists():
                    defect_imgs = _load_coco_images(split_dir)
                    all_imgs   += defect_imgs
                    all_labels += [1] * len(defect_imgs)
        return DatasetSplit(all_imgs, all_labels, [None] * len(all_imgs))
