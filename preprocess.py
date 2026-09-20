"""Split manifests and dataset statistics.

Writes to each dataset root:
  manifest_train.json   defect-free reference images
  manifest_test.json    test images with labels (0 normal, 1 defective)
  stats.json            image counts and an image-size summary

Splits:
  sdnet   Decks, pavements and walls pooled. The uncracked and the cracked
          images are each shuffled with random.Random(seed) and split 80/20;
          the reference set is the uncracked 80%, the test set the uncracked
          and cracked 20%, shuffled with the same generator. Seed 0.
  mvtec   Official split for config.MVTEC_CATEGORIES: train/good as reference
          images, test/ as test set, with the ground_truth/ mask paths.
          MVTec AD is downloaded manually (see setup_data.sh).
  vision  Per category of config.VISION_CATEGORIES, in that order, one
          numpy default_rng(seed) draws 80% of the images listed in
          inference/_annotations.coco.json as reference images (kept in COCO
          file order) and leaves the rest as normal test images; the
          generator is consumed once per category that has inference/. The
          images of train/ and then val/ are the defective test images. Seed 0.

Usage:
  python preprocess.py [--datasets sdnet mvtec vision]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import config as cfg

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
STATS_SAMPLE = 500


def _images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMG_EXTS)


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _save_manifest(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  wrote {path} ({len(records)} records)")


def _save_stats(path: Path, stats: dict) -> None:
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  wrote {path}")


def _image_stats(paths: list[Path], seed: int = 0) -> dict:
    """Width and height summary over a seeded sample of up to STATS_SAMPLE images."""
    sample = random.Random(seed).sample(paths, min(STATS_SAMPLE, len(paths)))
    sizes = [Image.open(p).size for p in sample]
    widths, heights = [w for w, _ in sizes], [h for _, h in sizes]
    return {
        "width":  {"min": min(widths),  "max": max(widths),  "mean": sum(widths) // len(widths)},
        "height": {"min": min(heights), "max": max(heights), "mean": sum(heights) // len(heights)},
        "sampled": len(sample),
    }


def _available(root: Path, categories: list[str]) -> list[str]:
    present = {p.name for p in root.iterdir() if p.is_dir()}
    missing = [c for c in categories if c not in present]
    if missing:
        print(f"  WARNING: categories not found: {missing}")
    return [c for c in categories if c in present]


# ── SDNET2018 ─────────────────────────────────────────────────────────────────

def preprocess_sdnet(root: Path, train_ratio: float = 0.8, seed: int = 0) -> bool:
    print("SDNET2018")
    if not root.exists():
        print(f"  SKIP: {root} not found")
        return False

    subsets = {"W": ("CW", "UW"), "D": ("CD", "UD"), "P": ("CP", "UP")}
    all_normal: list[Path] = []
    all_cracked: list[Path] = []
    for folder, (crack_dir, normal_dir) in subsets.items():
        for name, bucket in ((crack_dir, all_cracked), (normal_dir, all_normal)):
            d = root / folder / name
            if not d.exists():
                print(f"  WARNING: {d} not found")
                continue
            imgs = _images(d)
            print(f"  {folder}/{name}: {len(imgs)} images")
            bucket += imgs
    if not all_normal or not all_cracked:
        print("  SKIP: no images")
        return False

    rng = random.Random(seed)

    def split(paths):
        s = paths[:]
        rng.shuffle(s)
        cut = int(len(s) * train_ratio)
        return s[:cut], s[cut:]

    train_normal, test_normal = split(all_normal)
    _, test_cracked = split(all_cracked)

    train_records = [{"path": _relative(p, root), "label": 0, "mask": None}
                     for p in train_normal]
    test_records = (
        [{"path": _relative(p, root), "label": 0, "mask": None} for p in test_normal]
        + [{"path": _relative(p, root), "label": 1, "mask": None} for p in test_cracked])
    rng.shuffle(test_records)

    stats = {
        "total": len(all_normal) + len(all_cracked),
        "normal": len(all_normal),
        "cracked": len(all_cracked),
        "train_normal": len(train_normal),
        "test_normal": len(test_normal),
        "test_cracked": len(test_cracked),
        "train_ratio": train_ratio,
        "seed": seed,
        "has_pixel_masks": False,
        "image_stats": _image_stats(all_normal + all_cracked),
    }
    _save_manifest(root / "manifest_train.json", train_records)
    _save_manifest(root / "manifest_test.json", test_records)
    _save_stats(root / "stats.json", stats)
    return True


# ── MVTec AD ──────────────────────────────────────────────────────────────────

def preprocess_mvtec(root: Path, categories: list[str]) -> bool:
    print("MVTec AD")
    if not root.exists():
        print(f"  SKIP: {root} not found (manual download, see setup_data.sh)")
        return False
    use_cats = _available(root, categories)
    if not use_cats:
        print("  SKIP: no categories found")
        return False

    train_records: list[dict] = []
    test_records: list[dict] = []
    for cat in use_cats:
        good_train = root / cat / "train" / "good"
        if not good_train.exists():
            print(f"  WARNING: {good_train} not found, skipping {cat}")
            continue
        train_imgs = _images(good_train)
        train_records += [{"path": _relative(p, root), "label": 0, "mask": None,
                           "category": cat} for p in train_imgs]

        n_defect = 0
        gt_dir = root / cat / "ground_truth"
        for split_dir in sorted((root / cat / "test").iterdir()):
            is_normal = split_dir.name == "good"
            for p in _images(split_dir):
                mask = None
                if not is_normal:
                    m = gt_dir / split_dir.name / f"{p.stem}_mask.png"
                    mask = _relative(m, root) if m.exists() else None
                    n_defect += 1
                test_records.append({"path": _relative(p, root),
                                     "label": 0 if is_normal else 1,
                                     "mask": mask, "category": cat})
        print(f"  {cat}: {len(train_imgs)} reference | {n_defect} test-defective")

    stats = {
        "categories": use_cats,
        "train_normal": len(train_records),
        "test_normal": sum(r["label"] == 0 for r in test_records),
        "test_defective": sum(r["label"] == 1 for r in test_records),
        "has_pixel_masks": True,
        "image_stats": _image_stats([root / r["path"] for r in train_records + test_records]),
    }
    _save_manifest(root / "manifest_train.json", train_records)
    _save_manifest(root / "manifest_test.json", test_records)
    _save_stats(root / "stats.json", stats)
    return True


# ── VISION ────────────────────────────────────────────────────────────────────

def _coco_images(split_dir: Path) -> list[Path]:
    """Existing images listed in a split directory's COCO file, in file order."""
    with open(split_dir / "_annotations.coco.json") as f:
        coco = json.load(f)
    return [split_dir / img["file_name"] for img in coco["images"]
            if (split_dir / img["file_name"]).exists()]


def preprocess_vision(root: Path, categories: list[str], train_ratio: float = 0.8,
                      seed: int = 0) -> bool:
    print("VISION")
    if not root.exists():
        print(f"  SKIP: {root} not found (see setup_data.sh)")
        return False
    use_cats = _available(root, categories)
    if not use_cats:
        print("  SKIP: no categories found")
        return False

    def records(paths: list[Path], cat: str, label: int) -> list[dict]:
        return [{"path": _relative(p, root), "label": label, "mask": None,
                 "category": cat} for p in paths]

    rng = np.random.default_rng(seed)
    train_records: list[dict] = []
    test_records: list[dict] = []
    for cat in use_cats:
        reference: list[Path] = []
        normal_test: list[Path] = []
        inference_dir = root / cat / "inference"
        if inference_dir.exists():
            normal = _coco_images(inference_dir)
            idx = rng.choice(len(normal), int(train_ratio * len(normal)), replace=False)
            chosen = set(idx.tolist())
            reference = [normal[i] for i in sorted(idx)]
            normal_test = [p for i, p in enumerate(normal) if i not in chosen]
        defective = [p for split in ("train", "val") if (root / cat / split).exists()
                     for p in _coco_images(root / cat / split)]
        train_records += records(reference, cat, 0)
        test_records += records(normal_test, cat, 0) + records(defective, cat, 1)
        print(f"  {cat}: {len(reference)} reference | {len(normal_test)} test-normal "
              f"| {len(defective)} test-defective")

    stats = {
        "categories": use_cats,
        "train_normal": len(train_records),
        "test_total": len(test_records),
        "test_defective": sum(r["label"] == 1 for r in test_records),
        "has_pixel_masks": True,
        "mask_note": "COCO polygon annotations; manifest mask fields are null and "
                     "vision_masks.py rasterises the polygons",
        "image_stats": _image_stats([root / r["path"] for r in train_records + test_records]),
    }
    _save_manifest(root / "manifest_train.json", train_records)
    _save_manifest(root / "manifest_test.json", test_records)
    _save_stats(root / "stats.json", stats)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--datasets", nargs="+", choices=cfg.DATASETS, default=cfg.DATASETS)
    args = p.parse_args()

    ready = {}
    if "sdnet" in args.datasets:
        ready["sdnet"] = preprocess_sdnet(cfg.DATASET_PATHS["sdnet"])
    if "mvtec" in args.datasets:
        ready["mvtec"] = preprocess_mvtec(cfg.DATASET_PATHS["mvtec"], cfg.MVTEC_CATEGORIES)
    if "vision" in args.datasets:
        ready["vision"] = preprocess_vision(cfg.DATASET_PATHS["vision"], cfg.VISION_CATEGORIES)

    missing = [name for name, ok in ready.items() if not ok]
    if missing:
        sys.exit(f"not ready: {', '.join(missing)} (see setup_data.sh)")
    print("all datasets ready")


if __name__ == "__main__":
    main()
