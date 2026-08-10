from __future__ import annotations
"""Dataset preprocessing: verification, split manifests, and statistics.

Creates under each dataset root:
  manifest_train.json   — defect-free training images
  manifest_test.json    — test images with labels (and mask paths where available)
  stats.json            — image counts, size distribution, class balance

Supported datasets (those available or auto-downloadable):
  sdnet      SDNET2018           (already in data/sdnet2018/)
  mvtec      MVTec AD            (download via setup_data.sh)
  vision     VISION Workshop     (download via setup_data.sh)

Usage:
  python preprocess.py                          # all available datasets
  python preprocess.py --datasets sdnet mvtec   # specific datasets
  python preprocess.py --cache-resize 256       # also write resized copies
"""
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

# ── helpers ───────────────────────────────────────────────────────────────────

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMG_EXTS)


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _save_manifest(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"    Saved {path.name}: {len(records)} records")


def _save_stats(path: Path, stats: dict) -> None:
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"    Saved {path.name}")


def _image_stats(paths: list[Path]) -> dict:
    """Sample up to 500 images to estimate size distribution."""
    sample = random.sample(paths, min(500, len(paths)))
    widths, heights = [], []
    for p in sample:
        try:
            w, h = Image.open(p).size
            widths.append(w)
            heights.append(h)
        except Exception:
            pass
    if not widths:
        return {}
    return {
        "width":  {"min": min(widths),  "max": max(widths),  "mean": sum(widths)  // len(widths)},
        "height": {"min": min(heights), "max": max(heights), "mean": sum(heights) // len(heights)},
        "sampled": len(widths),
    }


def _resize_cache(src: Path, dst: Path, size: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src).convert("RGB").resize((size, size), Image.BILINEAR)
    img.save(dst, quality=92)


# ── SDNET2018 ─────────────────────────────────────────────────────────────────

def preprocess_sdnet(
    root: Path,
    train_ratio: float = 0.8,
    seed: int = 0,
    cache_size: Optional[int] = None,
) -> bool:
    print("\n── SDNET2018 ──────────────────────────────────────────")
    if not root.exists():
        print(f"  SKIP: {root} not found")
        return False

    subsets = {
        "W": ("CW", "UW"),   # walls
        "D": ("CD", "UD"),   # decks
        "P": ("CP", "UP"),   # pavements
    }

    all_normal: list[Path] = []
    all_cracked: list[Path] = []
    missing: list[str] = []

    for folder, (crack_dir, normal_dir) in subsets.items():
        c_path = root / folder / crack_dir
        u_path = root / folder / normal_dir
        for p, label in [(c_path, "cracked"), (u_path, "normal")]:
            if not p.exists():
                missing.append(str(p))
                continue
            imgs = _images(p)
            print(f"  {folder}/{p.name}: {len(imgs):,} images")
            if label == "cracked":
                all_cracked += imgs
            else:
                all_normal += imgs

    if missing:
        print(f"  WARNING: missing directories: {missing}")

    # Reproducible stratified split
    rng = random.Random(seed)
    def split(paths):
        s = paths[:]
        rng.shuffle(s)
        cut = int(len(s) * train_ratio)
        return s[:cut], s[cut:]

    train_normal, test_normal = split(all_normal)
    train_cracked, test_cracked = split(all_cracked)

    # Training manifest: defect-free only (for anomaly model fitting)
    train_records = [
        {"path": _relative(p, root), "label": 0, "mask": None}
        for p in train_normal
    ]

    # Test manifest: all images with labels
    test_records = (
        [{"path": _relative(p, root), "label": 0, "mask": None} for p in test_normal] +
        [{"path": _relative(p, root), "label": 1, "mask": None} for p in test_cracked]
    )
    rng.shuffle(test_records)

    all_paths = all_normal + all_cracked
    stats = {
        "total": len(all_paths),
        "normal": len(all_normal),
        "cracked": len(all_cracked),
        "train_normal": len(train_normal),
        "test_normal": len(test_normal),
        "test_cracked": len(test_cracked),
        "train_ratio": train_ratio,
        "seed": seed,
        "has_pixel_masks": False,
        "image_stats": _image_stats(all_paths[:500]),
    }

    _save_manifest(root / "manifest_train.json", train_records)
    _save_manifest(root / "manifest_test.json",  test_records)
    _save_stats(root / "stats.json", stats)

    if cache_size:
        _cache_dataset(root, train_records + test_records, cache_size)

    print(f"  ✓ SDNET2018: {stats['total']:,} total | "
          f"{stats['normal']:,} normal / {stats['cracked']:,} cracked")
    return True


# ── MVTec AD ──────────────────────────────────────────────────────────────────

def preprocess_mvtec(
    root: Path,
    categories: list[str],
    cache_size: Optional[int] = None,
) -> bool:
    print("\n── MVTec AD ───────────────────────────────────────────")
    if not root.exists():
        print(f"  SKIP: {root} not found. Run: bash setup_data.sh")
        return False

    available = sorted(p.name for p in root.iterdir() if p.is_dir())
    use_cats  = [c for c in categories if c in available]
    missing   = [c for c in categories if c not in available]

    if missing:
        print(f"  WARNING: categories not found: {missing}")
        print(f"  Available: {available}")
    if not use_cats:
        print("  SKIP: no usable categories found")
        return False

    all_train_records: list[dict] = []
    all_test_records: list[dict] = []
    total_normal = total_defective = 0

    for cat in use_cats:
        cat_dir = root / cat
        good_train = cat_dir / "train" / "good"
        if not good_train.exists():
            print(f"  WARNING: {cat}/train/good missing, skipping")
            continue

        train_imgs = _images(good_train)
        all_train_records += [
            {"path": _relative(p, root), "label": 0, "mask": None, "category": cat}
            for p in train_imgs
        ]
        total_normal += len(train_imgs)
        print(f"  {cat}: {len(train_imgs):>4} train-normal", end="")

        test_dir = cat_dir / "test"
        gt_dir   = cat_dir / "ground_truth"
        n_defect = 0

        for split_dir in sorted(test_dir.iterdir()):
            is_normal = (split_dir.name == "good")
            gt_sub = gt_dir / split_dir.name if not is_normal else None
            for p in _images(split_dir):
                mask = None
                if gt_sub:
                    m = gt_sub / (p.stem + "_mask.png")
                    mask = _relative(m, root) if m.exists() else None
                all_test_records.append({
                    "path":     _relative(p, root),
                    "label":    0 if is_normal else 1,
                    "mask":     mask,
                    "category": cat,
                })
                if not is_normal:
                    n_defect += 1

        total_defective += n_defect
        print(f" | {n_defect:>4} test-defective")

    stats = {
        "categories": use_cats,
        "train_normal": total_normal,
        "test_normal":  sum(1 for r in all_test_records if r["label"] == 0),
        "test_defective": total_defective,
        "has_pixel_masks": True,
        "image_stats": _image_stats(
            [root / r["path"] for r in all_train_records[:200]]
        ),
    }

    _save_manifest(root / "manifest_train.json", all_train_records)
    _save_manifest(root / "manifest_test.json",  all_test_records)
    _save_stats(root / "stats.json", stats)

    if cache_size:
        _cache_dataset(root, all_train_records + all_test_records, cache_size)

    print(f"  ✓ MVTec AD ({', '.join(use_cats)}): "
          f"{total_normal:,} train-normal | {total_defective:,} test-defective")
    return True


# ── VISION ────────────────────────────────────────────────────────────────────

def preprocess_vision(
    root: Path,
    categories: list[str],
    cache_size: Optional[int] = None,
) -> bool:
    import json as _json

    print("\n── VISION Dataset ─────────────────────────────────────")
    if not root.exists():
        print(f"  SKIP: {root} not found. Run: bash setup_data.sh")
        return False

    available = sorted(p.name for p in root.iterdir() if p.is_dir())
    use_cats  = [c for c in categories if c in available]
    missing   = [c for c in categories if c not in available]

    if missing:
        print(f"  WARNING: categories not found: {missing}")
        print(f"  Available: {available}")
    if not use_cats:
        print("  SKIP: no usable categories found")
        return False

    all_train_records: list[dict] = []
    all_test_records:  list[dict] = []

    def _parse_coco(split_dir: Path, cat: str, is_test: bool) -> list[dict]:
        ann_file = split_dir / "_annotations.coco.json"
        if not ann_file.exists():
            print(f"    WARNING: {ann_file} missing")
            return []
        with open(ann_file) as f:
            coco = _json.load(f)
        defective_ids = {a["image_id"] for a in coco.get("annotations", [])}
        records = []
        for img in coco["images"]:
            p = split_dir / img["file_name"]
            if not p.exists():
                continue
            records.append({
                "path":     _relative(p, root),
                "label":    1 if img["id"] in defective_ids else 0,
                "mask":     None,   # VISION has no pixel masks
                "category": cat,
            })
        return records

    for cat in use_cats:
        train_dir = root / cat / "train"
        val_dir   = root / cat / "val"

        train_recs = _parse_coco(train_dir, cat, is_test=False) if train_dir.exists() else []
        val_recs   = _parse_coco(val_dir,   cat, is_test=True)  if val_dir.exists()   else []

        # Training manifest: only defect-free from train split
        train_normal = [r for r in train_recs if r["label"] == 0]
        all_train_records += train_normal

        all_test_records += val_recs

        n_val_defect = sum(1 for r in val_recs if r["label"] == 1)
        print(f"  {cat}: {len(train_normal):>4} train-normal | "
              f"{n_val_defect:>4} val-defective / {len(val_recs)} val total")

    stats = {
        "categories": use_cats,
        "note": "No concrete categories. Using metallic subsets as corrosion proxy.",
        "train_normal":   len(all_train_records),
        "test_total":     len(all_test_records),
        "test_defective": sum(1 for r in all_test_records if r["label"] == 1),
        "has_pixel_masks": False,
        "image_stats": _image_stats(
            [root / r["path"] for r in all_train_records[:200]]
        ),
    }

    _save_manifest(root / "manifest_train.json", all_train_records)
    _save_manifest(root / "manifest_test.json",  all_test_records)
    _save_stats(root / "stats.json", stats)

    if cache_size:
        _cache_dataset(root, all_train_records + all_test_records, cache_size)

    print(f"  ✓ VISION ({', '.join(use_cats)}): "
          f"{stats['train_normal']:,} train-normal | "
          f"{stats['test_defective']:,} test-defective")
    return True


# ── optional image resize cache ───────────────────────────────────────────────

def _cache_dataset(root: Path, records: list[dict], size: int) -> None:
    cache_root = root.parent / f"{root.name}_cache{size}"
    print(f"  Caching resized images ({size}px) to {cache_root} …")
    errors = 0
    for rec in tqdm(records, desc="  resizing", leave=False):
        src = root / rec["path"]
        dst = cache_root / rec["path"]
        try:
            _resize_cache(src, dst, size)
        except Exception as e:
            errors += 1
    print(f"  Cache done ({errors} errors)")


# ── summary report ────────────────────────────────────────────────────────────

def print_summary(results: dict[str, bool]) -> None:
    print("\n" + "━" * 54)
    print(f"{'Dataset':<20} {'Status'}")
    print("━" * 54)
    for name, ok in results.items():
        status = "✓ ready" if ok else "✗ skipped (not downloaded)"
        print(f"  {name:<18} {status}")
    print("━" * 54)
    ready = [k for k, v in results.items() if v]
    if ready:
        print(f"\nReady to run experiments on: {', '.join(ready)}")
        print("  python run_experiments.py --datasets " + " ".join(ready))
    else:
        print("\nNo datasets ready. Run: bash setup_data.sh")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--datasets", nargs="+",
        choices=["sdnet", "mvtec", "vision"],
        default=["sdnet", "mvtec", "vision"],
        help="Which datasets to preprocess",
    )
    p.add_argument(
        "--cache-resize", type=int, default=None, metavar="SIZE",
        help="Also write resized copies at SIZE×SIZE px (e.g. 256). "
             "Speeds up training but uses ~2-5× extra disk.",
    )
    p.add_argument(
        "--train-ratio", type=float, default=0.8,
        help="Train fraction for datasets without official splits (SDNET2018)",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results: dict[str, bool] = {}

    if "sdnet" in args.datasets:
        results["sdnet"] = preprocess_sdnet(
            cfg.DATASET_PATHS["sdnet"],
            train_ratio=args.train_ratio,
            seed=args.seed,
            cache_size=args.cache_resize,
        )

    if "mvtec" in args.datasets:
        results["mvtec"] = preprocess_mvtec(
            cfg.DATASET_PATHS["mvtec"],
            categories=cfg.MVTEC_CATEGORIES,
            cache_size=args.cache_resize,
        )

    if "vision" in args.datasets:
        results["vision"] = preprocess_vision(
            cfg.DATASET_PATHS["vision"],
            categories=cfg.VISION_CATEGORIES,
            cache_size=args.cache_resize,
        )

    print_summary(results)
