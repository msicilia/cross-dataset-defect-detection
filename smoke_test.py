from __future__ import annotations
"""Smoke test: validates the full pipeline on whatever datasets are available.

Runs one forward pass per method on a tiny slice of each dataset to catch
import errors, shape mismatches, and broken paths before a long experiment run.

Usage:
  python smoke_test.py                  # test all available datasets + methods
  python smoke_test.py --datasets sdnet # test only SDNET2018
"""
import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import config as cfg
from datasets import MVTecDataset, SDNETDataset, VISIONDataset
from evaluate import image_level_metrics

N_IMAGES = 10   # number of images per split to test with


def check(condition: bool, msg: str) -> None:
    if condition:
        print(f"    ✓ {msg}")
    else:
        print(f"    ✗ FAIL: {msg}")
        sys.exit(1)


def test_dataset(name: str) -> bool:
    root = cfg.DATASET_PATHS[name]
    if not root.exists():
        print(f"  [{name}] SKIP — not downloaded")
        return False

    print(f"\n  [{name}] loading …")
    try:
        if name == "sdnet":
            ds = SDNETDataset(root, subsets=["walls"])
        elif name == "mvtec":
            ds = MVTecDataset(root, categories=cfg.MVTEC_CATEGORIES)
        elif name == "vision":
            ds = VISIONDataset(root, categories=cfg.VISION_CATEGORIES)
        else:
            print(f"  [{name}] unknown dataset, skip")
            return False

        train = ds.normal_train()
        test  = ds.test()

        check(len(train.image_paths) > 0,    f"train split non-empty ({len(train.image_paths)} images)")
        check(len(test.image_paths)  > 0,    f"test split non-empty ({len(test.image_paths)} images)")
        check(all(p.exists() for p in train.image_paths[:20]), "first 20 train paths exist")
        check(all(p.exists() for p in test.image_paths[:20]),  "first 20 test paths exist")
        check(1 in test.labels, f"test split contains defective samples")
        check(0 in test.labels, f"test split contains normal samples")

        print(f"    train: {len(train.image_paths):,} normal images")
        print(f"    test:  {len(test.image_paths):,} total  "
              f"({sum(test.labels):,} defective / {test.labels.count(0):,} normal)")
        return True

    except Exception:
        print(f"    ✗ EXCEPTION:")
        traceback.print_exc()
        return False


def test_method(method_name: str, train_paths: list[Path], test_paths: list[Path],
                test_labels: list[int], device: str) -> bool:
    from methods import DINOPatchCore, PatchCore, CLIPZS
    print(f"\n  [{method_name}] …")
    try:
        if method_name == "dino_patchcore":
            m = DINOPatchCore(backbone=cfg.DINOV2_MODELS["small"],
                              max_train_images=N_IMAGES, batch_size=4, device=device)
        elif method_name == "patchcore":
            m = PatchCore(max_train_images=N_IMAGES, batch_size=4, device=device)
        elif method_name == "clip_zs":
            m = CLIPZS(model_id=cfg.CLIP_MODEL, batch_size=4, device=device)
        else:
            print(f"    SKIP")
            return True

        m.fit(train_paths)
        scores = m.score(test_paths)

        check(scores.shape == (len(test_paths),), f"score shape {scores.shape}")
        check(not np.any(np.isnan(scores)),        "no NaN scores")
        check(scores.min() >= 0,                   "scores non-negative")

        labels = np.array(test_labels)
        if labels.sum() > 0 and (labels == 0).sum() > 0:
            result = image_level_metrics(scores, labels)
            print(f"    AUROC={result.image_auroc:.3f}  AP={result.image_ap:.3f}  "
                  f"(on {len(test_paths)} images, random subset — not meaningful)")
        return True

    except Exception:
        print(f"    ✗ EXCEPTION:")
        traceback.print_exc()
        return False



def main(datasets: list[str], methods: list[str], device: str) -> None:
    import torch
    print(f"\nDevice: {device}  |  PyTorch: {torch.__version__}")
    print("=" * 54)

    # ── dataset checks ────────────────────────────────────────────────────────
    print("\n[1/2] Dataset integrity checks")
    ds_ok: dict[str, bool] = {}
    for name in datasets:
        ds_ok[name] = test_dataset(name)

    # ── method smoke tests ────────────────────────────────────────────────────
    # Use SDNET2018 if available, otherwise first available dataset
    ref_name = next((n for n in ["sdnet", "mvtec", "vision"] if ds_ok.get(n)), None)
    if ref_name is None:
        print("\n[2/2] No dataset available — skipping method checks")
    else:
        print(f"\n[2/2] Method smoke tests  (dataset: {ref_name}, {N_IMAGES} images each)")

        root = cfg.DATASET_PATHS[ref_name]
        if ref_name == "sdnet":
            ds = SDNETDataset(root, subsets=["walls"])
        elif ref_name == "mvtec":
            ds = MVTecDataset(root, categories=cfg.MVTEC_CATEGORIES)
        else:
            ds = VISIONDataset(root, categories=cfg.VISION_CATEGORIES)

        train = ds.normal_train()
        test  = ds.test()
        train_sample = train.image_paths[:N_IMAGES]
        # make sure we have at least some of each label in test sample
        defective = [p for p, l in zip(test.image_paths, test.labels) if l == 1]
        normal    = [p for p, l in zip(test.image_paths, test.labels) if l == 0]
        test_sample = (normal[:N_IMAGES//2] + defective[:N_IMAGES//2])
        test_label_sample = [0]*(N_IMAGES//2) + [1]*(N_IMAGES//2)

        for m in methods:
            test_method(m, train_sample, test_sample, test_label_sample, device)

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 54)
    print("Datasets ready:")
    for name, ok in ds_ok.items():
        print(f"  {'✓' if ok else '✗'}  {name}")
    print("\nIf all checks passed, run the full experiments:")
    print("  python run_experiments.py --datasets " +
          " ".join(n for n, ok in ds_ok.items() if ok))


if __name__ == "__main__":
    import torch
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   choices=["sdnet", "mvtec", "vision"],
                   default=["sdnet", "mvtec", "vision"])
    p.add_argument("--methods", nargs="+",
                   choices=["dino_patchcore", "patchcore", "spade", "padim",
                            "winclip", "clip_zs"],
                   default=["dino_patchcore", "patchcore", "clip_zs"])
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()

    main(args.datasets, args.methods, args.device)
