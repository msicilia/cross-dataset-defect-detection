from __future__ import annotations
"""Controlled memory-bank contamination experiment.

Disentangles "does dataset X contaminate the memory bank?" from "did we just give
the bank fewer good images?" by holding the good-source count fixed and varying
ONLY what is added.

For every base source B in {sdnet, mvtec, vision}, and every seed, we build:

    {B}250         : 250 images of B                       (baseline)
    {B}500         : 500 images of B                       (placebo: +250 MORE of the same)
    {B}250+{X}250  : the same 250 of B, plus 250 of X      (one per other source X)

The 250-image core of B is identical across all conditions for a given seed
(it is the first 250 of the sampled-500 pool), so {B}500 and {B}250+{X}250 differ
from {B}250 by exactly +250 images -- the only variable is WHAT those 250 are.

Every bank is evaluated on ALL three targets T (the "exam"), so each role is
rotated: each dataset serves as base, as added source, and as target. The clean
contamination quantity is

    Delta(B, X -> T) = AUROC({B}250+{X}250 -> T) - AUROC({B}250 -> T)

with {B}500 -> T as the count-matched control (adding GOOD data should not hurt).

Targets larger than --max-test are stratified-subsampled (label-balanced) per
seed to bound runtime; this affects absolute AUROC slightly but not the
count-controlled differences the experiment is about.

Results: results/raw/dino_patchcore_contamination/<config>/seed<s>/<tgt>/result.json

Usage:
    python run_contamination.py [--seeds 0 1 2 3 4] [--max-test 2000]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from datasets import MVTecDataset, SDNETDataset, VISIONDataset
from methods import DINOPatchCore
from evaluate import image_level_metrics

DATASETS = ["sdnet", "mvtec", "vision"]
N_BASE   = 250
N_ADD    = 250


def build_dataset(name: str, seed: int = 0):
    root = cfg.DATASET_PATHS[name]
    if name == "mvtec":  return MVTecDataset(root, categories=cfg.MVTEC_CATEGORIES)
    if name == "sdnet":  return SDNETDataset(root, seed=seed)
    if name == "vision": return VISIONDataset(root, categories=cfg.VISION_CATEGORIES)
    raise ValueError(name)


def _rng(seed: int, tag: int):
    return np.random.default_rng([seed, tag])


def _sample(paths, n, rng):
    if len(paths) <= n:
        return list(paths)
    idx = rng.choice(len(paths), n, replace=False)
    return [paths[i] for i in sorted(idx)]


def _stratified_test(split, max_test: int, seed: int):
    """Label-balanced subsample of a test split to at most max_test images."""
    paths  = list(split.image_paths)
    labels = list(split.labels)
    if max_test is None or len(paths) <= max_test:
        return paths, np.array(labels)
    rng = np.random.default_rng([seed, 7777])
    idx = np.arange(len(paths))
    pos = idx[np.array(labels) == 1]
    neg = idx[np.array(labels) == 0]
    frac = max_test / len(paths)
    n_pos = max(1, int(round(len(pos) * frac)))
    n_neg = max(1, int(round(len(neg) * frac)))
    keep = np.concatenate([
        rng.choice(pos, min(n_pos, len(pos)), replace=False),
        rng.choice(neg, min(n_neg, len(neg)), replace=False),
    ])
    keep.sort()
    return [paths[i] for i in keep], np.array([labels[i] for i in keep])


def make_method(device: str):
    return DINOPatchCore(
        backbone=cfg.DINOV2_MODELS["base"],
        device=device,
        coreset_ratio=cfg.PATCHCORE["coreset_ratio"],
        max_train_images=10_000,   # we control the count ourselves; do not re-subsample
        image_size=cfg.PATCHCORE["image_size"],
        batch_size=cfg.PATCHCORE["batch_size"],
    )


def run(seeds: list[int], max_test: int | None) -> None:
    import torch
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    out_root = cfg.RESULTS_DIR / "raw" / "dino_patchcore_contamination"
    print(f"device={device}  seeds={seeds}  max_test={max_test}")

    for seed in seeds:
        # Pre-build (and cache) the target exams for this seed.
        targets = {}
        for t in DATASETS:
            paths, labels = _stratified_test(build_dataset(t, seed).test(), max_test, seed)
            targets[t] = (paths, labels)
            print(f"[seed {seed}] target {t}: {len(paths)} test images "
                  f"({int((labels==1).sum())} defective)")

        # Sampled pools per source (500 for bases, 250 for add-ons).
        normals = {s: build_dataset(s, seed).normal_train().image_paths for s in DATASETS}

        # Build the list of bank configurations.
        configs: dict[str, list] = {}
        for B in DATASETS:
            poolB = _sample(normals[B], 500, _rng(seed, DATASETS.index(B)))
            base250 = poolB[:N_BASE]
            configs[f"{B}{N_BASE}"]            = base250            # baseline
            if len(poolB) >= N_BASE + N_ADD:
                configs[f"{B}{N_BASE+N_ADD}"]  = poolB[:N_BASE + N_ADD]  # placebo: +250 same
            for X in DATASETS:
                if X == B:
                    continue
                addX = _sample(normals[X], N_ADD, _rng(seed, 10 + DATASETS.index(X)))
                configs[f"{B}{N_BASE}+{X}{N_ADD}"] = base250 + addX

        for cfg_name, imgs in configs.items():
            # Skip if all targets already done for this config/seed.
            done = all((out_root / cfg_name / f"seed{seed}" / t / "result.json").exists()
                       for t in DATASETS)
            if done:
                print(f"[seed {seed}] {cfg_name}: done, skipping")
                continue

            print(f"\n[seed {seed}] building bank '{cfg_name}' ({len(imgs)} images) ...")
            method = make_method(device)
            method.fit(imgs, seed=seed)

            for t in DATASETS:
                out_dir = out_root / cfg_name / f"seed{seed}" / t
                if (out_dir / "result.json").exists():
                    continue
                paths, labels = targets[t]
                scores = method.score(paths)
                res = image_level_metrics(scores, labels)
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_dir / "result.json", "w") as f:
                    json.dump({"image_auroc": res.image_auroc,
                               "image_ap": res.image_ap,
                               "config": cfg_name, "target": t, "seed": seed,
                               "n_images": len(imgs)}, f, indent=2)
                print(f"    -> {t}: AUROC={res.image_auroc:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--max-test", type=int, default=2000,
                    help="cap per-target test images (stratified); 0 = no cap")
    a = ap.parse_args()
    run(a.seeds, None if a.max_test == 0 else a.max_test)
