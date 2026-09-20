"""Reference-set size sweep for DINO-PatchCore.

For each seed and source dataset, one pool of 500 defect-free reference images
is drawn as a random permutation prefix (numpy default_rng(seed)); the bank for
size n is the first n images of that pool, so smaller sets are nested in larger
ones. The pool is in random order, so every prefix mixes categories. Each
bank is scored on the full test split of all three targets.

Results: results/raw/dino_patchcore_fewshot/n<n>/seed<s>/<source>__<target>/

Usage:
    python run_fewshot.py [--seeds S ...] [--device cuda|mps|cpu]
"""
from __future__ import annotations

import argparse

import numpy as np

import config as cfg
from common import build_dataset, cell_done, pick_device, random_subset, save_cell
from evaluate import image_level_metrics
from methods import DINOPatchCore

N_SHOTS = [5, 10, 25, 50, 100, 250, 500]
BACKBONE = cfg.DINOV2_MODELS["base"]


def reference_pool(source: str, seed: int) -> list:
    paths = build_dataset(source).normal_train().image_paths
    if len(paths) < max(N_SHOTS):
        raise ValueError(f"{source} has only {len(paths)} reference images")
    return random_subset(paths, max(N_SHOTS), np.random.default_rng(seed))


def cell_config(source: str, target: str, seed: int, n: int) -> dict:
    return {"experiment": "reference_set_size", "method": "dino_patchcore_base",
            "backbone": BACKBONE, "source": source, "target": target,
            "seed": seed, "n": n,
            "coreset_ratio": cfg.PATCHCORE["coreset_ratio"],
            "image_size": cfg.PATCHCORE["image_size"]}


def run(seeds: list[int], device: str) -> None:
    root = cfg.RESULTS_DIR / "raw" / "dino_patchcore_fewshot"
    tests = {}

    for seed in seeds:
        for src in cfg.DATASETS:
            pool = None
            for n in N_SHOTS:
                pending = []
                for tgt in cfg.DATASETS:
                    out_dir = root / f"n{n}" / f"seed{seed}" / f"{src}__{tgt}"
                    config = cell_config(src, tgt, seed, n)
                    if not cell_done(out_dir, config):
                        pending.append((tgt, out_dir, config))
                if not pending:
                    print(f"[seed {seed}] {src} n={n}: all targets done")
                    continue

                pool = pool or reference_pool(src, seed)
                print(f"\n[seed {seed}] {src} n={n}: fitting")
                model = DINOPatchCore(backbone=BACKBONE, device=device,
                                      coreset_ratio=cfg.PATCHCORE["coreset_ratio"],
                                      max_train_images=n,
                                      image_size=cfg.PATCHCORE["image_size"],
                                      batch_size=cfg.PATCHCORE["batch_size"])
                model.fit(pool[:n], seed=seed)

                for tgt, out_dir, config in pending:
                    if tgt not in tests:
                        tests[tgt] = build_dataset(tgt).test()
                    test = tests[tgt]
                    labels = np.array(test.labels)
                    scores = model.score(test.image_paths)
                    metrics = image_level_metrics(scores, labels)
                    save_cell(out_dir, config,
                              {"image_auroc": metrics.image_auroc,
                               "image_ap": metrics.image_ap},
                              device, scores=scores, labels=labels,
                              paths=test.image_paths)
                    print(f"    {src} -> {tgt}: AUROC={metrics.image_auroc:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", nargs="+", type=int, default=cfg.SUBSET_SEEDS)
    ap.add_argument("--device", default=None,
                    help="default: $DEVICE, else cuda, mps, cpu")
    a = ap.parse_args()
    run(a.seeds, pick_device(a.device))
