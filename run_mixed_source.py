"""Mixed-source memory banks for DINO-PatchCore.

For each pair of source datasets and each seed, the memory bank is built from
250 defect-free reference images of each source (500 in total) and scored on
the full test split of all three targets.

The reference images come from one generator, numpy default_rng(seed), used
for both sources in pair order: 250 indices without replacement from the first
source's reference list, then 250 from the second's.

Results: results/raw/dino_patchcore_mixed/seed<s>/<src1>+<src2>__<target>/

Usage:
    python run_mixed_source.py [--seeds S ...] [--device cuda|mps|cpu]
"""
from __future__ import annotations

import argparse

import numpy as np

import config as cfg
from common import build_dataset, cell_done, pick_device, save_cell
from evaluate import image_level_metrics
from methods import DINOPatchCore

PAIRS = [("sdnet", "mvtec"), ("sdnet", "vision"), ("mvtec", "vision")]
PER_SOURCE = 250
BACKBONE = cfg.DINOV2_MODELS["base"]


def reference_images(pair: tuple[str, str], seed: int) -> list:
    rng = np.random.default_rng(seed)
    images = []
    for src in pair:
        paths = build_dataset(src).normal_train().image_paths
        idx = rng.choice(len(paths), PER_SOURCE, replace=False)
        images.extend(paths[i] for i in idx)
    return images


def cell_config(pair: tuple[str, str], target: str, seed: int) -> dict:
    return {"experiment": "mixed_source", "method": "dino_patchcore_base",
            "backbone": BACKBONE, "sources": list(pair), "target": target,
            "seed": seed, "reference_images_per_source": PER_SOURCE,
            "coreset_ratio": cfg.PATCHCORE["coreset_ratio"],
            "image_size": cfg.PATCHCORE["image_size"]}


def run(seeds: list[int], device: str) -> None:
    root = cfg.RESULTS_DIR / "raw" / "dino_patchcore_mixed"
    tests = {}

    for seed in seeds:
        for pair in PAIRS:
            key = "+".join(pair)
            pending = []
            for tgt in cfg.DATASETS:
                out_dir = root / f"seed{seed}" / f"{key}__{tgt}"
                config = cell_config(pair, tgt, seed)
                if not cell_done(out_dir, config):
                    pending.append((tgt, out_dir, config))
            if not pending:
                print(f"[seed {seed}] {key}: all targets done")
                continue

            images = reference_images(pair, seed)
            print(f"\n[seed {seed}] {key}: fitting on {len(images)} images")
            # The cap equals the bank size, so fit keeps every selected image.
            model = DINOPatchCore(backbone=BACKBONE, device=device,
                                  coreset_ratio=cfg.PATCHCORE["coreset_ratio"],
                                  max_train_images=len(images),
                                  image_size=cfg.PATCHCORE["image_size"],
                                  batch_size=cfg.PATCHCORE["batch_size"])
            model.fit(images, seed=seed)

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
                print(f"    {key} -> {tgt}: AUROC={metrics.image_auroc:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", nargs="+", type=int, default=cfg.SUBSET_SEEDS)
    ap.add_argument("--device", default=None,
                    help="default: $DEVICE, else cuda, mps, cpu")
    a = ap.parse_args()
    run(a.seeds, pick_device(a.device))
