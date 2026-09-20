"""Controlled memory-bank contamination experiment (DINO-PatchCore, ViT-B/14).

For every base source B and seed, three kinds of bank:

    {B}250         250 reference images of B
    {B}500         the same 250 plus 250 more of B
    {B}250+{X}250  the same 250 plus 250 of another source X

The 250-image core is the first half of one random 500-image draw from B, so it
is identical across the three kinds for a given seed, and the added 250 images
of X are the same whichever base they are added to. Every bank is evaluated on
all three targets, with test sets capped at --max-test by label-stratified
subsampling.

Results: results/raw/dino_patchcore_contamination/<bank>/seed<s>/<target>/

Usage:
    python run_contamination.py [--seeds S ...] [--max-test N] [--device cuda|mps|cpu]
"""
from __future__ import annotations

import argparse
import sys

import config as cfg
from common import (build_dataset, cell_done, pick_device, random_subset,
                    save_cell, seeded_rng, stratified_test)
from evaluate import image_level_metrics
from methods import DINOPatchCore

BACKBONE = cfg.DINOV2_MODELS["base"]
N_BASE = 250
N_ADD = 250


def make_method(device: str) -> DINOPatchCore:
    # The caller fixes the bank, so the detector's own subsampling is disabled.
    return DINOPatchCore(
        backbone=BACKBONE,
        device=device,
        coreset_ratio=cfg.PATCHCORE["coreset_ratio"],
        max_train_images=sys.maxsize,
        image_size=cfg.PATCHCORE["image_size"],
        batch_size=cfg.PATCHCORE["batch_size"],
    )


def banks(normals: dict[str, list], seed: int) -> dict[str, tuple[dict, list]]:
    """{bank name: (composition, reference paths)} for one seed."""
    out = {}
    for b in cfg.DATASETS:
        pool = random_subset(normals[b], N_BASE + N_ADD,
                             seeded_rng(seed, cfg.DATASETS.index(b)))
        assert len(pool) == N_BASE + N_ADD, f"{b} has too few reference images"
        core = pool[:N_BASE]
        out[f"{b}{N_BASE}"] = ({"base": b, "n_base": N_BASE, "added": None, "n_added": 0},
                               core)
        out[f"{b}{N_BASE + N_ADD}"] = (
            {"base": b, "n_base": N_BASE + N_ADD, "added": None, "n_added": 0}, pool)
        for x in cfg.DATASETS:
            if x == b:
                continue
            add = random_subset(normals[x], N_ADD, seeded_rng(seed, 10 + cfg.DATASETS.index(x)))
            assert len(add) == N_ADD, f"{x} has too few reference images"
            out[f"{b}{N_BASE}+{x}{N_ADD}"] = (
                {"base": b, "n_base": N_BASE, "added": x, "n_added": N_ADD}, core + add)
    return out


def cell_config(composition: dict, seed: int, target: str, max_test: int | None) -> dict:
    return {
        "experiment": "contamination",
        "backbone": BACKBONE,
        **composition,
        "seed": seed,
        "target": target,
        "max_test": max_test,
        "coreset_ratio": cfg.PATCHCORE["coreset_ratio"],
        "image_size": cfg.PATCHCORE["image_size"],
    }


def run(seeds: list[int], max_test: int | None, device: str) -> None:
    out_root = cfg.RESULTS_DIR / "raw" / "dino_patchcore_contamination"
    datasets = {d: build_dataset(d) for d in cfg.DATASETS}
    normals = {d: datasets[d].normal_train().image_paths for d in cfg.DATASETS}
    print(f"device={device}  seeds={seeds}  max_test={max_test}")

    for seed in seeds:
        targets = {t: stratified_test(datasets[t].test(), max_test, seed) for t in cfg.DATASETS}
        for t, (paths, labels) in targets.items():
            print(f"[seed {seed}] target {t}: {len(paths)} test images "
                  f"({int(labels.sum())} defective)")

        for name, (composition, refs) in banks(normals, seed).items():
            cells = {t: (out_root / name / f"seed{seed}" / t,
                         cell_config(composition, seed, t, max_test))
                     for t in cfg.DATASETS}
            pending = {t: c for t, c in cells.items() if not cell_done(*c)}
            if not pending:
                print(f"[seed {seed}] {name}: done")
                continue

            print(f"\n[seed {seed}] bank {name} ({len(refs)} images)")
            method = make_method(device)
            method.fit(refs, seed=seed)
            for t, (out_dir, config) in pending.items():
                paths, labels = targets[t]
                scores = method.score(paths)
                res = image_level_metrics(scores, labels)
                save_cell(out_dir, config,
                          {"image_auroc": res.image_auroc, "image_ap": res.image_ap},
                          device, scores=scores, labels=labels, paths=paths)
                print(f"    -> {t}: AUROC={res.image_auroc:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", nargs="+", type=int, default=cfg.SEEDS)
    ap.add_argument("--max-test", type=int, default=cfg.MAX_TEST,
                    help="cap on test images per target (label-stratified); 0 = no cap")
    ap.add_argument("--device", default=None,
                    help="default: $DEVICE, else cuda, mps, cpu")
    a = ap.parse_args()
    run(a.seeds, a.max_test or None, pick_device(a.device))
