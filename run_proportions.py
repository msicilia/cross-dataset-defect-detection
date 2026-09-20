"""Memory-bank composition sweep (DINO-PatchCore, ViT-B/14).

Each bank holds 500 reference images, of which a fraction 0, 25, 50, 75 or 100 %
comes from VISION and the rest from a base source (MVTec AD or SDNET2018). Both
parts are prefixes of one random 500-image draw per source and seed, so the
VISION images of a bank are a subset of those of every bank with a larger
fraction. Every bank is evaluated on all three targets, with test sets capped at
--max-test by label-stratified subsampling.

The 100 % bank contains no base images and is the same for both bases: it is
fitted once per seed and its cells are written under both base names.

Results: results/raw/dino_patchcore_proportions/<base>_vision<pct>/seed<s>/<target>/
where <pct> is the VISION percentage with three digits (000, 025, ..., 100).

Usage:
    python run_proportions.py [--seeds S ...] [--max-test N] [--device cuda|mps|cpu]
"""
from __future__ import annotations

import argparse

import config as cfg
from common import (build_dataset, cell_done, pick_device, random_subset,
                    save_cell, seeded_rng, stratified_test)
from evaluate import image_level_metrics
from run_contamination import BACKBONE, make_method

BANK_SIZE = 500
BASES = ["mvtec", "sdnet"]
CONTAMINANT = "vision"
FRACTIONS = [0.0, 0.25, 0.50, 0.75, 1.0]


def banks(normals: dict[str, list], seed: int) -> dict[tuple[int, int], tuple[list, list]]:
    """{(n_base, n_contaminant): ([bank names], reference paths)} for one seed."""
    pool_cont = random_subset(normals[CONTAMINANT], BANK_SIZE, seeded_rng(seed, 101))
    assert len(pool_cont) == BANK_SIZE, f"{CONTAMINANT} has too few reference images"
    out: dict[tuple[int, int], tuple[list, list]] = {}
    for base in BASES:
        pool_base = random_subset(normals[base], BANK_SIZE, seeded_rng(seed, 102))
        assert len(pool_base) == BANK_SIZE, f"{base} has too few reference images"
        for frac in FRACTIONS:
            n_cont = int(round(BANK_SIZE * frac))
            n_base = BANK_SIZE - n_cont
            name = f"{base}_{CONTAMINANT}{int(frac * 100):03d}"
            key = (base if n_base else None, n_cont)
            if key in out:
                out[key][0].append(name)
            else:
                out[key] = ([name], pool_base[:n_base] + pool_cont[:n_cont])
    return out


def cell_config(base: str | None, n_cont: int, seed: int, target: str,
                max_test: int | None) -> dict:
    return {
        "experiment": "proportions",
        "backbone": BACKBONE,
        "base": base,
        "n_base": BANK_SIZE - n_cont,
        "contaminant": CONTAMINANT,
        "n_contaminant": n_cont,
        "fraction_contaminant": n_cont / BANK_SIZE,
        "seed": seed,
        "target": target,
        "max_test": max_test,
        "coreset_ratio": cfg.PATCHCORE["coreset_ratio"],
        "image_size": cfg.PATCHCORE["image_size"],
    }


def run(seeds: list[int], max_test: int | None, device: str) -> None:
    out_root = cfg.RESULTS_DIR / "raw" / "dino_patchcore_proportions"
    datasets = {d: build_dataset(d) for d in cfg.DATASETS}
    normals = {d: datasets[d].normal_train().image_paths for d in cfg.DATASETS}
    print(f"device={device}  seeds={seeds}  max_test={max_test}  bank={BANK_SIZE}")

    for seed in seeds:
        targets = {t: stratified_test(datasets[t].test(), max_test, seed) for t in cfg.DATASETS}

        for (base, n_cont), (names, refs) in banks(normals, seed).items():
            assert len(refs) == BANK_SIZE
            cells = [(name, t, out_root / name / f"seed{seed}" / t,
                      cell_config(base, n_cont, seed, t, max_test))
                     for name in names for t in cfg.DATASETS]
            pending = [c for c in cells if not cell_done(c[2], c[3])]
            label = "/".join(names)
            if not pending:
                print(f"[seed {seed}] {label}: done")
                continue

            print(f"\n[seed {seed}] bank {label}: {BANK_SIZE - n_cont} {base} "
                  f"+ {n_cont} {CONTAMINANT}")
            method = make_method(device)
            method.fit(refs, seed=seed)
            scored = {}
            for name, t, out_dir, config in pending:
                if t not in scored:
                    paths, labels = targets[t]
                    scores = method.score(paths)
                    scored[t] = (scores, image_level_metrics(scores, labels))
                scores, res = scored[t]
                paths, labels = targets[t]
                save_cell(out_dir, config,
                          {"image_auroc": res.image_auroc, "image_ap": res.image_ap},
                          device, scores=scores, labels=labels, paths=paths)
                print(f"    -> {name} {t}: AUROC={res.image_auroc:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=cfg.SEEDS)
    ap.add_argument("--max-test", type=int, default=cfg.MAX_TEST,
                    help="cap on test images per target (label-stratified); 0 = no cap")
    ap.add_argument("--device", default=None,
                    help="default: $DEVICE, else cuda, mps, cpu")
    a = ap.parse_args()
    run(a.seeds, a.max_test or None, pick_device(a.device))
