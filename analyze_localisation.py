"""Aggregate the DINO-PatchCore localisation runs across seeds.

Reads results/raw/dino_patchcore_localisation/<source>__<target>/seed<s>/result.json
for every source in cfg.DATASETS, every target with pixel ground truth and every
seed in cfg.SEEDS, and reports per source-target pair the mean and SD (ddof=1)
of pixel AUROC and of the argmax-in-mask rate: the fraction of scored defective
images whose highest-scoring patch lies inside the defect mask.

The chance level of that rate is the mean ground-truth mask area over the
scored defective images (mean_mask_area). It depends only on the masks and the
crop, so it must be identical in every run on a target; it is checked, not
averaged. Every metric must be defined in every seed of every cell.

Usage:  python analyze_localisation.py
Output: results/localisation.json
"""
from __future__ import annotations

import json
import os

import numpy as np

import config as cfg

ROOT = cfg.RESULTS_DIR / "raw" / "dino_patchcore_localisation"
TARGETS = ["mvtec", "vision"]
METRICS = ["pixel_auroc_defective", "pixel_auroc_all", "argmax_in_mask"]
COUNTS = ["n_defective_scored", "n_defective_outside_crop"]
REQUIRED = ["mean_mask_area", *COUNTS]
AREA_TOL = 1e-12


def summarise(runs: list[dict], key: str, name: str) -> dict:
    v = [r[key] for r in runs if r.get(key) is not None]
    if len(v) != len(cfg.SEEDS):
        raise SystemExit(f"{name}: {key} is defined in {len(v)} of {len(cfg.SEEDS)} seeds")
    return {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "n_seeds": len(v)}


def write_json(path, payload) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def main() -> None:
    expected = [ROOT / f"{s}__{t}" / f"seed{seed}" / "result.json"
                for t in TARGETS for s in cfg.DATASETS for seed in cfg.SEEDS]
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        raise SystemExit("missing localisation runs:\n  " + "\n  ".join(missing))

    cells, areas = {}, {t: [] for t in TARGETS}
    for t in TARGETS:
        for s in cfg.DATASETS:
            name = f"{s}__{t}"
            runs = [json.loads((ROOT / name / f"seed{seed}" / "result.json").read_text())
                    for seed in cfg.SEEDS]
            lacking = {seed: [k for k in REQUIRED if r.get(k) is None]
                       for seed, r in zip(cfg.SEEDS, runs)}
            lacking = {seed: ks for seed, ks in lacking.items() if ks}
            if lacking:
                raise SystemExit(f"{name}: result.json lacks required fields {lacking}")
            counts = {}
            for k in COUNTS:
                values = sorted({r[k] for r in runs})
                if len(values) != 1:
                    raise SystemExit(f"{name}: {k} differs across seeds: {values}")
                counts[k] = values[0]
            areas[t].extend(r["mean_mask_area"] for r in runs)
            cells[name] = {"source": s, "target": t, "seeds": list(cfg.SEEDS),
                           **{k: summarise(runs, k, name) for k in METRICS}, **counts}

    chance = {}
    for t, v in areas.items():
        if max(v) - min(v) > AREA_TOL:
            raise SystemExit(f"target {t}: mean_mask_area differs across runs "
                             f"({min(v)!r} to {max(v)!r})")
        chance[t] = v[0]
    for c in cells.values():
        c["argmax_in_mask_x_chance"] = c["argmax_in_mask"]["mean"] / chance[c["target"]]

    def fmt(summary: dict, scale: float = 1.0, width: int = 7, prec: int = 3) -> str:
        return (f"{summary['mean'] * scale:{width}.{prec}f} +-{summary['sd'] * scale:.{prec}f}"
                f" (n={summary['n_seeds']})")

    for t in TARGETS:
        print(f"\n=== target {t}  (argmax-in-mask chance {chance[t] * 100:.2f}%)")
        print(f"  {'source':7s} {'pixel AUROC (defective)':>24s} {'pixel AUROC (all)':>24s} "
              f"{'argmax-in-mask %':>24s} {'x chance':>8s} {'scored':>6s} {'outside':>7s}")
        for c in cells.values():
            if c["target"] != t:
                continue
            print(f"  {c['source']:7s} {fmt(c['pixel_auroc_defective']):>24s} "
                  f"{fmt(c['pixel_auroc_all']):>24s} "
                  f"{fmt(c['argmax_in_mask'], 100, 5, 1):>24s} "
                  f"{c['argmax_in_mask_x_chance']:8.2f} "
                  f"{c['n_defective_scored']:6d} {c['n_defective_outside_crop']:7d}")

    out = cfg.RESULTS_DIR / "localisation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, {"config": {"seeds": list(cfg.SEEDS), "sources": list(cfg.DATASETS),
                                "targets": TARGETS},
                     "cells": cells, "chance_level": chance})
    print(f"\n  written to {out}")


if __name__ == "__main__":
    main()
