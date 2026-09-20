"""Generalisation gap of DINO-PatchCore under the image-space normalisations of
run_confounders.py.

For each arm, the in-distribution mean is the mean AUROC over the three
source == target cells, the cross-dataset mean is the mean over the six
source != target cells, and the gap is their difference. Each quantity is
computed per seed and then averaged over seeds; a standard deviation (ddof=1)
is reported only when more than one seed is present.

Input:  results/raw/dino_patchcore_confounders/<arm>/seed<s>/<src>__<tgt>/result.json
Output: results/confounders.json

Usage:
    python analyze_confounders.py [--results-dir results]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
from pathlib import Path

import config as cfg

ARMS = ["baseline", "gray", "equalised", "lowres"]


def auroc(path: Path) -> float:
    if not path.is_file():
        raise FileNotFoundError(f"missing result: {path}")
    return float(json.loads(path.read_text())["image_auroc"])


def check_seeds(directory: Path, expected: list[int]) -> None:
    if not directory.is_dir():
        raise FileNotFoundError(f"missing directory: {directory}")
    found = sorted(int(m.group(1)) for p in directory.iterdir()
                   if (m := re.fullmatch(r"seed(\d+)", p.name)))
    if found != sorted(expected):
        raise RuntimeError(f"{directory}: seeds {found}, expected {sorted(expected)}")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def summarise(values: list[float]) -> dict:
    return {"mean": st.mean(values),
            "sd": st.stdev(values) if len(values) > 1 else None,
            "per_seed": values}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", type=Path, default=cfg.RESULTS_DIR)
    args = ap.parse_args()
    root = args.results_dir / "raw" / "dino_patchcore_confounders"

    datasets = cfg.DATASETS
    pairs = [(s, t) for s in datasets for t in datasets]
    out = {"seeds": cfg.CONFOUNDER_SEEDS, "arms": {}}
    print(f"{'arm':10s} {'in-dist':>8s} {'cross':>8s} {'gap':>8s}")
    for arm in ARMS:
        check_seeds(root / arm, cfg.CONFOUNDER_SEEDS)
        cells = {f"{s}__{t}": [auroc(root / arm / f"seed{seed}" / f"{s}__{t}" / "result.json")
                               for seed in cfg.CONFOUNDER_SEEDS] for s, t in pairs}
        ind, cro, gap = [], [], []
        for k in range(len(cfg.CONFOUNDER_SEEDS)):
            i = st.mean(cells[f"{d}__{d}"][k] for d in datasets)
            c = st.mean(cells[f"{s}__{t}"][k] for s, t in pairs if s != t)
            ind.append(i)
            cro.append(c)
            gap.append(i - c)
        out["arms"][arm] = {
            "in_distribution": summarise(ind),
            "cross_dataset": summarise(cro),
            "gap": summarise(gap),
            "cells": {key: summarise(v) for key, v in cells.items()},
        }
        print(f"{arm:10s} {st.mean(ind):8.4f} {st.mean(cro):8.4f} {st.mean(gap):8.4f}")

    dest = args.results_dir / "confounders.json"
    write_json(dest, out)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
