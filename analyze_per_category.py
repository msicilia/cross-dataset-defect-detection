"""Pooled versus per-category (macro-averaged) AUROC on the multi-category targets.

MVTec AD and VISION are evaluated with one memory bank over all their
categories and one ranking over the pooled test split. From the stored
per-image scores this script computes, per method, source, target and seed:

  pooled   AUROC over the whole test split
  macro    unweighted mean of the AUROCs computed within each category

The category of an image is the directory directly below the dataset root in
its stored path.

Summary table per method and target (values are means over seeds):

  in_dist  source = target (reference-based detectors only)
  cross    mean over the two foreign sources; for reference-free detectors,
           which have no source, their single value

Orderings of detectors are given for each target, condition and convention,
together with the table entries in which macro exceeds pooled.

Usage:  python analyze_per_category.py
Output: results/per_category.json
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import config as cfg

RAW = cfg.RESULTS_DIR / "raw"
REFERENCE_BASED = ["patchcore", "dino_patchcore_base", "spade", "padim", "winclip_plus"]
REFERENCE_FREE = ["winclip", "clip_zs"]
TARGETS = ["mvtec", "vision"]
CONVENTIONS = ["pooled", "macro"]


def seeds_for(method: str) -> list[int]:
    return [0] if method in REFERENCE_FREE else list(cfg.SEEDS)


def sources_for(method: str) -> list[str]:
    return ["none"] if method in REFERENCE_FREE else list(cfg.DATASETS)


def npz_path(method: str, src: str, tgt: str, seed: int) -> Path:
    return RAW / method / f"seed{seed}" / f"{src}__{tgt}" / "scores.npz"


def category_of(path, dataset: str) -> str:
    """Directory directly below the dataset root (SDNET2018 subset, MVTec AD or
    VISION category)."""
    parts = Path(path).parts
    root = cfg.DATASET_PATHS[dataset].name
    if root not in parts[:-1]:
        raise ValueError(f"{path} is not under a directory named {root}")
    return parts[parts.index(root) + 1]


def seed_metrics(npz: Path, target: str) -> tuple[float, float, dict]:
    z = np.load(npz, allow_pickle=False)
    s, y = z["scores"], z["labels"]
    cats = np.array([category_of(p, target) for p in z["paths"]])
    per_cat = {}
    for c in sorted(set(cats)):
        m = cats == c
        if len(np.unique(y[m])) < 2:
            raise ValueError(f"{npz}: category {c} lacks one of the two classes")
        per_cat[c] = float(roc_auc_score(y[m], s[m]))
    return float(roc_auc_score(y, s)), float(np.mean(list(per_cat.values()))), per_cat


def sd1(values) -> float | None:
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def write_json(path, payload) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def main() -> None:
    methods = REFERENCE_BASED + REFERENCE_FREE
    missing = [str(npz_path(m, s, t, seed))
               for m in methods for s in sources_for(m) for t in TARGETS
               for seed in seeds_for(m) if not npz_path(m, s, t, seed).exists()]
    if missing:
        raise SystemExit("missing benchmark runs:\n  " + "\n  ".join(missing))

    cells = {}
    for m in methods:
        print(f"\n=== {m}")
        print(f"  {'source->target':18s} {'pooled':>16s} {'macro':>16s} {'macro-pooled':>13s}")
        for tgt in TARGETS:
            for src in sources_for(m):
                pooled, macro, per_cat = [], [], defaultdict(list)
                for seed in seeds_for(m):
                    p, mac, pc = seed_metrics(npz_path(m, src, tgt, seed), tgt)
                    pooled.append(p)
                    macro.append(mac)
                    for c, v in pc.items():
                        per_cat[c].append(v)
                incomplete = {c: len(v) for c, v in per_cat.items()
                              if len(v) != len(seeds_for(m))}
                if incomplete:
                    raise ValueError(f"{m}/{src}__{tgt}: categories without a value for every "
                                     f"seed {seeds_for(m)}: {incomplete}")
                cell = {
                    "method": m, "source": src, "target": tgt, "seeds": seeds_for(m),
                    "pooled_per_seed": pooled, "macro_per_seed": macro,
                    "pooled_mean": float(np.mean(pooled)), "pooled_std": sd1(pooled),
                    "macro_mean": float(np.mean(macro)), "macro_std": sd1(macro),
                    "per_category_per_seed": dict(per_cat),
                    "per_category": {c: float(np.mean(v)) for c, v in per_cat.items()},
                }
                cells[f"{m}/{src}__{tgt}"] = cell
                ps = f"{cell['pooled_std']:.4f}" if cell["pooled_std"] is not None else "  -   "
                ms = f"{cell['macro_std']:.4f}" if cell["macro_std"] is not None else "  -   "
                print(f"  {src + '->' + tgt:18s} {cell['pooled_mean']:.4f}+/-{ps} "
                      f"{cell['macro_mean']:.4f}+/-{ms} "
                      f"{cell['macro_mean'] - cell['pooled_mean']:+13.4f}")
                if src == tgt or src == "none":
                    for c, v in cell["per_category"].items():
                        print(f"      {c:16s} {v:.4f}")

    table = {t: {} for t in TARGETS}
    for t in TARGETS:
        for m in methods:
            row = {}
            if m in REFERENCE_BASED:
                row["in_dist"] = {conv: cells[f"{m}/{t}__{t}"][f"{conv}_mean"]
                                  for conv in CONVENTIONS}
                foreign = [s for s in cfg.DATASETS if s != t]
                row["cross"] = {conv: float(np.mean([cells[f"{m}/{s}__{t}"][f"{conv}_mean"]
                                                     for s in foreign]))
                                for conv in CONVENTIONS}
                row["cross_sources"] = foreign
            else:
                row["cross"] = {conv: cells[f"{m}/none__{t}"][f"{conv}_mean"]
                                for conv in CONVENTIONS}
                row["cross_sources"] = ["none"]
            table[t][m] = row

    orderings, macro_vs_pooled = {}, {}
    for t in TARGETS:
        orderings[t] = {}
        for cond in ("in_dist", "cross"):
            members = [m for m in methods if cond in table[t][m]]
            ranked = {conv: sorted(members, key=lambda m: -table[t][m][cond][conv])
                      for conv in CONVENTIONS}
            ranked["identical"] = ranked["pooled"] == ranked["macro"]
            orderings[t][cond] = ranked
        entries = [(m, cond, table[t][m][cond]["macro"] - table[t][m][cond]["pooled"])
                   for m in methods for cond in ("in_dist", "cross") if cond in table[t][m]]
        macro_vs_pooled[t] = {}
        for group, allowed in (("reference_based", REFERENCE_BASED), ("all", methods)):
            sel = [e for e in entries if e[0] in allowed]
            macro_vs_pooled[t][group] = {
                "n_entries": len(sel),
                "n_macro_greater": sum(d > 0 for _, _, d in sel),
                "exceptions": [{"method": m, "condition": c, "macro_minus_pooled": d}
                               for m, c, d in sel if d <= 0],
            }

    for t in TARGETS:
        print(f"\n=== summary, target {t} (cross = mean over foreign sources)")
        print(f"  {'method':22s} {'in pooled':>9s} {'in macro':>9s} {'x pooled':>9s} {'x macro':>9s}")
        for m in methods:
            row = table[t][m]
            ind = (f"{row['in_dist']['pooled']:9.3f} {row['in_dist']['macro']:9.3f}"
                   if "in_dist" in row else f"{'-':>9s} {'-':>9s}")
            print(f"  {m:22s} {ind} {row['cross']['pooled']:9.3f} {row['cross']['macro']:9.3f}")
        for cond, r in orderings[t].items():
            print(f"  ordering {cond:7s} pooled: {r['pooled']}")
            print(f"  ordering {cond:7s} macro:  {r['macro']}  identical={r['identical']}")
        for group, r in macro_vs_pooled[t].items():
            exc = [(e["method"], e["condition"], round(e["macro_minus_pooled"], 4))
                   for e in r["exceptions"]]
            print(f"  macro > pooled ({group}): {r['n_macro_greater']}/{r['n_entries']}; "
                  f"otherwise: {exc}")

    out = cfg.RESULTS_DIR / "per_category.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, {"config": {"seeds": list(cfg.SEEDS), "targets": TARGETS,
                                "reference_based": REFERENCE_BASED,
                                "reference_free": REFERENCE_FREE},
                     "cells": cells, "table": table, "orderings": orderings,
                     "macro_vs_pooled": macro_vs_pooled})
    print(f"\n  written to {out}")


if __name__ == "__main__":
    main()
