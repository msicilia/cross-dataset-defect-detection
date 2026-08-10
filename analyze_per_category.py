from __future__ import annotations
"""Per-category and macro-averaged AUROC.

MVTec and VISION categories are pooled into a single memory bank:
`MVTecDataset(root, categories=[...])` builds one bank over all five categories
and evaluates on the pooled test split, and VISION likewise over its four
subsets. That is a deliberate choice for a cross-*dataset* study -- a dataset
stands in for "a domain", and a deployment would not know in advance which
category it is looking at -- but it departs from the per-category protocol usual
in MVTec papers and makes the absolute numbers incomparable with published
per-category results.

This script reports, from the stored per-image scores and without re-running
anything:

  pooled AUROC   one ranking over the whole test split (what the paper reports)
  per-category   AUROC computed within each category separately
  macro average  the unweighted mean of those, i.e. the usual convention

The two differ for a specific reason worth stating in the paper: a pooled
ranking must separate defective from normal images *across* categories, so any
systematic score offset between categories (a category that simply scores higher
overall) costs AUROC even when the within-category ranking is perfect.

Category is recovered from the stored image paths.

Usage:  python analyze_per_category.py [--methods ...]
"""
import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path("results/raw")
MULTI_CATEGORY = {"mvtec", "vision"}


def category_of(path: str, dataset: str) -> str:
    """MVTec stores <category>/test/<defect>/img.png; VISION <subset>/... ."""
    parts = Path(path).parts
    for i, p in enumerate(parts):
        if p in ("test", "train"):
            return parts[i - 1] if i else "?"
    return parts[-3] if len(parts) >= 3 else "?"


def analyse(method: str, src: str, tgt: str, seeds: list[int]):
    pooled, macro, per_cat = [], [], defaultdict(list)
    for seed in seeds:
        d = ROOT / method / f"seed{seed}" / f"{src}__{tgt}"
        npz = d / "scores.npz"
        if not npz.exists():
            continue
        z = np.load(npz, allow_pickle=False)
        s, y, paths = z["scores"], z["labels"], z["paths"]
        if len(np.unique(y)) < 2:
            continue
        pooled.append(roc_auc_score(y, s))
        cats = np.array([category_of(p, tgt) for p in paths])
        cat_aurocs = []
        for c in sorted(set(cats)):
            m = cats == c
            if len(np.unique(y[m])) < 2:
                continue
            a = roc_auc_score(y[m], s[m])
            cat_aurocs.append(a); per_cat[c].append(a)
        if cat_aurocs:
            macro.append(st.mean(cat_aurocs))
    return pooled, macro, per_cat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=None)
    a = ap.parse_args()
    methods = a.methods or sorted(
        p.name for p in ROOT.iterdir()
        if p.is_dir() and (p / "seed0").exists()
        and any((p / "seed0").glob("*__*/scores.npz")))

    out = {}
    for method in methods:
        seeds = sorted(int(p.name[4:]) for p in (ROOT / method).glob("seed*"))
        srcs = ["none"] if (ROOT / method / f"seed{seeds[0]}" / "none__mvtec").exists() \
            else ["sdnet", "mvtec", "vision"]
        printed = False
        for tgt in sorted(MULTI_CATEGORY):
            for src in srcs:
                pooled, macro, per_cat = analyse(method, src, tgt, seeds)
                if not pooled:
                    continue
                if not printed:
                    print(f"\n=== {method}"); printed = True
                    print(f"  {'source->target':22s} {'pooled':>16s} {'macro':>16s} {'diff':>8s}")
                pm, mm = st.mean(pooled), (st.mean(macro) if macro else float("nan"))
                ps = st.stdev(pooled) if len(pooled) > 1 else 0.0
                ms = st.stdev(macro) if len(macro) > 1 else 0.0
                print(f"  {src+'->'+tgt:22s} {pm:.4f}+/-{ps:.4f} {mm:.4f}+/-{ms:.4f} "
                      f"{mm-pm:+8.4f}")
                out[f"{method}/{src}__{tgt}"] = {
                    "pooled_mean": pm, "pooled_std": ps,
                    "macro_mean": mm, "macro_std": ms,
                    "per_category": {c: st.mean(v) for c, v in sorted(per_cat.items())},
                }
                if src == tgt:      # show the breakdown for the in-distribution cell
                    for c, v in sorted(per_cat.items()):
                        print(f"      {c:28s} {st.mean(v):.4f}")

    Path("results").mkdir(exist_ok=True)
    json.dump(out, open("results/per_category.json", "w"), indent=2)
    print("\n  written to results/per_category.json")


if __name__ == "__main__":
    main()
