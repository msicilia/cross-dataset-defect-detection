from __future__ import annotations
"""Confidence intervals and significance tests.

Reported differences could be random variation in reference-image selection.
Two distinct sources of variance are involved:

  seed variance    which reference images were drawn, and how the coreset was
                   initialised. Captured by the +/- in the tables.
  sampling variance which test images happened to be in the evaluation set.
                   Not the same thing: a cell can be stable across seeds yet
                   still rest on a small number of informative test images.

Both are reported here, per cell, plus paired tests for the comparisons the
paper's claims actually depend on. Pairing matters: comparing two methods across
independent runs discards the fact that they saw the same seeds and the same
test images, which is exactly the information that makes a small difference
detectable.

Usage:  python analyze_significance.py [--boot 1000]
"""
import argparse
import json
import statistics as st
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path("results/raw")
DATASETS = ["sdnet", "mvtec", "vision"]


def fast_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney form; far cheaper than sklearn inside a bootstrap loop."""
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties, otherwise tied scores bias the estimate
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_ci(scores, labels, B: int, rng) -> tuple[float, float]:
    n = len(scores)
    out = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        out[b] = fast_auroc(scores[idx], labels[idx])
    out = out[~np.isnan(out)]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def load(method, src, tgt, seed):
    p = ROOT / method / f"seed{seed}" / f"{src}__{tgt}" / "scores.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    return z["scores"], z["labels"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    methods = [m for m in ("patchcore", "dino_patchcore_base", "padim", "spade",
                           "winclip_plus", "clip_zs") if (ROOT / m).exists()]

    print(f"  {'method':20s} {'src->tgt':16s} {'mean':>7s} {'seed0':>6s} "
          f"{'95% CI (test imgs)':>18s} {'seed sd':>8s}")
    print("  " + "-" * 82)
    results = {}
    for m in methods:
        seeds = sorted(int(p.name[4:]) for p in (ROOT / m).glob("seed*"))
        srcs = ["none"] if (ROOT / m / f"seed{seeds[0]}" / "none__mvtec").exists() else DATASETS
        for src in srcs:
            for tgt in DATASETS:
                per_seed = [load(m, src, tgt, s) for s in seeds]
                per_seed = [x for x in per_seed if x is not None]
                if not per_seed:
                    continue
                aurocs = [fast_auroc(s, y) for s, y in per_seed]
                # The interval is a property of ONE run's test sample, so it is
                # reported against that run's AUROC. Pairing it with the
                # multi-seed mean would produce intervals that appear not to
                # contain their own point estimate, since seed variance is a
                # separate source of spread reported in the last column.
                a0 = fast_auroc(*per_seed[0])
                lo, hi = bootstrap_ci(*per_seed[0], a.boot, rng)
                sd = st.stdev(aurocs) if len(aurocs) > 1 else 0.0
                print(f"  {m:20s} {src+'->'+tgt:16s} {st.mean(aurocs):7.3f} "
                      f"{a0:6.3f} [{lo:.3f}, {hi:.3f}] {sd:8.4f}")
                results[f"{m}/{src}__{tgt}"] = {
                    "auroc_mean": st.mean(aurocs), "auroc_seed0": a0, "seed_sd": sd,
                    "boot_ci_low": lo, "boot_ci_high": hi, "n_seeds": len(aurocs)}

    # ── paired tests on the claims the paper actually makes ──────────────────
    print("\n  Paired comparisons of the cross-dataset gap (same seeds):")
    gaps = {}
    for m in methods:
        if m == "clip_zs":
            continue
        seeds = sorted(int(p.name[4:]) for p in (ROOT / m).glob("seed*"))
        per_seed = []
        for s in seeds:
            ind = [load(m, d, d, s) for d in DATASETS]
            cro = [load(m, x, y, s) for x in DATASETS for y in DATASETS if x != y]
            if any(v is None for v in ind + cro):
                per_seed = []; break
            per_seed.append(st.mean([fast_auroc(*v) for v in ind])
                            - st.mean([fast_auroc(*v) for v in cro]))
        if per_seed:
            gaps[m] = per_seed

    for x, y in combinations(gaps, 2):
        if len(gaps[x]) != len(gaps[y]):
            continue
        diff = [p - q for p, q in zip(gaps[x], gaps[y])]
        n = len(diff)
        mean_d = st.mean(diff)
        sd = st.stdev(diff) if n > 1 else 0.0
        t = mean_d / (sd / n ** 0.5) if sd > 0 else float("inf")
        crit = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78}.get(n - 1, 2.78)
        verdict = "significant" if abs(t) > crit else "not significant"
        print(f"    {x:20s} vs {y:20s} dDelta={mean_d:+.4f} t={t:+6.2f} "
              f"(df={n-1}, crit={crit}) -> {verdict}")

    Path("results").mkdir(exist_ok=True)
    json.dump({"cells": results, "gaps": gaps}, open("results/significance.json", "w"), indent=2)
    print("\n  written to results/significance.json")


if __name__ == "__main__":
    main()
