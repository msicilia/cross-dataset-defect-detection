"""Confidence intervals, generalisation gaps and paired tests for the benchmark.

Two sources of variance are reported per cell (method, source, target):

  seed spread      standard deviation (ddof=1) of AUROC over the seeds, which
                   vary the reference images drawn and the coreset start.
  test sampling    95% percentile bootstrap interval over test images for the
                   seed-0 run. Each cell has its own generator, seeded with
                   zlib.crc32("method/source__target"), so an interval does not
                   depend on which other cells are analysed.

Generalisation gaps, per seed:

  full         mean in-distribution AUROC over the three datasets minus mean
               cross-dataset AUROC over the six off-diagonal cells.
  VISION-free  the same on SDNET2018 and MVTec AD only (two diagonal and two
               off-diagonal cells).

Reference-free detectors (clip_zs, winclip) have a single seed-independent run
per target, no source dataset and a gap of zero by construction.

Paired t-tests (two-sided, df = n_seeds - 1) are grouped into four families,
with Holm-adjusted p-values within each family:

  paradigm_full         all pairs of reference-based detectors, full gap
  paradigm_vision_free  the same pairs, VISION-free gap
  backbone_cross        DINOv2 ViT-S/B/L, mean cross-dataset AUROC
  backbone_in_dist      DINOv2 ViT-S/B/L, mean in-distribution AUROC

Interval widths: the mean bootstrap width over all cells and per target, the
mean seed SD over reference-based cells, and the 95% interval about an n-seed
mean implied by that spread, 2 * t_{0.975, n-1} * sd / sqrt(n).

Usage:  python analyze_significance.py [--boot 1000]
Output: results/significance.json
"""
from __future__ import annotations

import argparse
import json
import os
import zlib
from itertools import combinations

import numpy as np
from scipy import stats

import config as cfg

RAW = cfg.RESULTS_DIR / "raw"
REFERENCE_BASED = ["patchcore", "dino_patchcore_base", "padim", "spade", "winclip_plus"]
REFERENCE_FREE = ["clip_zs", "winclip"]
BACKBONES = ["dino_patchcore_small", "dino_patchcore_base", "dino_patchcore_large"]
VISION_FREE = ["sdnet", "mvtec"]


def seeds_for(method: str) -> list[int]:
    return [0] if method in REFERENCE_FREE else list(cfg.SEEDS)


def sources_for(method: str) -> list[str]:
    return ["none"] if method in REFERENCE_FREE else list(cfg.DATASETS)


def cell_key(method: str, src: str, tgt: str) -> str:
    return f"{method}/{src}__{tgt}"


def npz_path(method: str, src: str, tgt: str, seed: int):
    return RAW / method / f"seed{seed}" / f"{src}__{tgt}" / "scores.npz"


def check_complete(methods: list[str]) -> None:
    missing = [str(npz_path(m, s, t, seed))
               for m in methods for s in sources_for(m) for t in cfg.DATASETS
               for seed in seeds_for(m) if not npz_path(m, s, t, seed).exists()]
    if missing:
        raise SystemExit("missing benchmark runs:\n  " + "\n  ".join(missing))


def load(method: str, src: str, tgt: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(npz_path(method, src, tgt, seed), allow_pickle=False)
    return z["scores"].astype(np.float64), z["labels"].astype(np.int64)


def auroc_rows(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Mann-Whitney AUROC for each row of 2-D arrays, ties given mid-ranks.

    Rows lacking one of the two classes give nan.
    """
    ranks = stats.rankdata(scores, axis=1)
    n_pos = labels.sum(axis=1)
    n_neg = labels.shape[1] - n_pos
    u = (ranks * labels).sum(axis=1) - n_pos * (n_pos + 1) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        out = u / (n_pos * n_neg)
    out[(n_pos == 0) | (n_neg == 0)] = np.nan
    return out


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(auroc_rows(scores[None, :], labels[None, :])[0])


def bootstrap_ci(scores, labels, n_boot: int, key: str, chunk: int = 50) -> dict:
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    n = len(scores)
    reps = []
    for start in range(0, n_boot, chunk):
        idx = rng.integers(0, n, size=(min(chunk, n_boot - start), n))
        reps.append(auroc_rows(scores[idx], labels[idx]))
    reps = np.concatenate(reps)
    valid = reps[~np.isnan(reps)]
    lo, hi = np.percentile(valid, [2.5, 97.5])
    return {"low": float(lo), "high": float(hi), "width": float(hi - lo),
            "rng_seed": zlib.crc32(key.encode()), "n_valid_resamples": int(len(valid))}


def sd1(values) -> float | None:
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def paired_test(a: list[float], b: list[float]) -> dict:
    diff = np.asarray(a) - np.asarray(b)
    n = len(diff)
    df = n - 1
    res = stats.ttest_rel(a, b)
    return {"mean_diff": float(diff.mean()), "t": float(res.statistic), "df": df,
            "p": float(res.pvalue), "crit_0.975": float(stats.t.ppf(0.975, df)),
            "n": n}


def holm(tests: dict) -> None:
    """Add Holm-adjusted p-values (key p_holm) in place."""
    names = sorted(tests, key=lambda k: tests[k]["p"])
    m = len(names)
    running = 0.0
    for i, k in enumerate(names):
        running = max(running, min(1.0, (m - i) * tests[k]["p"]))
        tests[k]["p_holm"] = running


def write_json(path, payload) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def per_seed_summary(aurocs: dict, method: str, datasets: list[str]) -> dict:
    """Per-seed in-distribution mean, cross-dataset mean and gap over `datasets`."""
    ind, cro = [], []
    for i, _ in enumerate(seeds_for(method)):
        ind.append(float(np.mean([aurocs[cell_key(method, d, d)][i] for d in datasets])))
        cro.append(float(np.mean([aurocs[cell_key(method, s, t)][i]
                                  for s in datasets for t in datasets if s != t])))
    return {"in_dist": ind, "cross": cro, "gap": [x - y for x, y in zip(ind, cro)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    methods = REFERENCE_BASED + REFERENCE_FREE
    check_complete(methods + [m for m in BACKBONES if m not in methods])

    # ── per-cell AUROC, seed spread and bootstrap interval ───────────────────
    cells, aurocs = {}, {}
    print(f"  {'cell':38s} {'mean':>6s} {'seed0':>6s} {'95% bootstrap CI':>17s} "
          f"{'seed sd':>8s}")
    for m in methods:
        seeds = seeds_for(m)
        for src in sources_for(m):
            for tgt in cfg.DATASETS:
                key = cell_key(m, src, tgt)
                vals = [auroc(*load(m, src, tgt, s)) for s in seeds]
                aurocs[key] = vals
                s0, y0 = load(m, src, tgt, 0)
                ci = bootstrap_ci(s0, y0, args.boot, key)
                sd = sd1(vals)
                cells[key] = {
                    "method": m, "source": src, "target": tgt,
                    "auroc_mean": float(np.mean(vals)),
                    "auroc_per_seed": dict(zip(map(str, seeds), vals)),
                    "auroc_seed0": vals[0], "seed_sd": sd, "n_seeds": len(vals),
                    "n_test": int(len(y0)), "n_test_defective": int(y0.sum()),
                    "boot_ci_low": ci["low"], "boot_ci_high": ci["high"],
                    "boot_ci_width": ci["width"], "boot_rng_seed": ci["rng_seed"],
                    "boot_n_valid": ci["n_valid_resamples"],
                }
                sd_txt = f"{sd:8.4f}" if sd is not None else f"{'-':>8s}"
                print(f"  {key:38s} {np.mean(vals):6.3f} {vals[0]:6.3f} "
                      f"[{ci['low']:.3f}, {ci['high']:.3f}] {sd_txt}")

    # ── gaps ─────────────────────────────────────────────────────────────────
    for m in BACKBONES:
        if m in methods:
            continue
        for src in cfg.DATASETS:
            for tgt in cfg.DATASETS:
                aurocs[cell_key(m, src, tgt)] = [auroc(*load(m, src, tgt, s))
                                                 for s in seeds_for(m)]

    gaps, gaps_vf, summary = {}, {}, {}
    for m in REFERENCE_BASED:
        full = per_seed_summary(aurocs, m, cfg.DATASETS)
        vf = per_seed_summary(aurocs, m, VISION_FREE)
        gaps[m], gaps_vf[m] = full["gap"], vf["gap"]
        summary[m] = {
            "reference_based": True, "seeds": seeds_for(m),
            "in_dist_per_seed": full["in_dist"], "cross_per_seed": full["cross"],
            "in_dist_mean": float(np.mean(full["in_dist"])),
            "in_dist_sd": sd1(full["in_dist"]),
            "cross_mean": float(np.mean(full["cross"])),
            "cross_sd": sd1(full["cross"]),
            "gap_mean": float(np.mean(full["gap"])),
            "gap_vision_free_mean": float(np.mean(vf["gap"])),
        }
    for m in REFERENCE_FREE:
        per_target = [aurocs[cell_key(m, "none", t)][0] for t in cfg.DATASETS]
        summary[m] = {"reference_based": False, "seeds": seeds_for(m),
                      "in_dist_mean": float(np.mean(per_target)),
                      "gap_mean": 0.0, "gap_vision_free_mean": 0.0}

    print(f"\n  {'method':22s} {'in-dist':>8s} {'cross':>8s} {'gap':>7s} {'gap excl. VISION':>17s}")
    for m, v in summary.items():
        cross = f"{v['cross_mean']:8.3f}" if "cross_mean" in v else f"{'-':>8s}"
        print(f"  {m:22s} {v['in_dist_mean']:8.3f} {cross} {v['gap_mean']:7.3f} "
              f"{v['gap_vision_free_mean']:17.3f}")

    backbone = {}
    for m in BACKBONES:
        full = per_seed_summary(aurocs, m, cfg.DATASETS)
        backbone[m] = {
            "seeds": seeds_for(m),
            "in_dist_per_seed": full["in_dist"], "cross_per_seed": full["cross"],
            "gap_per_seed": full["gap"],
            "in_dist_mean": float(np.mean(full["in_dist"])), "in_dist_sd": sd1(full["in_dist"]),
            "cross_mean": float(np.mean(full["cross"])), "cross_sd": sd1(full["cross"]),
            "gap_mean": float(np.mean(full["gap"])),
        }
    lowest_cross = [min(BACKBONES, key=lambda m: backbone[m]["cross_per_seed"][i])
                    for i in range(len(cfg.SEEDS))]
    highest_in_dist = [max(BACKBONES, key=lambda m: backbone[m]["in_dist_per_seed"][i])
                       for i in range(len(cfg.SEEDS))]

    # ── paired tests ─────────────────────────────────────────────────────────
    tests = {"paradigm_full": {}, "paradigm_vision_free": {},
             "backbone_cross": {}, "backbone_in_dist": {}}
    for a, b in combinations(REFERENCE_BASED, 2):
        tests["paradigm_full"][f"{a}-{b}"] = paired_test(gaps[a], gaps[b])
        tests["paradigm_vision_free"][f"{a}-{b}"] = paired_test(gaps_vf[a], gaps_vf[b])
    for a, b in combinations(BACKBONES, 2):
        tests["backbone_cross"][f"{a}-{b}"] = paired_test(
            backbone[a]["cross_per_seed"], backbone[b]["cross_per_seed"])
        tests["backbone_in_dist"][f"{a}-{b}"] = paired_test(
            backbone[a]["in_dist_per_seed"], backbone[b]["in_dist_per_seed"])
    for family in tests.values():
        holm(family)

    for family, results in tests.items():
        print(f"\n  paired t-tests: {family}")
        for name, r in results.items():
            print(f"    {name:48s} diff={r['mean_diff']:+.4f} t={r['t']:+7.2f} "
                  f"df={r['df']} crit={r['crit_0.975']:.3f} p={r['p']:.2e} "
                  f"p_holm={r['p_holm']:.2e}")
    print(f"\n  backbone with lowest cross-dataset AUROC per seed: {lowest_cross}")
    print(f"  backbone with highest in-distribution AUROC per seed: {highest_in_dist}")

    # ── interval widths ──────────────────────────────────────────────────────
    ref_cells = [k for k, c in cells.items() if c["method"] in REFERENCE_BASED]
    all_cells = list(cells)
    seed_widths = [2 * stats.t.ppf(0.975, cells[k]["n_seeds"] - 1) * cells[k]["seed_sd"]
                   / np.sqrt(cells[k]["n_seeds"]) for k in ref_cells]
    boot_ref = float(np.mean([cells[k]["boot_ci_width"] for k in ref_cells]))
    seed_w = float(np.mean(seed_widths))
    by_target = {}
    for t in cfg.DATASETS:
        ks = [k for k in all_cells if cells[k]["target"] == t]
        by_target[t] = {"mean_width": float(np.mean([cells[k]["boot_ci_width"] for k in ks])),
                        "n_test": sorted({cells[k]["n_test"] for k in ks}), "cells": ks}
    by_method = {m: float(np.mean([c["boot_ci_width"] for c in cells.values()
                                   if c["method"] == m])) for m in methods}
    widths = {
        "bootstrap_mean_width": float(np.mean([cells[k]["boot_ci_width"] for k in all_cells])),
        "bootstrap_cells": all_cells,
        "bootstrap_mean_width_by_target": by_target,
        "bootstrap_mean_width_by_method": by_method,
        "bootstrap_mean_width_reference_based": boot_ref,
        "seed_sd_mean": float(np.mean([cells[k]["seed_sd"] for k in ref_cells])),
        "seed_interval_mean_width": seed_w,
        "seed_cells": ref_cells,
        "ratio_bootstrap_to_seed_reference_based": boot_ref / seed_w,
        "n_boot": args.boot,
    }
    print(f"\n  bootstrap 95% width: {widths['bootstrap_mean_width']:.4f} over "
          f"{len(all_cells)} cells; reference-based {boot_ref:.4f} over {len(ref_cells)}")
    for t, v in by_target.items():
        print(f"    target {t:7s} {v['mean_width']:.4f}  (n_test {v['n_test']})")
    print(f"  seed sd mean {widths['seed_sd_mean']:.4f}; seed-spread 95% width (t) "
          f"{seed_w:.4f} over {len(ref_cells)} cells; ratio {boot_ref / seed_w:.2f}")

    out = cfg.RESULTS_DIR / "significance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, {
        "config": {"n_boot": args.boot, "seeds": list(cfg.SEEDS),
                   "reference_based": REFERENCE_BASED, "reference_free": REFERENCE_FREE,
                   "backbones": BACKBONES, "vision_free_datasets": VISION_FREE},
        "cells": cells, "gaps": gaps, "gaps_vision_free": gaps_vf,
        "methods": summary,
        "backbone": {"per_method": backbone, "lowest_cross_per_seed": lowest_cross,
                     "highest_in_dist_per_seed": highest_in_dist},
        "tests": tests, "interval_widths": widths,
    })
    print(f"\n  written to {out}")


if __name__ == "__main__":
    main()
