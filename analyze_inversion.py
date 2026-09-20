"""Score-distribution statistics for below-chance (inverted) transfer cells.

Per cell (method, source, target) and seed, from the stored per-image scores:

  auroc       image-level AUROC.
  cohens_d    (mean defective score - mean defect-free score) / pooled SD, where
              pooled SD = sqrt(((n1-1) s1^2 + (n0-1) s0^2) / (n1 + n0 - 2)) and
              s1, s0 are the class SDs with ddof=1. Negative when defect-free
              images score higher.
  overlap     overlap coefficient of the two class histograms: sum over bins of
              min(density_defective, density_normal) * bin width, with
              HIST_BINS equal-width bins spanning the pooled score range.
  norm_shift  |mean score of the cell - mean score of the in-distribution
              cell| / SD of the in-distribution cell, where the in-distribution
              cell is the same method and seed with source = target, and both
              mean and SD (ddof=1) are taken over all its test images,
              defective and defect-free. Undefined for reference-free detectors,
              which have no source dataset.

Each statistic is averaged over seeds (seed 0 only for reference-free
detectors). A cell is inverted when its mean AUROC is below INVERSION_THRESHOLD.

Usage:  python analyze_inversion.py
Output: results/inversion.json
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

import config as cfg

RAW = cfg.RESULTS_DIR / "raw"
REFERENCE_BASED = ["patchcore", "dino_patchcore_base", "padim", "spade", "winclip_plus"]
REFERENCE_FREE = ["winclip", "clip_zs"]
# Cut below chance by more than seed noise; the selected cells are unchanged for
# any threshold up to the lowest non-inverted AUROC, which the script reports.
INVERSION_THRESHOLD = 0.485
HIST_BINS = 49


def seeds_for(method: str) -> list[int]:
    return [0] if method in REFERENCE_FREE else list(cfg.SEEDS)


def sources_for(method: str) -> list[str]:
    return ["none"] if method in REFERENCE_FREE else list(cfg.DATASETS)


def npz_path(method: str, src: str, tgt: str, seed: int):
    return RAW / method / f"seed{seed}" / f"{src}__{tgt}" / "scores.npz"


def load(method: str, src: str, tgt: str, seed: int):
    z = np.load(npz_path(method, src, tgt, seed), allow_pickle=False)
    return z["scores"].astype(np.float64), z["labels"].astype(np.int64)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp)


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    edges = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), HIST_BINS + 1)
    ha, _ = np.histogram(a, bins=edges, density=True)
    hb, _ = np.histogram(b, bins=edges, density=True)
    return float(np.minimum(ha, hb).sum() * (edges[1] - edges[0]))


def write_json(path, payload) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def main() -> None:
    methods = REFERENCE_BASED + REFERENCE_FREE
    missing = [str(npz_path(m, s, t, seed))
               for m in methods for s in sources_for(m) for t in cfg.DATASETS
               for seed in seeds_for(m) if not npz_path(m, s, t, seed).exists()]
    if missing:
        raise SystemExit("missing benchmark runs:\n  " + "\n  ".join(missing))

    cells = {}
    print(f"  {'cell':38s} {'AUROC':>6s} {'d':>7s} {'overlap':>8s} {'shift':>6s}")
    for m in methods:
        for src in sources_for(m):
            for tgt in cfg.DATASETS:
                per_seed = {"auroc": [], "cohens_d": [], "overlap": [], "norm_shift": []}
                for seed in seeds_for(m):
                    s, y = load(m, src, tgt, seed)
                    per_seed["auroc"].append(float(roc_auc_score(y, s)))
                    per_seed["cohens_d"].append(cohens_d(s[y == 1], s[y == 0]))
                    per_seed["overlap"].append(overlap(s[y == 1], s[y == 0]))
                    if src != "none":
                        ref, _ = load(m, tgt, tgt, seed)
                        per_seed["norm_shift"].append(
                            float(abs(s.mean() - ref.mean()) / ref.std(ddof=1)))
                key = f"{m}/{src}__{tgt}"
                cell = {"method": m, "source": src, "target": tgt,
                        "seeds": seeds_for(m), "per_seed": per_seed}
                for k, v in per_seed.items():
                    cell[f"{k}_mean"] = float(np.mean(v)) if v else None
                cell["inverted"] = cell["auroc_mean"] < INVERSION_THRESHOLD
                cells[key] = cell
                shift = (f"{cell['norm_shift_mean']:6.2f}" if cell["norm_shift_mean"] is not None
                         else f"{'-':>6s}")
                print(f"  {key:38s} {cell['auroc_mean']:6.3f} {cell['cohens_d_mean']:7.3f} "
                      f"{cell['overlap_mean']:8.3f} {shift}{'  inverted' if cell['inverted'] else ''}")

    inverted = [k for k, c in cells.items() if c["inverted"]]
    inv_shift = [cells[k]["norm_shift_mean"] for k in inverted
                 if cells[k]["norm_shift_mean"] is not None]
    inv_d = [cells[k]["cohens_d_mean"] for k in inverted]
    not_inverted = {k: c["auroc_mean"] for k, c in cells.items() if not c["inverted"]}
    lowest_other = min(not_inverted, key=not_inverted.get)
    summary = {
        "n_inverted": len(inverted),
        "inverted_cells": inverted,
        "n_inverted_negative_d": sum(d < 0 for d in inv_d),
        "cohens_d_range_inverted": [min(inv_d), max(inv_d)] if inv_d else None,
        "norm_shift_range_inverted_with_source": [min(inv_shift), max(inv_shift)] if inv_shift else None,
        "n_inverted_with_source": len(inv_shift),
        "inverted_involving_vision": sum("vision" in (cells[k]["source"], cells[k]["target"])
                                         for k in inverted),
        "lowest_auroc_not_inverted": {"cell": lowest_other, "auroc_mean": not_inverted[lowest_other]},
    }

    print(f"\n  cells with mean AUROC < {INVERSION_THRESHOLD}: {len(inverted)}")
    for k in inverted:
        c = cells[k]
        print(f"    {k:38s} AUROC={c['auroc_mean']:.3f} d={c['cohens_d_mean']:+.3f}")
    print(f"  Cohen's d range over these: {summary['cohens_d_range_inverted']}")
    print(f"  normalised shift range over those with a source: "
          f"{summary['norm_shift_range_inverted_with_source']}")
    print(f"  lowest mean AUROC not below the threshold: {lowest_other} "
          f"{not_inverted[lowest_other]:.4f}")

    out = cfg.RESULTS_DIR / "inversion.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, {"config": {"threshold": INVERSION_THRESHOLD, "hist_bins": HIST_BINS,
                                "seeds": list(cfg.SEEDS), "methods": methods},
                     "cells": cells, "summary": summary})
    print(f"\n  written to {out}")


if __name__ == "__main__":
    main()
