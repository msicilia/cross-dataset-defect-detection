"""Memory-bank composition and detector-gap figures.

proportions  AUROC of DINO-PatchCore against the percentage of a
             run_proportions.BANK_SIZE-image bank drawn from VISION, one panel
             per base source and one line per target; mean ± SD (ddof=1) over
             cfg.SEEDS.
paradigm     Generalisation gap Δ = mean in-distribution AUROC (source == target)
             − mean cross-dataset AUROC (source != target) for each detector,
             computed per seed and averaged over cfg.SEEDS, sorted by Δ and
             coloured by whether the detector uses source-domain reference
             images. Reference-free detectors are stored once with source
             "none" and seed 0; they have no source domain, so Δ = 0.

Input:  results/raw/dino_patchcore_proportions/<base>_vision<pct:03d>/seed<s>/<target>/result.json
        results/raw/<method>/seed<s>/<src>__<tgt>/result.json
Output: results/proportions.json, results/paradigm.json
        results/figures/proportions.pdf, .png; results/figures/paradigm.pdf, .png

Usage:
    python make_figures.py [proportions] [paradigm] [--results-dir results]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
# TrueType, not matplotlib's default Type 3: Type 3 renders poorly at the sizes
# these figures are printed at, and IEEE does not accept it.
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt

import config as cfg
from run_proportions import BANK_SIZE, BASES, CONTAMINANT, FRACTIONS

LABEL = {"sdnet": "SDNET", "mvtec": "MVTec", "vision": "VISION"}
PERCENTAGES = [int(f * 100) for f in FRACTIONS]
DETECTORS = [  # (label, results directory, uses source-domain references)
    ("PatchCore", "patchcore", True),
    ("DINO-PatchCore", "dino_patchcore_base", True),
    ("PaDiM", "padim", True),
    ("SPADE", "spade", True),
    ("WinCLIP+", "winclip_plus", True),
    ("WinCLIP", "winclip", False),
    ("CLIP-ZS", "clip_zs", False),
]
COLOUR = {True: "#c44e52", False: "#4c72b0"}


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


def save_figure(fig, figures: Path, name: str) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(figures / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {figures / f'{name}.pdf'}")


def proportions(results: Path) -> None:
    root = results / "raw" / "dino_patchcore_proportions"
    data = {}
    for base in BASES:
        dirs = {pct: root / f"{base}_{CONTAMINANT}{pct:03d}" for pct in PERCENTAGES}
        for d in dirs.values():
            check_seeds(d, cfg.SEEDS)
        data[base] = {}
        for tgt in cfg.DATASETS:
            points = []
            for pct, d in dirs.items():
                values = [auroc(d / f"seed{s}" / tgt / "result.json") for s in cfg.SEEDS]
                points.append({"vision_pct": pct, "bank": d.name, "mean": st.mean(values),
                               "sd": st.stdev(values), "per_seed": values})
            data[base][tgt] = points

    print(f"\nProportions: AUROC mean ± SD over seeds {cfg.SEEDS}")
    print(f"  {'base':6s} {'target':7s}" + "".join(f"{str(p) + '%':>14s}" for p in PERCENTAGES))
    for base, by_target in data.items():
        for tgt, points in by_target.items():
            print(f"  {LABEL[base]:6s} {LABEL[tgt]:7s}"
                  + "".join(f"{p['mean']:8.3f}±{p['sd']:.3f}" for p in points))

    # Stacked rather than side by side: the figure sits in one column, where a
    # 2:1 landscape layout renders too small to read.
    fig, axes = plt.subplots(len(BASES), 1, figsize=(3.5, 3.2), sharex=True, sharey=True)
    for ax, base in zip(axes, BASES):
        for tgt, points in data[base].items():
            ax.errorbar(PERCENTAGES, [p["mean"] for p in points],
                        yerr=[p["sd"] for p in points],
                        marker="o", ms=3.5, capsize=2, label=LABEL[tgt])
        ax.axhline(0.5, ls=":", c="grey", lw=1)
        ax.set_title(f"base: {LABEL[base]}", fontsize=8)
        ax.set_ylabel("AUROC", fontsize=8)
        ax.set_xticks(PERCENTAGES)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel(f"% of the {BANK_SIZE}-image bank drawn from {LABEL[CONTAMINANT]}",
                        fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    save_figure(fig, results / "figures", "proportions")
    write_json(results / "proportions.json",
               {"seeds": cfg.SEEDS, "bank_size": BANK_SIZE, "bases": data})
    print(f"wrote {results / 'proportions.json'}")


def detector_gap(raw: Path, method: str, uses_references: bool) -> dict:
    d = raw / method
    datasets = cfg.DATASETS
    if not uses_references:
        check_seeds(d, [0])
        values = [auroc(d / "seed0" / f"none__{t}" / "result.json") for t in datasets]
        mean = st.mean(values)
        return {"seeds": [0], "in_distribution": mean, "cross_dataset": mean,
                "gap": 0.0, "gap_sd": None}
    check_seeds(d, cfg.SEEDS)
    ind, cro = [], []
    for s in cfg.SEEDS:
        if (d / f"seed{s}" / f"none__{datasets[0]}").exists():
            raise RuntimeError(f"{d}: reference-free cells found for a reference-based detector")
        ind.append(st.mean(auroc(d / f"seed{s}" / f"{t}__{t}" / "result.json") for t in datasets))
        cro.append(st.mean(auroc(d / f"seed{s}" / f"{a}__{b}" / "result.json")
                           for a in datasets for b in datasets if a != b))
    gaps = [i - c for i, c in zip(ind, cro)]
    return {"seeds": cfg.SEEDS, "in_distribution": st.mean(ind), "cross_dataset": st.mean(cro),
            "gap": st.mean(gaps), "gap_sd": st.stdev(gaps)}


def paradigm(results: Path) -> None:
    raw = results / "raw"
    rows = []
    for label, method, uses_references in DETECTORS:
        rows.append({"label": label, "method": method, "uses_references": uses_references,
                     **detector_gap(raw, method, uses_references)})
    rows.sort(key=lambda r: (-r["gap"], not r["uses_references"], r["label"]))

    print("\nGeneralisation gap")
    print(f"  {'detector':16s} {'references':>10s} {'in-dist':>8s} {'cross':>8s} {'gap':>8s} {'SD':>7s}")
    for r in rows:
        sd = f"{r['gap_sd']:7.3f}" if r["gap_sd"] is not None else f"{'':7s}"
        print(f"  {r['label']:16s} {'yes' if r['uses_references'] else 'no':>10s} "
              f"{r['in_distribution']:8.3f} {r['cross_dataset']:8.3f} {r['gap']:8.3f} {sd}")

    gaps = [r["gap"] for r in rows]
    span = max(max(gaps), 0.0) - min(min(gaps), 0.0)
    # Drawn at the single-column width it is printed at, so nothing is scaled down.
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    y = range(len(rows))
    colours = [COLOUR[r["uses_references"]] for r in rows]
    ax.barh(y, gaps, color=colours)
    for i, r in enumerate(rows):
        if r["gap"] == 0.0:
            ax.plot(0.0, i, marker="D", ms=4, color=colours[i], clip_on=False, zorder=3)
        ax.text(max(r["gap"], 0.0) + 0.02 * span, i, f"{r['gap']:.3f}", va="center", fontsize=7)
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_yticks(list(y))
    ax.set_yticklabels([r["label"] for r in rows], fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(r"generalisation gap $\Delta$", fontsize=8)
    ax.set_xlim(min(min(gaps), 0.0) - 0.05 * span, max(max(gaps), 0.0) + 0.2 * span)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOUR[True]),
               plt.Rectangle((0, 0), 1, 1, color=COLOUR[False])]
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    # Below the axes: inside, it overlapped the zero-gap rows.
    fig.legend(handles, ["uses source-domain references", "reference-free"],
               frameon=False, fontsize=7, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.04))
    save_figure(fig, results / "figures", "paradigm")
    write_json(results / "paradigm.json", {"detectors": rows})
    print(f"wrote {results / 'paradigm.json'}")


FIGURES = {"proportions": proportions, "paradigm": paradigm}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("figures", nargs="*", choices=list(FIGURES),
                    help="figures to build (default: all)")
    ap.add_argument("--results-dir", type=Path, default=cfg.RESULTS_DIR)
    args = ap.parse_args()
    for name in args.figures or FIGURES:
        FIGURES[name](args.results_dir)


if __name__ == "__main__":
    main()
