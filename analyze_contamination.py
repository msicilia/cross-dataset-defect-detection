"""Controlled contamination experiment (run_contamination.py).

With n = run_contamination.N_BASE and a = run_contamination.N_ADD, for every
base source B and target T three kinds of bank share the same n images of B per
seed (bank directory names in brackets):
    baseline    n images of B                        (B<n>)
    plus_same   a more images of B                   (B<n+a>)
    plus_X      a images of each other source X      (B<n>+X<a>)
Each condition is summarised as mean and standard deviation (ddof=1) of AUROC
over cfg.SEEDS, together with its per-seed difference from baseline and, for
plus_X, from plus_same (mean and SD of the paired per-seed differences).

The figure shows the cases in which neither addition comes from the target:
bases other than VISION, targets other than the base, comparing baseline,
plus_same and plus_vision.

Input:  results/raw/dino_patchcore_contamination/<bank>/seed<s>/<target>/result.json
Output: results/contamination.json
        results/figures/contamination.pdf, .png

Usage:
    python analyze_contamination.py [--results-dir results]
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
import numpy as np

import config as cfg
from run_contamination import N_ADD, N_BASE

LABEL = {"sdnet": "SDNET", "mvtec": "MVTec", "vision": "VISION"}
CONTAMINANT = "vision"


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
    return {"mean": st.mean(values), "sd": st.stdev(values), "per_seed": values}


def conditions(base: str) -> dict[str, str]:
    """{condition name: bank directory name} for one base source."""
    out = {"baseline": f"{base}{N_BASE}", "plus_same": f"{base}{N_BASE + N_ADD}"}
    out.update({f"plus_{x}": f"{base}{N_BASE}+{x}{N_ADD}" for x in cfg.DATASETS if x != base})
    return out


def analyse(raw: Path) -> dict:
    cells = {}
    for base in cfg.DATASETS:
        banks = conditions(base)
        for name in banks.values():
            check_seeds(raw / name, cfg.SEEDS)
        for tgt in cfg.DATASETS:
            per_seed = {c: [auroc(raw / name / f"seed{s}" / tgt / "result.json")
                            for s in cfg.SEEDS] for c, name in banks.items()}
            entry = {"base": base, "target": tgt, "conditions": {}}
            for c, values in per_seed.items():
                cond = {"bank": banks[c], **summarise(values)}
                if c != "baseline":
                    cond["delta_vs_baseline"] = summarise(
                        [v - b for v, b in zip(values, per_seed["baseline"])])
                if c.startswith("plus_") and c != "plus_same":
                    cond["delta_vs_plus_same"] = summarise(
                        [v - b for v, b in zip(values, per_seed["plus_same"])])
                entry["conditions"][c] = cond
            cells[f"{base}->{tgt}"] = entry
    return cells


def print_table(cells: dict) -> None:
    print(f"\nAUROC mean ± SD over seeds {cfg.SEEDS}; "
          "d_base = change vs baseline, d_same = change vs plus_same")
    print(f"  {'base->target':16s} {'condition':14s} {'AUROC':>13s} {'d_base':>8s} {'d_same':>8s}")
    for key, entry in cells.items():
        label = f"{LABEL[entry['base']]}->{LABEL[entry['target']]}"
        for c, cond in entry["conditions"].items():
            d_base = cond.get("delta_vs_baseline")
            d_same = cond.get("delta_vs_plus_same")
            name = c if c in ("baseline", "plus_same") else f"plus_{LABEL[c[5:]]}"
            print(f"  {label:16s} {name:14s} {cond['mean']:7.3f}±{cond['sd']:.3f} "
                  f"{(format(d_base['mean'], '+8.3f') if d_base else ''):>8s} "
                  f"{(format(d_same['mean'], '+8.3f') if d_same else ''):>8s}")
            label = ""


def figure(cells: dict, cases: list[tuple[str, str]], figures: Path) -> None:
    series = [("baseline", f"base source ({N_BASE})", "#b0b0b0"),
              ("plus_same", f"+{N_ADD} same-source", "#4c78a8"),
              (f"plus_{CONTAMINANT}", f"+{N_ADD} {LABEL[CONTAMINANT]}", "#d65f5f")]
    x = np.arange(len(cases))
    w = 0.26
    # Drawn at the single-column width it is printed at, so nothing is scaled down.
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    lows, highs = [0.5], [0.5]
    for k, (cond, label, colour) in enumerate(series):
        mean = [cells[f"{b}->{t}"]["conditions"][cond]["mean"] for b, t in cases]
        sd = [cells[f"{b}->{t}"]["conditions"][cond]["sd"] for b, t in cases]
        ax.bar(x + (k - 1) * w, mean, w, yerr=sd, capsize=3, color=colour, label=label)
        lows += [m - e for m, e in zip(mean, sd)]
        highs += [m + e for m, e in zip(mean, sd)]
    span = max(highs) - min(lows)
    ax.set_ylim(min(lows) - 0.1 * span, max(highs) + 0.3 * span)
    ax.axhline(0.5, color="black", ls=":", lw=1.0)
    ax.text(len(cases) - 0.5, 0.5, "chance", fontsize=7, ha="right", va="bottom")
    ax.set_xticks(x)
    # Two lines: the four labels collide on one line at column width.
    ax.set_xticklabels([f"{LABEL[b]}\n→{LABEL[t]}" for b, t in cases], fontsize=7)
    ax.set_ylabel("AUROC", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    # No title: the caption carries it, and it does not fit one column legibly.
    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.03))
    figures.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(figures / f"contamination.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {figures / 'contamination.pdf'}")


def delta_range(cells: dict, keys: list[str], cond: str, field: str) -> list[float]:
    values = [cells[k]["conditions"][cond][field]["mean"] for k in keys]
    return [min(values), max(values)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", type=Path, default=cfg.RESULTS_DIR)
    args = ap.parse_args()
    raw = args.results_dir / "raw" / "dino_patchcore_contamination"

    cells = analyse(raw)
    print_table(cells)
    cases = [(b, t) for b in cfg.DATASETS if b != CONTAMINANT
             for t in cfg.DATASETS if t != b]
    case_keys = [f"{b}->{t}" for b, t in cases]
    contaminant = f"plus_{CONTAMINANT}"
    summary = {
        "figure_cases": case_keys,
        "plus_same_delta_vs_baseline_range_figure_cases":
            delta_range(cells, case_keys, "plus_same", "delta_vs_baseline"),
        "plus_same_delta_vs_baseline_range_all_cells":
            delta_range(cells, list(cells), "plus_same", "delta_vs_baseline"),
        f"{contaminant}_delta_vs_plus_same_range_figure_cases":
            delta_range(cells, case_keys, contaminant, "delta_vs_plus_same"),
    }
    print("\nRanges of mean differences:")
    for k, v in summary.items():
        if k != "figure_cases":
            print(f"  {k}: {v[0]:+.3f} to {v[1]:+.3f}")
    figure(cells, cases, args.results_dir / "figures")
    dest = args.results_dir / "contamination.json"
    write_json(dest, {"seeds": cfg.SEEDS, "cells": cells, "summary": summary})
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
