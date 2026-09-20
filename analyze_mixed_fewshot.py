"""Mixed-source memory banks and reference-set size for DINO-PatchCore.

Mixed-source table: single-source rows are means over cfg.SEEDS of the main
benchmark (dino_patchcore_base); mixed rows (250 images from each of two
sources) are mean and standard deviation over cfg.SUBSET_SEEDS. For each mixed
bank and target the difference from the better of its two single-source rows is
also reported.

Reference-set size: for each source and number of reference images n, the
cross-dataset AUROC of a seed is the mean over the two other targets, and the
in-distribution AUROC is the source == target cell; both are then averaged over
cfg.SUBSET_SEEDS. Figure: (a) cross-dataset and (b) in-distribution AUROC
against n (log scale), bands = one standard deviation.

All standard deviations use ddof=1.

Input:  results/raw/dino_patchcore_base/seed<s>/<src>__<tgt>/result.json
        results/raw/dino_patchcore_mixed/seed<s>/<a>+<b>__<tgt>/result.json
        results/raw/dino_patchcore_fewshot/n<N>/seed<s>/<src>__<tgt>/result.json
Output: results/mixed_fewshot.json
        results/figures/fewshot_curve.pdf, .png

Usage:
    python analyze_mixed_fewshot.py [--results-dir results]
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
from run_fewshot import N_SHOTS

LABEL = {"sdnet": "SDNET", "mvtec": "MVTec", "vision": "VISION"}
COLOUR = {"sdnet": "#1f77b4", "mvtec": "#ff7f0e", "vision": "#2ca02c"}
PAIRS = [("sdnet", "mvtec"), ("sdnet", "vision"), ("mvtec", "vision")]


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


def mixed_source(raw: Path) -> dict:
    single_root = raw / "dino_patchcore_base"
    mixed_root = raw / "dino_patchcore_mixed"
    check_seeds(single_root, cfg.SEEDS)
    check_seeds(mixed_root, cfg.SUBSET_SEEDS)
    datasets = cfg.DATASETS

    single = {src: {tgt: summarise([auroc(single_root / f"seed{s}" / f"{src}__{tgt}" / "result.json")
                                    for s in cfg.SEEDS]) for tgt in datasets}
              for src in datasets}
    mixed = {}
    for a, b in PAIRS:
        key = f"{a}+{b}"
        mixed[key] = {}
        for tgt in datasets:
            cell = summarise([auroc(mixed_root / f"seed{s}" / f"{key}__{tgt}" / "result.json")
                              for s in cfg.SUBSET_SEEDS])
            best = max(single[a][tgt]["mean"], single[b][tgt]["mean"])
            cell["minus_best_single"] = cell["mean"] - best
            mixed[key][tgt] = cell

    print("\nMixed-source AUROC (rows: reference bank; columns: target)")
    print(f"  {'':16s}" + "".join(f"{LABEL[t]:>16s}" for t in datasets))
    for src in datasets:
        print(f"  {LABEL[src] + ' (single)':16s}"
              + "".join(f"{single[src][t]['mean']:16.3f}" for t in datasets))
    for key, row in mixed.items():
        name = "+".join(LABEL[d] for d in key.split("+"))
        print(f"  {name:16s}"
              + "".join(f"{row[t]['mean']:9.3f}±{row[t]['sd']:.3f}" for t in datasets))
    print("  mixed minus better single source:")
    for key, row in mixed.items():
        name = "+".join(LABEL[d] for d in key.split("+"))
        print(f"  {name:16s}" + "".join(f"{row[t]['minus_best_single']:+16.3f}" for t in datasets))
    return {"single_source_seeds": cfg.SEEDS, "mixed_seeds": cfg.SUBSET_SEEDS,
            "single_source": single, "mixed": mixed}


def fewshot(raw: Path) -> dict:
    root = raw / "dino_patchcore_fewshot"
    if not root.is_dir():
        raise FileNotFoundError(f"missing directory: {root}")
    n_values = sorted(int(m.group(1)) for p in root.iterdir()
                      if (m := re.fullmatch(r"n(\d+)", p.name)))
    if n_values != sorted(N_SHOTS):
        raise RuntimeError(f"{root}: reference-set sizes {n_values}, expected {sorted(N_SHOTS)}")
    datasets = cfg.DATASETS

    curves = {}
    for src in datasets:
        curves[src] = []
        for n in n_values:
            check_seeds(root / f"n{n}", cfg.SUBSET_SEEDS)
            cross, indist = [], []
            for s in cfg.SUBSET_SEEDS:
                cell = root / f"n{n}" / f"seed{s}"
                indist.append(auroc(cell / f"{src}__{src}" / "result.json"))
                cross.append(st.mean(auroc(cell / f"{src}__{t}" / "result.json")
                                     for t in datasets if t != src))
            curves[src].append({"n": n, "cross_dataset": summarise(cross),
                                "in_distribution": summarise(indist)})

    print("\nReference-set size: mean ± SD over seeds")
    print(f"  {'source':8s} {'n':>5s} {'cross-dataset':>16s} {'in-distribution':>16s}")
    for src in datasets:
        for p in curves[src]:
            c, i = p["cross_dataset"], p["in_distribution"]
            print(f"  {LABEL[src]:8s} {p['n']:5d} {c['mean']:9.3f}±{c['sd']:.3f} "
                  f"{i['mean']:9.3f}±{i['sd']:.3f}")
    return {"seeds": cfg.SUBSET_SEEDS, "n_values": n_values, "curves": curves}


def fewshot_figure(fs: dict, figures: Path) -> None:
    n_values = fs["n_values"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, metric, title in zip(axes, ["cross_dataset", "in_distribution"],
                                 ["(a) Cross-dataset AUROC", "(b) In-distribution AUROC"]):
        lows, highs = [0.5], [0.5]
        for src, points in fs["curves"].items():
            mean = [p[metric]["mean"] for p in points]
            sd = [p[metric]["sd"] for p in points]
            lo = [m - e for m, e in zip(mean, sd)]
            hi = [m + e for m, e in zip(mean, sd)]
            ax.plot(n_values, mean, marker="o", linewidth=1.8, markersize=5,
                    color=COLOUR[src], label=LABEL[src])
            ax.fill_between(n_values, lo, hi, color=COLOUR[src], alpha=0.18, linewidth=0)
            lows += lo
            highs += hi
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.0, label="Chance")
        pad = 0.05 * (max(highs) - min(lows))
        ax.set_ylim(min(lows) - pad, max(highs) + pad)
        ax.set_xscale("log")
        ax.set_xticks(n_values)
        ax.set_xticklabels([str(n) for n in n_values], fontsize=10)
        ax.minorticks_off()
        ax.tick_params(axis="y", labelsize=10)
        ax.set_xlabel("Number of normal reference images", fontsize=12)
        ax.set_ylabel("AUROC", fontsize=12)
        ax.set_title(title, fontsize=12)
    # One shared legend below the panels: per-axes legends overlapped the curves.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.06))
    figures.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(figures / f"fewshot_curve.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {figures / 'fewshot_curve.pdf'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", type=Path, default=cfg.RESULTS_DIR)
    args = ap.parse_args()
    raw = args.results_dir / "raw"

    mixed = mixed_source(raw)
    fs = fewshot(raw)
    fewshot_figure(fs, args.results_dir / "figures")
    dest = args.results_dir / "mixed_fewshot.json"
    write_json(dest, {"mixed_source": mixed, "fewshot": fs})
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
