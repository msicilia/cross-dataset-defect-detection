from __future__ import annotations
"""Analyze the multi-seed, mixed-source and few-shot results.

Produces the corresponding figures and tables.

Outputs:
  results/figures/multiseed_table.tex   — cross-dataset table with mean±std
  results/figures/mixed_source_table.tex — mixed-source AUROC table
  results/figures/fewshot_curve.pdf     — few-shot AUROC vs n_train figure
  results/experiments.json              — machine-readable summary
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = Path(__file__).parent
RAW     = ROOT / "results" / "raw"
FIGURES = ROOT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

DATASETS  = ["sdnet", "mvtec", "vision"]
DS_LABELS = {"sdnet": "SDNET", "mvtec": "MVTec", "vision": "VISION"}
N_SHOTS   = [5, 10, 25, 50, 100, 250, 500]
SEEDS     = [0, 1, 2, 3, 4]


# ── helpers ────────────────────────────────────────────────────────────────────

def load_auroc(path: Path) -> float | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f).get("image_auroc")


# ── 1. Multi-seed cross-dataset table ─────────────────────────────────────────

def analyze_multiseed():
    print("\n=== Multi-seed cross-dataset ===")
    method = "dino_patchcore_base"

    # Collect all seeds for each (src, tgt) pair
    matrix = {}  # (src, tgt) -> list of auroc values
    for src in DATASETS:
        for tgt in DATASETS:
            vals = []
            for seed in SEEDS:
                p = RAW / method / f"seed{seed}" / f"{src}__{tgt}" / "result.json"
                v = load_auroc(p)
                if v is not None:
                    vals.append(v)
            matrix[(src, tgt)] = vals
            label = f"{DS_LABELS[src]}→{DS_LABELS[tgt]}"
            if vals:
                print(f"  {label}: n={len(vals)}  "
                      f"mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")
            else:
                print(f"  {label}: no data")

    # Write LaTeX table
    lines = []
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"  \toprule")
    lines.append(r"  \textbf{Source} & \textbf{SDNET} & \textbf{MVTec} & \textbf{VISION} \\")
    lines.append(r"  \midrule")
    for src in DATASETS:
        row_cells = [DS_LABELS[src]]
        for tgt in DATASETS:
            vals = matrix[(src, tgt)]
            if not vals:
                row_cells.append("--")
            elif len(vals) == 1:
                row_cells.append(f"{vals[0]:.3f}")
            else:
                m, s = np.mean(vals), np.std(vals)
                cell = f"${m:.3f}\\pm{s:.3f}$"
                if src == tgt:
                    cell = r"\textbf{" + cell + "}"
                row_cells.append(cell)
        lines.append("  " + " & ".join(row_cells) + r" \\")
    lines.append(r"  \bottomrule")
    lines.append(r"\end{tabular}")

    out = FIGURES / "multiseed_table.tex"
    out.write_text("\n".join(lines))
    print(f"  → wrote {out}")
    return matrix


# ── 2. Mixed-source table ─────────────────────────────────────────────────────

def analyze_mixed():
    print("\n=== Mixed-source experiment ===")
    mixed_root = RAW / "dino_patchcore_mixed"
    pairs = [("sdnet", "mvtec"), ("sdnet", "vision"), ("mvtec", "vision")]

    results = {}  # (pair_key, tgt) -> list of auroc
    for src1, src2 in pairs:
        key = f"{src1}+{src2}"
        for tgt in DATASETS:
            vals = []
            for seed in [0, 1, 2]:
                p = mixed_root / f"seed{seed}" / f"{key}__{tgt}" / "result.json"
                v = load_auroc(p)
                if v is not None:
                    vals.append(v)
            results[(key, tgt)] = vals
            if vals:
                print(f"  {key}→{DS_LABELS[tgt]}: "
                      f"mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  n={len(vals)}")

    # Write LaTeX table
    lines = []
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"  \toprule")
    lines.append(r"  \textbf{Source combination} & \textbf{SDNET} & \textbf{MVTec} & \textbf{VISION} \\")
    lines.append(r"  \midrule")
    # Single-source baselines from seed0
    for src in DATASETS:
        row = [DS_LABELS[src] + " (single)"]
        for tgt in DATASETS:
            p = RAW / "dino_patchcore_base" / "seed0" / f"{src}__{tgt}" / "result.json"
            v = load_auroc(p)
            cell = f"{v:.3f}" if v is not None else "--"
            if src == tgt:
                cell = r"\textbf{" + cell + "}"
            row.append(cell)
        lines.append("  " + " & ".join(row) + r" \\")
    lines.append(r"  \midrule")
    for src1, src2 in pairs:
        key = f"{src1}+{src2}"
        label = f"{DS_LABELS[src1]}+{DS_LABELS[src2]}"
        row = [label]
        for tgt in DATASETS:
            vals = results[(key, tgt)]
            if not vals:
                row.append("--")
            elif len(vals) == 1:
                row.append(f"{vals[0]:.3f}")
            else:
                m, s = np.mean(vals), np.std(vals)
                row.append(f"${m:.3f}\\pm{s:.3f}$")
        lines.append("  " + " & ".join(row) + r" \\")
    lines.append(r"  \bottomrule")
    lines.append(r"\end{tabular}")

    out = FIGURES / "mixed_source_table.tex"
    out.write_text("\n".join(lines))
    print(f"  → wrote {out}")
    return results


# ── 3. Few-shot curve ──────────────────────────────────────────────────────────

def analyze_fewshot():
    print("\n=== Few-shot curve ===")
    fs_root = RAW / "dino_patchcore_fewshot"

    # Collect mean cross-dataset AUROC (off-diagonal) per (src, n)
    # and in-distribution (diagonal) per (src, n)
    curve_cross = {}   # (src, n) -> mean cross AUROC
    curve_indist = {}  # (src, n) -> in-dist AUROC

    for src in DATASETS:
        for n in N_SHOTS:
            cross_vals, diag_vals = [], []
            for seed in [0, 1, 2]:
                for tgt in DATASETS:
                    p = fs_root / f"n{n}" / f"seed{seed}" / f"{src}__{tgt}" / "result.json"
                    v = load_auroc(p)
                    if v is None:
                        continue
                    if src == tgt:
                        diag_vals.append(v)
                    else:
                        cross_vals.append(v)
            curve_cross[(src, n)]  = np.mean(cross_vals)  if cross_vals  else np.nan
            curve_indist[(src, n)] = np.mean(diag_vals)   if diag_vals   else np.nan
            if cross_vals:
                print(f"  {DS_LABELS[src]}  n={n:4d}:  "
                      f"cross={curve_cross[(src,n)]:.4f}  "
                      f"indist={curve_indist[(src,n)]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    colors = {"sdnet": "#1f77b4", "mvtec": "#ff7f0e", "vision": "#2ca02c"}

    for ax, metric, title in zip(
        axes,
        [curve_cross, curve_indist],
        ["(a) Cross-dataset AUROC", "(b) In-distribution AUROC"]
    ):
        for src in DATASETS:
            ns   = [n for n in N_SHOTS if not np.isnan(metric.get((src, n), np.nan))]
            vals = [metric[(src, n)] for n in ns]
            if ns:
                ax.plot(ns, vals, marker="o", linewidth=1.8, markersize=5,
                        color=colors[src], label=DS_LABELS[src])
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.0, label="Chance")
        ax.set_xscale("log")
        ax.set_xlabel("Number of normal reference images", fontsize=12)
        ax.set_ylabel("AUROC", fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=11)
        ax.set_xticks(N_SHOTS)
        ax.set_xticklabels([str(n) for n in N_SHOTS], fontsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_ylim(0.40, 1.0)

    plt.tight_layout()
    out = FIGURES / "fewshot_curve.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  → wrote {out}")
    plt.close()
    return curve_cross, curve_indist


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ms   = analyze_multiseed()
    mix  = analyze_mixed()
    fs_c, fs_i = analyze_fewshot()

    # Save machine-readable summary
    summary = {
        "multiseed": {
            f"{s}__{t}": {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
            for (s, t), v in ms.items() if v
        },
        "fewshot_cross": {
            f"{s}_{n}": float(v)
            for (s, n), v in fs_c.items() if not np.isnan(v)
        },
    }
    out = ROOT / "results" / "experiments.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n→ summary written to {out}")
