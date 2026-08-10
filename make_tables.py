"""Read results from results/raw/ and generate:

  1. results/tables.json            — machine-readable summary
  2. results/latex_tables.tex       — ready-to-paste LaTeX tables for the paper
  3. results/figures/               — AUROC heatmaps and bar charts

Usage:
    python make_tables.py [--seeds S [S ...]] [--outdir DIR]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import sys
sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

# ── dataset display labels ────────────────────────────────────────────────────
DS_LABELS = {
    "mvtec":  "MVTec AD",
    "sdnet":  "SDNET",
    "vision": "VISION",
}

METHOD_LABELS = {
    "dino_patchcore_small": r"DINO-PC (S/14)",
    "dino_patchcore_base":  r"DINO-PC (B/14)",
    "dino_patchcore_large": r"DINO-PC (L/14)",
    "patchcore":            r"PatchCore",
    "spade":                r"SPADE",
    "padim":                r"PaDiM",
    "winclip":              r"WinCLIP",
    "winclip_plus":         r"WinCLIP+",
    "clip_zs":              r"CLIP-ZS",
}

# ── helpers ───────────────────────────────────────────────────────────────────

def load_raw(results_root: Path, method: str, seeds: list[int]) -> dict[str, dict]:
    """Load results for one method, averaging over seeds.
    Returns dict keyed by 'src__tgt' with averaged EvalResult fields."""
    aggregated: dict[str, dict[str, list]] = {}
    for seed in seeds:
        seed_dir = results_root / method / f"seed{seed}"
        if not seed_dir.exists():
            continue
        for pair_dir in seed_dir.iterdir():
            key = pair_dir.name
            rp  = pair_dir / "result.json"
            if not rp.exists():
                continue
            with open(rp) as f:
                d = json.load(f)
            if key not in aggregated:
                aggregated[key] = {k: [] for k in d}
            for k, v in d.items():
                if v is not None:
                    aggregated[key][k].append(v)

    return {
        key: {k: float(np.mean(vs)) if vs else None for k, vs in vals.items()}
        for key, vals in aggregated.items()
    }


def auroc_matrix(data: dict[str, dict], datasets: list[str]) -> np.ndarray:
    """Build a len(datasets) × len(datasets) AUROC matrix.

    For dataset-agnostic methods (results keyed as 'none__<tgt>'), the same
    target score is broadcast across all source rows so that gap = 0 correctly
    reflects dataset-independence.
    """
    n = len(datasets)
    mat = np.full((n, n), np.nan)
    for i, src in enumerate(datasets):
        for j, tgt in enumerate(datasets):
            for key in (f"{src}__{tgt}", f"none__{tgt}"):
                if key in data and data[key]["image_auroc"] is not None:
                    mat[i, j] = data[key]["image_auroc"]
                    break
    return mat


def gap(mat: np.ndarray) -> float:
    """Generalisation gap; returns NaN when there are no off-diagonal elements."""
    diag = np.diag(mat)
    mask = ~np.eye(len(mat), dtype=bool)
    off  = mat[mask]
    if off.size == 0 or np.all(np.isnan(off)):
        return float("nan")
    return float(np.nanmean(diag) - np.nanmean(off))


# ── LaTeX formatting ──────────────────────────────────────────────────────────

def fmt(v) -> str:
    if v is None or np.isnan(v):
        return "--"
    return f"{v:.3f}"


def make_cross_dataset_table(
    all_data: dict[str, dict[str, dict]],
    datasets: list[str],
    methods: list[str],
    metric: str = "image_auroc",
) -> str:
    n = len(datasets)
    col_spec = "ll" + "c" * n
    ds_heads  = " & ".join(f"\\textbf{{{DS_LABELS.get(d, d)}}}" for d in datasets)
    lines = [
        r"\begin{table*}[!t]",
        r"  \centering",
        r"  \caption{Cross-Dataset " + metric.replace("_", " ").upper() +
        r". Rows: source (fit) dataset; Columns: target (eval) dataset.}",
        r"  \label{tab:cross_" + metric + r"}",
        r"  \setlength{\tabcolsep}{4pt}",
        rf"  \begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        rf"    \textbf{{Method}} & \textbf{{Source $\backslash$ Target}} & {ds_heads} \\",
        r"    \midrule",
    ]

    for m_idx, method in enumerate(methods):
        data = all_data.get(method, {})
        mat  = auroc_matrix(data, datasets)
        mlabel = METHOD_LABELS.get(method, method)
        first = True
        n_rows = n if method != "clip_zs" else 1  # clip_zs has no source dependency
        src_range = datasets if method != "clip_zs" else ["none"]

        for src in src_range:
            row_vals = []
            for tgt in datasets:
                key = f"{src}__{tgt}"
                v   = data.get(key, {}).get(metric)
                row_vals.append(fmt(v))
            src_label = DS_LABELS.get(src, src) if src != "none" else r"\textit{n/a}"
            method_cell = rf"\multirow{{{n_rows}}}{{*}}{{{mlabel}}}" if first else ""
            lines.append(
                f"    {method_cell} & {src_label} & " + " & ".join(row_vals) + r" \\"
            )
            first = False

        lines.append(r"    \midrule")

    lines += [
        r"  \end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def make_gap_table(
    all_data: dict[str, dict[str, dict]],
    datasets: list[str],
    methods: list[str],
) -> str:
    lines = [
        r"\begin{table}[!t]",
        r"  \centering",
        r"  \caption{Generalisation Gap $\Delta$ (in-dist AUROC $-$ mean cross-dataset AUROC). Lower is better.}",
        r"  \label{tab:gap}",
        r"  \begin{tabular}{lcc}",
        r"    \toprule",
        r"    \textbf{Method} & \textbf{In-dist AUROC} & $\boldsymbol{\Delta}$ \\",
        r"    \midrule",
    ]
    for method in methods:
        data = all_data.get(method, {})
        mat  = auroc_matrix(data, datasets)
        diag_mean = float(np.nanmean(np.diag(mat)))
        g         = gap(mat)
        lines.append(
            f"    {METHOD_LABELS.get(method, method)} & {fmt(diag_mean)} & {fmt(g)} \\\\"
        )
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def make_backbone_ablation_table(
    all_data: dict[str, dict[str, dict]],
    datasets: list[str],
) -> str:
    variants = ["dino_patchcore_small", "dino_patchcore_base", "dino_patchcore_large"]
    labels   = ["ViT-S/14", "ViT-B/14", "ViT-L/14"]
    lines = [
        r"\begin{table}[!t]",
        r"  \centering",
        r"  \caption{Backbone Ablation: DINOv2 size vs.\ cross-dataset AUROC.}",
        r"  \label{tab:ablation_backbone}",
        r"  \begin{tabular}{lcc}",
        r"    \toprule",
        r"    \textbf{Backbone} & \textbf{In-dist} & \textbf{Cross-dataset} \\",
        r"    \midrule",
    ]
    for var, lbl in zip(variants, labels):
        data = all_data.get(var, {})
        mat  = auroc_matrix(data, datasets)
        off  = mat[~np.eye(len(mat), dtype=bool)]
        diag    = np.diag(mat)
        indist  = fmt(np.nanmean(diag) if not np.all(np.isnan(diag)) else float("nan"))
        cross   = fmt(np.nanmean(off) if off.size and not np.all(np.isnan(off)) else float("nan"))
        lines.append(f"    {lbl} & {indist} & {cross} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ── figures ───────────────────────────────────────────────────────────────────

def plot_heatmap(mat: np.ndarray, datasets: list[str], title: str, out_path: Path) -> None:
    labels = [DS_LABELS.get(d, d) for d in datasets]
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(
        mat, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0.5, vmax=1.0,
        xticklabels=labels, yticklabels=labels, ax=ax,
        linewidths=0.5, linecolor="grey",
        annot_kws={"size": 11, "weight": "bold"},
    )
    ax.set_xlabel("Target", fontsize=12)
    ax.set_ylabel("Source", fontsize=12)
    ax.set_title(title, fontsize=11, pad=6)
    ax.tick_params(axis="both", labelsize=11)
    ax.collections[0].colorbar.ax.tick_params(labelsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main(seeds: list[int], outdir: Path, methods: list[str], datasets: list[str]) -> None:
    results_root = outdir / "raw"
    fig_dir      = outdir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    all_data: dict[str, dict] = {}
    for method in methods:
        all_data[method] = load_raw(results_root, method, seeds)

    # ── tables ────────────────────────────────────────────────────────────────
    latex_path = outdir / "latex_tables.tex"
    with open(latex_path, "w") as f:
        f.write("% ── Cross-dataset AUROC ───────────────────────────────────\n")
        f.write(make_cross_dataset_table(all_data, datasets, methods, "image_auroc"))
        f.write("\n\n% ── Generalisation gap ────────────────────────────────\n")
        f.write(make_gap_table(all_data, datasets, methods))
        f.write("\n\n% ── Backbone ablation ─────────────────────────────────\n")
        f.write(make_backbone_ablation_table(all_data, datasets))
    print(f"LaTeX tables written to {latex_path}")

    # ── figures ───────────────────────────────────────────────────────────────
    for method in methods:
        mat = auroc_matrix(all_data.get(method, {}), datasets)
        if not np.all(np.isnan(mat)):
            plot_heatmap(
                mat, datasets,
                title=f"{METHOD_LABELS.get(method, method)} — cross-dataset AUROC",
                out_path=fig_dir / f"heatmap_{method}.pdf",
            )

    print(f"Figures written to {fig_dir}")

    # ── machine-readable summary ──────────────────────────────────────────────
    def _safe(v: float) -> "float | None":
        return None if (v is None or np.isnan(v)) else v

    summary: dict = {}
    for method in methods:
        mat = auroc_matrix(all_data.get(method, {}), datasets)
        off = mat[~np.eye(len(mat), dtype=bool)]
        summary[method] = {
            "matrix": [[_safe(x) for x in row] for row in mat.tolist()],
            "in_dist_auroc":      _safe(float(np.nanmean(np.diag(mat)))),
            "cross_dataset_auroc": _safe(float(np.nanmean(off))) if off.size and not np.all(np.isnan(off)) else None,
            "gap":                 _safe(gap(mat)),
        }

    with open(outdir / "tables.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {outdir / 'tables.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # Defaults reproduce the paper figures/tables: 5-seed means, the three
    # evaluated methods, and the three datasets in the same order as the tables.
    p.add_argument("--seeds",    nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--outdir",   type=Path, default=cfg.RESULTS_DIR)
    p.add_argument("--methods",  nargs="+",
                   default=["dino_patchcore_base", "patchcore", "clip_zs"])
    p.add_argument("--datasets", nargs="+", default=["sdnet", "mvtec", "vision"])
    args = p.parse_args()
    main(args.seeds, args.outdir, args.methods, args.datasets)
