from __future__ import annotations
"""Figures for the contamination-proportion and paradigm analyses.

  proportions.pdf   AUROC against the fraction of a fixed-size bank drawn
                    from VISION. The bank is always 500 images, so the x-axis is
                    composition, not size -- which is what lets the shape of the
                    curve distinguish dilution from interference.
  paradigm.pdf      generalisation gap by how each detector models normality,
                    grouped as reference-dependent vs reference-free.

Usage:  python make_figures.py
"""
import glob
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as cfg

FIGS = cfg.RESULTS_DIR / "figures"
DATASETS = ["sdnet", "mvtec", "vision"]
FR = [0, 25, 50, 75, 100]


def proportions() -> None:
    root = cfg.RESULTS_DIR / "raw" / "dino_patchcore_proportions"
    if not root.exists():
        print("  no proportion results"); return
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharey=True)
    for ax, base in zip(axes, ["mvtec", "sdnet"]):
        for tgt in DATASETS:
            m, e = [], []
            for f in FR:
                v = [json.load(open(x))["image_auroc"]
                     for x in glob.glob(f"{root}/{base}_vision{f:03d}/seed*/{tgt}/result.json")]
                m.append(st.mean(v) if v else np.nan)
                e.append(st.stdev(v) if len(v) > 1 else 0.0)
            ax.errorbar(FR, m, yerr=e, marker="o", ms=4, capsize=2, label=tgt.upper())
        ax.axhline(0.5, ls=":", c="grey", lw=1)
        ax.set_title(f"base: {base.upper()}", fontsize=10)
        ax.set_xlabel("% of the 500-image bank drawn from VISION")
        ax.set_xticks(FR)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("AUROC")
    axes[1].legend(frameon=False, fontsize=8, title="target", title_fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"proportions.{ext}", dpi=200)
    print(f"  wrote {FIGS/'proportions.pdf'}")


def paradigm() -> None:
    root = cfg.RESULTS_DIR / "raw"
    spec = [
        ("PatchCore", "patchcore", "ref"),
        ("DINO-PatchCore", "dino_patchcore_base", "ref"),
        ("PaDiM", "padim", "ref"),
        ("SPADE", "spade", "ref"),
        ("WinCLIP+", "winclip_plus", "ref"),
        ("WinCLIP", "winclip", "free"),
        ("CLIP-ZS", "clip_zs", "free"),
    ]
    names, gaps, kinds = [], [], []
    for label, m, kind in spec:
        d = root / m
        if not d.exists():
            continue
        seeds = sorted(int(p.name[4:]) for p in d.glob("seed*"))
        if (d / f"seed{seeds[0]}" / "none__mvtec").exists():      # reference-free
            gaps.append(0.0)
        else:
            def cell(s, t):
                v = [json.load(open(d / f"seed{sd}" / f"{s}__{t}" / "result.json"))["image_auroc"]
                     for sd in seeds if (d / f"seed{sd}" / f"{s}__{t}" / "result.json").exists()]
                return st.mean(v) if v else None
            ind = [cell(x, x) for x in DATASETS]
            cro = [cell(x, y) for x in DATASETS for y in DATASETS if x != y]
            if any(v is None for v in ind + cro):
                continue
            gaps.append(st.mean(ind) - st.mean(cro))
        names.append(label); kinds.append(kind)

    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    colours = ["#c44e52" if k == "ref" else "#4c72b0" for k in kinds]
    ax.barh(range(len(names)), gaps, color=colours)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(r"generalisation gap $\Delta$")
    for i, g in enumerate(gaps):
        ax.text(g + 0.004, i, f"{g:.3f}", va="center", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#c44e52"),
               plt.Rectangle((0, 0), 1, 1, color="#4c72b0")]
    ax.legend(handles, ["uses source-domain references", "reference-free"],
              frameon=False, fontsize=8, loc="lower right")
    ax.set_xlim(0, max(gaps) * 1.25 if gaps else 1)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"paradigm.{ext}", dpi=200)
    print(f"  wrote {FIGS/'paradigm.pdf'}")


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    proportions()
    paradigm()
