from __future__ import annotations
"""Analyse the controlled contamination experiment (run_contamination.py).

For every (base source B, target T), reports:
    baseline   AUROC( B250        -> T )
    +same      AUROC( B500        -> T )   (count-matched control: add 250 more B)
    +X         AUROC( B250+X250   -> T )   for each foreign source X

and the deltas vs baseline:
    d_same = +same - baseline   (adding GOOD data; expected >= 0)
    d_X    = +X    - baseline   (adding foreign data; < 0 == contamination)

Because the baseline 250 of B is identical across all three banks (per seed), each
delta isolates the effect of WHAT was added, with image count held fixed.

Outputs:
    results/contamination.json                  machine-readable
    results/figures/contamination.pdf / .png    grouped-bar figure
    (prints a LaTeX-ready summary table to stdout)

Usage:
    python analyze_contamination.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg

DATASETS = ["sdnet", "mvtec", "vision"]
LABEL    = {"sdnet": "SDNET", "mvtec": "MVTec", "vision": "VISION"}
RAW      = cfg.RESULTS_DIR / "raw" / "dino_patchcore_contamination"
FIG      = cfg.RESULTS_DIR / "figures"


def load(cfg_name: str, tgt: str):
    """Mean, std, n over seeds for one (config, target)."""
    vals = []
    for d in sorted(RAW.glob(f"{cfg_name}/seed*/{tgt}/result.json")):
        vals.append(json.load(open(d))["image_auroc"])
    if not vals:
        return None
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def main() -> None:
    rows = []   # (base, target, baseline, +same, {X: +X})
    summary = {}
    for B in DATASETS:
        for T in DATASETS:
            base = load(f"{B}250", T)
            same = load(f"{B}500", T)
            adds = {X: load(f"{B}250+{X}250", T) for X in DATASETS if X != B}
            if base is None:
                continue
            rows.append((B, T, base, same, adds))
            summary[f"{B}->{T}"] = {
                "baseline": base, "plus_same": same,
                **{f"plus_{X}": v for X, v in adds.items()},
            }

    # ── console table ──────────────────────────────────────────────────────────
    print(f"\n{'base->target':16}{'base250':>10}{'+250same':>11}"
          f"{'+250(otherGood)':>17}{'+250VISION':>13}")
    print("-" * 67)
    for B, T, base, same, adds in rows:
        other = [X for X in adds if X != "vision"]
        og = adds.get(other[0]) if other else None
        vis = adds.get("vision")
        def cell(x): return f"{x[0]:.3f}" if x else "  -- "
        def dcell(x): return f"({x[0]-base[0]:+.3f})" if x else ""
        og_s  = f"{cell(og)}{dcell(og)}" if "vision" != "" else ""
        print(f"{LABEL[B]}->{LABEL[T]:<8}{cell(base):>10}"
              f"{cell(same):>8}{dcell(same):>9}"
              f"{(cell(og)+dcell(og)) if og else '--':>17}"
              f"{(cell(vis)+dcell(vis)) if vis else '--':>13}")

    FIG.mkdir(parents=True, exist_ok=True)
    (cfg.RESULTS_DIR / "contamination.json").write_text(json.dumps(summary, indent=2))

    # ── figure: count-matched control, cross-domain cases ──────────────────────
    # For each non-VISION base and a target != base, compare three banks at the
    # same image budget: 250 good (base), +250 MORE good, +250 VISION. Both
    # additions are NON-target data, so the only difference is what was added.
    cases = [("mvtec", "sdnet"), ("mvtec", "vision"),
             ("sdnet", "mvtec"), ("sdnet", "vision")]
    labels = [f"{LABEL[b]}→{LABEL[t]}" for b, t in cases]
    base_v, base_e, same_v, same_e, vis_v, vis_e = [], [], [], [], [], []
    for b, t in cases:
        B = load(f"{b}250", t); S = load(f"{b}500", t); V = load(f"{b}250+vision250", t)
        base_v.append(B[0]); base_e.append(B[1])
        same_v.append(S[0]); same_e.append(S[1])
        vis_v.append(V[0]);  vis_e.append(V[1])

    x = np.arange(len(cases)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.bar(x - w, base_v, w, yerr=base_e, capsize=3, color="#b0b0b0", label="base source (250)")
    ax.bar(x,     same_v, w, yerr=same_e, capsize=3, color="#4c78a8", label="+250 same-source")
    ax.bar(x + w, vis_v,  w, yerr=vis_e,  capsize=3, color="#d65f5f", label="+250 VISION")
    ax.axhline(0.5, color="black", ls=":", lw=1.0)
    ax.text(len(cases) - 0.5, 0.505, "chance", fontsize=8, ha="right", va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("AUROC", fontsize=11)
    ax.set_ylim(0.45, 0.70)
    ax.set_title("Matched image budget: extra same-source vs. VISION references",
                 fontsize=11)
    ax.legend(fontsize=8, ncol=3, loc="upper center")
    fig.tight_layout()
    fig.savefig(FIG / "contamination.pdf", bbox_inches="tight")
    fig.savefig(FIG / "contamination.png", dpi=140, bbox_inches="tight")
    print(f"\n-> wrote {FIG/'contamination.pdf'} and contamination.json")


if __name__ == "__main__":
    main()
