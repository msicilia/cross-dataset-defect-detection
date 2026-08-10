from __future__ import annotations
"""Why do some transfers land *below* chance?

Below-chance AUROC is not "bad detection" -- it is systematically *inverted*
detection: defect-free images receive higher anomaly scores than defective ones.
Noise alone produces 0.5, so an entry at 0.34 has to be explained.

Note what is NOT evidence here: AUROC(-s) = 1 - AUROC(s) identically, so
"flipping the sign recovers above-chance performance" is a tautology and says
nothing. The question is why the ordering inverts, which needs the score
distributions themselves.

Reported per cell:

  d           standardised mean difference between defective and normal scores
              (Cohen's d). Positive = defective score higher, as intended.
              Negative = the ranking is inverted, not merely uninformative.
  sep         overlap of the two distributions
  norm_shift  how far the whole score distribution moves between the
              in-distribution and cross-domain case, in in-distribution
              standard deviations. A large shift with a collapsed |d| means the
              detector is responding to domain identity rather than to defects.

Read alongside run_localisation.py: cross-domain, the peak-scoring patch lands
on the defect only 6-9 % of the time (vs 39 % in-distribution). The image score
*is* that peak, so it is being set by image content unrelated to the defect --
which is how an ordering can invert rather than merely degrade.

Usage:  python analyze_inversion.py
"""
import json
import statistics as st
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path("results/raw")
DATASETS = ["sdnet", "mvtec", "vision"]


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of the pooled range where both distributions have mass."""
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    bins = np.linspace(lo, hi, 50)
    ha, _ = np.histogram(a, bins=bins, density=True)
    hb, _ = np.histogram(b, bins=bins, density=True)
    return float(np.minimum(ha, hb).sum() * (bins[1] - bins[0]))


def load(method: str, src: str, tgt: str, seed: int):
    p = ROOT / method / f"seed{seed}" / f"{src}__{tgt}" / "scores.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    return z["scores"], z["labels"]


def main() -> None:
    methods = [m for m in ("patchcore", "dino_patchcore_base", "padim", "spade",
                           "winclip_plus")
               if (ROOT / m).exists()]
    print(f"  {'method':20s} {'src->tgt':16s} {'AUROC':>7s} {'d':>7s} "
          f"{'overlap':>8s} {'norm shift':>11s}")
    print("  " + "-" * 76)
    inverted = []
    for method in methods:
        seeds = sorted(int(p.name[4:]) for p in (ROOT / method).glob("seed*"))
        # In-distribution reference statistics per target, for the shift measure.
        ref = {}
        for t in DATASETS:
            got = [load(method, t, t, s) for s in seeds]
            got = [g for g in got if g]
            if got:
                allsc = np.concatenate([g[0] for g in got])
                ref[t] = (allsc.mean(), allsc.std())
        for src in DATASETS:
            for tgt in DATASETS:
                got = [load(method, src, tgt, s) for s in seeds]
                got = [g for g in got if g]
                if not got:
                    continue
                aurocs, ds, ovs, shifts = [], [], [], []
                for s, y in got:
                    if len(np.unique(y)) < 2:
                        continue
                    aurocs.append(roc_auc_score(y, s))
                    ds.append(cohens_d(s[y == 1], s[y == 0]))
                    ovs.append(overlap(s[y == 1], s[y == 0]))
                    if tgt in ref and ref[tgt][1] > 0:
                        shifts.append(abs(s.mean() - ref[tgt][0]) / ref[tgt][1])
                if not aurocs:
                    continue
                a, d = st.mean(aurocs), st.mean(ds)
                mark = "  <-- INVERTED" if a < 0.485 else ""
                print(f"  {method:20s} {src+'->'+tgt:16s} {a:7.3f} {d:7.3f} "
                      f"{st.mean(ovs):8.3f} {(st.mean(shifts) if shifts else float('nan')):11.2f}{mark}")
                if a < 0.485:
                    inverted.append((method, src, tgt, a, d))

    print()
    if inverted:
        print(f"  {len(inverted)} inverted cells (AUROC < 0.485):")
        neg_d = [x for x in inverted if x[4] < 0]
        print(f"    of these, {len(neg_d)} have negative Cohen's d, i.e. defect-free "
              f"images score HIGHER than defective ones.")
        print("    This is a systematic reversal of the score ordering, not an "
              "absence of signal;")
        print("    a detector with no information would sit at d=0 and AUROC=0.5.")
    else:
        print("  no inverted cells found")


if __name__ == "__main__":
    main()
