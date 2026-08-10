from __future__ import annotations
"""Is the generalisation gap a property of the memory-bank paradigm?

Two detectors that differ only in backbone cannot separate a paradigm-level
effect from a representation-level one. This compares detectors that differ in
how normality is *modelled*, holding the backbone fixed at WideResNet-50
wherever possible:

    patchcore  memory bank, coreset of patches, NN distance    (non-parametric)
    spade      memory bank, whole-image descriptors, kNN       (non-parametric)
    padim      per-position Gaussian, Mahalanobis distance     (parametric)
    dino_patchcore_base                                        (PatchCore, different backbone)
    clip_zs    language-aligned, no reference images

Reading: if padim's gap matches the memory-bank methods', the bottleneck is
cross-domain transfer of unsupervised anomaly detection generally, rather than
the memory-bank paradigm. If padim's gap is clearly smaller, the memory-bank
reading is supported.

Usage:  python analyze_paradigm.py
"""
import json
import statistics
from pathlib import Path

DATASETS = ["sdnet", "mvtec", "vision"]
SEEDS = [0, 1, 2, 3, 4]
FAMILY = {
    "patchcore": "memory bank (patch coreset)",
    "spade": "memory bank (image kNN)",
    "dino_patchcore_base": "memory bank (patch coreset, DINOv2)",
    "padim": "parametric Gaussian",
    "clip_zs": "reference-free (language)",
}
ROOT = Path("results/raw")


def mean_cell(method: str, src: str, tgt: str) -> float | None:
    seeds = [0] if method == "clip_zs" else SEEDS
    vals = []
    for s in seeds:
        p = ROOT / method / f"seed{s}" / f"{src}__{tgt}" / "result.json"
        if p.exists():
            vals.append(json.load(open(p))["image_auroc"])
    return statistics.mean(vals) if vals else None


def main() -> None:
    print(f"{'method':22s} {'family':38s} {'in-dist':>8s} {'cross':>8s} {'delta':>8s}")
    print("-" * 88)
    rows = []
    for m, fam in FAMILY.items():
        if m == "clip_zs":
            vals = [mean_cell(m, "none", t) for t in DATASETS]
            if any(v is None for v in vals):
                continue
            i = statistics.mean(vals)
            print(f"{m:22s} {fam:38s} {i:8.3f} {i:8.3f} {0.0:8.3f}")
            rows.append((m, fam, i, i, 0.0))
            continue
        ind = [mean_cell(m, d, d) for d in DATASETS]
        cro = [mean_cell(m, s, t) for s in DATASETS for t in DATASETS if s != t]
        if any(v is None for v in ind + cro):
            print(f"{m:22s} {fam:38s}   (incomplete)")
            continue
        i, c = statistics.mean(ind), statistics.mean(cro)
        print(f"{m:22s} {fam:38s} {i:8.3f} {c:8.3f} {i - c:8.3f}")
        rows.append((m, fam, i, c, i - c))

    # A raw gap is not comparable across detectors of different strength: a
    # method that only reaches 0.60 in-distribution has less room to fall than
    # one reaching 0.73. Report the gap as a fraction of the available headroom
    # above chance, (in-dist - 0.5), alongside the raw value.
    print(f"\n{'method':22s} {'delta':>8s} {'headroom':>9s} {'delta/headroom':>15s}")
    print("-" * 88)
    for m, _, i, _, d in rows:
        head = i - 0.5
        norm = d / head if head > 0.05 else float("nan")
        print(f"{m:22s} {d:8.3f} {head:9.3f} {norm:15.2f}")

    # The claim under test concerns coreset-based memory banks, so those are the
    # reference. SPADE is reported alongside but not used to widen the range:
    # its in-distribution accuracy is much lower, so its gap is not on the same
    # footing, and letting it stretch the interval would make "inside" trivial.
    ref = [r for r in rows if r[0] in ("patchcore", "dino_patchcore_base")]
    par = [r for r in rows if r[1].startswith("parametric")]
    if ref and par:
        lo, hi = min(r[4] for r in ref), max(r[4] for r in ref)
        print(f"\n  coreset memory-bank gaps span {lo:.3f}-{hi:.3f} "
              f"(patchcore, dino_patchcore_base)")
        for m, _, i, _, d in par:
            print(f"  {m}: gap = {d:.3f}, in-dist = {i:.3f}")
            margin = 0.02   # a difference below this is not meaningful here
            if lo - margin <= d <= hi + margin:
                print("  -> COMPARABLE to the coreset methods. The gap is then NOT specific\n"
                      "     to the memory-bank paradigm, but a property of cross-domain\n"
                      "     transfer of unsupervised anomaly detection generally.")
            elif d < lo - margin:
                print("  -> CLEARLY SMALLER. The memory-bank reading is supported: storing\n"
                      "     exemplars does carry a larger gap than modelling a distribution.")
            else:
                print("  -> CLEARLY LARGER. Memory banks transfer better than the parametric\n"
                      "     alternative, the opposite of the memory-bank reading.")
        for m, _, i, _, d in [r for r in rows if r[0] == "spade"]:
            print(f"\n  spade: gap = {d:.3f}, in-dist = {i:.3f} -- reported for completeness.\n"
                  "  Interpret with care: its in-distribution accuracy is well below the\n"
                  "  coreset methods', so its gap is not directly comparable.")


if __name__ == "__main__":
    main()
