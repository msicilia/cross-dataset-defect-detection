from __future__ import annotations
"""Consistency audit over every result in the tree.

Run before deriving any reported number. Four independent checks:

  1. INTEGRITY   every scores.npz reproduces the result.json beside it
  2. COMPLETENESS every experiment set has the cell count it should
  3. SANITY      no AUROC outside [0,1], no NaN, no empty score vector
  4. PROVENANCE  no result file predates the code that produced it, which is
                 how a silently inconsistent set would look

Usage:  python check_consistency.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path("results/raw")
CODE = Path(".")

EXPECTED = {
    "patchcore": 45, "dino_patchcore_base": 45, "dino_patchcore_small": 45,
    "dino_patchcore_large": 45, "spade": 45, "padim": 45, "clip_zs": 3,
    "dino_patchcore_mvtecsplit": 80, "dino_patchcore_mixed": 27,
    "dino_patchcore_contamination": 180, "dino_patchcore_fewshot": 189,
    "dino_patchcore_proportions": 150, "dino_patchcore_confounders": 36,
}
# Which source file governs which result set, for the provenance check.
GOVERNS = {
    "patchcore": "methods/patchcore.py", "spade": "methods/spade.py",
    "padim": "methods/padim.py",
    **{k: "methods/dino_patchcore.py" for k in
       ("dino_patchcore_base", "dino_patchcore_small", "dino_patchcore_large",
        "dino_patchcore_mvtecsplit", "dino_patchcore_mixed",
        "dino_patchcore_contamination", "dino_patchcore_fewshot",
        "dino_patchcore_proportions", "dino_patchcore_confounders")},
    "clip_zs": "methods/clip_zs.py",
}

_exc_file = Path("provenance_exceptions.json")
EXCEPTIONS = ({e["set"]: e for e in json.load(open(_exc_file))["exceptions"]}
              if _exc_file.exists() else {})

fails: list[str] = []


def main() -> None:
    print("1. INTEGRITY — scores.npz reproduces result.json")
    n_ok = n_skip = 0
    for npz in sorted(ROOT.rglob("scores.npz")):
        rj = npz.parent / "result.json"
        if not rj.exists():
            fails.append(f"scores without result: {npz.parent}"); continue
        try:
            d = np.load(npz, allow_pickle=False)
            saved = json.load(open(rj))["image_auroc"]
            if len(np.unique(d["labels"])) < 2:
                n_skip += 1; continue          # AUROC undefined for one class
            if abs(saved - roc_auc_score(d["labels"], d["scores"])) > 1e-12:
                fails.append(f"metric mismatch: {npz.parent}")
            else:
                n_ok += 1
        except Exception as e:
            fails.append(f"unreadable {npz}: {type(e).__name__}")
    print(f"   {n_ok} verified, {n_skip} skipped (single-class)")

    print("2. COMPLETENESS — expected cell counts")
    for name, want in EXPECTED.items():
        got = len(list((ROOT / name).rglob("result.json"))) if (ROOT / name).exists() else 0
        status = "ok" if got == want else "MISMATCH"
        if got != want:
            fails.append(f"{name}: {got} results, expected {want}")
        print(f"   {name:32s} {got:4d}/{want:<4d} {status}")

    print("3. SANITY — value ranges")
    bad = 0
    for rj in ROOT.rglob("result.json"):
        v = json.load(open(rj)).get("image_auroc")
        if v is None or not (0.0 <= v <= 1.0) or v != v:
            fails.append(f"implausible AUROC in {rj}: {v}"); bad += 1
    print(f"   {bad} implausible values")

    print("4. PROVENANCE — results must postdate the code that made them")
    for name, src in GOVERNS.items():
        d = ROOT / name
        if not d.exists() or not (CODE / src).exists():
            continue
        code_t = (CODE / src).stat().st_mtime
        earliest = min((p.stat().st_mtime for p in d.rglob("result.json")), default=None)
        if earliest is None:
            continue
        if earliest < code_t:
            ex = EXCEPTIONS.get(name)
            if ex and ex.get("identical") and ex.get("source") == src:
                print(f"   {name:32s} predates {src}, but verified inert "
                      f"({ex['verified_cell']} reproduces bit-for-bit)")
            else:
                fails.append(f"{name}: results predate {src} "
                             f"(verify inertness or re-run)")
                print(f"   {name:32s} STALE vs {src}")
        else:
            print(f"   {name:32s} ok")

    print()
    if fails:
        print(f"{len(fails)} PROBLEM(S):")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
