from __future__ import annotations
"""How much of a distant domain does it take to spoil a memory bank?

The contamination experiment tests a single mixture (250 base + 250 VISION) and
shows that adding VISION harms an otherwise-useful bank. That answers "does it
hurt" but not "how much is too much", and it cannot separate the candidate
explanations -- domain imbalance, feature interference,
reference redundancy, conflicting distributions.

Here the bank size is held at exactly 500 images throughout and only the
*proportion* drawn from VISION varies: 0, 25, 50, 75, 100 %. Because the total
never changes, any trend is attributable to composition rather than to bank
size, and the shape of the curve distinguishes the explanations:

  - a linear decline suggests simple dilution of useful references;
  - a sharp drop at a small VISION fraction suggests active interference, i.e.
    a few distant references are enough to capture the nearest-neighbour lookups;
  - a flat curve until VISION dominates suggests the two populations coexist in
    the bank without competing.

Usage:
    python run_proportions.py [--seeds 0 1 2 3 4] [--max-test 2000]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from datasets import MVTecDataset, SDNETDataset, VISIONDataset
from methods import DINOPatchCore
from evaluate import image_level_metrics
from run_contamination import _stratified_test, _sample, _rng, build_dataset, make_method

DATASETS = ["sdnet", "mvtec", "vision"]
BANK_SIZE = 500
# Bases are the two sources that transfer usefully; VISION is the contaminant.
BASES = ["mvtec", "sdnet"]
FRACTIONS = [0.0, 0.25, 0.50, 0.75, 1.0]
CONTAMINANT = "vision"


def run(seeds: list[int], max_test: int | None) -> None:
    import torch
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    out_root = cfg.RESULTS_DIR / "raw" / "dino_patchcore_proportions"
    print(f"device={device}  seeds={seeds}  max_test={max_test}  bank={BANK_SIZE}")

    for seed in seeds:
        targets = {}
        for t in DATASETS:
            paths, labels = _stratified_test(build_dataset(t, seed).test(), max_test, seed)
            targets[t] = (paths, labels)

        normals = {s: build_dataset(s, seed).normal_train().image_paths for s in DATASETS}
        # Draw generous pools once per seed, then take nested prefixes, so that
        # the 25 % bank's VISION images are a subset of the 50 % bank's. Without
        # this the comparison would confound proportion with which images were
        # drawn.
        pool_cont = _sample(normals[CONTAMINANT], BANK_SIZE, _rng(seed, 101))

        for base in BASES:
            pool_base = _sample(normals[base], BANK_SIZE, _rng(seed, 102))
            for frac in FRACTIONS:
                n_cont = int(round(BANK_SIZE * frac))
                n_base = BANK_SIZE - n_cont
                if n_base > len(pool_base) or n_cont > len(pool_cont):
                    print(f"  [seed {seed}] {base} frac={frac}: not enough images, skipping")
                    continue
                name = f"{base}_vision{int(frac*100):03d}"
                done = all((out_root / name / f"seed{seed}" / t / "result.json").exists()
                           for t in DATASETS)
                if done:
                    print(f"[seed {seed}] {name}: done, skipping")
                    continue

                imgs = pool_base[:n_base] + pool_cont[:n_cont]
                assert len(imgs) == BANK_SIZE, f"bank size drifted: {len(imgs)}"
                print(f"\n[seed {seed}] bank '{name}': {n_base} {base} + {n_cont} {CONTAMINANT}")
                method = make_method(device)
                method.fit(imgs, seed=seed)

                for t in DATASETS:
                    out_dir = out_root / name / f"seed{seed}" / t
                    if (out_dir / "result.json").exists():
                        continue
                    paths, labels = targets[t]
                    scores = method.score(paths)
                    res = image_level_metrics(scores, labels)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    with open(out_dir / "result.json", "w") as f:
                        json.dump({"image_auroc": res.image_auroc, "image_ap": res.image_ap,
                                   "base": base, "fraction_contaminant": frac,
                                   "n_base": n_base, "n_contaminant": n_cont,
                                   "target": t, "seed": seed}, f, indent=2)
                    np.savez_compressed(out_dir / "scores.npz",
                                        scores=np.asarray(scores, dtype=np.float64),
                                        labels=np.asarray(labels, dtype=np.int8))
                    print(f"    -> {t}: AUROC={res.image_auroc:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--max-test", type=int, default=2000)
    a = ap.parse_args()
    run(a.seeds, a.max_test)
