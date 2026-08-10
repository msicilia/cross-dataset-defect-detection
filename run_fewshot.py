from __future__ import annotations
"""Few-shot training-size ablation for DINO-PatchCore.

Sweeps the number of normal training images available from the source domain
and measures cross-dataset AUROC on all targets. Quantifies how quickly the
memory bank becomes useful as reference data grows from zero toward the full
500-image cap.

n_shots sweep: [5, 10, 25, 50, 100, 250, 500]

Results saved to:
    results/raw/dino_patchcore_fewshot/n<k>/seed<s>/<src>__<tgt>/result.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from datasets import MVTecDataset, SDNETDataset, VISIONDataset
from methods import DINOPatchCore
from evaluate import image_level_metrics

DATASETS = ["sdnet", "mvtec", "vision"]
N_SHOTS  = [5, 10, 25, 50, 100, 250, 500]
SEEDS    = [0, 1, 2]


def build_dataset(name: str, seed: int = 0):
    root = cfg.DATASET_PATHS[name]
    if name == "mvtec":
        return MVTecDataset(root, categories=cfg.MVTEC_CATEGORIES)
    if name == "sdnet":
        return SDNETDataset(root, seed=seed)
    if name == "vision":
        return VISIONDataset(root, categories=cfg.VISION_CATEGORIES)
    raise ValueError(name)


def run():
    device = "mps" if __import__("torch").backends.mps.is_available() else "cpu"
    results_root = cfg.RESULTS_DIR / "raw" / "dino_patchcore_fewshot"

    for seed in SEEDS:
        for src_name in DATASETS:
            src_ds = build_dataset(src_name, seed)
            all_train = src_ds.normal_train().image_paths

            # Sample once at the largest size; subsets are nested prefixes
            rng = np.random.default_rng(seed)
            max_n = min(max(N_SHOTS), len(all_train))
            if len(all_train) > max_n:
                pool_idx = rng.choice(len(all_train), max_n, replace=False)
                pool = [all_train[i] for i in sorted(pool_idx)]
            else:
                pool = list(all_train)

            for n in N_SHOTS:
                actual_n = min(n, len(pool))
                train_paths = pool[:actual_n]

                print(f"\n[seed={seed}  src={src_name}  n={actual_n}]")

                # Check if all targets already done for this (seed, src, n)
                all_done = all(
                    (results_root / f"n{actual_n}" / f"seed{seed}"
                     / f"{src_name}__{tgt}" / "result.json").exists()
                    for tgt in DATASETS
                )
                if all_done:
                    print("  all targets done, skipping")
                    continue

                method = DINOPatchCore(
                    backbone=cfg.DINOV2_MODELS["base"],
                    device=device,
                    coreset_ratio=cfg.PATCHCORE["coreset_ratio"],
                    max_train_images=actual_n + 1,   # allow exactly n images
                    image_size=cfg.PATCHCORE["image_size"],
                    batch_size=cfg.PATCHCORE["batch_size"],
                )
                method.fit(train_paths, seed=seed)

                for tgt_name in DATASETS:
                    out_dir = (results_root / f"n{actual_n}" / f"seed{seed}"
                               / f"{src_name}__{tgt_name}")
                    if (out_dir / "result.json").exists():
                        print(f"  → {tgt_name}: already done")
                        continue

                    tgt_ds = build_dataset(tgt_name, seed)
                    test = tgt_ds.test()
                    scores = method.score(test.image_paths)
                    labels = np.array(test.labels)
                    result = image_level_metrics(scores, labels)

                    out_dir.mkdir(parents=True, exist_ok=True)
                    with open(out_dir / "result.json", "w") as f:
                        json.dump({"image_auroc": result.image_auroc,
                                   "image_ap": result.image_ap,
                                   "n_train": actual_n,
                                   "src": src_name,
                                   "tgt": tgt_name,
                                   "seed": seed}, f, indent=2)
                    print(f"  → {tgt_name}: AUROC={result.image_auroc:.4f}")


if __name__ == "__main__":
    run()
