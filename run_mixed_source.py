from __future__ import annotations
"""Mixed-source memory bank experiment.

Fits DINO-PatchCore on the combined normal training images from two source
datasets and evaluates on all three target datasets. Tests the paper's
recommendation that a mixed bank outperforms any single-domain bank when the
target domain is heterogeneous.

Source combinations (leave-one-out):
    sdnet + mvtec  → evaluate on vision, sdnet, mvtec
    sdnet + vision → evaluate on mvtec, sdnet, vision
    mvtec + vision → evaluate on sdnet, mvtec, vision

Results saved to:
    results/raw/dino_patchcore_mixed/seed<s>/<src1>+<src2>__<tgt>/result.json
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
SEEDS = [0, 1, 2]
MAX_PER_SOURCE = 250   # 250×2 = 500 total, same cap as single-source


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
    results_root = cfg.RESULTS_DIR / "raw" / "dino_patchcore_mixed"

    pairs = [
        ("sdnet", "mvtec"),
        ("sdnet", "vision"),
        ("mvtec", "vision"),
    ]

    for seed in SEEDS:
        for src1_name, src2_name in pairs:
            key = f"{src1_name}+{src2_name}"
            print(f"\n{'='*60}")
            print(f"Mixed source: {key}  seed={seed}")
            print(f"{'='*60}")

            # Collect normal training images from both sources
            rng = np.random.default_rng(seed)
            combined_paths = []
            for src_name in (src1_name, src2_name):
                ds = build_dataset(src_name, seed)
                paths = ds.normal_train().image_paths
                if len(paths) > MAX_PER_SOURCE:
                    idx = rng.choice(len(paths), MAX_PER_SOURCE, replace=False)
                    paths = [paths[i] for i in idx]
                combined_paths.extend(paths)

            print(f"  Combined training set: {len(combined_paths)} images")

            method = DINOPatchCore(
                backbone=cfg.DINOV2_MODELS["base"],
                device=device,
                **{k: v for k, v in cfg.PATCHCORE.items() if k != "image_size"},
                image_size=cfg.PATCHCORE["image_size"],
            )
            # Override max_train_images to allow the combined set
            method.max_train_images = len(combined_paths) + 1
            method.fit(combined_paths, seed=seed)

            for tgt_name in DATASETS:
                out_dir = results_root / f"seed{seed}" / f"{key}__{tgt_name}"
                if (out_dir / "result.json").exists():
                    print(f"  [{key} → {tgt_name}] already done, skipping")
                    continue

                tgt_ds = build_dataset(tgt_name, seed)
                test = tgt_ds.test()
                scores = method.score(test.image_paths)
                labels = np.array(test.labels)
                result = image_level_metrics(scores, labels)

                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_dir / "result.json", "w") as f:
                    json.dump({"image_auroc": result.image_auroc,
                               "image_ap": result.image_ap}, f, indent=2)
                print(f"  [{key} → {tgt_name}]  AUROC={result.image_auroc:.4f}")


if __name__ == "__main__":
    run()
