"""Transfer between groups with MVTec AD split by category type.

Groups, each used as source and as target:
    sdnet       SDNET2018
    mvtec_tex   MVTec AD tile, wood, grid        (textures; MVTec-tex in the paper)
    mvtec_obj   MVTec AD metal_nut, screw        (metallic objects; MVTec-metal)
    vision      VISION Casting, Ring, Screw, Cylinder

For each (source, target) and seed, DINO-PatchCore builds its memory bank from
the source's defect-free reference images (at most
PATCHCORE["max_train_images"], drawn with the seed) and scores the target test
split, label-stratified to at most --max-test images with the seed.

Results: results/raw/dino_patchcore_mvtecsplit/<source>__<target>/seed<s>/

Usage:
    python run_mvtec_split.py [--seeds S ...] [--max-test N] [--device cuda|mps|cpu]
"""
from __future__ import annotations

import argparse

import config as cfg
from common import build_dataset, cell_done, pick_device, save_cell, stratified_test
from datasets import MVTecDataset
from evaluate import image_level_metrics
from methods import DINOPatchCore

MVTEC_GROUPS = {"mvtec_tex": ["tile", "wood", "grid"],
                "mvtec_obj": ["metal_nut", "screw"]}
GROUPS = ["sdnet", "mvtec_tex", "mvtec_obj", "vision"]
BACKBONE = cfg.DINOV2_MODELS["base"]


def build_group(name: str):
    if name in MVTEC_GROUPS:
        return MVTecDataset(cfg.DATASET_PATHS["mvtec"], categories=MVTEC_GROUPS[name])
    return build_dataset(name)


def cell_config(source: str, target: str, seed: int, max_test: int | None) -> dict:
    return {"experiment": "mvtec_split", "method": "dino_patchcore_base",
            "backbone": BACKBONE, "source": source, "target": target,
            "seed": seed, "max_test": max_test,
            "max_reference_images": cfg.PATCHCORE["max_train_images"],
            "coreset_ratio": cfg.PATCHCORE["coreset_ratio"],
            "image_size": cfg.PATCHCORE["image_size"]}


def run(seeds: list[int], max_test: int | None, device: str) -> None:
    root = cfg.RESULTS_DIR / "raw" / "dino_patchcore_mvtecsplit"
    print(f"device={device} seeds={seeds} max_test={max_test}")

    for seed in seeds:
        targets = {}
        for src in GROUPS:
            pending = []
            for tgt in GROUPS:
                out_dir = root / f"{src}__{tgt}" / f"seed{seed}"
                config = cell_config(src, tgt, seed, max_test)
                if not cell_done(out_dir, config):
                    pending.append((tgt, out_dir, config))
            if not pending:
                print(f"[seed {seed}] {src}: all targets done")
                continue

            references = build_group(src).normal_train().image_paths
            print(f"\n[seed {seed}] {src}: fitting ({len(references)} candidate images)")
            model = DINOPatchCore(backbone=BACKBONE, device=device,
                                  coreset_ratio=cfg.PATCHCORE["coreset_ratio"],
                                  max_train_images=cfg.PATCHCORE["max_train_images"],
                                  image_size=cfg.PATCHCORE["image_size"],
                                  batch_size=cfg.PATCHCORE["batch_size"])
            model.fit(references, seed=seed)

            for tgt, out_dir, config in pending:
                if tgt not in targets:
                    targets[tgt] = stratified_test(build_group(tgt).test(), max_test, seed)
                paths, labels = targets[tgt]
                scores = model.score(paths)
                metrics = image_level_metrics(scores, labels)
                save_cell(out_dir, config,
                          {"image_auroc": metrics.image_auroc,
                           "image_ap": metrics.image_ap},
                          device, scores=scores, labels=labels, paths=paths)
                print(f"    {src} -> {tgt}: AUROC={metrics.image_auroc:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", nargs="+", type=int, default=cfg.SEEDS)
    ap.add_argument("--max-test", type=int, default=cfg.MAX_TEST,
                    help="stratified cap on each target test split; 0 = no cap")
    ap.add_argument("--device", default=None,
                    help="default: $DEVICE, else cuda, mps, cpu")
    a = ap.parse_args()
    run(a.seeds, a.max_test or None, pick_device(a.device))
