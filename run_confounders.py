"""Cross-dataset matrix under image-space normalisations (DINO-PatchCore, ViT-B/14).

Each arm applies one transform after the square resize, identically to
reference and test images:

    baseline    no transform
    gray        colour removed, kept as three channels
    equalised   histogram equalisation of the luminance channel, chroma kept
    lowres      downsampled to 64x64 and back to the loaded size

Reference banks follow the main benchmark (up to 500 images drawn by the
detector with the run seed); test sets are capped at --max-test by
label-stratified subsampling.

Results: results/raw/dino_patchcore_confounders/<arm>/seed<s>/<source>__<target>/

Usage:
    python run_confounders.py [--seeds S ...] [--max-test N] [--device cuda|mps|cpu]
"""
from __future__ import annotations

import argparse

from PIL import Image, ImageOps

import config as cfg
from common import build_dataset, cell_done, pick_device, save_cell, stratified_test
from evaluate import image_level_metrics
from methods import DINOPatchCore

BACKBONE = cfg.DINOV2_MODELS["base"]
LOWRES_SIZE = 64


def _gray(img: Image.Image) -> Image.Image:
    return img.convert("L").convert("RGB")


def _equalised(img: Image.Image) -> Image.Image:
    # Luminance only, so that this arm does not also remove colour.
    y, cb, cr = img.convert("YCbCr").split()
    return Image.merge("YCbCr", (ImageOps.equalize(y), cb, cr)).convert("RGB")


def _lowres(img: Image.Image) -> Image.Image:
    return img.resize((LOWRES_SIZE, LOWRES_SIZE), Image.BILINEAR).resize(img.size, Image.BILINEAR)


ARMS = {"baseline": None, "gray": _gray, "equalised": _equalised, "lowres": _lowres}


def cell_config(arm: str, seed: int, source: str, target: str, max_test: int | None) -> dict:
    return {
        "experiment": "confounders",
        "backbone": BACKBONE,
        "arm": arm,
        "source": source,
        "n_reference": cfg.PATCHCORE["max_train_images"],
        "seed": seed,
        "target": target,
        "max_test": max_test,
        "coreset_ratio": cfg.PATCHCORE["coreset_ratio"],
        "image_size": cfg.PATCHCORE["image_size"],
    }


def run(seeds: list[int], max_test: int | None, device: str) -> None:
    out_root = cfg.RESULTS_DIR / "raw" / "dino_patchcore_confounders"
    datasets = {d: build_dataset(d) for d in cfg.DATASETS}
    print(f"device={device}  seeds={seeds}  max_test={max_test}")

    for seed in seeds:
        targets = {t: stratified_test(datasets[t].test(), max_test, seed) for t in cfg.DATASETS}

        for arm, transform in ARMS.items():
            for src in cfg.DATASETS:
                cells = {t: (out_root / arm / f"seed{seed}" / f"{src}__{t}",
                             cell_config(arm, seed, src, t, max_test))
                         for t in cfg.DATASETS}
                pending = {t: c for t, c in cells.items() if not cell_done(*c)}
                if not pending:
                    print(f"[seed {seed}] {arm}/{src}: done")
                    continue

                print(f"\n[seed {seed}] arm={arm} source={src}")
                method = DINOPatchCore(
                    backbone=BACKBONE, device=device,
                    coreset_ratio=cfg.PATCHCORE["coreset_ratio"],
                    max_train_images=cfg.PATCHCORE["max_train_images"],
                    image_size=cfg.PATCHCORE["image_size"],
                    batch_size=cfg.PATCHCORE["batch_size"],
                    preprocess=transform, variant=arm,
                )
                method.fit(datasets[src].normal_train().image_paths, seed=seed)
                for t, (out_dir, config) in pending.items():
                    paths, labels = targets[t]
                    scores = method.score(paths)
                    res = image_level_metrics(scores, labels)
                    save_cell(out_dir, config,
                              {"image_auroc": res.image_auroc, "image_ap": res.image_ap},
                              device, scores=scores, labels=labels, paths=paths)
                    print(f"    -> {t}: AUROC={res.image_auroc:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=cfg.CONFOUNDER_SEEDS)
    ap.add_argument("--max-test", type=int, default=cfg.MAX_TEST,
                    help="cap on test images per target (label-stratified); 0 = no cap")
    ap.add_argument("--device", default=None,
                    help="default: $DEVICE, else cuda, mps, cpu")
    a = ap.parse_args()
    run(a.seeds, a.max_test or None, pick_device(a.device))
