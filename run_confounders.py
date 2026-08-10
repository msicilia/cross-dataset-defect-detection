from __future__ import annotations
"""Does the transfer gap survive when each confound is removed?

The degradation could come from illumination, imaging resolution, object
geometry, annotation quality or defect scale rather than from morphology;
without controlled ablations a morphological attribution would be
speculative. The MVTec split already holds *acquisition* constant by
construction, but it cannot address confounds that differ across the three
datasets.

This runs the cross-dataset matrix under image-space normalisations that each
remove one candidate explanation, applied identically to reference and test
images:

    baseline    unmodified pipeline (reference point)
    gray        colour removed -- if the gap shrinks, it was partly chromatic
    equalised   per-image histogram equalisation on luminance, removing global
                exposure and contrast differences between datasets
    lowres      downsample to 64x64 and back, destroying fine texture and scale
                cues while preserving coarse shape and layout

Reading: a gap that persists under all three is not explained by colour,
illumination or fine-scale texture. The `lowres` arm is the most informative
for the morphology claim -- it strips texture detail but keeps object shape, so
if surface-vs-object morphology is what matters, transfer within a morphological
group should degrade least there.

This is a diagnostic, not a headline benchmark: it uses one seed per arm and a
capped test set, which is enough to see whether a gap of ~0.17 moves.

Usage:
    python run_confounders.py [--seeds 0] [--max-test 2000]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from methods import DINOPatchCore
from evaluate import image_level_metrics
from run_contamination import _stratified_test, build_dataset

DATASETS = ["sdnet", "mvtec", "vision"]


def _gray(img: Image.Image) -> Image.Image:
    # Kept as 3-channel so the backbone's input shape is unchanged; only the
    # chromatic information is destroyed.
    return img.convert("L").convert("RGB")


def _equalised(img: Image.Image) -> Image.Image:
    # Equalise luminance only, leaving chroma untouched. Equalising a greyscale
    # conversion instead would remove colour as well, which would make this arm
    # a compound of the `gray` manipulation and confound the two.
    y, cb, cr = img.convert("YCbCr").split()
    return Image.merge("YCbCr", (ImageOps.equalize(y), cb, cr)).convert("RGB")


def _lowres(img: Image.Image) -> Image.Image:
    w, h = img.size
    return img.resize((64, 64), Image.BILINEAR).resize((w, h), Image.BILINEAR)


VARIANTS = {"baseline": None, "gray": _gray, "equalised": _equalised, "lowres": _lowres}


def run(seeds: list[int], max_test: int | None) -> None:
    import torch
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    out_root = cfg.RESULTS_DIR / "raw" / "dino_patchcore_confounders"
    print(f"device={device}  seeds={seeds}  max_test={max_test}")

    for seed in seeds:
        targets = {}
        for t in DATASETS:
            paths, labels = _stratified_test(build_dataset(t, seed).test(), max_test, seed)
            targets[t] = (paths, labels)

        for vname, fn in VARIANTS.items():
            for src in DATASETS:
                done = all((out_root / vname / f"seed{seed}" / f"{src}__{t}"
                            / "result.json").exists() for t in DATASETS)
                if done:
                    print(f"[seed {seed}] {vname}/{src}: done, skipping")
                    continue

                print(f"\n[seed {seed}] variant={vname} source={src}")
                m = DINOPatchCore(
                    backbone=cfg.DINOV2_MODELS["base"], device=device,
                    coreset_ratio=cfg.PATCHCORE["coreset_ratio"],
                    max_train_images=cfg.PATCHCORE["max_train_images"],
                    image_size=cfg.PATCHCORE["image_size"],
                    batch_size=cfg.PATCHCORE["batch_size"],
                    preprocess=fn, variant=vname,
                )
                m.fit(build_dataset(src, seed).normal_train().image_paths, seed=seed)

                for t in DATASETS:
                    out_dir = out_root / vname / f"seed{seed}" / f"{src}__{t}"
                    if (out_dir / "result.json").exists():
                        continue
                    paths, labels = targets[t]
                    scores = m.score(paths)
                    res = image_level_metrics(scores, labels)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    json.dump({"image_auroc": res.image_auroc, "image_ap": res.image_ap,
                               "variant": vname, "source": src, "target": t, "seed": seed},
                              open(out_dir / "result.json", "w"), indent=2)
                    np.savez_compressed(out_dir / "scores.npz",
                                        scores=np.asarray(scores, dtype=np.float64),
                                        labels=np.asarray(labels, dtype=np.int8))
                    print(f"    -> {t}: AUROC={res.image_auroc:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--max-test", type=int, default=2000)
    a = ap.parse_args()
    run(a.seeds, a.max_test)
