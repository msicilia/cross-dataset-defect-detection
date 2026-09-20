"""Localisation of DINO-PatchCore (ViT-B/14) under cross-dataset transfer.

Banks are built from each source as in the main benchmark and scored on the two
targets with pixel ground truth: MVTec AD (shipped masks) and VISION (COCO
polygons rasterised by vision_masks.py). Test splits are not capped.

Metrics per run:
    pixel_auroc_defective, pixel_ap_defective   pixels of defective images only
    pixel_auroc_all                              pixels of all test images
    argmax_in_mask     fraction of defective images whose map maximum lies in the mask
    mean_mask_area     mean mask area fraction, the chance level of argmax_in_mask
    image_auroc, image_ap                        image scores = map maxima

Geometry: the backbone sees only a centre crop of the loaded image
(DINOPatchCore.map_box), so maps cover that crop. Masks are reduced to the
loaded frame by area averaging, keeping any nonzero coverage ("box_any"), and
cropped to the same box. Defective images whose mask lies entirely outside the crop are counted
in n_defective_outside_crop and excluded from the defective-only metrics.

Each (seed, source) bank is fitted once, if any of its target cells is
missing, and scores every pending target.

Results: results/raw/dino_patchcore_localisation/<source>__<target>/seed<s>/

Usage:
    python run_localisation.py [--seeds S ...] [--targets mvtec vision] [--device cuda|mps|cpu]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

import config as cfg
from common import build_dataset, cell_done, pick_device, save_cell
from evaluate import image_level_metrics
from methods import DINOPatchCore
from vision_masks import build_masks

BACKBONE = cfg.DINOV2_MODELS["base"]
TARGETS = ["mvtec", "vision"]


def _mask(path, frame: int, box: tuple[int, int, int]) -> np.ndarray:
    """Binary mask in the frame_x_frame loaded image, cropped to box."""
    small = np.array(Image.open(path).convert("L").resize((frame, frame), Image.BOX)) > 0
    top, left, size = box
    return small[top:top + size, left:left + size]


def evaluate(maps, mask_paths, frame: int, box: tuple[int, int, int]) -> dict:
    d_scores, d_labels, a_scores, a_labels, areas = [], [], [], [], []
    hits = total = outside = 0
    for smap, mpath in zip(maps, mask_paths):
        assert smap.shape == (box[2], box[2]), f"map {smap.shape} does not match box {box}"
        if mpath is not None:
            m = _mask(mpath, frame, box)
            if m.any():
                d_scores.append(smap.ravel())
                d_labels.append(m.ravel())
                hits += int(m[np.unravel_index(np.argmax(smap), smap.shape)])
                total += 1
                areas.append(float(m.mean()))
            else:
                outside += 1
        else:
            m = np.zeros(smap.shape, dtype=bool)
        a_scores.append(smap.ravel())
        a_labels.append(m.ravel())

    out = {"pixel_auroc_defective": None, "pixel_ap_defective": None}
    if d_scores:
        s, lab = np.concatenate(d_scores), np.concatenate(d_labels)
        out["pixel_auroc_defective"] = float(roc_auc_score(lab, s))
        out["pixel_ap_defective"] = float(average_precision_score(lab, s))
    s, lab = np.concatenate(a_scores), np.concatenate(a_labels)
    out["pixel_auroc_all"] = float(roc_auc_score(lab, s)) if lab.any() else None
    out["argmax_in_mask"] = hits / total if total else None
    out["mean_mask_area"] = float(np.mean(areas)) if areas else None
    out["n_defective_scored"] = total
    out["n_defective_outside_crop"] = outside
    return out


def target_split(target: str):
    """Test split of a target with a mask path for every defective image."""
    split = build_dataset(target).test()
    if target == "vision":
        masks = build_masks(cfg.DATASET_PATHS["vision"], cfg.VISION_CATEGORIES)
        split.mask_paths = [masks.get(Path(p)) if lab else None
                            for p, lab in zip(split.image_paths, split.labels)]
    elif target != "mvtec":
        raise ValueError(f"{target} has no pixel ground truth")
    missing = [p for p, lab, m in zip(split.image_paths, split.labels, split.mask_paths)
               if lab and m is None]
    assert not missing, f"{len(missing)} defective {target} images lack a mask, e.g. {missing[0]}"
    return split


def cell_config(source: str, target: str, seed: int, map_box: tuple[int, int, int]) -> dict:
    return {
        "experiment": "localisation",
        "backbone": BACKBONE,
        "source": source,
        "n_reference": cfg.PATCHCORE["max_train_images"],
        "seed": seed,
        "target": target,
        "max_test": None,
        "coreset_ratio": cfg.PATCHCORE["coreset_ratio"],
        "image_size": cfg.PATCHCORE["image_size"],
        "map_box": list(map_box),
        "mask_reduction": "box_any",
    }


def _fmt(v, spec: str) -> str:
    return "n/a" if v is None else format(v, spec)


def run(seeds: list[int], targets: list[str], device: str) -> None:
    out_root = cfg.RESULTS_DIR / "raw" / "dino_patchcore_localisation"
    # One detector serves every bank: fit replaces the memory bank, and the
    # crop box it reports enters each cell's configuration.
    method = DINOPatchCore(
        backbone=BACKBONE, device=device,
        coreset_ratio=cfg.PATCHCORE["coreset_ratio"],
        max_train_images=cfg.PATCHCORE["max_train_images"],
        image_size=cfg.PATCHCORE["image_size"],
        batch_size=cfg.PATCHCORE["batch_size"],
    )
    box = method.map_box()
    splits = {}
    print(f"device={device}  targets={targets}  map_box={box}")

    for seed in seeds:
        for src in cfg.DATASETS:
            pending = []
            for target in targets:
                out_dir = out_root / f"{src}__{target}" / f"seed{seed}"
                config = cell_config(src, target, seed, box)
                if cell_done(out_dir, config):
                    print(f"[seed {seed}] {src}->{target}: done")
                else:
                    pending.append((target, out_dir, config))
            if not pending:
                continue

            print(f"\n[seed {seed}] bank from {src}")
            method.fit(build_dataset(src).normal_train().image_paths, seed=seed)
            for target, out_dir, config in pending:
                if target not in splits:
                    splits[target] = target_split(target)
                split = splits[target]
                paths, labels = list(split.image_paths), np.array(split.labels)
                print(f"  -> {target}: {len(paths)} test images ({int(labels.sum())} defective)")
                maps = method.score_maps(paths)
                metrics = evaluate(maps, split.mask_paths, method.image_size, box)
                scores = np.array([float(m.max()) for m in maps])
                img = image_level_metrics(scores, labels)
                metrics.update({"image_auroc": img.image_auroc, "image_ap": img.image_ap})
                save_cell(out_dir, config, metrics, device,
                          scores=scores, labels=labels, paths=paths)
                print(f"    pixel AUROC defective={_fmt(metrics['pixel_auroc_defective'], '.4f')}"
                      f"  all={_fmt(metrics['pixel_auroc_all'], '.4f')}"
                      f"  argmax-in-mask={_fmt(metrics['argmax_in_mask'], '.3f')}"
                      f"  image AUROC={img.image_auroc:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=cfg.SEEDS)
    ap.add_argument("--targets", nargs="+", default=TARGETS, choices=TARGETS)
    ap.add_argument("--device", default=None,
                    help="default: $DEVICE, else cuda, mps, cpu")
    a = ap.parse_args()
    run(a.seeds, a.targets, pick_device(a.device))
