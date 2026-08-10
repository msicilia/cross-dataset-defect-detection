from __future__ import annotations
"""Does cross-dataset transfer break detection, localisation, or both?

PatchCore-style detectors produce localisation maps for free, and evaluating
them says something about *why* transfer fails. It can only be done where pixel
ground truth exists: of the three domains, only MVTec AD has masks (413 defective test images, all masked);
SDNET2018 is image-level by construction and our VISION subset carries no masks.
MVTec is therefore the only usable target, and the experiment asks whether a
bank built elsewhere still finds MVTec's defects in the right *place*.

Three quantities per source:

  pixel AUROC (defective only)  ranking of pixels within defective images
  pixel AUROC (all images)      the usual benchmark convention, which also
                                penalises false positives on normal images
  argmax-in-mask rate           fraction of defective images whose single
                                highest-scoring patch falls inside the defect

The last is the mechanism metric. The image-level score *is* the max patch
distance, so if that argmax sits outside the defect the detector reached its
decision by looking at the wrong thing -- which is exactly how an anomaly
ranking can invert under domain shift.

Usage:  python run_localisation.py [--seeds 0 1 2 3 4]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from datasets import MVTecDataset
from methods import DINOPatchCore
from run_contamination import build_dataset

SOURCES = ["sdnet", "mvtec", "vision"]
TARGET = "mvtec"


def _mask(path, shape) -> np.ndarray:
    m = np.array(Image.open(path).convert("L").resize(shape[::-1], Image.NEAREST)) > 0
    return m


def evaluate(maps, mask_paths, labels):
    from sklearn.metrics import roc_auc_score, average_precision_score
    d_scores, d_labels, a_scores, a_labels = [], [], [], []
    hits = total = 0
    for smap, mpath, lab in zip(maps, mask_paths, labels):
        if mpath is not None:
            m = _mask(mpath, smap.shape)
            if m.any():
                d_scores.append(smap.ravel()); d_labels.append(m.ravel().astype(np.uint8))
                # Does the peak of the map land on the defect?
                idx = np.unravel_index(np.argmax(smap), smap.shape)
                hits += int(m[idx]); total += 1
        else:
            # Normal image: every pixel is a true negative under the usual
            # benchmark convention, so it can only add false positives.
            m = np.zeros_like(smap, dtype=np.uint8)
        a_scores.append(smap.ravel())
        a_labels.append((m.ravel().astype(np.uint8) if mpath is not None
                         else np.zeros(smap.size, dtype=np.uint8)))

    out = {}
    if d_scores:
        s, l = np.concatenate(d_scores), np.concatenate(d_labels)
        out["pixel_auroc_defective"] = float(roc_auc_score(l, s))
        out["pixel_ap_defective"] = float(average_precision_score(l, s))
    s, l = np.concatenate(a_scores), np.concatenate(a_labels)
    if l.sum() > 0:
        out["pixel_auroc_all"] = float(roc_auc_score(l, s))
    out["argmax_in_mask"] = (hits / total) if total else None
    out["n_defective_scored"] = total
    return out


def run(seeds: list[int]) -> None:
    import torch
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    out_root = cfg.RESULTS_DIR / "raw" / "dino_patchcore_localisation"
    tgt = MVTecDataset(cfg.DATASET_PATHS[TARGET], categories=cfg.MVTEC_CATEGORIES).test()
    print(f"device={device}  target={TARGET}  test={len(tgt.image_paths)} "
          f"({sum(1 for m in tgt.mask_paths if m is not None)} masked)")

    for seed in seeds:
        for src in SOURCES:
            out_dir = out_root / f"{src}__{TARGET}" / f"seed{seed}"
            if (out_dir / "result.json").exists():
                print(f"[seed {seed}] {src}: done, skipping"); continue
            print(f"\n[seed {seed}] bank from {src} -> localise on {TARGET}")
            m = DINOPatchCore(
                backbone=cfg.DINOV2_MODELS["base"], device=device,
                coreset_ratio=cfg.PATCHCORE["coreset_ratio"],
                max_train_images=cfg.PATCHCORE["max_train_images"],
                image_size=cfg.PATCHCORE["image_size"],
                batch_size=cfg.PATCHCORE["batch_size"],
            )
            m.fit(build_dataset(src, seed).normal_train().image_paths, seed=seed)
            maps = m.score_maps(tgt.image_paths)
            res = evaluate(maps, tgt.mask_paths, tgt.labels)
            res.update({"source": src, "target": TARGET, "seed": seed})
            out_dir.mkdir(parents=True, exist_ok=True)
            json.dump(res, open(out_dir / "result.json", "w"), indent=2)
            print(f"    pixel AUROC (defective)={res.get('pixel_auroc_defective'):.4f}  "
                  f"(all)={res.get('pixel_auroc_all'):.4f}  "
                  f"argmax-in-mask={res.get('argmax_in_mask'):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    run(a.seeds)
