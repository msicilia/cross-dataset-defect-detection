from __future__ import annotations
"""Cross-group transfer with MVTec split into textures and metallic objects.

Groups (each acts as source and as target):
    sdnet       SDNET2018
    mvtec_tex   MVTec tile + wood + grid
    mvtec_obj   MVTec metal_nut + screw
    vision      VISION Casting/Ring/Screw/Cylinder

For each (source, target) and seed, DINO-PatchCore builds a memory bank from the
source's defect-free images (<=500) and is scored on the target test split.
Targets larger than --max-test are stratified-subsampled per seed.

Results: results/raw/dino_patchcore_mvtecsplit/<src>__<tgt>/seed<s>/result.json
Usage:   python run_mvtec_split.py [--seeds 0 1 2 3 4] [--max-test 2000]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from datasets import SDNETDataset, VISIONDataset
from methods import DINOPatchCore
from evaluate import image_level_metrics

GROUPS = ["sdnet", "mvtec_tex", "mvtec_obj", "vision"]
MVTEC_TEX = ["tile", "wood", "grid"]
MVTEC_OBJ = ["metal_nut", "screw"]


def _mvtec_subset(cats: list[str], split: str):
    """Read the MVTec manifest and keep only records whose category (path prefix)
    is in `cats`. Returns (image_paths, labels)."""
    root = cfg.DATASET_PATHS["mvtec"]
    recs = json.load(open(root / f"manifest_{split}.json"))
    recs = [r for r in recs if r["path"].split("/")[0] in cats]
    return [root / r["path"] for r in recs], [r["label"] for r in recs]


def train_paths(name: str, seed: int) -> list[Path]:
    if name == "sdnet":
        return SDNETDataset(cfg.DATASET_PATHS["sdnet"], seed=seed).normal_train().image_paths
    if name == "vision":
        return VISIONDataset(cfg.DATASET_PATHS["vision"],
                             categories=cfg.VISION_CATEGORIES).normal_train().image_paths
    if name == "mvtec_tex":
        return _mvtec_subset(MVTEC_TEX, "train")[0]
    if name == "mvtec_obj":
        return _mvtec_subset(MVTEC_OBJ, "train")[0]
    raise ValueError(name)


def test_split(name: str, seed: int):
    if name == "sdnet":
        t = SDNETDataset(cfg.DATASET_PATHS["sdnet"], seed=seed).test()
        return list(t.image_paths), list(t.labels)
    if name == "vision":
        t = VISIONDataset(cfg.DATASET_PATHS["vision"], categories=cfg.VISION_CATEGORIES).test()
        return list(t.image_paths), list(t.labels)
    if name == "mvtec_tex":
        return _mvtec_subset(MVTEC_TEX, "test")
    if name == "mvtec_obj":
        return _mvtec_subset(MVTEC_OBJ, "test")
    raise ValueError(name)


def _stratified(paths, labels, max_test, seed):
    if max_test is None or len(paths) <= max_test:
        return paths, np.array(labels)
    rng = np.random.default_rng([seed, 7777]); lab = np.array(labels); idx = np.arange(len(paths))
    pos, neg = idx[lab == 1], idx[lab == 0]; frac = max_test / len(paths)
    keep = np.concatenate([rng.choice(pos, max(1, int(round(len(pos) * frac))), replace=False),
                           rng.choice(neg, max(1, int(round(len(neg) * frac))), replace=False)])
    keep.sort()
    return [paths[i] for i in keep], lab[keep]


def make_method(device):
    return DINOPatchCore(backbone=cfg.DINOV2_MODELS["base"], device=device,
                         coreset_ratio=cfg.PATCHCORE["coreset_ratio"],
                         max_train_images=cfg.PATCHCORE["max_train_images"],
                         image_size=cfg.PATCHCORE["image_size"],
                         batch_size=cfg.PATCHCORE["batch_size"])


def run(seeds, max_test):
    import torch
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    out = cfg.RESULTS_DIR / "raw" / "dino_patchcore_mvtecsplit"
    print(f"device={device} seeds={seeds} max_test={max_test}")

    for seed in seeds:
        targets = {g: _stratified(*test_split(g, seed), max_test, seed) for g in GROUPS}
        for g in GROUPS:
            p, l = targets[g]
            print(f"[seed {seed}] target {g}: {len(p)} ({int((l == 1).sum())} defective)")

        for src in GROUPS:
            done = all((out / f"{src}__{t}" / f"seed{seed}" / "result.json").exists() for t in GROUPS)
            if done:
                print(f"[seed {seed}] source {src}: done, skipping"); continue
            tp = train_paths(src, seed)
            print(f"\n[seed {seed}] building bank '{src}' ({len(tp)} images) ...")
            method = make_method(device); method.fit(tp, seed=seed)
            for t in GROUPS:
                od = out / f"{src}__{t}" / f"seed{seed}"
                if (od / "result.json").exists():
                    continue
                paths, labels = targets[t]
                res = image_level_metrics(method.score(paths), labels)
                od.mkdir(parents=True, exist_ok=True)
                json.dump({"image_auroc": res.image_auroc, "image_ap": res.image_ap,
                           "src": src, "tgt": t, "seed": seed}, open(od / "result.json", "w"), indent=2)
                print(f"    -> {t}: AUROC={res.image_auroc:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--max-test", type=int, default=2000)
    a = ap.parse_args()
    run(a.seeds, None if a.max_test == 0 else a.max_test)
