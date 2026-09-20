"""Cross-dataset benchmark.

Each reference-based detector builds its model from the defect-free reference
images of a source dataset (at most PATCHCORE["max_train_images"], drawn with
the run seed) and scores the full test split of every target dataset.
Reference-free detectors (CLIP-ZS, WinCLIP) are evaluated once per target and
stored with source "none" and seed 0.

Results: results/raw/<method>/seed<s>/<source>__<target>/{result.json,scores.npz}

Complete cells produced with the same configuration are skipped, and a model
is built only if at least one of its cells is missing. A complete cell with a
different configuration stops the run (see common.cell_done).

Usage:
    python run_experiments.py [--methods M ...] [--datasets D ...]
                              [--seeds S ...] [--device cuda|mps|cpu]
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json

import numpy as np

import config as cfg
from common import build_dataset, cell_done, pick_device, save_cell
from evaluate import image_level_metrics
from methods import CLIPZS, DINOPatchCore, PaDiM, PatchCore, SPADE, WinCLIP

METHODS = [
    "dino_patchcore_small", "dino_patchcore_base", "dino_patchcore_large",
    "patchcore", "spade", "padim", "winclip", "winclip_plus", "clip_zs",
]
PAPER_METHODS = [
    "dino_patchcore_base", "patchcore", "spade", "padim",
    "winclip", "winclip_plus", "clip_zs",
]
REFERENCE_FREE = {"clip_zs", "winclip"}


def build_method(name: str, device: str):
    pc = cfg.PATCHCORE
    if name.startswith("dino_patchcore_"):
        size = name.removeprefix("dino_patchcore_")
        return DINOPatchCore(backbone=cfg.DINOV2_MODELS[size], device=device,
                             coreset_ratio=pc["coreset_ratio"],
                             max_train_images=pc["max_train_images"],
                             image_size=pc["image_size"],
                             batch_size=pc["batch_size"])
    if name == "patchcore":
        return PatchCore(device=device, coreset_ratio=pc["coreset_ratio"],
                         max_train_images=pc["max_train_images"],
                         batch_size=pc["batch_size"])
    if name == "spade":
        return SPADE(device=device, k=cfg.SPADE["k"],
                     max_train_images=pc["max_train_images"],
                     batch_size=pc["batch_size"])
    if name == "padim":
        return PaDiM(device=device, n_dims=cfg.PADIM["n_dims"], eps=cfg.PADIM["eps"],
                     max_train_images=pc["max_train_images"],
                     batch_size=pc["batch_size"])
    if name in ("winclip", "winclip_plus"):
        return WinCLIP(model_id=cfg.CLIP_MODEL, device=device,
                       prompts=cfg.CLIP_ZS["prompts"],
                       image_size=cfg.CLIP_ZS["image_size"],
                       few_shot=(name == "winclip_plus"),
                       max_train_images=pc["max_train_images"],
                       batch_size=cfg.WINCLIP["batch_size"])
    if name == "clip_zs":
        return CLIPZS(model_id=cfg.CLIP_MODEL, device=device,
                      prompts=cfg.CLIP_ZS["prompts"],
                      image_size=cfg.CLIP_ZS["image_size"],
                      batch_size=cfg.CLIP_ZS["batch_size"])
    raise ValueError(f"unknown method: {name}")


def prompts_digest() -> str:
    """Digest of the CLIP prompt pairs, so that editing them invalidates stored cells."""
    text = json.dumps(cfg.CLIP_ZS["prompts"], sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _default(cls, arg: str):
    """Default of a constructor argument that build_method does not set."""
    return list(inspect.signature(cls.__init__).parameters[arg].default)


def method_params(name: str) -> dict:
    """The build_method settings that affect scores."""
    pc = cfg.PATCHCORE
    if name.startswith("dino_patchcore_"):
        return {"backbone": cfg.DINOV2_MODELS[name.removeprefix("dino_patchcore_")],
                "max_reference_images": pc["max_train_images"],
                "coreset_ratio": pc["coreset_ratio"],
                "image_size": pc["image_size"]}
    if name == "patchcore":
        return {"layers": _default(PatchCore, "layers"),
                "max_reference_images": pc["max_train_images"],
                "coreset_ratio": pc["coreset_ratio"]}
    if name == "spade":
        return {"layers": _default(SPADE, "layers"),
                "max_reference_images": pc["max_train_images"], "k": cfg.SPADE["k"]}
    if name == "padim":
        return {"layers": _default(PaDiM, "layers"),
                "max_reference_images": pc["max_train_images"],
                "n_dims": cfg.PADIM["n_dims"], "eps": cfg.PADIM["eps"]}
    if name == "winclip_plus":
        return {"model": cfg.CLIP_MODEL, "max_reference_images": pc["max_train_images"],
                "image_size": cfg.CLIP_ZS["image_size"], "prompts": prompts_digest()}
    if name in ("winclip", "clip_zs"):
        return {"model": cfg.CLIP_MODEL, "image_size": cfg.CLIP_ZS["image_size"],
                "prompts": prompts_digest()}
    raise ValueError(f"unknown method: {name}")


def cell_config(method: str, source: str, target: str, seed: int) -> dict:
    config = {"experiment": "benchmark", "method": method,
              "source": source, "target": target}
    if method not in REFERENCE_FREE:
        config["seed"] = seed
    return {**config, **method_params(method)}


def run(methods: list[str], datasets: list[str], seeds: list[int], device: str) -> None:
    raw = cfg.RESULTS_DIR / "raw"
    tests = {}

    def test_split(name):
        if name not in tests:
            tests[name] = build_dataset(name).test()
        return tests[name]

    for method_name in methods:
        print(f"\n=== {method_name} ===")
        if method_name in REFERENCE_FREE:
            banks = [("none", 0)]
        else:
            banks = [(src, seed) for src in datasets for seed in seeds]

        for src, seed in banks:
            pending = []
            for tgt in datasets:
                out_dir = raw / method_name / f"seed{seed}" / f"{src}__{tgt}"
                config = cell_config(method_name, src, tgt, seed)
                if not cell_done(out_dir, config):
                    pending.append((tgt, out_dir, config))
            if not pending:
                print(f"  [{src}, seed {seed}] all targets done")
                continue

            model = build_method(method_name, device)
            if method_name in REFERENCE_FREE:
                model.fit([])
            else:
                references = build_dataset(src).normal_train().image_paths
                print(f"  fitting on {src} (seed {seed})")
                model.fit(references, seed=seed)

            for tgt, out_dir, config in pending:
                test = test_split(tgt)
                labels = np.array(test.labels)
                scores = model.score(test.image_paths)
                metrics = image_level_metrics(scores, labels)
                save_cell(out_dir, config,
                          {"image_auroc": metrics.image_auroc,
                           "image_ap": metrics.image_ap},
                          device, scores=scores, labels=labels,
                          paths=test.image_paths)
                print(f"    {src} -> {tgt}: AUROC={metrics.image_auroc:.4f}  "
                      f"AP={metrics.image_ap:.4f}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--methods", nargs="+", choices=METHODS, default=PAPER_METHODS)
    p.add_argument("--datasets", nargs="+", choices=cfg.DATASETS, default=cfg.DATASETS)
    p.add_argument("--seeds", nargs="+", type=int, default=cfg.SEEDS,
                   help="reference-sampling seeds (ignored by reference-free methods)")
    p.add_argument("--device", default=None,
                   help="default: $DEVICE, else cuda, mps, cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.methods, args.datasets, args.seeds, pick_device(args.device))
