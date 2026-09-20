"""Preflight check before the full runs.

Loads each dataset's splits, then fits every detector of the benchmark on a
few SDNET2018 reference images and scores a few test images of each label,
checking that scores are finite and one per image. Exits non-zero on any
failure.

Usage:
  python smoke_test.py [--device cpu] [--n-ref 8] [--n-test 4]
"""
from __future__ import annotations

import argparse
import sys
import traceback

import numpy as np

import config as cfg
from common import build_dataset, pick_device, random_subset
from evaluate import image_level_metrics
from methods import CLIPZS, DINOPatchCore, PaDiM, PatchCore, SPADE, WinCLIP


def build(name: str, device: str, n_ref: int):
    prompts = cfg.CLIP_ZS["prompts"]
    if name == "patchcore":
        return PatchCore(coreset_ratio=cfg.PATCHCORE["coreset_ratio"], batch_size=4, device=device)
    if name == "dino_patchcore_base":
        return DINOPatchCore(backbone=cfg.DINOV2_MODELS["base"],
                             coreset_ratio=cfg.PATCHCORE["coreset_ratio"],
                             image_size=cfg.PATCHCORE["image_size"], batch_size=4, device=device)
    if name == "spade":
        return SPADE(k=cfg.SPADE["k"], batch_size=4, device=device)
    if name == "padim":
        # The covariance needs more reference images than channels.
        return PaDiM(n_dims=min(cfg.PADIM["n_dims"], n_ref - 1), eps=cfg.PADIM["eps"],
                     batch_size=4, device=device)
    if name in ("winclip", "winclip_plus"):
        return WinCLIP(prompts=prompts, model_id=cfg.CLIP_MODEL,
                       image_size=cfg.CLIP_ZS["image_size"],
                       few_shot=name == "winclip_plus", batch_size=2, device=device)
    if name == "clip_zs":
        return CLIPZS(prompts=prompts, model_id=cfg.CLIP_MODEL,
                      image_size=cfg.CLIP_ZS["image_size"], batch_size=4, device=device)
    raise ValueError(name)


DETECTORS = ["patchcore", "dino_patchcore_base", "spade", "padim",
             "winclip", "winclip_plus", "clip_zs"]


def check_datasets() -> list[str]:
    failures = []
    for name in cfg.DATASETS:
        try:
            ds = build_dataset(name)
            train, test = ds.normal_train(), ds.test()
            assert train.image_paths and test.image_paths, "empty split"
            assert set(train.labels) == {0}, "reference set is not all normal"
            assert set(test.labels) == {0, 1}, "test split lacks a label"
            assert all(p.exists() for p in train.image_paths[:20] + test.image_paths[:20]), \
                "missing image files"
            print(f"  {name}: {len(train.image_paths)} reference, {len(test.image_paths)} test "
                  f"({sum(test.labels)} defective)")
        except Exception as e:
            print(f"  {name}: FAIL {e!r}")
            failures.append(name)
    return failures


def check_detectors(device: str, n_ref: int, n_test: int) -> list[str]:
    ds = build_dataset("sdnet")
    rng = np.random.default_rng(0)
    reference = random_subset(ds.normal_train().image_paths, n_ref, rng)
    test = ds.test()
    normal = [p for p, y in zip(test.image_paths, test.labels) if y == 0][: n_test // 2]
    defective = [p for p, y in zip(test.image_paths, test.labels) if y == 1][: n_test // 2]
    paths, labels = normal + defective, [0] * len(normal) + [1] * len(defective)

    failures = []
    for name in DETECTORS:
        try:
            method = build(name, device, len(reference))
            method.fit(reference, seed=0)
            scores = np.asarray(method.score(paths))
            assert scores.shape == (len(paths),), f"shape {scores.shape}"
            assert np.isfinite(scores).all(), "non-finite scores"
            m = image_level_metrics(scores, labels)
            print(f"  {name}: ok (AUROC {m.image_auroc:.2f} on {len(paths)} images)")
        except Exception:
            traceback.print_exc()
            print(f"  {name}: FAIL")
            failures.append(name)
    return failures


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--device", default="cpu",
                   help="cpu (default), cuda, mps or auto (see common.pick_device)")
    p.add_argument("--n-ref", type=int, default=8)
    p.add_argument("--n-test", type=int, default=4)
    args = p.parse_args()
    device = pick_device(args.device)

    print("datasets")
    failures = check_datasets()
    if "sdnet" in failures:
        sys.exit("smoke test failed: sdnet")
    print(f"detectors on {device}")
    failures += check_detectors(device, args.n_ref, args.n_test)
    if failures:
        sys.exit(f"smoke test failed: {', '.join(failures)}")
    print("smoke test passed")


if __name__ == "__main__":
    main()
