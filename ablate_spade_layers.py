"""SPADE's feature layers: PatchCore's two layers versus SPADE's three.

SPADE retrieves whole images, for which Cohen and Hoshen use deep features, so
it draws on layer2-layer4 while PatchCore uses layer2-layer3. This ablation
measures what restricting SPADE to PatchCore's two layers costs, so that the
choice of layers in the benchmark is backed by a measurement rather than an
assertion.

Both arms are measured in-distribution on MVTec AD, where the benchmark's
in-distribution scores are highest and a loss is therefore visible, over the
same reference draws (cfg.SEEDS) and the full test split. Everything except
the layer set is held fixed at the benchmark's configuration.

Results: results/raw/spade_layers/<arm>/seed<s>/mvtec__mvtec/{result.json,scores.npz}
Summary: results/spade_layers.json

Complete cells produced with the same configuration are skipped, and a model is
built only if its cell is missing. A complete cell with a different
configuration stops the run (see common.cell_done).

Usage:
    python ablate_spade_layers.py [--seeds S ...] [--device cuda|mps|cpu]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import config as cfg
from common import build_dataset, cell_done, pick_device, save_cell
from evaluate import image_level_metrics
from methods import SPADE

DATASET = "mvtec"
ARMS = {
    "patchcore_layers": ("layer2", "layer3"),
    "spade_layers": ("layer2", "layer3", "layer4"),
}


def cell_config(arm: str, layers: tuple[str, ...], seed: int) -> dict:
    return {
        "experiment": "spade_layers",
        "method": "spade",
        "arm": arm,
        "layers": list(layers),
        "source": DATASET,
        "target": DATASET,
        "seed": seed,
        "k": cfg.SPADE["k"],
        "max_reference_images": cfg.PATCHCORE["max_train_images"],
    }


def run(seeds: list[int], device: str) -> None:
    raw = cfg.RESULTS_DIR / "raw"
    ds = build_dataset(DATASET)
    references = ds.normal_train().image_paths
    test = ds.test()
    labels = np.array(test.labels)

    auroc: dict[str, dict[int, float]] = {arm: {} for arm in ARMS}
    for arm, layers in ARMS.items():
        print(f"\n=== {arm}: {'+'.join(layers)} ===")
        for seed in seeds:
            out_dir = raw / "spade_layers" / arm / f"seed{seed}" / f"{DATASET}__{DATASET}"
            config = cell_config(arm, layers, seed)
            if cell_done(out_dir, config):
                auroc[arm][seed] = json.loads(
                    (out_dir / "result.json").read_text())["image_auroc"]
                print(f"  seed {seed}: done, AUROC={auroc[arm][seed]:.4f}")
                continue
            model = SPADE(layers=layers, k=cfg.SPADE["k"],
                          max_train_images=cfg.PATCHCORE["max_train_images"],
                          batch_size=cfg.PATCHCORE["batch_size"], device=device)
            model.fit(references, seed=seed)
            scores = model.score(test.image_paths)
            metrics = image_level_metrics(scores, labels)
            save_cell(out_dir, config,
                      {"image_auroc": metrics.image_auroc,
                       "image_ap": metrics.image_ap},
                      device, scores=scores, labels=labels, paths=test.image_paths)
            auroc[arm][seed] = metrics.image_auroc
            print(f"  seed {seed}: AUROC={metrics.image_auroc:.4f}  AP={metrics.image_ap:.4f}")

    summarise(auroc, seeds, device)


def summarise(auroc: dict[str, dict[int, float]], seeds: list[int], device: str) -> None:
    full = [auroc["spade_layers"][s] for s in seeds]
    restricted = [auroc["patchcore_layers"][s] for s in seeds]
    # Paired over reference draws: both arms see the same reference images.
    loss = [f - r for f, r in zip(full, restricted)]
    record = {
        "dataset": DATASET,
        "device": device,
        "seeds": seeds,
        "arms": {arm: list(layers) for arm, layers in ARMS.items()},
        "in_dist_auroc": {
            "spade_layers": {"mean": float(np.mean(full)), "sd": float(np.std(full, ddof=1))},
            "patchcore_layers": {"mean": float(np.mean(restricted)),
                                 "sd": float(np.std(restricted, ddof=1))},
        },
        "loss_from_restricting_to_patchcore_layers": {
            "mean": float(np.mean(loss)), "sd": float(np.std(loss, ddof=1)),
        },
    }
    print(f"\n  layer2-layer4 (SPADE)      {record['in_dist_auroc']['spade_layers']['mean']:.4f}"
          f" +- {record['in_dist_auroc']['spade_layers']['sd']:.4f}")
    print(f"  layer2-layer3 (PatchCore)  "
          f"{record['in_dist_auroc']['patchcore_layers']['mean']:.4f}"
          f" +- {record['in_dist_auroc']['patchcore_layers']['sd']:.4f}")
    print(f"  cost of restricting        "
          f"{record['loss_from_restricting_to_patchcore_layers']['mean']:.4f}"
          f" +- {record['loss_from_restricting_to_patchcore_layers']['sd']:.4f}")

    out = cfg.RESULTS_DIR / "spade_layers.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(json.dumps(record, indent=2))
    os.replace(tmp, out)
    print(f"\n  written to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", nargs="+", type=int, default=cfg.SEEDS)
    ap.add_argument("--device", default=None,
                    help="default: $DEVICE, else cuda, mps, cpu")
    a = ap.parse_args()
    run(a.seeds, pick_device(a.device))


if __name__ == "__main__":
    main()
