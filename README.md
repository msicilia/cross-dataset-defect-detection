# Cross-Dataset Defect Detection

A benchmark for **cross-dataset generalisation of unsupervised defect / anomaly
detection** with frozen vision foundation models. It measures how well a detector
built on one image domain transfers to a completely different one — no model is
ever trained or fine-tuned.

## Methods

| Method | Backbone (frozen) | Reference images |
|---|---|---|
| **DINO-PatchCore** | DINOv2 ViT-B/14 | normal images → memory bank, nearest-neighbour scoring |
| **PatchCore** | WideResNet-50 (ImageNet) | same machinery, different backbone |
| **CLIP-ZS** | CLIP ViT-L/14 | none → zero-shot text-prompt scoring |

## Datasets

| Dataset | Domain | Used | License |
|---|---|---|---|
| [SDNET2018](https://digitalcommons.usu.edu/all_datasets/48/) | concrete cracks | walls/decks/pavements | CC BY |
| [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) | industrial textures & objects | tile, wood, grid, metal_nut, screw | Research-only |
| [VISION](https://huggingface.co/datasets/VISION-Workshop/VISION-Datasets) | metallic parts | Casting, Ring, Screw, Cylinder | CC-BY-NC-4.0 |

Datasets are **not** redistributed here; download them from the links above into
`data/<name>/` (expected layouts are documented in the loaders under `datasets/`).

## Setup

Uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync                       # create .venv and install dependencies
uv run python preprocess.py   # build split manifests + stats under each dataset
```

Runs on CPU, CUDA, or Apple Silicon (MPS); the device is auto-detected.
Per-image AUROC results are written as JSON under `results/raw/`.

## Running the experiments

```bash
# Main cross-dataset benchmark (leave-one-dataset-out, 5 seeds)
uv run python run_experiments.py \
    --methods dino_patchcore_base patchcore clip_zs \
    --datasets sdnet mvtec vision --seeds 0 1 2 3 4

# Backbone ablation (ViT-S / B / L)
uv run python run_experiments.py \
    --methods dino_patchcore_small dino_patchcore_base dino_patchcore_large \
    --datasets sdnet mvtec vision --seeds 0

uv run python run_fewshot.py         # reference-set-size sweep
uv run python run_mixed_source.py    # two-source memory banks
uv run python run_contamination.py   # count-matched contamination test
uv run python run_mvtec_split.py     # MVTec textures vs. metallic-objects transfer
```

Each run is reproducible under a fixed seed (CLIP-ZS is fully deterministic) and
saves per-(source, target, seed) AUROC to `results/raw/`.

## License

Released under the [MIT License](LICENSE). Datasets retain their own licenses.
