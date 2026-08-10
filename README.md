# Cross-Dataset Defect Detection

A benchmark for **cross-dataset generalisation of unsupervised defect / anomaly
detection** with frozen vision foundation models. It measures how well a detector
built on one image domain transfers to a completely different one — no model is
ever trained or fine-tuned.

The central quantity is the **generalisation gap** Δ, the drop from
in-distribution to cross-domain AUROC. Across the detectors below, Δ tracks how
much a method depends on **reference images from the source domain**, rather than
which backbone or which architecture it uses.

## Methods

| Method | Backbone (frozen) | How normality is modelled | Reference images |
|---|---|---|---|
| **DINO-PatchCore** | DINOv2 ViT-S/B/L-14 | coreset of patch features, NN distance | yes |
| **PatchCore** | WideResNet-50 (ImageNet) | coreset of patch features, NN distance | yes |
| **SPADE** | WideResNet-50 (ImageNet) | whole-image descriptors, kNN | yes |
| **PaDiM** | WideResNet-50 (ImageNet) | per-position Gaussian, Mahalanobis | yes |
| **WinCLIP+** | CLIP ViT-L/14 | language + windowed reference features | yes |
| **WinCLIP** | CLIP ViT-L/14 | windowed language alignment | no |
| **CLIP-ZS** | CLIP ViT-L/14 | text-prompt scoring | no |

PaDiM is the parametric control: it uses reference images without storing
exemplars, which separates *storing exemplars* from *depending on reference
data*. WinCLIP and WinCLIP+ are the within-architecture control: same backbone,
same windows, same prompts, differing only in whether reference images are used.

Detectors requiring gradient-based training are out of scope by protocol, since
the study uses frozen backbones only.

## Datasets

| Dataset | Domain | Used | License |
|---|---|---|---|
| [SDNET2018](https://digitalcommons.usu.edu/all_datasets/48/) | concrete cracks | walls/decks/pavements | CC BY |
| [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) | industrial textures & objects | tile, wood, grid, metal_nut, screw | Research-only |
| [VISION](https://huggingface.co/datasets/VISION-Workshop/VISION-Datasets) | metallic parts | Casting, Ring, Screw, Cylinder | CC-BY-NC-4.0 |

Datasets are **not** redistributed here; download them from the links above into
`data/<name>/` (expected layouts are documented in the loaders under `datasets/`).

Categories within a dataset are **pooled** into a single memory bank, since a
dataset stands in for "a domain" and a deployment would not know in advance which
category it is looking at. This departs from the per-category protocol usual in
MVTec papers, so absolute numbers are not comparable with per-category results
published elsewhere; `analyze_per_category.py` reports both conventions.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python preprocess.py               # build split manifests + stats per dataset
python smoke_test.py               # one forward pass per method, checks wiring
```

Runs on CPU, CUDA, or Apple Silicon (MPS); the device is auto-detected in that
order of preference. Each run writes per-(source, target, seed) metrics to
`results/raw/<method>/seed<N>/<src>__<tgt>/result.json`, alongside the raw
per-image scores in `scores.npz` so that per-category breakdowns, bootstrap
intervals and score-distribution analyses are pure post-hoc operations.

`--skip-existing` makes any run resumable; a cell counts as complete only when
both its metrics and its raw scores are present.

## Running the experiments

```bash
# Main cross-dataset benchmark (leave-one-dataset-out, 5 seeds)
python run_experiments.py \
    --methods dino_patchcore_base patchcore spade padim \
              winclip winclip_plus clip_zs \
    --datasets sdnet mvtec vision --seeds 0 1 2 3 4

# Backbone ablation (ViT-S / B / L)
python run_experiments.py \
    --methods dino_patchcore_small dino_patchcore_base dino_patchcore_large \
    --datasets sdnet mvtec vision --seeds 0 1 2 3 4

python run_fewshot.py         # reference-set-size sweep
python run_mixed_source.py    # two-source memory banks
python run_contamination.py   # count-matched contamination test
python run_proportions.py     # fixed 500-image bank, varying VISION fraction
python run_mvtec_split.py     # MVTec textures vs. metallic-objects transfer
python run_confounders.py     # gap under greyscale / equalised / low-resolution
python run_localisation.py    # pixel AUROC and argmax-in-mask rate on MVTec
python measure_cost.py        # storage, fit time, latency, scaling
```

Each run is reproducible under a fixed seed (CLIP-ZS and WinCLIP are fully
deterministic, having no reference sampling).

## Analysis

```bash
python check_consistency.py     # audit the result tree before deriving numbers
python analyze_paradigm.py      # gap by how normality is modelled
python analyze_per_category.py  # pooled vs per-category vs macro AUROC
python analyze_significance.py  # bootstrap CIs and paired tests
python analyze_inversion.py     # why some transfers land below chance
python analyze_divergence.py    # MMD, Fréchet distance, t-SNE between domains
python analyze_domain_gap.py    # mean pairwise cosine distance, separability
python analyze_mvtec_split.py   # texture/object split within MVTec
python analyze_contamination.py
python analyze_experiments.py   # multi-seed, mixed-source and few-shot summaries
python make_tables.py           # LaTeX tables
python make_figures.py          # contamination-proportion and paradigm figures
```

`check_consistency.py` verifies that every `scores.npz` reproduces the
`result.json` beside it, that each experiment set has the cell count it should,
that no metric is out of range, and that no result file predates the code that
produced it.

## License

Released under the [MIT License](LICENSE). Datasets retain their own licenses.
