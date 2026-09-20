# Cross-dataset defect detection benchmark

Code for a benchmark of cross-dataset generalisation in unsupervised visual
defect detection. Seven detectors (PatchCore, DINO-PatchCore, SPADE, PaDiM,
WinCLIP, WinCLIP+ and CLIP-ZS) model normality from defect-free reference images
of one source dataset, or use no reference images, and are evaluated on the test
splits of SDNET2018, MVTec AD and VISION. Further experiments vary the backbone,
the composition and size of the reference set, the MVTec AD categories, and the
image preprocessing, and measure localisation and cost.

## Environment

Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All scripts are run from this directory with the `python` of the active
environment. Pretrained weights (DINOv2, CLIP ViT-L/14, WideResNet-50) are
downloaded on first use into the Hugging Face and torch caches; once cached,
`HF_HUB_OFFLINE=1` avoids network access.

## Data

```bash
bash setup_data.sh
```

The script downloads VISION from Hugging Face, prints download instructions for
SDNET2018 and MVTec AD (both manual), and checks the file counts of all three
under `data/`. Dataset locations and the categories used are set in
`config.py`.

## Reproduction

```bash
./reproduce.sh            # all stages
./reproduce.sh analysis   # one or more stages
```

Stages, in order: `data` (split manifests, VISION masks), `preflight` (smoke
test on CPU), `benchmark`, `experiments`, `check` (result integrity),
`analysis` (integrity check, statistics, figures) and `tables`. Each step
appends to `results/logs/<step>.log`. Per-cell results are written to
`results/raw/<result set>/.../{result.json,scores.npz}`; complete cells are not
recomputed, so an interrupted run resumes at the first incomplete result cell.

## Hardware

`DEVICE=auto` (default) uses CUDA if available, then Apple MPS, then CPU;
`DEVICE=cuda|mps|cpu ./reproduce.sh` forces one. The reported results were
computed on Apple M-series GPUs through MPS. Other devices or library versions
can change the last reported digit.

Backbone forward passes run on the selected device. Image decoding and resizing
run on a thread pool, and the greedy coreset selection of PatchCore and
DINO-PatchCore runs on the CPU with its distance updates split across threads;
neither changes the computed values. Nearest-neighbour distances are computed on
the CPU. On macOS, `reproduce.sh` runs under `caffeinate` so the machine does
not sleep during long stages.

## Paper tables and figures

Every numeric table is written by `make_tables.py` to
`results/tables/<name>.tex` (the complete tabular environment, to be `\input`
in the paper) from the JSON outputs of the analysis scripts. The defect category
mapping table holds no results and is static.

| Paper table | `results/tables/` | Inputs (under `results/`) |
|---|---|---|
| Datasets | `datasets.tex` | dataset directories under `data/` |
| Cross-dataset AUROC | `transfer_auroc.tex` | `significance.json` |
| Generalisation gap | `gap.tex` | `significance.json`, `paradigm.json` |
| Distributional distance | `divergence.tex` | `divergence.json` |
| Pooled vs macro-averaged AUROC | `pooling.tex` | `per_category.json` |
| Per-category AUROC | `percat.tex` | `per_category.json` |
| Localisation | `localisation.tex` | `localisation.json` |
| Backbone ablation | `ablation_backbone.tex` | `significance.json` |
| Mixed-source banks | `mixed_source.tex` | `mixed_fewshot.json`, `significance.json` |
| Cost | `cost.tex` | `cost.json` |

The inputs are written by `analyze_significance.py` (`significance.json`),
`analyze_divergence.py` (`divergence.json`), `analyze_per_category.py`
(`per_category.json`), `analyze_localisation.py` (`localisation.json`),
`analyze_mixed_fewshot.py` (`mixed_fewshot.json`), `analyze_inversion.py`
(`inversion.json`), `analyze_mvtec_split.py` (`mvtec_split.json`),
`analyze_contamination.py` (`contamination.json`), `analyze_confounders.py`
(`confounders.json`), `make_figures.py` (`paradigm.json`, `proportions.json`)
and `measure_cost.py` (`cost.json`). The values quoted in the text are written,
unrounded, to `results/tables/numbers.json`.

To compare the generated tables with those in the manuscript, given a path to
its LaTeX source (not part of this repository):

```bash
python make_tables.py --compare /path/to/main.tex
```

This prints the differing numbers of each table; `reproduce.sh` does not run it.

| Paper figure | Script | Output |
|---|---|---|
| Dataset samples | `make_sample_montage.py` | `results/figures/dataset_samples.pdf` |
| t-SNE of domains | `analyze_divergence.py` | `results/figures/tsne_domains.pdf` |
| Gap by detector | `make_figures.py` | `results/figures/paradigm.pdf` |
| MVTec split | `analyze_mvtec_split.py` | `results/figures/mvtec_split.pdf` |
| Contamination | `analyze_contamination.py` | `results/figures/contamination.pdf` |
| Contamination proportions | `make_figures.py` | `results/figures/proportions.pdf` |
| Reference-set size | `analyze_mixed_fewshot.py` | `results/figures/fewshot_curve.pdf` |
