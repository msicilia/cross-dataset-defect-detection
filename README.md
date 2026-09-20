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
