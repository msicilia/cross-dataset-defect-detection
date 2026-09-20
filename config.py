"""Configuration shared by all experiments."""
from pathlib import Path

# ── datasets ──────────────────────────────────────────────────────────────────
DATA_ROOT = Path("data")

DATASET_PATHS = {
    "sdnet":  DATA_ROOT / "sdnet2018",
    "mvtec":  DATA_ROOT / "mvtec_anomaly_detection",
    "vision": DATA_ROOT / "vision_dataset",
}
DATASETS = ["sdnet", "mvtec", "vision"]

# Two metallic objects (metal_nut, screw) and three flat textures.
MVTEC_CATEGORIES = ["metal_nut", "screw", "tile", "wood", "grid"]

# The VISION metallic subsets used.
VISION_CATEGORIES = ["Casting", "Ring", "Screw", "Cylinder"]

# ── models ────────────────────────────────────────────────────────────────────
DINOV2_MODELS = {
    "small": "facebook/dinov2-small",   # ViT-S/14
    "base":  "facebook/dinov2-base",    # ViT-B/14
    "large": "facebook/dinov2-large",   # ViT-L/14
}
CLIP_MODEL = "openai/clip-vit-large-patch14"

# ── detectors ─────────────────────────────────────────────────────────────────
PATCHCORE = {
    # PatchCore, DINO-PatchCore: fraction of all reference patches kept.
    "coreset_ratio":    0.01,
    # All reference-based detectors (PatchCore, DINO-PatchCore, SPADE, PaDiM,
    # WinCLIP+): reference images drawn per source bank.
    "max_train_images": 500,
    # DINO-PatchCore: square resize before the processor's crop; also the frame
    # of the VISION masks (vision_masks.py).
    "image_size":       256,
    # PatchCore, SPADE, PaDiM (WideResNet-50) and DINO-PatchCore.
    "batch_size":       32,
}

CLIP_ZS = {
    "image_size": 224,
    "batch_size": 64,
    # Prompt pairs per defect category: (positive, negative). The image score
    # is the largest positive-class probability over all pairs.
    "prompts": {
        "crack": [
            ("a photo of a concrete wall with a crack",
             "a photo of a smooth intact concrete wall"),
            ("structural damage: crack on a building surface",
             "undamaged building surface"),
            ("a hairline crack on concrete",
             "pristine concrete surface with no defects"),
        ],
        "spalling": [
            ("concrete surface with spalling and exposed aggregate",
             "smooth concrete surface"),
            ("deteriorated concrete: chunks falling off",
             "well-maintained concrete wall"),
        ],
        "corrosion": [
            ("corroded metal surface with rust",
             "clean unpainted metal surface"),
            ("rusty metal component with corrosion stains",
             "intact metal surface"),
        ],
        "staining": [
            ("concrete wall with humidity stains and efflorescence",
             "dry clean concrete wall"),
        ],
    },
}

# SPADE: k nearest reference images averaged for the image score (Cohen and
# Hoshen use k=50; capped at the reference set size).
SPADE = {"k": 50}

# PaDiM: n_dims random feature channels per patch position (the covariance of
# each position needs more reference images than channels); eps is the
# covariance regulariser as a fraction of the mean per-channel variance.
PADIM = {"n_dims": 200, "eps": 0.01}

# WinCLIP: each image expands to 1 + 4 + 9 = 14 windows, so the CLIP batch is
# 14 times this.
WINCLIP = {"batch_size": 8}

# ── experiments ───────────────────────────────────────────────────────────────
SEEDS = [0, 1, 2, 3, 4]
SUBSET_SEEDS = [0, 1, 2]     # mixed-source and reference-set-size experiments
CONFOUNDER_SEEDS = [0]       # confounder ablation
# Label-stratified cap on target test sets in the contamination, proportion,
# MVTec-split and confounder experiments; only SDNET2018 exceeds it.
MAX_TEST = 2000
RESULTS_DIR = Path("results")
