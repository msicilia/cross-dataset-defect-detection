"""Central configuration for all experiments."""
from pathlib import Path

# ── Dataset root directories ──────────────────────────────────────────────────
DATA_ROOT = Path("data")

DATASET_PATHS = {
    "mvtec":  DATA_ROOT / "mvtec_anomaly_detection",
    "sdnet":  DATA_ROOT / "sdnet2018",
    "vision": DATA_ROOT / "vision_dataset",
}

# MVTec categories used (flat-surface textures + metallic objects)
MVTEC_CATEGORIES = ["metal_nut", "screw", "tile", "wood", "grid"]

# VISION dataset — metallic/mechanical surface categories.
# Full list: Cable, Capacitor, Casting, Console, Cylinder, Electronics,
#            Groove, Hemisphere, Lens, PCB_1, PCB_2, Ring, Screw, Wood
VISION_CATEGORIES = ["Casting", "Ring", "Screw", "Cylinder"]

# ── Model identifiers ─────────────────────────────────────────────────────────
DINOV2_MODELS = {
    "small":  "facebook/dinov2-small",   # ViT-S/14
    "base":   "facebook/dinov2-base",    # ViT-B/14
    "large":  "facebook/dinov2-large",   # ViT-L/14
}
DINOV2_DEFAULT = "base"

CLIP_MODEL   = "openai/clip-vit-large-patch14"
WRNET_LAYERS = ("layer2", "layer3")          # WideResNet-50 layers for PatchCore

# ── Method hyperparameters ────────────────────────────────────────────────────
PATCHCORE = {
    "coreset_ratio":    0.01,   # fraction of patch features to keep in memory bank
    "max_train_images": 500,    # cap to avoid OOM on large datasets
    "image_size":       256,
    "batch_size":       32,
}

CLIP_ZS = {
    "image_size": 224,
    "batch_size": 64,
    # Prompt pairs per defect category: (positive, negative)
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

# ── Experiment settings ───────────────────────────────────────────────────────
SEEDS        = [0, 1, 2, 3, 4]
RESULTS_DIR  = Path("results")

DATASET_NAMES = ["mvtec", "sdnet", "vision"]
METHODS       = ["dino_patchcore_small", "dino_patchcore_base",
                 "dino_patchcore_large", "patchcore", "clip_zs"]
