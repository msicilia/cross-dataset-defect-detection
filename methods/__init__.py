"""Anomaly detectors of the benchmark."""
from .base import AnomalyMethod
from .clip_zs import CLIPZS
from .dino_patchcore import DINOPatchCore
from .padim import PaDiM
from .patchcore import PatchCore
from .spade import SPADE
from .winclip import WinCLIP

__all__ = ["DINOPatchCore", "PatchCore", "SPADE", "PaDiM", "WinCLIP", "CLIPZS", "AnomalyMethod"]
