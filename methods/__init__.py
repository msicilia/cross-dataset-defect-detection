from .dino_patchcore import DINOPatchCore
from .patchcore import PatchCore
from .spade import SPADE
from .padim import PaDiM
from .winclip import WinCLIP
from .clip_zs import CLIPZS
from .base import AnomalyMethod

__all__ = ["DINOPatchCore", "PatchCore", "SPADE", "PaDiM", "WinCLIP", "CLIPZS",
           "AnomalyMethod"]
