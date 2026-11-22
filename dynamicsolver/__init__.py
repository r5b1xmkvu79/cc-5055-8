"""Dynamic-scene shot solver: layered motion decomposition package."""
from .decomposition import BACKGROUND, BOX, DEFERRED, DecompositionResult, decompose
from .priors import Priors
from .synth import SCENARIOS

__all__ = ["BACKGROUND", "BOX", "DEFERRED", "DecompositionResult", "decompose",
           "Priors", "SCENARIOS"]
