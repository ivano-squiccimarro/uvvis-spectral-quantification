"""UV-Vis quantification in interfering matrices. Concentrations are in uM."""

from .model import SpectralPLS
from .preprocessing import SpectralConfig

__version__ = "0.1.0"
__all__ = ["SpectralPLS", "SpectralConfig"]
