"""AstroVision — Classification morphologique de galaxies."""

from .data_loader import load_galaxy10_splits, CLASS_NAMES, download_galaxy10
from .models import SimpleCNN, EfficientNetGalaxy, build_model
from .trainer import train
from .gradcam import GradCAM, plot_gradcam_grid

__all__ = [
    "load_galaxy10_splits", "CLASS_NAMES", "download_galaxy10",
    "SimpleCNN", "EfficientNetGalaxy", "build_model",
    "train",
    "GradCAM", "plot_gradcam_grid",
]
