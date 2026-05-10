"""
AstroVision - Architectures de modèles

SimpleCNN  : CNN léger entraîné from scratch (baseline)
EfficientNetGalaxy : EfficientNet-B0 fine-tuné depuis ImageNet
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

NUM_CLASSES = 10


# ── SimpleCNN ─────────────────────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    """CNN 4-blocs from scratch.

    Input  : (B, 3, 256, 256)
    Output : (B, NUM_CLASSES)

    Attribut gradcam_target : couche cible pour Grad-CAM.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.4):
        super().__init__()

        def conv_block(in_c, out_c, drop2d=0.1):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(drop2d),
            )

        self.features = nn.Sequential(
            conv_block(3,   32,  0.10),   # → 128×128
            conv_block(32,  64,  0.10),   # →  64×64
            conv_block(64,  128, 0.20),   # →  32×32
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),              # →  16×16
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        # Couche cible Grad-CAM : dernier conv (avant le pool global)
        self.gradcam_target = self.features[-3]  # Conv2d(128→256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ── EfficientNetGalaxy ────────────────────────────────────────────────────────
class EfficientNetGalaxy(nn.Module):
    """EfficientNet-B0 pré-entraîné ImageNet, tête remplacée pour Galaxy10.

    Input  : (B, 3, 256, 256)  — redimensionné automatiquement par EfficientNet
    Output : (B, NUM_CLASSES)

    Args:
        freeze_backbone: si True, gèle tout sauf la tête (feature extraction).
    """

    def __init__(self, num_classes: int = NUM_CLASSES, freeze_backbone: bool = False):
        super().__init__()

        self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3, inplace=True),
            nn.Linear(in_features, num_classes),
        )

        # Couche cible Grad-CAM : dernier bloc de features
        self.gradcam_target = self.backbone.features[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ── Factory ───────────────────────────────────────────────────────────────────
def build_model(name: str = "simplecnn", **kwargs) -> nn.Module:
    """Instancie un modèle par nom.

    Args:
        name: "simplecnn" | "efficientnet"
    """
    registry = {
        "simplecnn":   SimpleCNN,
        "efficientnet": EfficientNetGalaxy,
    }
    if name not in registry:
        raise ValueError(f"Modèle inconnu '{name}'. Options : {list(registry)}")
    return registry[name](**kwargs)
