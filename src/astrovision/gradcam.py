"""
AstroVision - Grad-CAM (Gradient-weighted Class Activation Mapping)

Référence : Selvaraju et al., 2017 — https://arxiv.org/abs/1610.02391
"""

from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FIGURES_DIR = Path(__file__).parent.parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "Disturbed", "Merging", "Round Smooth", "In-between Round Smooth",
    "Cigar Shaped Smooth", "Barred Spiral", "Unbarred Tight Spiral",
    "Unbarred Loose Spiral", "Edge-on without Bulge", "Edge-on with Bulge",
]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])


# ── GradCAM ───────────────────────────────────────────────────────────────────
class GradCAM:
    """Calcule les cartes d'activation Grad-CAM pour une couche cible.

    Usage:
        gcam = GradCAM(model, model.gradcam_target)
        heatmap, pred_idx = gcam.compute(img_tensor)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self._gradients: torch.Tensor | None = None
        self._activations: torch.Tensor | None = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _, __, output):
        self._activations = output.detach()

    def _save_gradient(self, _, __, grad_output):
        self._gradients = grad_output[0].detach()

    def compute(
        self,
        img_tensor: torch.Tensor,
        class_idx: int | None = None,
    ) -> tuple[np.ndarray, int]:
        """Calcule la heatmap Grad-CAM.

        Args:
            img_tensor : (1, C, H, W) sur le device du modèle.
            class_idx  : classe cible. Si None, utilise la classe prédite.

        Returns:
            heatmap (H, W) normalisée [0, 1], class_idx prédit.
        """
        self.model.eval()
        logits = self.model(img_tensor)

        if class_idx is None:
            class_idx = int(logits.argmax(1).item())

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Pondération des activations par les gradients moyennés
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Redimensionnement à la taille de l'image d'entrée
        cam = F.interpolate(cam, size=img_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalisation [0, 1]
        vmin, vmax = cam.min(), cam.max()
        cam = (cam - vmin) / (vmax - vmin + 1e-8)

        return cam, class_idx


# ── Visualisation ─────────────────────────────────────────────────────────────
def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Inverse la normalisation ImageNet → image RGB [0, 1]."""
    img = tensor.clone().cpu()
    for c in range(3):
        img[c] = img[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
    return img.permute(1, 2, 0).clamp(0, 1).numpy()


def plot_gradcam_grid(
    model: nn.Module,
    gradcam: GradCAM,
    images: torch.Tensor,
    labels: torch.Tensor,
    device,
    n_samples: int = 8,
    alpha: float = 0.45,
    filename: str = "gradcam_grid.png",
) -> plt.Figure:
    """Grille : image originale | heatmap | overlay.

    Args:
        n_samples : nombre de lignes dans la grille.
        alpha     : opacité de la heatmap sur l'overlay.
        filename  : nom du fichier de sortie dans figures/.
    """
    model.eval()
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, n_samples * 3.5), dpi=120)
    fig.suptitle("Grad-CAM — AstroVision", fontsize=14, fontweight="bold", y=1.01)

    cols = ["Image originale", "Heatmap Grad-CAM", "Overlay (prédiction)"]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=9, fontweight="bold")

    for i in range(n_samples):
        img_t = images[i:i+1].to(device)
        true_lbl = int(labels[i].item())

        img_np = _denormalize(images[i])
        cam, pred_idx = gradcam.compute(img_t)

        heatmap = cm.jet(cam)[..., :3]
        overlay = (1 - alpha) * img_np + alpha * heatmap
        overlay = np.clip(overlay, 0, 1)

        correct = "✓" if pred_idx == true_lbl else "✗"

        axes[i, 0].imshow(img_np);          axes[i, 0].set_ylabel(CLASS_NAMES[true_lbl], fontsize=7)
        axes[i, 1].imshow(cam, cmap="jet")
        axes[i, 2].imshow(overlay);         axes[i, 2].set_title(
            f"{correct} {CLASS_NAMES[pred_idx]}", fontsize=7,
            color="green" if pred_idx == true_lbl else "red"
        )

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    save_path = FIGURES_DIR / filename
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    print(f"✓ Figure sauvegardée : {save_path}")
    return fig
