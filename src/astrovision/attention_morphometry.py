"""
AstroVision — attention_morphometry.py
Morphométrie guidée par les attention maps DINOv2 (contribution originale).

Motivation scientifique :
    Les métriques CAS/Gini/M20 classiques utilisent un seuillage Otsu pour
    définir la région galactique. Cette approche est aveugle à la sémantique :
    elle réagit à la brillance, pas à la structure.

    Les attention maps du ViT DINOv2 (après fine-tuning) focalisent sur la
    structure galactique physique. En les utilisant comme masque, on obtient
    des métriques morphologiques plus robustes, notamment pour :
        - Galaxies edge-on (la luminosité de bord masque la structure interne)
        - Galaxies en fusion (multiples composantes)
        - Galaxies à faible surface brightness

Contribution originale :
    AttentionMorphometry remplace le masque Otsu par un masque attentionnel
    et calcule les métriques CAS/Gini/M20 sur ce masque. On compare ensuite
    les deux approches (Otsu vs Attention) par classe morphologique.

Usage :
    from astrovision.attention_morphometry import AttentionMorphometry

    am  = AttentionMorphometry(dino_model, device='cuda')
    df  = am.batch_compute(images, labels)
    fig = am.plot_comparison(df_otsu, df_attn)
"""

import warnings
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_fill_holes, label as nd_label

warnings.filterwarnings("ignore")

CLASS_NAMES = [
    "Disturbed", "Merging", "Round Smooth", "In-between Round Smooth",
    "Cigar Shaped Smooth", "Barred Spiral", "Unbarred Tight Spiral",
    "Unbarred Loose Spiral", "Edge-on without Bulge", "Edge-on with Bulge",
]
PALETTE = [
    "#6e40aa","#4c6edb","#23abd8","#1ac7c2","#1ddfa3",
    "#52f667","#aff05b","#e2b72f","#fb8a27","#f83e4b",
]
DARK = {
    "figure.facecolor":"#0d1117","axes.facecolor":"#161b22",
    "axes.edgecolor":"#30363d","axes.labelcolor":"#c9d1d9",
    "axes.titlecolor":"#f0f6fc","xtick.color":"#8b949e",
    "ytick.color":"#8b949e","text.color":"#c9d1d9",
    "grid.color":"#21262d","figure.dpi":150,
    "savefig.facecolor":"#0d1117","savefig.bbox":"tight",
}


# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACTION DES ATTENTION MAPS
# ═══════════════════════════════════════════════════════════════════════════════

class _AttentionHook:
    """Hook pour capturer les attention maps du dernier bloc ViT."""

    def __init__(self):
        self.attention = None

    def __call__(self, module, input, output):
        # output shape : (B, num_heads, N+1, N+1) — inclut le token CLS
        self.attention = output.detach()


def extract_attention_map(model:  nn.Module,
                           image:  torch.Tensor,
                           device: torch.device,
                           head_fusion: str = "mean") -> np.ndarray:
    """Extrait la carte d'attention du dernier bloc ViT-B/14.

    Implémente la méthode de Caron et al. (DINO 2021) :
    attention[0, :, 0, 1:] = attention du token CLS vers les patches.

    Args:
        model       : DINOv2 ViT-B/14 (doit avoir un attribut .blocks)
        image       : (1, 3, H, W) tensor normalisé
        device      : device PyTorch
        head_fusion : 'mean' | 'max' | 'min' pour la fusion des têtes

    Returns:
        (h_patch, w_patch) array float32 — carte d'attention 2D
    """
    hook  = _AttentionHook()
    # Accrocher au dernier bloc d'attention
    # Compatibilité : chercher l'attribut 'attn' dans le dernier bloc
    last_block = None
    if hasattr(model, "blocks"):
        last_block = model.blocks[-1]
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        last_block = model.encoder.layer[-1]
    else:
        raise AttributeError(
            "Structure de modèle non reconnue — "
            "fournir un ViT DINOv2 standard (model.blocks[-1].attn)")

    attn_module = None
    if hasattr(last_block, "attn"):
        attn_module = last_block.attn
    elif hasattr(last_block, "attention"):
        attn_module = last_block.attention

    if attn_module is None:
        raise AttributeError("Module d'attention non trouvé dans le dernier bloc")

    handle = attn_module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            _ = model(image.to(device))
    finally:
        handle.remove()

    if hook.attention is None:
        raise RuntimeError("Hook d'attention n'a pas capturé de données")

    attn = hook.attention  # (1, n_heads, N+1, N+1)
    # Attention du token CLS (index 0) vers les patches (indices 1:)
    cls_attn = attn[0, :, 0, 1:]  # (n_heads, N_patches)
    n_patches = cls_attn.shape[-1]
    h_patch   = int(n_patches ** 0.5)
    w_patch   = h_patch

    # Fusion des têtes
    if head_fusion == "mean":
        fused = cls_attn.mean(0)
    elif head_fusion == "max":
        fused = cls_attn.max(0).values
    elif head_fusion == "min":
        fused = cls_attn.min(0).values
    else:
        fused = cls_attn.mean(0)

    fused = fused.cpu().numpy().reshape(h_patch, w_patch)
    return fused


def attention_to_mask(attn_map:   np.ndarray,
                       threshold:  Optional[float] = None,
                       percentile: float = 60.0) -> np.ndarray:
    """Convertit une carte d'attention en masque binaire.

    Args:
        attn_map   : (h, w) attention map 2D
        threshold  : seuil fixe (si None → percentile)
        percentile : percentile pour le seuil adaptatif (défaut 60%)

    Returns:
        (h, w) masque binaire bool
    """
    if threshold is None:
        threshold = np.percentile(attn_map, percentile)
    mask = attn_map >= threshold
    mask = binary_fill_holes(mask)
    return mask


# ═══════════════════════════════════════════════════════════════════════════════
# MÉTRIQUES MORPHOLOGIQUES SUR MASQUE ATTENTIONNEL
# ═══════════════════════════════════════════════════════════════════════════════

def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return (0.299*image[...,0] + 0.587*image[...,1] + 0.114*image[...,2])
    return image.astype(np.float64)


def _concentration(img_gray: np.ndarray, mask: np.ndarray,
                    r_inner: float = 0.3, r_outer: float = 0.5) -> float:
    """Concentration C = 5 * log10(r_outer / r_inner) variant.

    Utilise la définition Conselice 2003 basée sur les pixels sorted.
    """
    pix = img_gray[mask].astype(np.float64)
    if len(pix) == 0:
        return np.nan
    pix_sorted = np.sort(pix)
    total      = pix_sorted.sum()
    if total <= 0:
        return np.nan
    cum     = np.cumsum(pix_sorted[::-1])
    n       = len(pix_sorted)
    n_inner = max(1, int(r_inner * n))
    n_outer = max(n_inner + 1, int(r_outer * n))
    c_inner = cum[n_inner - 1] / total
    c_outer = cum[n_outer - 1] / total
    if c_inner <= 0 or c_outer <= c_inner:
        return np.nan
    return 5.0 * np.log10(c_outer / c_inner)


def _asymmetry(img_gray: np.ndarray, mask: np.ndarray) -> float:
    """Asymétrie A (Conselice 2003) : |I - I_rot| / |I|."""
    img_m = img_gray * mask
    img_r = np.rot90(img_m, 2)
    denom = np.abs(img_m).sum()
    if denom <= 0:
        return np.nan
    return float(np.abs(img_m - img_r).sum() / denom)


def _smoothness(img_gray: np.ndarray, mask: np.ndarray,
                sigma: float = 0.25) -> float:
    """Smoothness S (Conselice 2003) : résidu de lissage."""
    from scipy.ndimage import gaussian_filter
    img_m    = img_gray * mask
    smoothed = gaussian_filter(img_m, sigma=max(1, sigma * img_gray.shape[0]))
    denom    = np.abs(img_m).sum()
    if denom <= 0:
        return np.nan
    return float(np.abs(img_m - smoothed).sum() / denom)


def _gini(img_gray: np.ndarray, mask: np.ndarray) -> float:
    """Coefficient de Gini G (Lotz et al. 2004)."""
    pix = img_gray[mask].astype(np.float64)
    if len(pix) == 0:
        return np.nan
    pix  = np.abs(pix)
    pix  = np.sort(pix)
    n    = len(pix)
    mean = pix.mean()
    if mean <= 0:
        return np.nan
    indices = np.arange(1, n + 1)
    return float((2.0 * (indices * pix).sum() - (n + 1) * pix.sum())
                 / (mean * n * (n - 1)))


def _m20(img_gray: np.ndarray, mask: np.ndarray) -> float:
    """Second moment du flux des 20% les plus brillants M20 (Lotz et al. 2004)."""
    from scipy.ndimage import center_of_mass
    pix_coords = np.argwhere(mask)
    if len(pix_coords) == 0:
        return np.nan
    pix_vals   = img_gray[mask].astype(np.float64)
    total_flux = pix_vals.sum()
    if total_flux <= 0:
        return np.nan

    # Centre de masse
    cy, cx = center_of_mass(img_gray * mask)

    # Moment total
    ys, xs = pix_coords[:, 0], pix_coords[:, 1]
    m_tot  = (pix_vals * ((ys - cy)**2 + (xs - cx)**2)).sum()
    if m_tot <= 0:
        return np.nan

    # Moment des 20% les plus brillants
    order    = np.argsort(-pix_vals)
    cum_flux = np.cumsum(pix_vals[order])
    top20    = cum_flux <= 0.2 * total_flux
    if top20.sum() == 0:
        top20[0] = True
    idx20     = order[top20]
    pix_top20 = pix_vals[idx20]
    ys20, xs20 = ys[idx20], xs[idx20]
    m20_val   = (pix_top20 * ((ys20 - cy)**2 + (xs20 - cx)**2)).sum()

    return float(np.log10(m20_val / m_tot)) if m_tot > 0 else np.nan


def compute_attn_morphometry(image: np.ndarray,
                              attn_map: np.ndarray,
                              percentile: float = 60.0) -> dict:
    """Calcule toutes les métriques sur le masque attentionnel.

    Args:
        image      : (H, W, 3) uint8 ou float image galaxie
        attn_map   : (h, w) attention map DINOv2 (sera resizée vers (H, W))
        percentile : seuil pour le masque attentionnel

    Returns:
        dict : C, A, S, Gini, M20, n_components, mask_coverage
    """
    H, W = image.shape[:2]

    # Upsampler la carte d'attention vers la résolution image
    attn_hr = F.interpolate(
        torch.tensor(attn_map[None, None], dtype=torch.float32),
        size=(H, W), mode="bilinear", align_corners=False,
    ).squeeze().numpy()

    mask = attention_to_mask(attn_hr, percentile=percentile)

    # Nombre de composantes connexes (utile pour Merging/Disturbed)
    labeled, n_comp = nd_label(mask)

    img_gray   = _to_gray(image)
    mask_coverage = float(mask.mean())

    return {
        "C":              _concentration(img_gray, mask),
        "A":              _asymmetry(img_gray, mask),
        "S":              _smoothness(img_gray, mask),
        "Gini":           _gini(img_gray, mask),
        "M20":            _m20(img_gray, mask),
        "n_components":   n_comp,
        "mask_coverage":  mask_coverage,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSE PRINCIPALE
# ═══════════════════════════════════════════════════════════════════════════════

class AttentionMorphometry:
    """Pipeline complet : extraction attention + calcul métriques morphologiques.

    Args:
        model       : DINOv2 ViT-B/14 fine-tuné (AstroVision)
        device      : device PyTorch
        head_fusion : 'mean' | 'max' | 'min'
        percentile  : seuil du masque attentionnel (défaut 60%)
        transform   : transforms PyTorch (normalisation ImageNet)
    """

    def __init__(self, model: nn.Module,
                 device: str = "cuda",
                 head_fusion: str = "mean",
                 percentile: float = 60.0,
                 transform=None):
        from torchvision import transforms as T
        self.model       = model.eval()
        self.device      = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model       = self.model.to(self.device)
        self.head_fusion = head_fusion
        self.percentile  = percentile
        self.transform   = transform or T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def compute_single(self, image: np.ndarray) -> dict:
        """Calcule les métriques pour une seule image.

        Args:
            image : (H, W, 3) uint8

        Returns:
            dict avec C, A, S, Gini, M20, n_components, mask_coverage,
                      attn_map (pour visualisation)
        """
        img_t  = self.transform(image).unsqueeze(0)
        attn   = extract_attention_map(self.model, img_t, self.device,
                                        self.head_fusion)
        result = compute_attn_morphometry(image, attn, self.percentile)
        result["attn_map"] = attn
        return result

    def batch_compute(self, images:  np.ndarray,
                       labels:  np.ndarray,
                       verbose: bool = True) -> "pd.DataFrame":
        """Calcule les métriques pour tout le dataset.

        Args:
            images : (N, H, W, 3) uint8
            labels : (N,) int labels

        Returns:
            DataFrame avec colonnes : g10_idx, label, class_name,
                                      C, A, S, Gini, M20,
                                      n_components, mask_coverage
        """
        import pandas as pd
        records = []
        n = len(images)

        for i, (img, lbl) in enumerate(zip(images, labels)):
            if verbose and i % 500 == 0:
                print(f"  {i:>5}/{n}  ({i/n*100:.1f}%)", end="\r")
            try:
                res = self.compute_single(img)
                res.pop("attn_map", None)
                res["g10_idx"]    = i
                res["label"]      = int(lbl)
                res["class_name"] = CLASS_NAMES[int(lbl)]
                records.append(res)
            except Exception as e:
                records.append({
                    "g10_idx": i, "label": int(lbl),
                    "class_name": CLASS_NAMES[int(lbl)],
                    "C": np.nan, "A": np.nan, "S": np.nan,
                    "Gini": np.nan, "M20": np.nan,
                    "n_components": 0, "mask_coverage": 0.0,
                })
        print()
        return pd.DataFrame(records)

    def visualize_attention(self, image:    np.ndarray,
                             label:    int,
                             save_path: Optional[str] = None) -> plt.Figure:
        """Visualise l'attention map et le masque sur une galaxie.

        Panels : Original | Attention (heatmap) | Masque binaire | Overlay
        """
        plt.rcParams.update(DARK)
        result    = self.compute_single(image)
        attn_map  = result["attn_map"]

        H, W = image.shape[:2]
        attn_hr = F.interpolate(
            torch.tensor(attn_map[None, None], dtype=torch.float32),
            size=(H, W), mode="bilinear", align_corners=False,
        ).squeeze().numpy()
        mask = attention_to_mask(attn_hr, percentile=self.percentile)

        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
        cls_name  = CLASS_NAMES[label] if 0 <= label < len(CLASS_NAMES) else str(label)
        fig.suptitle(
            f"Attention Morphometry — {cls_name}  |  "
            f"C={result['C']:.2f}  A={result['A']:.3f}  "
            f"Gini={result['Gini']:.3f}  M20={result['M20']:.2f}  "
            f"n_comp={result['n_components']}",
            fontsize=11, fontweight="bold",
        )

        # Original
        axes[0].imshow(image)
        axes[0].set_title("Original", fontsize=10)
        axes[0].axis("off")

        # Attention heatmap
        im = axes[1].imshow(attn_hr, cmap="inferno")
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        axes[1].set_title("DINOv2 Attention", fontsize=10)
        axes[1].axis("off")

        # Masque binaire
        axes[2].imshow(mask.astype(float), cmap="Blues", vmin=0, vmax=1)
        axes[2].set_title(f"Masque attention\n({self.percentile:.0f}% seuil)", fontsize=10)
        axes[2].axis("off")

        # Overlay attention sur image
        axes[3].imshow(image)
        axes[3].imshow(attn_hr, cmap="hot", alpha=0.55)
        axes[3].contour(mask, colors="#1ddfa3", linewidths=1.5)
        axes[3].set_title("Overlay", fontsize=10)
        axes[3].axis("off")

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARAISON OTSU vs ATTENTION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_morphometry_comparison(df_otsu: "pd.DataFrame",
                                 df_attn: "pd.DataFrame",
                                 save_path: Optional[str] = None) -> plt.Figure:
    """Compare les métriques Otsu vs Attention par classe.

    Montre si l'attention améliore la discrimination morphologique.

    Args:
        df_otsu : DataFrame morphométrie Otsu (colonnes C, A, Gini, M20, class_name)
        df_attn : DataFrame morphométrie Attention (mêmes colonnes)

    Returns:
        plt.Figure
    """
    import pandas as pd
    from scipy.stats import spearmanr

    plt.rcParams.update(DARK)
    metrics = [c for c in ["C", "A", "Gini", "M20"] if c in df_otsu.columns]
    n_met   = len(metrics)

    fig, axes = plt.subplots(2, n_met, figsize=(5 * n_met, 10))
    if n_met == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle("Morphométrie — Otsu vs Attention DINOv2",
                 fontsize=14, fontweight="bold")

    # Ligne 0 — médiane par classe pour Otsu
    for j, metric in enumerate(metrics):
        ax = axes[0, j]
        for df, color, label in [(df_otsu, "#23abd8", "Otsu"),
                                   (df_attn, "#f83e4b", "Attention")]:
            if metric not in df.columns:
                continue
            meds = (df.groupby("class_name")[metric]
                      .median()
                      .reindex(CLASS_NAMES)
                      .dropna())
            cls_colors = [PALETTE[CLASS_NAMES.index(c)] for c in meds.index]
            offset = -0.15 if label == "Otsu" else 0.15
            ax.barh([i + offset for i in range(len(meds))], meds.values,
                    height=0.3, color=cls_colors, alpha=0.7,
                    label=label, edgecolor="none")
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_yticklabels([c[:14] for c in CLASS_NAMES], fontsize=8)
        ax.set_xlabel(f"Médiane {metric}", fontsize=10)
        ax.set_title(f"{metric} — Otsu vs Attention", fontsize=11)
        if j == 0:
            ax.legend(fontsize=9)

    # Ligne 1 — scatter Otsu vs Attention (corrélation)
    merged = df_otsu.merge(df_attn, on="g10_idx", suffixes=("_otsu", "_attn"))
    for j, metric in enumerate(metrics):
        ax = axes[1, j]
        x_col = f"{metric}_otsu"
        y_col = f"{metric}_attn"
        if x_col not in merged.columns or y_col not in merged.columns:
            continue
        sub = merged.dropna(subset=[x_col, y_col])
        if "label_otsu" in sub.columns:
            colors = [PALETTE[int(l)] for l in sub["label_otsu"]]
        else:
            colors = "#23abd8"
        ax.scatter(sub[x_col], sub[y_col],
                   c=colors, s=6, alpha=0.4, rasterized=True)
        # Ligne y=x
        lims = [min(sub[x_col].min(), sub[y_col].min()),
                max(sub[x_col].max(), sub[y_col].max())]
        ax.plot(lims, lims, "--", color="#8b949e", lw=1.5)
        # Corrélation
        rho, pval = spearmanr(sub[x_col], sub[y_col])
        ax.set_xlabel(f"{metric} (Otsu)", fontsize=10)
        ax.set_ylabel(f"{metric} (Attention)", fontsize=10)
        ax.set_title(f"{metric} : Otsu vs Attention\nρ={rho:.3f}  p={pval:.3f}",
                     fontsize=11)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
