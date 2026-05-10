"""
AstroVision — segmenter.py
Segmentation morphologique de galaxies par trois approches complémentaires.

1. OtsuSegmenter       — seuillage rapide (baseline, numpy only)
2. DINOv2PatchSegmenter — segmentation self-supervised (patch tokens ViT-B/14)
3. SAMSegmenter        — Segment Anything Model (optionnel, ~2.5 GB)

Usage :
    from astrovision.segmenter import OtsuSegmenter, DINOv2PatchSegmenter

    # Baseline rapide
    seg = OtsuSegmenter()
    masks = seg.segment(image_uint8)   # dict : {0:bg, 1:disk, 2:bulge}

    # Self-supervised DINOv2
    dseg = DINOv2PatchSegmenter(dino_model, device='cuda')
    seg_map = dseg.segment(image_uint8, n_clusters=4)
"""

import warnings
from pathlib import Path
from typing import Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import (binary_fill_holes, center_of_mass,
                            gaussian_filter, label)

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

SEG_COLORS = np.array([
    [13,  17,  23],   # 0 — fond (#0d1117)
    [78, 105, 219],   # 1 — disque (#4c6edb)
    [26, 199, 194],   # 2 — bulbe (#1ac7c2)
    [242, 130,  39],  # 3 — bras / structures (#fb8a27)
    [248,  62,  75],  # 4 — noyaux multiples / fusion (#f83e4b)
], dtype=np.uint8)


# ── Utilitaires ────────────────────────────────────────────────────────────────
def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return (0.299*image[...,0] + 0.587*image[...,1]
                + 0.114*image[...,2]).astype(np.float32)
    return image.astype(np.float32)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, n = label(mask)
    if n == 0:
        return mask
    sizes = [np.sum(labeled == i) for i in range(1, n+1)]
    return labeled == (np.argmax(sizes) + 1)


def colorize_segmap(seg: np.ndarray, n_classes: int = None) -> np.ndarray:
    """Convertit un mask de classes en image RGB colorée."""
    n = n_classes or (seg.max() + 1)
    h, w = seg.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(min(n, len(SEG_COLORS))):
        rgb[seg == c] = SEG_COLORS[c]
    return rgb


# ══════════════════════════════════════════════════════════════════════════════
class OtsuSegmenter:
    """Segmentation en 3 classes (fond / disque / bulbe) par seuillage.

    Labels :
        0 — fond de ciel
        1 — disque galactique (entre seuil Otsu et 80e percentile)
        2 — bulbe / noyau brillant (>80e percentile des pixels de la galaxie)

    C'est la méthode utilisée pour générer les pseudo-labels du U-Net.
    """

    def __init__(self, bg_pct: float = 5.0, bulge_pct: float = 80.0):
        self.bg_pct    = bg_pct
        self.bulge_pct = bulge_pct

    def segment(self, image: np.ndarray) -> dict:
        """Retourne un dict {mask, seg_map, n_components}.

        seg_map : (H, W) int avec classes 0/1/2
        """
        gray = _to_gray(image)

        # Soustraction fond
        bg = np.percentile(gray, self.bg_pct)
        img_sub = np.maximum(gray - bg, 0.0)

        # Masque galaxie (Otsu)
        try:
            from skimage.filters import threshold_otsu
            thresh_galaxy = threshold_otsu(img_sub)
        except ImportError:
            thresh_galaxy = img_sub.mean() * 0.5
        galaxy_mask = img_sub > thresh_galaxy
        galaxy_mask = binary_fill_holes(galaxy_mask)
        galaxy_mask = _largest_component(galaxy_mask)

        # Masque bulbe
        galaxy_pix = img_sub[galaxy_mask]
        if len(galaxy_pix) > 0:
            thresh_bulge = np.percentile(galaxy_pix, self.bulge_pct)
            bulge_mask = (img_sub > thresh_bulge) & galaxy_mask
        else:
            bulge_mask = np.zeros_like(galaxy_mask)

        # Assemblage du mask de segmentation
        seg_map = np.zeros(gray.shape, dtype=np.uint8)
        seg_map[galaxy_mask] = 1   # disque
        seg_map[bulge_mask]  = 2   # bulbe

        return {
            "seg_map":        seg_map,
            "galaxy_mask":    galaxy_mask,
            "bulge_mask":     bulge_mask,
            "disk_mask":      galaxy_mask & ~bulge_mask,
            "n_galaxy_pix":   int(galaxy_mask.sum()),
            "bulge_fraction": float(bulge_mask.sum() / galaxy_mask.sum())
                              if galaxy_mask.sum() > 0 else 0.0,
        }

    def batch_segment(self, images: np.ndarray,
                      verbose: bool = True) -> np.ndarray:
        """Segmente un batch → ndarray (N, H, W) uint8."""
        n = len(images)
        h, w = images[0].shape[:2]
        result = np.zeros((n, h, w), dtype=np.uint8)
        for i, img in enumerate(images):
            if verbose and i % 500 == 0:
                print(f"  Otsu segmentation {i}/{n}", end="\r")
            result[i] = self.segment(img)["seg_map"]
        if verbose:
            print(f"  Otsu segmentation {n}/{n} ✓   ")
        return result


# ══════════════════════════════════════════════════════════════════════════════
class DINOv2PatchSegmenter:
    """Segmentation self-supervised via les patch tokens de DINOv2.

    Principe (Caron+ 2021, Oquab+ 2023) :
        - DINOv2 ViT-B/14 divise l'image en 16×16 = 256 patches de 14×14 px
        - Chaque patch → vecteur de features 768-dim
        - k-means sur ces 256 vecteurs → clusters morphologiques
        - Cluster 0 = fond (plus basse énergie), clusters 1+ = structures

    Aucun label requis — segmentation 100% non-supervisée.
    """

    def __init__(self, model, device: str = "cuda"):
        import torch
        self.model  = model.eval()
        self.device = torch.device(device)
        self._build_transform()

    def _build_transform(self):
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225]),
        ])

    def _get_patch_features(self, image: np.ndarray) -> np.ndarray:
        """Retourne les features de patch DINOv2 — robuste à toutes les versions."""
        import torch
        img_t = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out   = self.model.get_intermediate_layers(img_t, n=1)[0]
            tokens = out[0].cpu().numpy()   # (N, 768) — N varie selon la version

        n = len(tokens)

        # Cas 1 : N est un carré parfait → patch tokens purs (pas de CLS inclus)
        hw = int(n ** 0.5)
        if hw * hw == n:
            return tokens                   # ex: 256 → 16×16 ✓

        # Cas 2 : N-1 est un carré parfait → CLS au début, on le saute
        hw2 = int((n - 1) ** 0.5)
        if hw2 * hw2 == n - 1:
            return tokens[1:]               # ex: 257 → 256 patches ✓

        # Cas 3 : fallback — garder les hw*hw derniers tokens
        # (couvre les versions avec register_tokens ou autre overhead)
        hw = int(n ** 0.5)                  # ex: 255 → hw=15 → 225 patches
        return tokens[n - hw * hw:]

    def segment(self, image: np.ndarray, n_clusters: int = 4) -> dict:
        """Segmente une image via k-means sur les patch features.

        Returns:
            seg_map : (224, 224) int — labels de 0 à n_clusters-1
            patch_labels : (16, 16) int
            bg_cluster : int — cluster identifié comme fond de ciel
        """
        import torch
        import torch.nn.functional as F
        from sklearn.cluster import KMeans

        patch_feats = self._get_patch_features(image)  # (256, 768)

        # k-means
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels_km = km.fit_predict(patch_feats)  # (256,)

        # Reshape 16×16
        hw  = int(patch_feats.shape[0] ** 0.5)
        seg_patch = labels_km.reshape(hw, hw)  # (16, 16)

        # Upsample 16×16 → 224×224 (nearest neighbor)
        seg_up = torch.from_numpy(seg_patch).float().unsqueeze(0).unsqueeze(0)
        seg_up = F.interpolate(seg_up, size=(224, 224), mode="nearest")
        seg_map = seg_up.squeeze().numpy().astype(np.uint8)

        # Identifier le cluster fond = celui avec le minimum d'énergie moyenne
        gray = _to_gray(
            np.array(
                __import__("PIL").Image.fromarray(image).resize((224,224))
            )
        )
        cluster_energy = [
            gray[seg_map == c].mean() if (seg_map == c).any() else 9999
            for c in range(n_clusters)
        ]
        bg_cluster = int(np.argmin(cluster_energy))

        # Réordonner : 0=fond, reste par énergie croissante
        order = [bg_cluster] + [c for c in np.argsort(cluster_energy) if c != bg_cluster]
        remap = {old: new for new, old in enumerate(order)}
        seg_remap = np.vectorize(remap.get)(seg_map).astype(np.uint8)

        return {
            "seg_map":     seg_remap,
            "patch_labels": seg_patch,
            "patch_feats":  patch_feats,
            "bg_cluster":   0,
            "n_clusters":   n_clusters,
        }

    def segment_batch_gallery(self, images: np.ndarray,
                               labels: np.ndarray,
                               n_per_class: int = 3,
                               n_clusters: int = 4,
                               save_path: str = None):
        """Grille de segmentation DINOv2 : une ligne par classe."""
        plt.rcParams.update(DARK)
        n_cls = len(CLASS_NAMES)
        fig, axes = plt.subplots(n_cls, n_per_class * 3,
                                 figsize=(n_per_class * 9, n_cls * 3.2),
                                 dpi=100)
        fig.patch.set_facecolor("#0d1117")
        np.random.seed(42)

        for row, cls_idx in enumerate(range(n_cls)):
            cls_imgs = images[labels == cls_idx]
            sample   = cls_imgs[np.random.choice(len(cls_imgs),
                                                  min(n_per_class, len(cls_imgs)),
                                                  replace=False)]
            for col, img in enumerate(sample):
                try:
                    res = self.segment(img, n_clusters=n_clusters)
                    seg_rgb = colorize_segmap(res["seg_map"], n_clusters)
                    overlay = np.clip(
                        0.5 * np.array(
                            __import__("PIL").Image.fromarray(img).resize((224,224))
                        ).astype(np.float32)/255. + 0.5 * seg_rgb.astype(np.float32)/255.,
                        0, 1)
                except Exception:
                    seg_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
                    overlay = np.zeros((224, 224, 3))

                img_small = np.array(
                    __import__("PIL").Image.fromarray(img).resize((224,224))
                )
                base_col = col * 3
                for j, (data, ttl) in enumerate([
                    (img_small, "Original"),
                    (seg_rgb,   "DINOv2 seg"),
                    (overlay,   "Overlay"),
                ]):
                    ax = axes[row, base_col + j]
                    ax.imshow(data); ax.axis("off")
                    if row == 0:
                        ax.set_title(ttl, fontsize=7.5, color="#8b949e", pad=3)
                    if base_col + j == 0:
                        ax.set_ylabel(f"{cls_idx}.{CLASS_NAMES[cls_idx][:14]}",
                                      fontsize=7, color=PALETTE[cls_idx],
                                      rotation=0, labelpad=80, va="center")

        fig.suptitle(f"DINOv2 Patch Segmentation — {n_clusters} clusters (self-supervised)",
                     fontsize=12, fontweight="bold", color="#f0f6fc", y=1.005)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            print(f"✓ Galerie sauvegardée : {save_path}")
        return fig


# ══════════════════════════════════════════════════════════════════════════════
class SAMSegmenter:
    """Segmentation via Segment Anything Model (optionnel, ~2.5 GB).

    Téléchargement du checkpoint :
        wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

    Installation :
        pip install segment-anything
    """

    MODEL_URLS = {
        "vit_b": ("https://dl.fbaipublicfiles.com/segment_anything/"
                  "sam_vit_b_01ec64.pth"),
        "vit_l": ("https://dl.fbaipublicfiles.com/segment_anything/"
                  "sam_vit_l_0b3195.pth"),
    }

    def __init__(self, checkpoint_path: str, model_type: str = "vit_b",
                 device: str = "cuda"):
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            sam.to(device)
            self.generator = SamAutomaticMaskGenerator(
                sam,
                points_per_side=16,
                pred_iou_thresh=0.88,
                stability_score_thresh=0.92,
                min_mask_region_area=200,
            )
            self._available = True
            print(f"✓ SAM {model_type} chargé")
        except ImportError:
            print("⚠ segment-anything non installé : pip install segment-anything")
            self._available = False
        except Exception as e:
            print(f"⚠ SAM non disponible : {e}")
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def segment(self, image: np.ndarray) -> dict:
        """Retourne les masques SAM pour une image RGB uint8."""
        if not self._available:
            return {"masks": [], "n_masks": 0}
        masks = self.generator.generate(image)
        # Trier par aire décroissante
        masks = sorted(masks, key=lambda x: x["area"], reverse=True)
        return {"masks": masks, "n_masks": len(masks)}

    def to_seg_map(self, sam_result: dict, image_shape: tuple) -> np.ndarray:
        """Convertit les masques SAM en map de segmentation."""
        h, w = image_shape[:2]
        seg = np.zeros((h, w), dtype=np.uint8)
        for i, mask_dict in enumerate(sam_result["masks"][:4]):  # max 4 composantes
            seg[mask_dict["segmentation"]] = i + 1
        return seg

    @classmethod
    def download_checkpoint(cls, model_type: str = "vit_b",
                            dest_dir: str = "../data/") -> str:
        """Télécharge le checkpoint SAM si absent."""
        import urllib.request
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        url   = cls.MODEL_URLS.get(model_type)
        fname = Path(url).name
        dest  = Path(dest_dir) / fname
        if dest.exists():
            print(f"✓ Checkpoint déjà présent : {dest}")
            return str(dest)
        print(f"Téléchargement {fname} (~375 MB pour vit_b)...")
        urllib.request.urlretrieve(url, dest,
            lambda c, b, t: print(f"\r  {min(c*b*100//t,100)}%", end="", flush=True))
        print(f"\n✓ Sauvegardé : {dest}")
        return str(dest)


# ══════════════════════════════════════════════════════════════════════════════
#  Visualisation comparative
# ══════════════════════════════════════════════════════════════════════════════
def plot_segmentation_gallery(images: np.ndarray, labels: np.ndarray,
                               otsu_seg: "OtsuSegmenter",
                               dino_seg: "DINOv2PatchSegmenter" = None,
                               sam_seg: "SAMSegmenter" = None,
                               n_per_class: int = 2,
                               save_path: str = None) -> plt.Figure:
    """Galerie comparative : Original | Otsu | DINOv2 | SAM (si dispo)."""
    plt.rcParams.update(DARK)
    import PIL.Image as PILImage

    methods = ["Original", "Otsu (3 classes)"]
    if dino_seg is not None:
        methods.append("DINOv2 patches")
    if sam_seg is not None and sam_seg.available:
        methods.append("SAM zero-shot")

    n_methods = len(methods)
    n_classes = len(CLASS_NAMES)
    np.random.seed(42)

    fig, axes = plt.subplots(n_classes, n_per_class * n_methods,
                             figsize=(n_per_class * n_methods * 3, n_classes * 3),
                             dpi=90)
    fig.patch.set_facecolor("#0d1117")

    for row, cls_idx in enumerate(range(n_classes)):
        cls_imgs = images[labels == cls_idx]
        n_samp   = min(n_per_class, len(cls_imgs))
        sample   = cls_imgs[np.random.choice(len(cls_imgs), n_samp, replace=False)]

        for s, img in enumerate(sample):
            img_224 = np.array(PILImage.fromarray(img).resize((224, 224)))
            base = s * n_methods

            results = [img_224]

            # Otsu
            ores = otsu_seg.segment(img)
            results.append(colorize_segmap(
                np.array(PILImage.fromarray(ores["seg_map"]).resize((224,224),
                         PILImage.NEAREST)), 3))

            # DINOv2
            if dino_seg is not None:
                try:
                    dres = dino_seg.segment(img, n_clusters=4)
                    results.append(colorize_segmap(dres["seg_map"], 4))
                except Exception:
                    results.append(np.zeros((224,224,3), dtype=np.uint8))

            # SAM
            if sam_seg is not None and sam_seg.available:
                try:
                    sres = sam_seg.segment(img_224)
                    smap = sam_seg.to_seg_map(sres, (224,224))
                    results.append(colorize_segmap(smap, smap.max()+1))
                except Exception:
                    results.append(np.zeros((224,224,3), dtype=np.uint8))

            for m_idx, (data, method) in enumerate(zip(results, methods)):
                ax = axes[row, base + m_idx]
                ax.imshow(data); ax.axis("off")
                if row == 0:
                    ax.set_title(method, fontsize=8, color="#8b949e", pad=3)
                if base + m_idx == 0:
                    ax.set_ylabel(f"{cls_idx}.{CLASS_NAMES[cls_idx][:14]}",
                                  fontsize=7, color=PALETTE[cls_idx],
                                  rotation=0, labelpad=90, va="center")

    fig.suptitle("Comparaison des méthodes de segmentation — Galaxy10 DECaLS",
                 fontsize=12, fontweight="bold", color="#f0f6fc", y=1.005)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=90, bbox_inches="tight")
        print(f"✓ Galerie sauvegardée : {save_path}")
    return fig
