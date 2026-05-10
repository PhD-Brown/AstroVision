"""
AstroVision — morphometry.py
Métriques morphologiques non-paramétriques standards en astrophysique.

Implémente les statistiques CAS (Concentration-Asymmetry-Smoothness),
le coefficient de Gini et le moment M20 — métriques publiées dans :
    - Conselice et al. 2003, ApJS 147, 1–28
    - Lotz et al. 2004, AJ 128, 163–182
    - Abraham et al. 1996, ApJ 471, 694

Usage rapide :
    from astrovision.morphometry import GalaxyMorphometry, batch_morphometry

    m = GalaxyMorphometry(image_uint8)
    print(m.compute_all())          # dict complet

    df = batch_morphometry(images, labels)   # traitement en lot
"""

import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import (binary_fill_holes, center_of_mass, gaussian_filter,
                            label, rotate)
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

CLASS_NAMES = [
    "Disturbed", "Merging", "Round Smooth", "In-between Round Smooth",
    "Cigar Shaped Smooth", "Barred Spiral", "Unbarred Tight Spiral",
    "Unbarred Loose Spiral", "Edge-on without Bulge", "Edge-on with Bulge",
]

DARK_PARAMS = {
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d", "axes.labelcolor": "#c9d1d9",
    "axes.titlecolor": "#f0f6fc", "xtick.color": "#8b949e",
    "ytick.color": "#8b949e", "text.color": "#c9d1d9",
    "grid.color": "#21262d", "figure.dpi": 150,
    "savefig.facecolor": "#0d1117", "savefig.bbox": "tight",
}


# ── Utilitaires ────────────────────────────────────────────────────────────────
def _to_gray(image: np.ndarray) -> np.ndarray:
    """Convertit en niveaux de gris si RGB."""
    if image.ndim == 3:
        return (0.299 * image[..., 0]
                + 0.587 * image[..., 1]
                + 0.114 * image[..., 2]).astype(np.float64)
    return image.astype(np.float64)


def _otsu_mask(img_gray: np.ndarray) -> np.ndarray:
    """Masque binaire par seuillage d'Otsu sur les pixels au-dessus du fond."""
    try:
        from skimage.filters import threshold_otsu
        thresh = threshold_otsu(img_gray)
    except ImportError:
        thresh = img_gray.mean()
    mask = img_gray > thresh
    # Remplir les trous internes et garder la plus grande composante
    mask = binary_fill_holes(mask)
    labeled, n_comp = label(mask)
    if n_comp == 0:
        return mask
    sizes = [np.sum(labeled == i) for i in range(1, n_comp + 1)]
    largest = np.argmax(sizes) + 1
    return labeled == largest


def _radial_profile(img: np.ndarray, cy: float, cx: float,
                    mask: np.ndarray) -> tuple:
    """Retourne (radii_sorted, flux_cumsum_sorted) sur les pixels du masque."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_flat = r[mask]
    f_flat = img[mask]
    order   = np.argsort(r_flat)
    return r_flat[order], np.cumsum(f_flat[order])


# ══════════════════════════════════════════════════════════════════════════════
class GalaxyMorphometry:
    """Calcule les métriques morphologiques non-paramétriques d'une galaxie.

    Args:
        image   : ndarray (H, W) en niveaux de gris ou (H, W, 3) RGB, uint8 ou float
        bg_pct  : percentile pour l'estimation du fond de ciel (défaut 10)
        smooth_sigma : sigma du filtre Gaussien pour S (défaut 3 px)
    """

    def __init__(self, image: np.ndarray, bg_pct: float = 10.0,
                 smooth_sigma: float = 3.0):
        self.img_gray    = _to_gray(image)
        self.smooth_sigma = smooth_sigma

        # Soustraction du fond de ciel
        self.bg          = np.percentile(self.img_gray, bg_pct)
        self.img_sub     = np.maximum(self.img_gray - self.bg, 0.0)

        # Masque galaxie
        self.mask        = _otsu_mask(self.img_sub)
        self._n_pix      = self.mask.sum()

        # Centre de masse
        if self._n_pix > 0:
            self.cy, self.cx = center_of_mass(self.img_sub * self.mask)
        else:
            h, w = self.img_gray.shape
            self.cy, self.cx = h / 2, w / 2

    # ── Concentration C ────────────────────────────────────────────────────────
    def concentration(self) -> float:
        """C = 5 × log₁₀(r₈₀ / r₂₀) (Conselice 2003).

        r_p = rayon contenant p% du flux total dans le masque.
        C ↑ → galaxie plus concentrée (elliptiques > spirales > irrégulières).
        """
        if self._n_pix < 10:
            return np.nan
        r_sorted, f_cumsum = _radial_profile(self.img_sub, self.cy, self.cx,
                                             self.mask)
        f_total = f_cumsum[-1]
        if f_total <= 0:
            return np.nan

        r20 = r_sorted[np.searchsorted(f_cumsum, 0.20 * f_total)]
        r80 = r_sorted[np.searchsorted(f_cumsum, 0.80 * f_total)]
        if r20 <= 0:
            return np.nan
        return float(5.0 * np.log10(r80 / r20))

    # ── Asymétrie A ────────────────────────────────────────────────────────────
    def asymmetry(self) -> float:
        """A = Σ|I − I₁₈₀| / (2 Σ|I|) (Abraham+ 1996, Conselice+ 2000).

        I₁₈₀ = image tournée de 180° autour du centre de masse.
        A ↑ → galaxie perturbée / en fusion.
        """
        if self._n_pix < 10:
            return np.nan

        img = self.img_sub
        h, w = img.shape

        # Translation pour centrer sur le pixel central avant rotation
        shift_y = int(round(self.cy - h / 2))
        shift_x = int(round(self.cx - w / 2))

        # Découpage sur la région du masque
        ys, xs = np.where(self.mask)
        if len(ys) == 0:
            return np.nan
        y0, y1 = max(ys.min() - 2, 0), min(ys.max() + 2, h)
        x0, x1 = max(xs.min() - 2, 0), min(xs.max() + 2, w)
        crop    = img[y0:y1, x0:x1]
        mask_c  = self.mask[y0:y1, x0:x1]

        I_rot = rotate(crop, 180.0, reshape=False, order=1)

        num   = np.abs((crop - I_rot) * mask_c).sum()
        denom = np.abs(crop * mask_c).sum()
        if denom <= 0:
            return np.nan
        return float(num / (2.0 * denom))

    # ── Lissé S ────────────────────────────────────────────────────────────────
    def smoothness(self) -> float:
        """S = Σ|I − I_smooth| / Σ|I| (Conselice 2003).

        I_smooth = image convoluée avec un noyau Gaussien (σ = smooth_sigma).
        S ↑ → structure à haute fréquence spatiale (bras spiraux, régions HII).
        """
        if self._n_pix < 10:
            return np.nan

        img_m    = self.img_sub * self.mask
        img_s    = gaussian_filter(img_m, sigma=self.smooth_sigma) * self.mask
        num      = np.abs((img_m - img_s) * self.mask).sum()
        denom    = np.abs(img_m).sum()
        if denom <= 0:
            return np.nan
        return float(num / denom)

    # ── Gini G ────────────────────────────────────────────────────────────────
    def gini(self) -> float:
        """Coefficient de Gini (Abraham+ 2003, Lotz+ 2004).

        Mesure l'inégalité de distribution du flux entre pixels.
        G → 1 : tout le flux concentré dans quelques pixels (galaxies compactes).
        G → 0 : flux uniformément distribué (galaxies diffuses).
        """
        px = np.sort(np.abs(self.img_sub[self.mask]))
        n  = len(px)
        if n < 2 or px.sum() == 0:
            return np.nan
        idx = np.arange(1, n + 1)
        return float(
            np.sum((2 * idx - n - 1) * px) / (n * px.sum() + 1e-12)
        )

    # ── M20 ───────────────────────────────────────────────────────────────────
    def m20(self) -> float:
        """Moment M₂₀ (Lotz+ 2004).

        M₂₀ = log₁₀(M_bright20 / M_total)
        où M = Σᵢ fᵢ × rᵢ² est le moment d'ordre 2 du flux.
        Les 20% des pixels les plus brillants sont sélectionnés.

        M₂₀ ↓ (plus négatif) → flux concentré (elliptiques).
        M₂₀ proche de 0 → multiples noyaux brillants (fusions / perturbées).
        """
        ys, xs = np.where(self.mask)
        if len(ys) < 10:
            return np.nan

        f = self.img_sub[self.mask]
        r2 = (xs - self.cx) ** 2 + (ys - self.cy) ** 2

        M_tot = np.sum(f * r2)
        if M_tot <= 0:
            return np.nan

        # Sélectionner les 20% de pixels les plus brillants
        sorted_idx = np.argsort(f)[::-1]
        cumflux    = np.cumsum(f[sorted_idx])
        threshold  = 0.20 * cumflux[-1]
        n_bright   = np.searchsorted(cumflux, threshold) + 1
        bright_idx = sorted_idx[:n_bright]

        M_20 = np.sum(f[bright_idx] * r2[bright_idx])
        if M_20 <= 0:
            return np.nan
        return float(np.log10(M_20 / M_tot))

    # ── Rayon de Petrosian ────────────────────────────────────────────────────
    def petrosian_radius(self, eta: float = 0.2) -> float:
        """Rayon de Petrosian r_p tel que η(r_p) = 0.2.

        η(r) = I(r) / <I>(r) où I(r) est l'intensité à r et <I>(r) la
        moyenne dans l'ellipse de rayon r.
        Utilisé comme échelle de longueur standard en astronomie.
        """
        if self._n_pix < 20:
            return np.nan

        r_sorted, f_cumsum = _radial_profile(self.img_sub, self.cy, self.cx,
                                             self.mask)
        if r_sorted[-1] < 2:
            return np.nan

        # Discrétiser en anneaux
        r_bins  = np.linspace(0, r_sorted[-1], 50)
        etas    = []
        r_mids  = []
        for i in range(1, len(r_bins)):
            r0, r1 = r_bins[i-1], r_bins[i]
            in_ann = (r_sorted >= r0) & (r_sorted < r1)
            in_ell = r_sorted < r1
            if in_ann.sum() < 1 or in_ell.sum() < 1:
                continue
            f_ann = f_cumsum[in_ann][-1] - (f_cumsum[in_ann][0] if in_ann.sum() > 1 else 0)
            area_ann = in_ann.sum()
            area_ell = in_ell.sum()
            mean_ann  = f_ann / area_ann if area_ann > 0 else 0
            mean_ell  = (f_cumsum[in_ell][-1] / area_ell) if area_ell > 0 else 1
            if mean_ell > 0:
                etas.append(mean_ann / mean_ell)
                r_mids.append((r0 + r1) / 2)

        if not etas:
            return np.nan

        etas, r_mids = np.array(etas), np.array(r_mids)
        # Trouver le premier rayon où η < 0.2
        idx = np.where(etas < eta)[0]
        return float(r_mids[idx[0]]) if len(idx) > 0 else float(r_sorted[-1])

    # ── Indice de Sérsic (proxy) ───────────────────────────────────────────────
    def sersic_index_proxy(self) -> float:
        """Ajustement d'un profil de Sérsic 1D au profil radial.

        I(r) = I_e × exp(−b_n × ((r/r_e)^(1/n) − 1))

        n ≈ 1 → disque exponentiel (spirales)
        n ≈ 4 → loi de de Vaucouleurs (elliptiques)

        Note : ajustement simplifié sur le profil radial moyen (non elliptique).
        """
        if self._n_pix < 20:
            return np.nan

        h, w = self.img_sub.shape
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.sqrt((xx - self.cx) ** 2 + (yy - self.cy) ** 2)[self.mask]
        f = self.img_sub[self.mask]

        r_bins  = np.linspace(0, r.max(), 30)
        r_mids, f_means = [], []
        for i in range(1, len(r_bins)):
            sel = (r >= r_bins[i-1]) & (r < r_bins[i])
            if sel.sum() > 0:
                r_mids.append((r_bins[i-1] + r_bins[i]) / 2)
                f_means.append(f[sel].mean())
        if len(r_mids) < 5:
            return np.nan

        r_arr, f_arr = np.array(r_mids), np.array(f_means)
        f_arr = np.maximum(f_arr, 1e-6)

        def sersic(r, I_e, r_e, n):
            bn = 1.9992 * n - 0.3271
            return I_e * np.exp(-bn * ((r / r_e) ** (1.0 / n) - 1.0))

        try:
            popt, _ = curve_fit(sersic, r_arr, f_arr,
                                p0=[f_arr[0], r_arr[len(r_arr)//2], 2.0],
                                bounds=([0, 0.1, 0.3], [1e6, 100, 10]),
                                maxfev=2000)
            n_fit = float(popt[2])
            return n_fit if 0.3 <= n_fit <= 10 else np.nan
        except Exception:
            return np.nan

    # ── Tout calculer ─────────────────────────────────────────────────────────
    def compute_all(self) -> dict:
        """Calcule toutes les métriques et retourne un dictionnaire."""
        return {
            "C":          self.concentration(),
            "A":          self.asymmetry(),
            "S":          self.smoothness(),
            "Gini":       self.gini(),
            "M20":        self.m20(),
            "r_petrosian": self.petrosian_radius(),
            "sersic_n":   self.sersic_index_proxy(),
            "n_pixels":   int(self._n_pix),
            "cx":         float(self.cx),
            "cy":         float(self.cy),
        }


# ══════════════════════════════════════════════════════════════════════════════
def batch_morphometry(images: np.ndarray, labels: np.ndarray,
                      max_n: int = 5000,
                      verbose: bool = True) -> "pd.DataFrame":
    """Calcule toutes les métriques sur un batch d'images.

    Args:
        images  : (N, H, W, 3) uint8 ou (N, H, W) float
        labels  : (N,) int — classes Galaxy10
        max_n   : nombre maximum d'images à traiter (sous-échantillonnage aléatoire)
        verbose : affiche la progression

    Returns:
        DataFrame avec colonnes [label, class_name, C, A, S, Gini, M20, ...]
    """
    import pandas as pd

    np.random.seed(42)
    if len(images) > max_n:
        idx = np.random.choice(len(images), max_n, replace=False)
    else:
        idx = np.arange(len(images))

    rows = []
    n = len(idx)
    for k, i in enumerate(idx):
        if verbose and k % 200 == 0:
            print(f"  {k:>5}/{n}  ({k/n*100:.1f}%)", end="\r")
        try:
            m = GalaxyMorphometry(images[i])
            row = m.compute_all()
            row["label"]      = int(labels[i])
            row["class_name"] = CLASS_NAMES[int(labels[i])]
            row["g10_idx"]    = int(i)
        except Exception as e:
            row = {"label": int(labels[i]),
                   "class_name": CLASS_NAMES[int(labels[i])],
                   "g10_idx": int(i)}
        rows.append(row)

    if verbose:
        print(f"  {n}/{n}  (100.0%) ✓")

    df = pd.DataFrame(rows)
    numeric_cols = ["C","A","S","Gini","M20","r_petrosian","sersic_n"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Visualisations
# ══════════════════════════════════════════════════════════════════════════════
PALETTE = [
    "#6e40aa","#4c6edb","#23abd8","#1ac7c2","#1ddfa3",
    "#52f667","#aff05b","#e2b72f","#fb8a27","#f83e4b",
]


def plot_cas_diagram(df: "pd.DataFrame", save_path: str = None):
    """Diagramme C-A (Conselice 2003) et Gini-M20 (Lotz 2004)."""
    plt.rcParams.update(DARK_PARAMS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=150)

    # ── C vs A ────────────────────────────────────────────────────────────────
    ax = axes[0]
    sub = df.dropna(subset=["C","A"])
    for i, cls in enumerate(CLASS_NAMES):
        m = sub["class_name"] == cls
        if m.sum() < 3:
            continue
        ax.scatter(sub.loc[m,"A"], sub.loc[m,"C"], c=PALETTE[i], s=8,
                   alpha=0.5, label=f"{i}.{cls[:12]}", rasterized=True)
    # Ligne de séparation typique (Conselice 2003 : A=0.35 sépare les fusions)
    ax.axvline(0.35, color="#f83e4b", lw=1.5, ls="--", alpha=0.8,
               label="Fusions A>0.35")
    ax.set_xlabel("Asymétrie A"); ax.set_ylabel("Concentration C")
    ax.set_title("Diagramme C-A (Conselice 2003)\nA>0.35 → galaxies en fusion")
    ax.legend(fontsize=6, markerscale=2, ncol=2); ax.grid(alpha=0.2)

    # ── Gini vs M20 ───────────────────────────────────────────────────────────
    ax2 = axes[1]
    sub2 = df.dropna(subset=["Gini","M20"])
    for i, cls in enumerate(CLASS_NAMES):
        m = sub2["class_name"] == cls
        if m.sum() < 3:
            continue
        ax2.scatter(sub2.loc[m,"M20"], sub2.loc[m,"Gini"], c=PALETTE[i], s=8,
                    alpha=0.5, label=f"{i}.{cls[:12]}", rasterized=True)

    # Lignes de séparation Lotz+ 2004
    m20_arr = np.linspace(-3.5, -0.5, 100)
    gini_lotz = -0.14 * m20_arr + 0.778  # E/S0 line
    ax2.plot(m20_arr, gini_lotz, "#23abd8", lw=1.5, ls="--",
             label="E/S0 (Lotz+2004)")
    ax2.set_xlabel("M₂₀"); ax2.set_ylabel("Gini G")
    ax2.set_title("Diagramme Gini-M₂₀ (Lotz 2004)\nGauche = concentré, Gini↑ = inégal")
    ax2.legend(fontsize=6, markerscale=2, ncol=2); ax2.grid(alpha=0.2)

    fig.suptitle("Métriques morphologiques non-paramétriques — Galaxy10 DECaLS",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150); print(f"✓ Sauvegardé : {save_path}")
    return fig


def plot_morphometry_by_class(df: "pd.DataFrame",
                               metrics: list = None,
                               save_path: str = None):
    """Violin plots de chaque métrique par classe morphologique."""
    plt.rcParams.update(DARK_PARAMS)
    if metrics is None:
        metrics = [m for m in ["C","A","S","Gini","M20","sersic_n"]
                   if m in df.columns]
    n = len(metrics)
    fig, axes = plt.subplots(2, (n+1)//2, figsize=(14, 8), dpi=150)
    axes = axes.flat

    for ax, metric in zip(axes, metrics):
        data, cls_labels, colors = [], [], []
        for i, cls in enumerate(CLASS_NAMES):
            vals = df.loc[df["class_name"]==cls, metric].dropna().values
            if len(vals) < 5:
                continue
            data.append(vals); cls_labels.append(f"{i}"); colors.append(PALETTE[i])

        if not data:
            ax.axis("off"); continue

        vp = ax.violinplot(data, showmedians=True, showextrema=False)
        for body, col in zip(vp["bodies"], colors):
            body.set_facecolor(col); body.set_alpha(0.7); body.set_edgecolor("none")
        vp["cmedians"].set_color("#f0f6fc"); vp["cmedians"].set_linewidth(2)
        ax.set_xticks(range(1, len(cls_labels)+1))
        ax.set_xticklabels(cls_labels, fontsize=8)
        ax.set_ylabel(metric); ax.set_title(metric); ax.grid(axis="y", alpha=0.3)

    # Masquer les axes vides
    for ax in list(axes)[len(metrics):]:
        ax.axis("off")

    fig.suptitle("Métriques CAS/Gini/M20 par classe — Galaxy10 DECaLS",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150); print(f"✓ Sauvegardé : {save_path}")
    return fig
