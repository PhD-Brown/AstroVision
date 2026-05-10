"""
AstroVision — spectral_viz.py
Visualisations scientifiques pour l'analyse morphologie × spectroscopie.

Fonctions principales :
    plot_cmd()              Diagramme Couleur-Magnitude
    plot_color_by_class()   Distribution des couleurs par classe
    plot_umap_spectral()    UMAP DINOv2 coloré par propriétés spectrales
    plot_bpt_proxy()        Diagramme BPT proxy (couleurs photométriques)
    plot_correlation()      Heatmap corrélation features × propriétés
    plot_population_umap()  Blue Cloud / Red Sequence / Green Valley
    plot_sersic_morphology() Sérsic index vs classe (si NSA disponible)
"""

from pathlib import Path

import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy import stats

PALETTE = [
    "#6e40aa", "#4c6edb", "#23abd8", "#1ac7c2", "#1ddfa3",
    "#52f667", "#aff05b", "#e2b72f", "#fb8a27", "#f83e4b",
]
CLASS_NAMES = [
    "Disturbed", "Merging", "Round Smooth", "In-between Round Smooth",
    "Cigar Shaped Smooth", "Barred Spiral", "Unbarred Tight Spiral",
    "Unbarred Loose Spiral", "Edge-on without Bulge", "Edge-on with Bulge",
]
DARK = {
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d", "axes.labelcolor": "#c9d1d9",
    "axes.titlecolor": "#f0f6fc", "axes.titlesize": 12,
    "axes.titleweight": "bold", "xtick.color": "#8b949e",
    "ytick.color": "#8b949e", "text.color": "#c9d1d9",
    "grid.color": "#21262d", "grid.linewidth": 0.7,
    "figure.dpi": 150, "savefig.facecolor": "#0d1117",
    "savefig.bbox": "tight", "font.size": 10,
}
POP_COLORS = {
    "Blue Cloud":   "#23abd8",
    "Green Valley": "#1ddfa3",
    "Red Sequence": "#f83e4b",
}


def _apply_dark():
    plt.rcParams.update(DARK)


# ── 1. Diagramme Couleur-Magnitude ─────────────────────────────────────────────
def plot_cmd(df: pd.DataFrame, color_col: str = "g-r",
             mag_col: str = "Mr", save_path: str = None) -> plt.Figure:
    """CMD coloré par classe morphologique et par population stellaire.

    Args:
        df         : DataFrame du cross-match (avec colonnes couleur + magnitude)
        color_col  : colonne couleur (g-r, u-r, etc.)
        mag_col    : colonne magnitude absolue (Mr, etc.)
        save_path  : chemin de sauvegarde
    """
    _apply_dark()
    sub = df.dropna(subset=[color_col, mag_col])
    if sub.empty:
        print(f"✗ CMD : colonnes '{color_col}' ou '{mag_col}' absentes/vides")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # ── Gauche : par classe morphologique ────────────────────────────────────
    ax = axes[0]
    for i, cls in enumerate(CLASS_NAMES):
        mask = sub["class_name"] == cls
        if mask.sum() < 3:
            continue
        ax.scatter(sub.loc[mask, color_col], sub.loc[mask, mag_col],
                   c=PALETTE[i], s=4, alpha=0.5, label=f"{i}.{cls[:14]}",
                   rasterized=True)
    ax.invert_yaxis()
    ax.set_xlabel(f"Couleur {color_col}")
    ax.set_ylabel(f"Magnitude absolue {mag_col}")
    ax.set_title("CMD — par classe morphologique")
    ax.legend(fontsize=6, markerscale=3, ncol=2)
    ax.grid(alpha=0.2)

    # Séparation Blue Cloud / Red Sequence (Baldry+ 2004 si u-r)
    if color_col == "u-r":
        ax.axvline(2.22, color="#23abd8", lw=1.5, ls="--",
                   alpha=0.7, label="Blue/Green")
        ax.axvline(2.55, color="#f83e4b", lw=1.5, ls="--",
                   alpha=0.7, label="Green/Red")
    elif color_col == "g-r":
        ax.axvline(0.55, color="#23abd8", lw=1.5, ls="--", alpha=0.7)
        ax.axvline(0.70, color="#f83e4b", lw=1.5, ls="--", alpha=0.7)

    # ── Droite : par population ───────────────────────────────────────────────
    ax2 = axes[1]
    if "population" in sub.columns:
        for pop, col in POP_COLORS.items():
            mask = sub["population"] == pop
            ax2.scatter(sub.loc[mask, color_col], sub.loc[mask, mag_col],
                        c=col, s=4, alpha=0.5, label=f"{pop} ({mask.sum()})",
                        rasterized=True)
    else:
        ax2.scatter(sub[color_col], sub[mag_col], c="#6e40aa",
                    s=4, alpha=0.4, rasterized=True)

    ax2.invert_yaxis()
    ax2.set_xlabel(f"Couleur {color_col}")
    ax2.set_ylabel(f"Magnitude absolue {mag_col}")
    ax2.set_title("CMD — Blue Cloud / Green Valley / Red Sequence")
    ax2.legend(fontsize=9, markerscale=3)
    ax2.grid(alpha=0.2)

    fig.suptitle(f"Diagramme Couleur-Magnitude Galaxy10 DECaLS × SDSS DR16",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✓ CMD sauvegardé : {save_path}")
    return fig


# ── 2. Distribution couleurs par classe ────────────────────────────────────────
def plot_color_by_class(df: pd.DataFrame, color_col: str = "g-r",
                        save_path: str = None) -> plt.Figure:
    """Violins + boxplots de la couleur par classe morphologique."""
    _apply_dark()
    sub = df.dropna(subset=[color_col])
    if sub.empty:
        return None

    data_per_class = [
        sub.loc[sub["class_name"] == cls, color_col].values
        for cls in CLASS_NAMES
    ]
    data_per_class = [d for d in data_per_class if len(d) >= 5]
    classes_present = [
        cls for cls, d in zip(CLASS_NAMES, [
            sub.loc[sub["class_name"] == cls, color_col].values
            for cls in CLASS_NAMES
        ]) if len(d) >= 5
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # ── Violin plot ───────────────────────────────────────────────────────────
    ax = axes[0]
    vp = ax.violinplot(data_per_class, positions=range(len(classes_present)),
                       showmedians=True, showextrema=False)
    colors_used = [PALETTE[CLASS_NAMES.index(c)] for c in classes_present]
    for body, col in zip(vp["bodies"], colors_used):
        body.set_facecolor(col); body.set_alpha(0.7); body.set_edgecolor("none")
    vp["cmedians"].set_color("#f0f6fc"); vp["cmedians"].set_linewidth(2)
    ax.set_xticks(range(len(classes_present)))
    ax.set_xticklabels([f"{CLASS_NAMES.index(c)}.{c[:10]}"
                        for c in classes_present], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(f"Couleur {color_col}")
    ax.set_title(f"Distribution de {color_col} par classe")

    # Lignes de référence Blue Cloud / Red Sequence
    if color_col == "g-r":
        ax.axhline(0.55, color="#23abd8", lw=1.5, ls="--", alpha=0.7, label="Blue/Green")
        ax.axhline(0.70, color="#f83e4b", lw=1.5, ls="--", alpha=0.7, label="Green/Red")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # ── Médiane + IC 95% ──────────────────────────────────────────────────────
    ax2 = axes[1]
    medians = [np.median(d) for d in data_per_class]
    ci_low  = [np.percentile(d, 2.5) for d in data_per_class]
    ci_high = [np.percentile(d, 97.5) for d in data_per_class]
    x = range(len(classes_present))
    ax2.barh(list(x), medians, color=colors_used, alpha=0.8, edgecolor="none")
    for i, (lo, hi) in enumerate(zip(ci_low, ci_high)):
        ax2.plot([lo, hi], [i, i], color="#8b949e", lw=2)
    ax2.set_yticks(list(x))
    ax2.set_yticklabels(classes_present, fontsize=8)
    ax2.set_xlabel(f"Médiane {color_col} (IC 95%)")
    ax2.set_title("Médiane par classe morphologique")
    if color_col == "g-r":
        ax2.axvline(0.55, color="#23abd8", lw=1.5, ls="--", alpha=0.7)
        ax2.axvline(0.70, color="#f83e4b", lw=1.5, ls="--", alpha=0.7)
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle(f"Couleur photométrique {color_col} par classe — Galaxy10 × SDSS",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✓ Figure sauvegardée : {save_path}")
    return fig


# ── 3. UMAP DINOv2 coloré par propriétés spectrales ───────────────────────────
def plot_umap_spectral(emb_2d: np.ndarray, df_matched: pd.DataFrame,
                       g10_indices: np.ndarray,
                       properties: list = None,
                       save_path: str = None) -> plt.Figure:
    """UMAP DINOv2 coloré par différentes propriétés du catalogue croisé.

    Args:
        emb_2d      : (N, 2) coordonnées UMAP de TOUTES les galaxies
        df_matched  : DataFrame du cross-match
        g10_indices : indices g10_idx dans emb_2d
        properties  : liste de colonnes à visualiser (défaut : g-r, Mr, population)
    """
    _apply_dark()

    if properties is None:
        properties = []
        for col in ["g-r", "u-r", "Mr", "population"]:
            if col in df_matched.columns:
                properties.append(col)
        if not properties:
            properties = ["g-r"]

    n = len(properties)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), dpi=150)
    if n == 1:
        axes = [axes]

    # Créer un mapping g10_idx → row dans df_matched
    idx_map = dict(zip(df_matched["g10_idx"].values,
                       range(len(df_matched))))

    for ax, prop in zip(axes, properties):
        # Récupérer les valeurs pour les points matchés
        mask_matched = np.array([i in idx_map for i in g10_indices])
        matched_rows = [idx_map[i] for i in g10_indices[mask_matched]]
        emb_m  = emb_2d[mask_matched]
        emb_um = emb_2d[~mask_matched]

        # Fond : galaxies non matchées
        ax.scatter(emb_um[:, 0], emb_um[:, 1], c="#21262d",
                   s=2, alpha=0.3, rasterized=True)

        if prop == "population":
            vals = df_matched.iloc[matched_rows]["population"].values
            for pop, col in POP_COLORS.items():
                m = vals == pop
                ax.scatter(emb_m[m, 0], emb_m[m, 1], c=col, s=6,
                           alpha=0.7, label=pop, rasterized=True)
            ax.legend(fontsize=8, markerscale=2)
            ax.set_title(f"Population stellaire (Blue/Green/Red)")
        else:
            vals = pd.to_numeric(df_matched.iloc[matched_rows][prop],
                                 errors="coerce").values
            valid = ~np.isnan(vals)
            vmin, vmax = np.percentile(vals[valid], [2, 98])
            sc = ax.scatter(emb_m[valid, 0], emb_m[valid, 1],
                            c=vals[valid], cmap="RdYlBu_r" if "color" in prop.lower() or "-" in prop
                            else "viridis",
                            s=6, alpha=0.8, vmin=vmin, vmax=vmax, rasterized=True)
            plt.colorbar(sc, ax=ax, pad=0.02).set_label(prop, fontsize=9)
            ax.set_title(f"DINOv2 UMAP coloré par {prop}")

        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.grid(alpha=0.15)

    fig.suptitle("Espace latent DINOv2 × Propriétés spectrales SDSS DR16",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✓ UMAP spectral sauvegardé : {save_path}")
    return fig


# ── 4. Corrélation features DINOv2 × propriétés ────────────────────────────────
def plot_feature_correlation(feats: np.ndarray, df_matched: pd.DataFrame,
                             g10_indices: np.ndarray,
                             n_components: int = 10,
                             save_path: str = None) -> plt.Figure:
    """Heatmap de corrélation PCA(features DINOv2) × propriétés spectrales.

    Permet de voir si les dimensions de l'espace DINOv2 encodent
    la couleur, la luminosité, le type de population, etc.
    """
    from sklearn.decomposition import PCA
    _apply_dark()

    spectral_cols = [c for c in ["g-r", "u-r", "g-i", "Mr", "z_photo"]
                     if c in df_matched.columns]
    if not spectral_cols:
        print("✗ Aucune colonne spectrale disponible pour la corrélation")
        return None

    # PCA sur features
    pca = PCA(n_components=n_components, random_state=42)
    pcs = pca.fit_transform(feats)

    # Aligner avec les matchés
    idx_map = dict(zip(df_matched["g10_idx"].values, range(len(df_matched))))
    mask = np.array([i in idx_map for i in g10_indices])
    matched_rows = [idx_map[i] for i in g10_indices[mask]]
    pcs_m = pcs[mask]
    df_m  = df_matched.iloc[matched_rows][spectral_cols]

    # Matrice de corrélation Spearman
    corr_matrix = np.zeros((n_components, len(spectral_cols)))
    pval_matrix = np.zeros((n_components, len(spectral_cols)))

    for i in range(n_components):
        for j, col in enumerate(spectral_cols):
            vals = pd.to_numeric(df_m[col], errors="coerce").values
            valid = ~np.isnan(vals)
            if valid.sum() < 10:
                continue
            r, p = stats.spearmanr(pcs_m[valid, i], vals[valid])
            corr_matrix[i, j] = r
            pval_matrix[i, j] = p

    fig, ax = plt.subplots(figsize=(max(6, len(spectral_cols) * 1.5),
                                    n_components * 0.6 + 2), dpi=150)
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("ρ Spearman", fontsize=9)

    ax.set_xticks(range(len(spectral_cols)))
    ax.set_xticklabels(spectral_cols, rotation=30, ha="right")
    ax.set_yticks(range(n_components))
    ax.set_yticklabels([f"PC{i+1}\n({pca.explained_variance_ratio_[i]*100:.1f}%)"
                        for i in range(n_components)], fontsize=8)
    ax.set_title("Corrélation Spearman : PCA(DINOv2) × Propriétés SDSS\n"
                 "(* p<0.05)", fontsize=11)

    # Annoter les corrélations significatives
    for i in range(n_components):
        for j in range(len(spectral_cols)):
            if abs(corr_matrix[i, j]) > 0.1 and pval_matrix[i, j] < 0.05:
                ax.text(j, i, f"{corr_matrix[i,j]:.2f}*",
                        ha="center", va="center", fontsize=7.5,
                        color="white" if abs(corr_matrix[i, j]) > 0.35 else "#c9d1d9")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✓ Corrélation sauvegardée : {save_path}")
    return fig, corr_matrix, spectral_cols


# ── 5. KS test par paires de classes ───────────────────────────────────────────
def ks_test_classes(df: pd.DataFrame, color_col: str = "g-r") -> pd.DataFrame:
    """KS test deux à deux entre classes morphologiques sur une couleur.

    Permet de savoir quelles classes sont statistiquement distinguables
    par leur distribution de couleur.
    """
    results = []
    data = {cls: df.loc[df["class_name"] == cls, color_col].dropna().values
            for cls in CLASS_NAMES}

    for i, c1 in enumerate(CLASS_NAMES):
        for j, c2 in enumerate(CLASS_NAMES):
            if j <= i:
                continue
            d1, d2 = data.get(c1, []), data.get(c2, [])
            if len(d1) < 5 or len(d2) < 5:
                continue
            ks_stat, p_val = stats.ks_2samp(d1, d2)
            results.append({
                "class_1": c1, "class_2": c2,
                "ks_stat": round(ks_stat, 4),
                "p_value": round(p_val, 6),
                "significant": p_val < 0.01,
                "n1": len(d1), "n2": len(d2),
            })

    df_ks = pd.DataFrame(results).sort_values("ks_stat", ascending=False)
    return df_ks


# ── 6. Synthetic BPT (proxy photométrique) ────────────────────────────────────
def plot_bpt_proxy(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """Diagramme BPT proxy en utilisant les couleurs SDSS.

    Méthode : Blanton & Moustakas 2009 montrent que u-r corrèle avec
    [NII]/Hα. On utilise ce proxy pour séparer star-forming / AGN / quiescent.

    NOTE : Ce n'est PAS un vrai BPT (qui requiert des spectres) mais un
    proxy photométrique bien établi dans la littérature.
    """
    _apply_dark()

    needed = ["g-r", "u-r"]
    for col in needed:
        if col not in df.columns:
            print(f"✗ Colonne {col} manquante pour le BPT proxy")
            return None

    sub = df.dropna(subset=needed)
    if sub.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # ── BPT proxy ─────────────────────────────────────────────────────────────
    ax = axes[0]
    for i, cls in enumerate(CLASS_NAMES):
        mask = sub["class_name"] == cls
        if mask.sum() < 3:
            continue
        ax.scatter(sub.loc[mask, "g-r"], sub.loc[mask, "u-r"],
                   c=PALETTE[i], s=5, alpha=0.5,
                   label=f"{i}.{cls[:12]}", rasterized=True)

    # Zones de référence
    ax.axhspan(0,   2.22, alpha=0.05, color="#23abd8", label="Blue Cloud zone")
    ax.axhspan(2.55, 5.0, alpha=0.05, color="#f83e4b", label="Red Sequence zone")
    ax.axhline(2.22, color="#23abd8", lw=1.5, ls="--", alpha=0.7)
    ax.axhline(2.55, color="#f83e4b", lw=1.5, ls="--", alpha=0.7)
    ax.set_xlabel("g - r"); ax.set_ylabel("u - r")
    ax.set_title("Diagramme couleur-couleur proxy (u-r vs g-r)")
    ax.legend(fontsize=6, markerscale=2, ncol=2)
    ax.grid(alpha=0.2)

    # ── Fraction Blue Cloud / Red Sequence par classe ─────────────────────────
    ax2 = axes[1]
    if "population" not in sub.columns:
        ax2.text(0.5, 0.5, "Colonne 'population' absente\n(u-r requis)",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=11)
    else:
        fracs = sub.groupby(["class_name", "population"]).size().unstack(fill_value=0)
        fracs = fracs.div(fracs.sum(axis=1), axis=0)
        classes_order = [c for c in CLASS_NAMES if c in fracs.index]
        fracs = fracs.loc[classes_order]

        bottom = np.zeros(len(fracs))
        for pop, col in POP_COLORS.items():
            if pop in fracs.columns:
                vals = fracs[pop].values
                ax2.bar(range(len(fracs)), vals, bottom=bottom,
                        color=col, alpha=0.85, label=pop, edgecolor="none")
                bottom += vals

        ax2.set_xticks(range(len(fracs)))
        ax2.set_xticklabels([f"{CLASS_NAMES.index(c)}.{c[:10]}"
                             for c in classes_order],
                            rotation=30, ha="right", fontsize=8)
        ax2.set_ylabel("Fraction")
        ax2.set_title("Fraction Blue/Green/Red par classe morphologique")
        ax2.legend(fontsize=9)
        ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Analyse couleur-population — Galaxy10 × SDSS DR16",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✓ BPT proxy sauvegardé : {save_path}")
    return fig


# ── 7. Synthèse par classe ─────────────────────────────────────────────────────
def plot_class_summary(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """Tableau de bord complet par classe morphologique."""
    _apply_dark()

    color_cols = [c for c in ["g-r", "u-r", "g-i", "Mr", "z_photo"]
                  if c in df.columns]
    if not color_cols:
        return None

    n_cols = len(color_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 7), dpi=150)
    if n_cols == 1:
        axes = [axes]

    for ax, col in zip(axes, color_cols):
        data = []
        labels_plot = []
        colors_plot = []
        for i, cls in enumerate(CLASS_NAMES):
            vals = df.loc[df["class_name"] == cls, col].dropna()
            if len(vals) < 5:
                continue
            data.append(vals.values)
            labels_plot.append(f"{i}.{cls[:10]}")
            colors_plot.append(PALETTE[i])

        if not data:
            continue

        bxp = ax.boxplot(data, patch_artist=True, vert=False,
                         medianprops={"color": "#f0f6fc", "linewidth": 2},
                         whiskerprops={"color": "#8b949e"},
                         capprops={"color": "#8b949e"},
                         flierprops={"marker": ".", "color": "#8b949e",
                                     "markersize": 2, "alpha": 0.4})
        for patch, col_c in zip(bxp["boxes"], colors_plot):
            patch.set_facecolor(col_c); patch.set_alpha(0.75)
        ax.set_yticks(range(1, len(labels_plot) + 1))
        ax.set_yticklabels(labels_plot, fontsize=7.5)
        ax.set_xlabel(col); ax.grid(axis="x", alpha=0.3)
        ax.set_title(col)

    fig.suptitle("Propriétés photométriques par classe — Galaxy10 × SDSS",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✓ Résumé par classe sauvegardé : {save_path}")
    return fig
