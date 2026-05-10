"""
AstroVision — synthesis.py
Synthèse finale : Morphologie × Photométrie × Segmentation × Spectroscopie.

Croise les résultats de toutes les phases du projet :
    Phase 1 : Classification CNN / DINOv2 (balanced_acc 0.839)
    Phase 2 : EDA + Grad-CAM + erreurs
    Phase 3A : Cross-match Galaxy10 × SDSS DR16 (R²=0.536, z_photo ρ=0.998)
    Phase 3B : Segmentation Otsu / DINOv2 / U-Net (mIoU=0.540) + CAS/Gini/M20
    Phase 4  : Synthèse — fraction_bulbe × g-r × [Fe/H] (LAMOST tentative)

Usage :
    from astrovision.synthesis import AstroVisionSynthesis
    synth = AstroVisionSynthesis(data_dir='../data/', figures_dir='../figures/')
    df    = synth.build_master_dataframe()
    synth.run_all_figures(df)
"""

import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, kruskal

warnings.filterwarnings("ignore")

# ── Constantes ──────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Disturbed", "Merging", "Round Smooth", "In-between Round Smooth",
    "Cigar Shaped Smooth", "Barred Spiral", "Unbarred Tight Spiral",
    "Unbarred Loose Spiral", "Edge-on without Bulge", "Edge-on with Bulge",
]
CLASS_SHORT = [
    "Disturbed", "Merging", "Rnd Smth", "In-betw", "Cigar",
    "Barred Sp", "Unb Tight", "Unb Loose", "Edge-on", "Edge+Bulge",
]
PALETTE = [
    "#6e40aa", "#4c6edb", "#23abd8", "#1ac7c2", "#1ddfa3",
    "#52f667", "#aff05b", "#e2b72f", "#fb8a27", "#f83e4b",
]
DARK = {
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",   "axes.labelcolor": "#c9d1d9",
    "axes.titlecolor": "#f0f6fc",  "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",      "text.color": "#c9d1d9",
    "grid.color": "#21262d",       "figure.dpi": 150,
    "savefig.facecolor": "#0d1117","savefig.bbox": "tight",
}
POP_COLORS = {
    "Blue Cloud":   "#23abd8",
    "Green Valley": "#1ddfa3",
    "Red Sequence": "#f83e4b",
}
LAMOST_CATALOG = "vizier:V/154/sdss16"   # fallback SDSS si LAMOST indispo
LAMOST_VIZ     = "vizier:J/AJ/157/168"   # LAMOST DR6 galaxy catalog (Yuan+2019)


# ── Classe principale ────────────────────────────────────────────────────────────
class AstroVisionSynthesis:
    """Charge, fusionne et analyse tous les résultats AstroVision.

    Args:
        data_dir    : répertoire contenant les caches .csv.gz et .npy
        figures_dir : répertoire de sortie pour les figures
    """

    def __init__(self, data_dir: str = "../data/",
                 figures_dir: str = "../figures/"):
        self.data_dir    = Path(data_dir)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update(DARK)

    # ── Chargement des données ────────────────────────────────────────────────
    def _load_sdss(self) -> pd.DataFrame:
        """Charge le cross-match Galaxy10 × SDSS DR16."""
        candidates = [
            self.data_dir / "galaxy10_sdss_xmatch.csv.gz",
            self.data_dir / "galaxy10_sdss_xmatch.csv",
            self.data_dir / "sdss_crossmatch.csv",
        ]
        for path in candidates:
            if path.exists():
                df = pd.read_csv(path)
                print(f"  ✓ SDSS chargé : {len(df):,} galaxies ← {path.name}")
                return df
        print("  ⚠ Aucun cache SDSS trouvé — DataFrame vide")
        return pd.DataFrame()

    def _load_morphometry(self) -> pd.DataFrame:
        """Charge les métriques CAS/Gini/M20 calculées en notebook 06."""
        candidates = [
            self.data_dir / "morphometry_df.pkl",
            self.data_dir / "morphometry_results.pkl",
            self.data_dir / "morphometry_df.csv",
        ]
        for path in candidates:
            if path.exists():
                if path.suffix == ".pkl":
                    df = pd.read_pickle(path)
                else:
                    df = pd.read_csv(path)
                print(f"  ✓ Morphométrie chargée : {len(df):,} galaxies ← {path.name}")
                return df
        print("  ⚠ Aucun cache morphométrie trouvé — DataFrame vide")
        return pd.DataFrame()

    def _load_unet_predictions(self) -> Optional[np.ndarray]:
        """Charge les prédictions U-Net (N × H × W) en uint8."""
        candidates = [
            self.data_dir / "unet_preds_all.npy",
            self.data_dir / "unet_predictions.npy",
            self.data_dir / "seg_predictions.npy",
        ]
        for path in candidates:
            if path.exists():
                arr = np.load(path)
                print(f"  ✓ Prédictions U-Net : {arr.shape} ← {path.name}")
                return arr
        print("  ⚠ Aucune prédiction U-Net trouvée — bulge fraction ignorée")
        return None

    def _load_dino_features(self) -> Optional[np.ndarray]:
        """Charge les features DINOv2 (N × 768)."""
        candidates = [
            self.data_dir / "dino_features.npy",
            self.data_dir / "features_dino.npy",
        ]
        for path in candidates:
            if path.exists():
                arr = np.load(path)
                print(f"  ✓ Features DINOv2 : {arr.shape} ← {path.name}")
                return arr
        print("  ⚠ Features DINOv2 non trouvées")
        return None

    def _load_galaxy10_labels(self) -> Optional[np.ndarray]:
        """Charge les labels Galaxy10 (N,) en int."""
        candidates = [
            self.data_dir / "galaxy10_labels.npy",
            self.data_dir / "labels_all.npy",
        ]
        for path in candidates:
            if path.exists():
                arr = np.load(path)
                print(f"  ✓ Labels Galaxy10 : {len(arr):,} ← {path.name}")
                return arr
        # Fallback : lire depuis HDF5
        h5_candidates = [
            self.data_dir / "Galaxy10_DECals.h5",
            self.data_dir.parent / "data" / "Galaxy10_DECals.h5",
            Path("../data/Galaxy10_DECals.h5"),
        ]
        for h5 in h5_candidates:
            if h5.exists():
                import h5py
                with h5py.File(h5, "r") as f:
                    arr = f["ans"][:].astype(int)
                print(f"  ✓ Labels Galaxy10 (HDF5) : {len(arr):,}")
                return arr
        print("  ⚠ Labels Galaxy10 non trouvés")
        return None

    # ── Calcul de la fraction bulbe depuis U-Net ──────────────────────────────
    @staticmethod
    def compute_bulge_fractions(unet_preds: np.ndarray) -> pd.DataFrame:
        """Calcule les fractions de pixels fond/disque/bulbe par galaxie.

        Args:
            unet_preds : array (N, H, W) uint8 — labels {0:fond, 1:disque, 2:bulbe}

        Returns:
            DataFrame avec colonnes : g10_idx, frac_fond, frac_disque, frac_bulbe,
                                      n_components (nombre de composantes connexes bulbe)
        """
        from scipy.ndimage import label as nd_label
        records = []
        n_pix = unet_preds.shape[1] * unet_preds.shape[2]

        for i, pred in enumerate(unet_preds):
            frac_fond    = (pred == 0).sum() / n_pix
            frac_disque  = (pred == 1).sum() / n_pix
            frac_bulbe   = (pred == 2).sum() / n_pix

            # Nombre de composantes connexes dans la région "bulbe"
            bulbe_mask        = (pred == 2).astype(np.uint8)
            _, n_comp_bulbe   = nd_label(bulbe_mask)

            records.append({
                "g10_idx":      i,
                "frac_fond":    frac_fond,
                "frac_disque":  frac_disque,
                "frac_bulbe":   frac_bulbe,
                "n_comp_bulbe": n_comp_bulbe,
            })
        return pd.DataFrame(records)

    # ── Construction du DataFrame maître ─────────────────────────────────────
    def build_master_dataframe(self,
                               unet_preds: Optional[np.ndarray] = None,
                               labels: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Fusionne toutes les sources de données par g10_idx.

        Args:
            unet_preds : si None, tente de charger depuis data_dir
            labels     : si None, tente de charger depuis data_dir

        Returns:
            DataFrame maître avec toutes les colonnes disponibles.
        """
        print("═" * 55)
        print("  CHARGEMENT DES DONNÉES")
        print("═" * 55)

        sdss_df  = self._load_sdss()
        morph_df = self._load_morphometry()

        if unet_preds is None:
            unet_preds = self._load_unet_predictions()

        if labels is None:
            labels = self._load_galaxy10_labels()

        # Base : index complet Galaxy10
        n_total = 17736  # Galaxy10 DECaLS
        if labels is not None:
            n_total = len(labels)
        base_df = pd.DataFrame({
            "g10_idx":    np.arange(n_total),
            "label":      labels if labels is not None else np.zeros(n_total, int),
        })
        if labels is not None:
            base_df["class_name"] = base_df["label"].map(
                lambda x: CLASS_NAMES[int(x)] if 0 <= int(x) < len(CLASS_NAMES) else "Unknown"
            )

        # Fusion morphométrie
        master = base_df.copy()
        if not morph_df.empty:
            merge_col = "g10_idx" if "g10_idx" in morph_df.columns else morph_df.index.name
            if merge_col is None or merge_col not in morph_df.columns:
                morph_df = morph_df.reset_index()
                morph_df = morph_df.rename(columns={"index": "g10_idx"})
            master = master.merge(morph_df, on="g10_idx", how="left")

        # Fusion SDSS
        if not sdss_df.empty:
            sdss_cols = ["g10_idx"] + [c for c in sdss_df.columns
                                        if c not in master.columns and c != "label"]
            master = master.merge(sdss_df[sdss_cols], on="g10_idx", how="left")

        # Fractions U-Net
        if unet_preds is not None:
            print("  Calcul des fractions bulbe...")
            bulge_df = self.compute_bulge_fractions(unet_preds)
            master   = master.merge(bulge_df, on="g10_idx", how="left")

        print(f"\n  ✓ DataFrame maître : {len(master):,} galaxies × {master.shape[1]} colonnes")
        matched = master["g-r"].notna().sum() if "g-r" in master.columns else 0
        print(f"  ✓ Avec données SDSS : {matched:,} galaxies")
        morph_ok = master["C"].notna().sum() if "C" in master.columns else 0
        print(f"  ✓ Avec morphométrie : {morph_ok:,} galaxies")
        bulge_ok = master["frac_bulbe"].notna().sum() if "frac_bulbe" in master.columns else 0
        print(f"  ✓ Avec U-Net segm. : {bulge_ok:,} galaxies")
        print("═" * 55)

        return master

    # ── Cross-match LAMOST ────────────────────────────────────────────────────
    def attempt_lamost_xmatch(self, master: pd.DataFrame,
                               radius_as: float = 5.0) -> pd.DataFrame:
        """Tente un cross-match CDS XMatch Galaxy10 × LAMOST DR7.

        LAMOST observe aussi des galaxies (class=GALAXY dans DR5/DR7).
        Si trouvé, ajoute les colonnes : lamost_z, lamost_snr, [Fe/H] proxy.

        Args:
            master    : DataFrame maître avec colonnes ra / dec
            radius_as : rayon de cross-match en arcsec

        Returns:
            master enrichi avec colonnes lamost_* si des matches sont trouvés.
        """
        cache_path = self.data_dir / "galaxy10_lamost_xmatch.csv.gz"

        if cache_path.exists():
            lamost_df = pd.read_csv(cache_path)
            print(f"✓ Cache LAMOST trouvé : {len(lamost_df):,} matches")
        else:
            print("Cross-match Galaxy10 × LAMOST DR7 (vizier:V/156/spectra)")
            print(f"  Rayon : {radius_as}\" | Sources : {len(master):,}")

            if "ra" not in master.columns or "dec" not in master.columns:
                print("  ⚠ Colonnes ra/dec manquantes — cross-match impossible")
                return master

            try:
                import time
                from astroquery.xmatch import XMatch
                from astropy import units as u
                from astropy.table import Table

                valid = master.dropna(subset=["ra", "dec"])[["g10_idx", "ra", "dec"]]
                tbl   = Table.from_pandas(valid)

                # LAMOST DR7 general catalog sur VizieR
                LAMOST_CAT = "vizier:V/156/spectra"
                res = XMatch.query(
                    cat1=tbl, cat2=LAMOST_CAT,
                    max_distance=radius_as * u.arcsec,
                    colRA1="ra", colDec1="dec",
                )

                if res is None or len(res) == 0:
                    print("  ⚠ Aucun match LAMOST DR7 — essai LAMOST DR5...")
                    # Fallback : LAMOST DR5 via VizieR
                    LAMOST_DR5 = "vizier:J/AJ/157/168"
                    res = XMatch.query(
                        cat1=tbl, cat2=LAMOST_DR5,
                        max_distance=radius_as * u.arcsec,
                        colRA1="ra", colDec1="dec",
                    )

                if res is not None and len(res) > 0:
                    lamost_df = res.to_pandas()
                    lamost_df = lamost_df.sort_values("angDist").drop_duplicates(
                        subset=["g10_idx"], keep="first")
                    lamost_df.to_csv(cache_path, index=False)
                    print(f"  ✓ {len(lamost_df):,} matches LAMOST trouvés")
                else:
                    print("  ⚠ Aucun match LAMOST — le footprint LAMOST ne couvre pas assez Galaxy10")
                    return master

            except Exception as e:
                print(f"  ⚠ Erreur XMatch LAMOST : {e}")
                return master

        # Renommer les colonnes LAMOST utiles
        rename_map = {}
        for col in lamost_df.columns:
            cl = col.lower()
            if "feh" in cl or "fe_h" in cl:
                rename_map[col] = "lamost_feh"
            elif "snr" in cl and "snrg" not in cl:
                rename_map[col] = "lamost_snr"
            elif cl in ("z", "redshift"):
                rename_map[col] = "lamost_z"
            elif "class" in cl:
                rename_map[col] = "lamost_class"
        lamost_df = lamost_df.rename(columns=rename_map)

        cols_to_add = ["g10_idx"] + [c for c in lamost_df.columns
                                      if c.startswith("lamost_")]
        master = master.merge(lamost_df[cols_to_add], on="g10_idx", how="left")
        n_matched = master["lamost_z"].notna().sum() if "lamost_z" in master.columns else 0
        print(f"✓ {n_matched:,} galaxies avec données LAMOST fusionnées")
        return master

    # ── Tests statistiques ────────────────────────────────────────────────────
    @staticmethod
    def spearman_matrix(df: pd.DataFrame,
                        x_cols: list, y_cols: list) -> pd.DataFrame:
        """Matrice de corrélation Spearman avec p-values.

        Returns:
            DataFrame (x_cols × y_cols) avec format 'ρ (p=p_val)'
        """
        rows = []
        for x in x_cols:
            row = {}
            for y in y_cols:
                sub = df[[x, y]].dropna()
                if len(sub) < 10:
                    row[y] = "n/a"
                    continue
                rho, pval = spearmanr(sub[x], sub[y])
                sig = "*" if pval < 0.05 else ""
                row[y] = f"{rho:+.3f}{sig}"
            rows.append(row)
        return pd.DataFrame(rows, index=x_cols)

    @staticmethod
    def kruskal_by_class(df: pd.DataFrame, col: str,
                          class_col: str = "class_name") -> dict:
        """Test de Kruskal-Wallis : est-ce que `col` diffère entre les classes ?

        Returns:
            dict : stat, pval, significatif (bool)
        """
        groups = [g[col].dropna().values
                  for _, g in df.groupby(class_col)
                  if g[col].notna().sum() >= 5]
        if len(groups) < 2:
            return {"stat": np.nan, "pval": np.nan, "significatif": False}
        stat, pval = kruskal(*groups)
        return {"stat": float(stat), "pval": float(pval), "significatif": pval < 0.05}

    # ═══════════════════════════════════════════════════════════════════════════
    # FIGURES
    # ═══════════════════════════════════════════════════════════════════════════

    def fig_bulge_vs_color(self, master: pd.DataFrame,
                            save: bool = True) -> plt.Figure:
        """Figure 1 : Fraction bulbe U-Net × couleur g-r SDSS.

        Montre si les galaxies avec un grand bulbe sont plus rouges.
        """
        needed = ["frac_bulbe", "g-r", "class_name"]
        df = master.dropna(subset=[c for c in needed if c in master.columns])
        if df.empty or "frac_bulbe" not in df.columns or "g-r" not in df.columns:
            print("  ⚠ Données insuffisantes pour fig_bulge_vs_color")
            return plt.figure()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Fraction bulbe U-Net × Couleur photométrique g-r (SDSS DR16)",
                     fontsize=14, fontweight="bold", y=1.02)

        # 1 — Scatter global
        ax = axes[0]
        sc = ax.scatter(df["frac_bulbe"], df["g-r"],
                        c=df["label"] if "label" in df.columns else "#23abd8",
                        cmap="turbo", alpha=0.4, s=8, rasterized=True)
        # Régression
        rho, pval = spearmanr(df["frac_bulbe"], df["g-r"])
        x_lin = np.linspace(df["frac_bulbe"].min(), df["frac_bulbe"].max(), 100)
        m, b   = np.polyfit(df["frac_bulbe"], df["g-r"].clip(-1, 3), 1)
        ax.plot(x_lin, m * x_lin + b, color="#f83e4b", lw=2, ls="--",
                label=f"Régression (ρ={rho:+.3f}, p={pval:.3f})")
        ax.axhline(0.6,  color="#23abd8", ls=":", lw=1.5, label="Blue/Green (g-r=0.6)")
        ax.axhline(0.75, color="#f83e4b", ls=":", lw=1.5, label="Green/Red (g-r=0.75)")
        ax.set_xlabel("Fraction bulbe (U-Net)", fontsize=11)
        ax.set_ylabel("Couleur g-r (SDSS)", fontsize=11)
        ax.set_title("Scatter global", fontsize=11)
        ax.set_ylim(-0.5, 3)
        ax.legend(fontsize=8)

        # 2 — Médiane par classe (barh)
        ax = axes[1]
        order = (df.groupby("class_name")["frac_bulbe"]
                   .median()
                   .reindex(CLASS_NAMES)
                   .dropna()
                   .sort_values())
        colors_bar = [PALETTE[CLASS_NAMES.index(c)] for c in order.index]
        ax.barh(range(len(order)), order.values, color=colors_bar, alpha=0.85)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([CLASS_SHORT[CLASS_NAMES.index(c)] for c in order.index],
                           fontsize=9)
        ax.axvline(order.values.mean(), color="#f0f6fc", ls="--", lw=1.5,
                   label=f"Moyenne = {order.values.mean():.3f}")
        ax.set_xlabel("Fraction bulbe médiane", fontsize=11)
        ax.set_title("Par classe morphologique", fontsize=11)
        ax.legend(fontsize=9)

        # 3 — Boxplot frac_bulbe × population stellaire
        ax = axes[2]
        if "population" in df.columns:
            pops = ["Blue Cloud", "Green Valley", "Red Sequence"]
            data_pop = [df.loc[df["population"] == p, "frac_bulbe"].dropna()
                        for p in pops]
            bp = ax.boxplot(data_pop, patch_artist=True, notch=False,
                            medianprops=dict(color="#f0f6fc", lw=2))
            for patch, pop in zip(bp["boxes"], pops):
                patch.set_facecolor(POP_COLORS[pop])
                patch.set_alpha(0.75)
            ax.set_xticklabels(pops, fontsize=9)
            ax.set_ylabel("Fraction bulbe (U-Net)", fontsize=11)
            ax.set_title("Par population stellaire", fontsize=11)
            # Kruskal-Wallis
            stat, pval2 = kruskal(*[d for d in data_pop if len(d) > 0])
            ax.text(0.98, 0.98, f"Kruskal-Wallis\np={pval2:.4f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, color="#8b949e")
        else:
            ax.text(0.5, 0.5, "Données population\nnon disponibles",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#8b949e", fontsize=12)

        plt.tight_layout()
        if save:
            path = self.figures_dir / "syn_bulge_vs_color.png"
            fig.savefig(path, dpi=150)
            print(f"  ✓ Sauvegardé : {path.name}")
        return fig

    def fig_cas_population(self, master: pd.DataFrame,
                            save: bool = True) -> plt.Figure:
        """Figure 2 : Métriques CAS × Population stellaire.

        Montre si l'asymétrie et le Gini diffèrent entre Blue Cloud / Red Sequence.
        """
        metrics = ["C", "A", "S", "Gini", "M20"]
        available = [m for m in metrics if m in master.columns]
        if not available or "population" not in master.columns:
            print("  ⚠ Données insuffisantes pour fig_cas_population")
            return plt.figure()

        df = master.dropna(subset=available + ["population"])
        pops = ["Blue Cloud", "Green Valley", "Red Sequence"]
        n = len(available)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 6))
        if n == 1:
            axes = [axes]
        fig.suptitle("Métriques morphologiques CAS/Gini/M20 × Population stellaire",
                     fontsize=13, fontweight="bold")

        for ax, metric in zip(axes, available):
            data_pop = [df.loc[df["population"] == p, metric].dropna() for p in pops]
            bp = ax.violinplot(data_pop, showmedians=True, showextrema=False)
            for body, pop in zip(bp["bodies"], pops):
                body.set_facecolor(POP_COLORS[pop])
                body.set_alpha(0.7)
            bp["cmedians"].set_color("#f0f6fc")
            bp["cmedians"].set_linewidth(2)
            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels(["Blue\nCloud", "Green\nValley", "Red\nSeq."], fontsize=9)
            ax.set_title(metric, fontsize=12, fontweight="bold")
            ax.set_ylabel(metric, fontsize=10)

            # Kruskal-Wallis p-value
            valid_groups = [d for d in data_pop if len(d) >= 5]
            if len(valid_groups) >= 2:
                _, pv = kruskal(*valid_groups)
                sig = "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else "ns"))
                ax.text(0.98, 0.98, f"KW: {sig}\n(p={pv:.3f})",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=8, color="#8b949e")

        plt.tight_layout()
        if save:
            path = self.figures_dir / "syn_cas_population.png"
            fig.savefig(path, dpi=150)
            print(f"  ✓ Sauvegardé : {path.name}")
        return fig

    def fig_dino_synthesis(self, master: pd.DataFrame,
                            dino_features: Optional[np.ndarray] = None,
                            save: bool = True) -> plt.Figure:
        """Figure 3 : DINOv2 PCA × (g-r, Gini, population).

        Montre que PC2 encode à la fois la structure morphologique et la couleur.
        """
        if dino_features is None:
            dino_features = self._load_dino_features()
        if dino_features is None:
            print("  ⚠ Features DINOv2 manquantes pour fig_dino_synthesis")
            return plt.figure()

        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        # PCA sur les features DINOv2
        scaler = StandardScaler()
        feats  = scaler.fit_transform(dino_features)
        pca    = PCA(n_components=5, random_state=42)
        pcs    = pca.fit_transform(feats)   # (N, 5)

        # Aligner les indices avec master
        n_min  = min(len(pcs), len(master))
        df_pca = master.iloc[:n_min].copy().reset_index(drop=True)
        for i in range(pcs.shape[1]):
            df_pca[f"PC{i+1}"] = pcs[:n_min, i]

        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle(
            f"Espace latent DINOv2 — PCA synthèse  "
            f"(PC1={pca.explained_variance_ratio_[0]*100:.1f}%  "
            f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%)",
            fontsize=14, fontweight="bold",
        )

        # Panel 0,0 — PC1 vs PC2 coloré par classe
        ax = axes[0, 0]
        if "label" in df_pca.columns:
            for cls_i, cls_name in enumerate(CLASS_NAMES):
                mask = df_pca["label"] == cls_i
                ax.scatter(df_pca.loc[mask, "PC1"], df_pca.loc[mask, "PC2"],
                           color=PALETTE[cls_i], label=CLASS_SHORT[cls_i],
                           s=6, alpha=0.5, rasterized=True)
        ax.set_xlabel("PC1", fontsize=10)
        ax.set_ylabel("PC2", fontsize=10)
        ax.set_title("PC1 vs PC2 — classes morphologiques", fontsize=11)
        ax.legend(fontsize=6, ncol=2, markerscale=2)

        # Panel 0,1 — PC2 vs g-r
        ax = axes[0, 1]
        if "g-r" in df_pca.columns:
            sub = df_pca.dropna(subset=["g-r"])
            gr_clip = sub["g-r"].clip(-0.5, 2.5)
            sc = ax.scatter(sub["PC2"], gr_clip,
                            c=gr_clip, cmap="RdYlBu_r",
                            s=8, alpha=0.5, rasterized=True, vmin=0, vmax=1.5)
            plt.colorbar(sc, ax=ax, label="g-r")
            rho, pval = spearmanr(sub["PC2"], sub["g-r"])
            ax.set_title(f"PC2 vs g-r  (ρ={rho:+.3f}, p={pval:.3f})", fontsize=11)
            ax.set_xlabel("PC2", fontsize=10)
            ax.set_ylabel("g-r (SDSS)", fontsize=10)
        else:
            ax.text(0.5, 0.5, "g-r non disponible\n(SDSS requis)",
                    ha="center", va="center", transform=ax.transAxes, color="#8b949e")

        # Panel 0,2 — PC2 vs Gini
        ax = axes[0, 2]
        if "Gini" in df_pca.columns:
            sub = df_pca.dropna(subset=["Gini"])
            sc = ax.scatter(sub["PC2"], sub["Gini"],
                            c=sub["Gini"], cmap="plasma",
                            s=8, alpha=0.5, rasterized=True)
            plt.colorbar(sc, ax=ax, label="Gini")
            rho, pval = spearmanr(sub["PC2"], sub["Gini"])
            ax.set_title(f"PC2 vs Gini  (ρ={rho:+.3f}, p={pval:.3f})", fontsize=11)
            ax.set_xlabel("PC2", fontsize=10)
            ax.set_ylabel("Gini", fontsize=10)
        else:
            ax.text(0.5, 0.5, "Gini non disponible\n(morphométrie requise)",
                    ha="center", va="center", transform=ax.transAxes, color="#8b949e")

        # Panel 1,0 — Variance expliquée
        ax = axes[1, 0]
        var = pca.explained_variance_ratio_[:10] * 100 if len(pca.explained_variance_ratio_) >= 10 else pca.explained_variance_ratio_ * 100
        ax.bar(range(1, len(var)+1), var, color=PALETTE[:len(var)], alpha=0.8)
        ax.set_xlabel("Composante principale", fontsize=10)
        ax.set_ylabel("Variance expliquée (%)", fontsize=10)
        ax.set_title("Variance expliquée PCA(DINOv2)", fontsize=11)
        for i, v in enumerate(var):
            ax.text(i+1, v+0.1, f"{v:.1f}%", ha="center", fontsize=7)

        # Panel 1,1 — PC1/PC2 coloré par g-r (heatmap)
        ax = axes[1, 1]
        if "g-r" in df_pca.columns:
            sub = df_pca.dropna(subset=["g-r"])
            sc = ax.scatter(sub["PC1"], sub["PC2"],
                            c=sub["g-r"].clip(0, 1.5), cmap="RdYlBu_r",
                            s=6, alpha=0.5, rasterized=True, vmin=0, vmax=1.5)
            plt.colorbar(sc, ax=ax, label="g-r")
            ax.set_xlabel("PC1", fontsize=10)
            ax.set_ylabel("PC2", fontsize=10)
            ax.set_title("Espace latent coloré par g-r", fontsize=11)
        else:
            ax.text(0.5, 0.5, "g-r non disponible",
                    ha="center", va="center", transform=ax.transAxes, color="#8b949e")

        # Panel 1,2 — PC1/PC2 coloré par population
        ax = axes[1, 2]
        if "population" in df_pca.columns:
            pops_order = ["Blue Cloud", "Green Valley", "Red Sequence"]
            for pop in pops_order:
                sub = df_pca[df_pca["population"] == pop]
                ax.scatter(sub["PC1"], sub["PC2"],
                           color=POP_COLORS[pop], label=pop,
                           s=6, alpha=0.6, rasterized=True)
            ax.set_xlabel("PC1", fontsize=10)
            ax.set_ylabel("PC2", fontsize=10)
            ax.set_title("Espace latent — population stellaire", fontsize=11)
            ax.legend(fontsize=9, markerscale=2)
        else:
            ax.text(0.5, 0.5, "Population non disponible\n(SDSS requis)",
                    ha="center", va="center", transform=ax.transAxes, color="#8b949e")

        plt.tight_layout()
        if save:
            path = self.figures_dir / "syn_dino_synthesis.png"
            fig.savefig(path, dpi=150)
            print(f"  ✓ Sauvegardé : {path.name}")
        return fig

    def fig_triple_synthesis(self, master: pd.DataFrame,
                              save: bool = True) -> plt.Figure:
        """Figure 4 : Triple croisement — fraction_bulbe × g-r × Gini.

        C'est la figure de synthèse principale pour la publication.
        """
        needed = ["frac_bulbe", "g-r", "Gini", "class_name"]
        available = [c for c in needed if c in master.columns]
        df = master.dropna(subset=available)

        if len(available) < 2 or df.empty:
            print("  ⚠ Données insuffisantes pour fig_triple_synthesis")
            return plt.figure()

        fig = plt.figure(figsize=(18, 10))
        gs  = gridspec.GridSpec(2, 3, figure=fig,
                                hspace=0.35, wspace=0.3)
        fig.suptitle(
            "Synthèse AstroVision — Morphologie × Photométrie × Segmentation",
            fontsize=14, fontweight="bold",
        )

        # Panel A — Gini vs frac_bulbe (si les deux disponibles)
        ax = fig.add_subplot(gs[0, 0])
        if "Gini" in df.columns and "frac_bulbe" in df.columns:
            sub = df.dropna(subset=["Gini", "frac_bulbe"])
            if "label" in sub.columns:
                for i, cls in enumerate(CLASS_NAMES):
                    m = sub["label"] == i
                    ax.scatter(sub.loc[m, "frac_bulbe"], sub.loc[m, "Gini"],
                               color=PALETTE[i], s=8, alpha=0.5, label=CLASS_SHORT[i],
                               rasterized=True)
            rho, pval = spearmanr(sub["frac_bulbe"], sub["Gini"])
            ax.set_xlabel("Fraction bulbe (U-Net)", fontsize=10)
            ax.set_ylabel("Gini", fontsize=10)
            ax.set_title(f"Gini vs Fraction bulbe\nρ={rho:+.3f} (p={pval:.3f})", fontsize=10)
        else:
            ax.text(0.5, 0.5, "Gini ou frac_bulbe\nnon disponible",
                    ha="center", va="center", transform=ax.transAxes, color="#8b949e")

        # Panel B — g-r vs frac_bulbe
        ax = fig.add_subplot(gs[0, 1])
        if "g-r" in df.columns and "frac_bulbe" in df.columns:
            sub = df.dropna(subset=["g-r", "frac_bulbe"])
            gr_clip = sub["g-r"].clip(-0.5, 2)
            sc = ax.scatter(sub["frac_bulbe"], gr_clip,
                            c=gr_clip, cmap="RdYlBu_r", s=8, alpha=0.4,
                            rasterized=True, vmin=0.2, vmax=1.4)
            plt.colorbar(sc, ax=ax, label="g-r")
            rho, pval = spearmanr(sub["frac_bulbe"], sub["g-r"])
            ax.axhline(0.6,  color="#23abd8", ls=":", lw=1, alpha=0.8)
            ax.axhline(0.75, color="#f83e4b", ls=":", lw=1, alpha=0.8)
            ax.set_xlabel("Fraction bulbe (U-Net)", fontsize=10)
            ax.set_ylabel("g-r (SDSS)", fontsize=10)
            ax.set_title(f"g-r vs Fraction bulbe\nρ={rho:+.3f} (p={pval:.3f})", fontsize=10)
        else:
            ax.text(0.5, 0.5, "g-r ou frac_bulbe\nnon disponible",
                    ha="center", va="center", transform=ax.transAxes, color="#8b949e")

        # Panel C — Gini vs g-r
        ax = fig.add_subplot(gs[0, 2])
        if "Gini" in df.columns and "g-r" in df.columns:
            sub = df.dropna(subset=["Gini", "g-r"])
            gr_clip = sub["g-r"].clip(-0.5, 2)
            sc = ax.scatter(sub["Gini"], gr_clip,
                            c=gr_clip, cmap="RdYlBu_r", s=8, alpha=0.4,
                            rasterized=True, vmin=0.2, vmax=1.4)
            plt.colorbar(sc, ax=ax, label="g-r")
            rho, pval = spearmanr(sub["Gini"], sub["g-r"])
            ax.set_xlabel("Gini", fontsize=10)
            ax.set_ylabel("g-r (SDSS)", fontsize=10)
            ax.set_title(f"g-r vs Gini\nρ={rho:+.3f} (p={pval:.3f})", fontsize=10)
        else:
            ax.text(0.5, 0.5, "Gini ou g-r\nnon disponible",
                    ha="center", va="center", transform=ax.transAxes, color="#8b949e")

        # Panel D — Heatmap corrélations Spearman
        ax = fig.add_subplot(gs[1, 0])
        morph_cols   = [c for c in ["C", "A", "S", "Gini", "M20"] if c in df.columns]
        phot_cols    = [c for c in ["g-r", "u-r", "Mr"] if c in df.columns]
        seg_cols     = [c for c in ["frac_bulbe", "frac_disque"] if c in df.columns]
        all_x = morph_cols + seg_cols
        all_y = phot_cols
        if all_x and all_y:
            rho_mat = np.zeros((len(all_x), len(all_y)))
            sig_mat = np.zeros((len(all_x), len(all_y)), dtype=bool)
            for i, x in enumerate(all_x):
                for j, y in enumerate(all_y):
                    sub = df[[x, y]].dropna()
                    if len(sub) >= 10:
                        rho, pval = spearmanr(sub[x], sub[y])
                        rho_mat[i, j] = rho
                        sig_mat[i, j] = pval < 0.05
            im = ax.imshow(rho_mat, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
            plt.colorbar(im, ax=ax, label="ρ Spearman")
            ax.set_xticks(range(len(all_y)))
            ax.set_xticklabels(all_y, fontsize=9)
            ax.set_yticks(range(len(all_x)))
            ax.set_yticklabels(all_x, fontsize=9)
            for i in range(len(all_x)):
                for j in range(len(all_y)):
                    txt  = f"{rho_mat[i,j]:+.2f}"
                    txt += "*" if sig_mat[i, j] else ""
                    ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                            color="#f0f6fc")
            ax.set_title("Corrélation Spearman\n(* p<0.05)", fontsize=10)
        else:
            ax.text(0.5, 0.5, "Données insuffisantes\npour heatmap",
                    ha="center", va="center", transform=ax.transAxes, color="#8b949e")

        # Panel E — Fraction bulbe par population (violin)
        ax = fig.add_subplot(gs[1, 1])
        if "frac_bulbe" in df.columns and "population" in df.columns:
            pops = ["Blue Cloud", "Green Valley", "Red Sequence"]
            data_pop = [df.loc[df["population"] == p, "frac_bulbe"].dropna() for p in pops]
            vp = ax.violinplot(data_pop, showmedians=True, showextrema=False)
            for body, pop in zip(vp["bodies"], pops):
                body.set_facecolor(POP_COLORS[pop])
                body.set_alpha(0.75)
            vp["cmedians"].set_color("#f0f6fc")
            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels(["Blue\nCloud", "Green\nValley", "Red\nSeq."], fontsize=9)
            ax.set_ylabel("Fraction bulbe (U-Net)", fontsize=10)
            ax.set_title("Fraction bulbe × Population", fontsize=10)
            valid = [d for d in data_pop if len(d) >= 5]
            if len(valid) >= 2:
                _, pval = kruskal(*valid)
                ax.text(0.98, 0.98, f"KW p={pval:.3f}",
                        transform=ax.transAxes, ha="right", va="top", fontsize=8)
        else:
            ax.text(0.5, 0.5, "Données population\nnon disponibles",
                    ha="center", va="center", transform=ax.transAxes, color="#8b949e")

        # Panel F — Résumé par classe : médiane Gini + médiane g-r
        ax = fig.add_subplot(gs[1, 2])
        if "Gini" in df.columns and "g-r" in df.columns and "class_name" in df.columns:
            summary = (df.groupby("class_name")
                         .agg(gini_med=("Gini", "median"),
                              gr_med=("g-r", "median"))
                         .reindex(CLASS_NAMES)
                         .dropna())
            colors_cls = [PALETTE[CLASS_NAMES.index(c)] for c in summary.index]
            sc = ax.scatter(summary["gini_med"], summary["gr_med"],
                            c=colors_cls, s=120, zorder=5)
            for name, row in summary.iterrows():
                ax.annotate(CLASS_SHORT[CLASS_NAMES.index(name)],
                            (row["gini_med"], row["gr_med"]),
                            fontsize=7, ha="left", va="bottom",
                            xytext=(3, 3), textcoords="offset points")
            ax.axhline(0.6,  color="#23abd8", ls=":", lw=1, alpha=0.8)
            ax.axhline(0.75, color="#f83e4b", ls=":", lw=1, alpha=0.8)
            ax.set_xlabel("Gini médian", fontsize=10)
            ax.set_ylabel("g-r médian (SDSS)", fontsize=10)
            ax.set_title("Résumé par classe\n(médiane Gini × g-r)", fontsize=10)
        else:
            ax.text(0.5, 0.5, "Données insuffisantes",
                    ha="center", va="center", transform=ax.transAxes, color="#8b949e")

        if save:
            path = self.figures_dir / "syn_triple_synthesis.png"
            fig.savefig(path, dpi=150)
            print(f"  ✓ Sauvegardé : {path.name}")
        return fig

    def fig_lamost_bridge(self, master: pd.DataFrame,
                           save: bool = True) -> plt.Figure:
        """Figure 5 : Bridge AstroSpectro × AstroVision.

        Si des données LAMOST sont disponibles, montre [Fe/H] × morphologie.
        Sinon, affiche un diagramme récapitulatif de l'écosystème AstroAI.
        """
        has_lamost = "lamost_feh" in master.columns and master["lamost_feh"].notna().any()

        if has_lamost:
            df = master.dropna(subset=["lamost_feh"])
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle("[Fe/H] LAMOST × Morphologie Galaxy10 — Bridge AstroSpectro × AstroVision",
                         fontsize=13, fontweight="bold")

            # [Fe/H] par classe
            ax = axes[0]
            order = (df.groupby("class_name")["lamost_feh"]
                       .median()
                       .reindex(CLASS_NAMES)
                       .dropna()
                       .sort_values())
            colors_bar = [PALETTE[CLASS_NAMES.index(c)] for c in order.index]
            ax.barh(range(len(order)), order.values, color=colors_bar, alpha=0.85)
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels([CLASS_SHORT[CLASS_NAMES.index(c)] for c in order.index])
            ax.axvline(df["lamost_feh"].median(), color="#f0f6fc", ls="--", lw=1.5)
            ax.set_xlabel("[Fe/H] médian (LAMOST)", fontsize=11)
            ax.set_title("[Fe/H] par classe morphologique", fontsize=11)

            # [Fe/H] vs g-r
            ax = axes[1]
            if "g-r" in df.columns:
                sub = df.dropna(subset=["lamost_feh", "g-r"])
                ax.scatter(sub["lamost_feh"], sub["g-r"].clip(-0.5, 2),
                           c=[PALETTE[int(l)] if "label" in sub.columns
                              else "#23abd8" for l in sub.get("label", [0]*len(sub))],
                           s=10, alpha=0.5, rasterized=True)
                rho, pval = spearmanr(sub["lamost_feh"], sub["g-r"])
                ax.set_xlabel("[Fe/H] (LAMOST)", fontsize=11)
                ax.set_ylabel("g-r (SDSS)", fontsize=11)
                ax.set_title(f"[Fe/H] vs g-r\nρ={rho:+.3f} (p={pval:.3f})", fontsize=11)

            # [Fe/H] vs Gini
            ax = axes[2]
            if "Gini" in df.columns:
                sub = df.dropna(subset=["lamost_feh", "Gini"])
                ax.scatter(sub["lamost_feh"], sub["Gini"],
                           c=[PALETTE[int(l)] for l in sub.get("label", [0]*len(sub))],
                           s=10, alpha=0.5, rasterized=True)
                rho, pval = spearmanr(sub["lamost_feh"], sub["Gini"])
                ax.set_xlabel("[Fe/H] (LAMOST)", fontsize=11)
                ax.set_ylabel("Gini", fontsize=11)
                ax.set_title(f"[Fe/H] vs Gini\nρ={rho:+.3f} (p={pval:.3f})", fontsize=11)

        else:
            # Pas de données LAMOST → diagramme de l'écosystème
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 6)
            ax.axis("off")
            fig.suptitle("Écosystème AstroAI — Bridge spectroscopie × imagerie",
                         fontsize=14, fontweight="bold")

            # Boîtes
            boxes = [
                (0.5, 2.5, 2.5, 3, "#6e40aa", "AstroSpectro\n(LAMOST DR5\n43k spectres)"),
                (3.5, 2.5, 2.5, 3, "#1ac7c2", "Cross-match\n(CDS XMatch\n1 arcsec)"),
                (6.5, 2.5, 2.5, 3, "#1ddfa3", "AstroVision\n(Galaxy10\n17k images)"),
            ]
            for x, y, w, h, col, txt in boxes:
                rect = plt.Rectangle((x, y), w, h, fc=col, ec="none", alpha=0.25)
                ax.add_patch(rect)
                ax.text(x + w/2, y + h/2, txt, ha="center", va="center",
                        fontsize=12, fontweight="bold", color="#f0f6fc")

            # Flèches
            for x1, x2 in [(3.0, 3.5), (6.0, 6.5)]:
                ax.annotate("", xy=(x2, 4.0), xytext=(x1, 4.0),
                            arrowprops=dict(arrowstyle="->", color="#8b949e", lw=2))

            ax.text(3.25, 4.3, "RA/Dec\nmatch", ha="center", fontsize=9, color="#8b949e")
            ax.text(6.25, 4.3, "Features\nDINOv2", ha="center", fontsize=9, color="#8b949e")
            ax.text(5, 1.5,
                    "→  Corrélation [Fe/H] × morphologie visuelle  ←\n"
                    "       (cross-match LAMOST : tentative future)      ",
                    ha="center", fontsize=12, color="#e2b72f", style="italic",
                    fontweight="bold")

        plt.tight_layout()
        if save:
            path = self.figures_dir / "syn_lamost_bridge.png"
            fig.savefig(path, dpi=150)
            print(f"  ✓ Sauvegardé : {path.name}")
        return fig

    def build_summary_table(self, master: pd.DataFrame) -> pd.DataFrame:
        """Construit la table de résultats statistiques par classe.

        Colonnes : classe, n, g-r médian, Gini médian, frac_bulbe médiane,
                   % Blue Cloud, % Red Sequence, mIoU U-Net (si disponible)
        """
        records = []
        for i, cls in enumerate(CLASS_NAMES):
            if "class_name" in master.columns:
                sub = master[master["class_name"] == cls]
            elif "label" in master.columns:
                sub = master[master["label"] == i]
            else:
                continue

            rec = {"Classe": cls, "N total": len(sub)}

            for col, label in [("g-r", "g-r médian"), ("Gini", "Gini médian"),
                                ("frac_bulbe", "Frac. bulbe médiane"),
                                ("C", "C médian"), ("A", "A médian")]:
                if col in sub.columns:
                    rec[label] = f"{sub[col].median():.3f}" if sub[col].notna().any() else "—"

            if "population" in sub.columns:
                n_match = sub["population"].notna().sum()
                rec["N SDSS"] = n_match
                if n_match > 0:
                    rec["% Blue Cloud"]   = f"{(sub['population']=='Blue Cloud').sum()/n_match*100:.0f}%"
                    rec["% Red Sequence"] = f"{(sub['population']=='Red Sequence').sum()/n_match*100:.0f}%"

            records.append(rec)

        df_summary = pd.DataFrame(records)
        return df_summary

    # ── Runner principal ──────────────────────────────────────────────────────
    def run_all_figures(self, master: pd.DataFrame,
                         dino_features: Optional[np.ndarray] = None) -> dict:
        """Génère toutes les figures de synthèse.

        Returns:
            dict figure_name → plt.Figure
        """
        print("\n" + "═"*55)
        print("  GÉNÉRATION DES FIGURES DE SYNTHÈSE")
        print("═"*55)
        figs = {}
        figs["bulge_vs_color"]   = self.fig_bulge_vs_color(master)
        figs["cas_population"]   = self.fig_cas_population(master)
        figs["dino_synthesis"]   = self.fig_dino_synthesis(master, dino_features)
        figs["triple_synthesis"] = self.fig_triple_synthesis(master)
        figs["lamost_bridge"]    = self.fig_lamost_bridge(master)
        print(f"\n  ✓ {len(figs)} figures générées → {self.figures_dir.resolve()}")
        return figs
