"""
AstroVision — crossmatcher.py
Cross-match Galaxy10 DECaLS avec les catalogues astronomiques via CDS XMatch.

Catalogues supportés :
    - SDSS DR16  (vizier:V/154/sdss16)  — photométrie + z_spec si disponible
    - NSA v1.0.1 (vizier:J/AJ/149/77)   — masses stellaires, SFR Hα, Sérsic index
    - 2MRS       (vizier:J/ApJS/199/26)  — redshifts du 2MASS Redshift Survey

Usage:
    from astrovision.crossmatcher import Galaxy10Crossmatcher
    xm = Galaxy10Crossmatcher(h5_path='../data/Galaxy10_DECals.h5',
                               cache_dir='../data/')
    df = xm.run_sdss()
    df = xm.run_nsa()
"""

import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Constantes ─────────────────────────────────────────────────────────────────
SDSS_CATALOG = "vizier:V/154/sdss16"
NSA_CATALOG  = "vizier:J/AJ/149/77"

CLASS_NAMES = [
    "Disturbed", "Merging", "Round Smooth", "In-between Round Smooth",
    "Cigar Shaped Smooth", "Barred Spiral", "Unbarred Tight Spiral",
    "Unbarred Loose Spiral", "Edge-on without Bulge", "Edge-on with Bulge",
]


class Galaxy10Crossmatcher:
    """Cross-match Galaxy10 DECaLS avec les catalogues via CDS XMatch.

    Toutes les requêtes sont mises en cache (CSV compressé) pour éviter
    de solliciter CDS à chaque run.

    Args:
        h5_path   : chemin vers Galaxy10_DECals.h5
        cache_dir : dossier de cache (créé si absent)
        radius_as : rayon de cross-match en arcsec (défaut 3.0)
    """

    def __init__(self, h5_path: str, cache_dir: str = "../data/",
                 radius_as: float = 3.0):
        self.h5_path   = Path(h5_path)
        self.cache_dir = Path(cache_dir)
        self.radius_as = radius_as
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Chargement des positions Galaxy10
        self._g10 = self._load_galaxy10()
        print(f"✓ Galaxy10 chargé : {len(self._g10)} galaxies")
        print(f"  Positions valides : {(~self._g10['ra'].isna()).sum():,}")
        print(f"  Cache dir : {self.cache_dir.resolve()}")

    # ── Chargement Galaxy10 ────────────────────────────────────────────────────
    def _load_galaxy10(self) -> pd.DataFrame:
        """Lit les colonnes utiles depuis le HDF5 Galaxy10."""
        with h5py.File(self.h5_path, "r") as f:
            data = {
                "g10_idx": np.arange(len(f["ans"][:])),
                "label":   f["ans"][:].astype(int),
                "ra":      f["ra"][:],
                "dec":     f["dec"][:],
                "z_photo": f["redshift"][:],
                "pxscale": f["pxscale"][:],
            }
        df = pd.DataFrame(data)
        df["class_name"] = df["label"].map(lambda x: CLASS_NAMES[x])

        # Nettoyage
        df["ra"]      = pd.to_numeric(df["ra"],      errors="coerce")
        df["dec"]     = pd.to_numeric(df["dec"],     errors="coerce")
        df["z_photo"] = pd.to_numeric(df["z_photo"], errors="coerce")
        df["pxscale"] = pd.to_numeric(df["pxscale"], errors="coerce")
        return df

    # ── Cross-match générique ──────────────────────────────────────────────────
    def _xmatch(self, catalog: str, cache_name: str,
                batch_size: int = 5000) -> pd.DataFrame:
        """Lance le cross-match CDS XMatch par batches pour éviter les timeouts.

        Args:
            catalog    : identifiant VizieR (ex: 'vizier:V/154/sdss16')
            cache_name : nom du fichier CSV de cache
            batch_size : taille des batches envoyés à CDS

        Returns:
            DataFrame avec les colonnes Galaxy10 + colonnes du catalogue externe
        """
        cache_path = self.cache_dir / f"{cache_name}.csv.gz"

        if cache_path.exists():
            print(f"✓ Cache trouvé : {cache_path.name} — rechargement...")
            df = pd.read_csv(cache_path)
            print(f"  {len(df):,} correspondances chargées")
            return df

        try:
            from astroquery.xmatch import XMatch
            from astropy import units as u
            from astropy.table import Table
        except ImportError:
            raise ImportError(
                "astroquery requis : pip install astroquery astropy")

        valid = self._g10.dropna(subset=["ra", "dec"])
        print(f"\nCross-match {catalog}")
        print(f"  {len(valid):,} sources × rayon {self.radius_as}\"")
        print(f"  Batches de {batch_size} — peut prendre quelques minutes...")

        results = []
        n_batches = (len(valid) + batch_size - 1) // batch_size

        for i in range(n_batches):
            batch = valid.iloc[i * batch_size:(i + 1) * batch_size]
            tbl   = Table.from_pandas(batch[["g10_idx", "ra", "dec", "label",
                                             "z_photo"]])
            try:
                res = XMatch.query(
                    cat1=tbl,
                    cat2=catalog,
                    max_distance=self.radius_as * u.arcsec,
                    colRA1="ra",
                    colDec1="dec",
                )
                if res is not None and len(res) > 0:
                    results.append(res.to_pandas())
                print(f"  Batch {i+1}/{n_batches} → {len(res) if res else 0} hits",
                      end="\r")
            except Exception as e:
                print(f"\n  ⚠ Batch {i+1} échoué : {e} — on continue")
            time.sleep(0.3)  # respecter les rate limits CDS

        print()
        if not results:
            print("  ✗ Aucune correspondance trouvée")
            return pd.DataFrame()

        df = pd.concat(results, ignore_index=True)
        df = self._add_derived_columns(df, catalog)

        # Garder seulement la meilleure correspondance par source
        if "angDist" in df.columns:
            df = df.sort_values("angDist").drop_duplicates(
                subset=["g10_idx"], keep="first")

        df = df.merge(
            self._g10[["g10_idx", "class_name", "pxscale"]],
            on="g10_idx", how="left")

        df.to_csv(cache_path, index=False)
        print(f"✓ {len(df):,} correspondances sauvegardées → {cache_path.name}")
        return df

    # ── Colonnes dérivées ─────────────────────────────────────────────────────
    def _add_derived_columns(self, df: pd.DataFrame, catalog: str) -> pd.DataFrame:
        """Ajoute les colonnes dérivées selon le catalogue."""
        if "sdss" in catalog.lower() or "154" in catalog:
            df = self._add_sdss_columns(df)
        elif "nsa" in catalog.lower() or "149" in catalog:
            df = self._add_nsa_columns(df)
        return df

    def _add_sdss_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Couleurs et proxies SDSS."""
        mag_cols = ["umag", "gmag", "rmag", "imag", "zmag"]
        for col in mag_cols:
            if col not in df.columns:
                # Chercher des variantes de noms
                for alt in [col.capitalize(), col.upper(),
                            col.replace("mag", "Mag"),
                            "mag" + col[0].upper()]:
                    if alt in df.columns:
                        df[col] = df[alt]
                        break

        # Couleurs standards
        for c1, c2 in [("u","g"), ("g","r"), ("r","i"), ("i","z"), ("u","r")]:
            c1m, c2m = f"{c1}mag", f"{c2}mag"
            if c1m in df.columns and c2m in df.columns:
                df[f"{c1}-{c2}"] = df[c1m] - df[c2m]

        # Proxy SFR : u-r (Baldry+ 2004 : u-r < 2.22 → Blue Cloud)
        if "u-r" in df.columns:
            df["blue_cloud"]   = df["u-r"] < 2.22
            df["red_sequence"] = df["u-r"] > 2.55
            df["green_valley"] = (~df["blue_cloud"]) & (~df["red_sequence"])
            df["population"]   = df.apply(
                lambda row: "Blue Cloud" if row["blue_cloud"]
                else ("Red Sequence" if row["red_sequence"]
                      else "Green Valley"), axis=1)

        # Magnitude absolue (proxy masse) si redshift disponible
        z_col = "zsp" if "zsp" in df.columns else "z_photo"
        if z_col in df.columns and "rmag" in df.columns:
            z = pd.to_numeric(df[z_col], errors="coerce")
            valid = z > 0.001
            df.loc[valid, "Mr"] = (
                df.loc[valid, "rmag"]
                - 5 * np.log10(z[valid] * 3e5 / 70 * 1e6 / 10)
            )

        return df

    def _add_nsa_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Propriétés physiques NSA."""
        # Masse stellaire log(M/Msun)
        for col in ["logMass", "logm", "Mass"]:
            if col in df.columns:
                df["log_mass"] = pd.to_numeric(df[col], errors="coerce")
                break
        # SFR
        for col in ["SFR", "logSFR", "sfr"]:
            if col in df.columns:
                df["sfr"] = pd.to_numeric(df[col], errors="coerce")
                break
        # Sérsic index
        for col in ["nsersic", "Nsersic", "n"]:
            if col in df.columns:
                df["sersic_n"] = pd.to_numeric(df[col], errors="coerce")
                break
        return df

    # ── API publique ───────────────────────────────────────────────────────────
    def run_sdss(self) -> pd.DataFrame:
        """Cross-match avec SDSS DR16."""
        return self._xmatch(SDSS_CATALOG, "galaxy10_sdss_xmatch")

    def run_nsa(self) -> pd.DataFrame:
        """Cross-match avec NASA-Sloan Atlas v1.0.1."""
        return self._xmatch(NSA_CATALOG, "galaxy10_nsa_xmatch")

    def stats(self, df: pd.DataFrame) -> None:
        """Affiche les statistiques du catalogue croisé."""
        if df.empty:
            print("DataFrame vide.")
            return

        n_total    = len(self._g10)
        n_matched  = df["g10_idx"].nunique()
        match_rate = n_matched / n_total * 100

        print(f"\n{'═'*55}")
        print(f"  STATISTIQUES DU CROSS-MATCH")
        print(f"{'═'*55}")
        print(f"  Galaxy10 total    : {n_total:>7,}")
        print(f"  Correspondances   : {n_matched:>7,}  ({match_rate:.1f}%)")
        print(f"  Non-matchés       : {n_total - n_matched:>7,}")
        print()

        if "class_name" in df.columns:
            print("  Par classe morphologique :")
            g = df.groupby("class_name")["g10_idx"].count()
            for cls in CLASS_NAMES:
                n = g.get(cls, 0)
                src_n = (self._g10["class_name"] == cls).sum()
                pct = n / src_n * 100 if src_n > 0 else 0
                print(f"    {cls:<30} {n:>5} / {src_n:>5}  ({pct:.0f}%)")

        if "g-r" in df.columns:
            print(f"\n  g-r médian global : {df['g-r'].median():.3f}")
        if "population" in df.columns:
            pop = df["population"].value_counts()
            print(f"\n  Blue Cloud   : {pop.get('Blue Cloud',0):>5}")
            print(f"  Green Valley : {pop.get('Green Valley',0):>5}")
            print(f"  Red Sequence : {pop.get('Red Sequence',0):>5}")
        print(f"{'═'*55}")
