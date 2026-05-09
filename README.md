<div align="center">

# 🔭 AstroVision

**Classification morphologique de galaxies par deep learning**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.7](https://img.shields.io/badge/PyTorch-2.7-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![W&B](https://img.shields.io/badge/Tracked_with-W%26B-FFBE00?style=flat-square&logo=weightsandbiases&logoColor=black)](https://wandb.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![Companion](https://img.shields.io/badge/Companion-AstroSpectro-6e40aa?style=flat-square)](https://github.com/PhD-Brown/AstroSpectro)

*Projet compagnon de [AstroSpectro](https://github.com/PhD-Brown/AstroSpectro) — même philosophie, domaine 2D.*  
*Université Laval · Département de physique · 2025–2026*

</div>

---

## Vue d'ensemble

AstroVision est un pipeline de deep learning pour la **classification morphologique automatisée de galaxies** sur le dataset Galaxy10 DECaLS (17 736 images 256×256 RGB, 10 classes GalZoo-style). Le projet va au-delà de la simple classification : il combine **segmentation sémantique** (U-Net), **analyse morphométrique** (CAS/Gini/M20), **cross-match photométrique** (SDSS DR16) et un **bridge spectroscopique** avec AstroSpectro pour former un écosystème AstroAI complet.

```
Galaxy10 DECaLS ──► CNN / DINOv2 ──► Classification morphologique (0.839 bal. acc)
        │                                          │
        ├──► U-Net ──────────────────────► Segmentation fond/disque/bulbe (mIoU 0.540)
        │
        ├──► CDS XMatch × SDSS DR16 ────► Photométrie g-r, populations stellaires
        │
        └──► CDS XMatch × AstroSpectro ─► Bridge [Fe/H] × morphologie visuelle
```

---

## Résultats

### Classification morphologique

| Méthode | Balanced Acc | Params | Notes |
|---|---|---|---|
| Baseline aléatoire | 0.100 | — | 10 classes équilibrées |
| SimpleCNN from scratch | 0.595 | 651k | CNN 4-blocs, AMP FP16 |
| DINOv2 zero-shot probe | 0.555 | 86M (gelé) | Features pré-calculées |
| EfficientNet-B0 fine-tuned | 0.827 | 5.3M | Transfer learning ImageNet |
| **DINOv2 ViT-B/14 fine-tuned** ★ | **0.839** | **86M** | **10 epochs, lr=5e-6, FP16** |

### Recall par classe — DINOv2 ViT-B/14

| # | Classe | Recall | Difficulté |
|---|---|---|---|
| 0 | Disturbed | 0.53 | ⚠️ Ambiguïté physique intrinsèque |
| 1 | Merging | 0.86 | ✅ |
| 2 | Round Smooth | 0.94 | ✅ |
| 3 | In-between Round Smooth | 0.95 | ✅ |
| 4 | Cigar Shaped Smooth | 0.72 | 🔶 |
| 5 | Barred Spiral | 0.87 | ✅ |
| 6 | Unbarred Tight Spiral | 0.83 | ✅ |
| 7 | Unbarred Loose Spiral | 0.78 | ✅ |
| 8 | Edge-on without Bulge | 0.96 | ✅ |
| 9 | Edge-on with Bulge | 0.93 | ✅ |

### Segmentation sémantique — U-Net (17.3M params)

| Méthode | mIoU test | Notes |
|---|---|---|
| Baseline aléatoire | ~0.167 | 3 classes équilibrées |
| Otsu (pseudo-labels teacher) | ~0.500 | Auto-référence |
| **U-Net (Dice + CE loss, 20 epochs)** | **0.540** | **+4% vs teacher** |

> Le U-Net dépasse son propre teacher — il apprend à **régulariser et généraliser** au-delà des pseudo-labels bruités.

### Cross-match Galaxy10 × SDSS DR16

| Métrique | Valeur |
|---|---|
| Galaxies matchées | 5 651 / 17 736 (31.9%) |
| z_photo vs z_spec | ρ = 0.998 |
| R² DINOv2 → g-r (Lasso) | 0.536 |
| PC2(DINOv2) × Gini | ρ = 0.53 \* |
| PC2(DINOv2) × g-r | ρ = 0.30 \* |

---

## Findings scientifiques

### 1 — Domain gap DINOv2 zero-shot

Les features DINOv2 non-adaptées (0.555) sont **inférieures au SimpleCNN from scratch** (0.595). Les attention maps révèlent que le modèle fixe les étoiles de fond brillantes plutôt que les structures galactiques — signal d'un domain gap ImageNet → astronomie.

### 2 — Le fine-tuning restaure et dépasse

Après 10 epochs (lr=5e-6, FP16, probe linéaire comme initialisation), DINOv2 atteint **0.839 > EfficientNet** (0.827). Les attention maps post-FT couvrent la totalité de la structure galactique. La hiérarchie de Hubble est clairement encodée dans l'espace latent.

### 3 — PC2(DINOv2) est un axe physique dual

La 2e composante principale de l'espace DINOv2 (13.2% de variance) corrèle simultanément avec :
- **Gini ρ=0.53** — distribution de la lumière inégale (étoiles concentrées)
- **g-r ρ=0.30** — couleur photométrique (population stellaire)

Ces deux propriétés sont physiquement liées via la **séquence de Hubble**, et DINOv2 les capture dans le même axe latent **sans avoir jamais vu de données astronomiques**.

### 4 — Clase Disturbed — plafond physique à 0.53

Une galaxie perturbée peut ressembler à n'importe quelle autre classe selon l'angle de vue et la distance. Ce recall plafonné n'est **pas un échec du modèle** — c'est une limite physique de la classification morphologique projective.

### 5 — Blue Cloud / Red Sequence visibles dans l'UMAP DINOv2

La bimodalité stellaire (Blue Cloud ↔ Red Sequence via g-r) est **spatialement visible** dans l'UMAP des features DINOv2, confirmant que le modèle encode la physique sous-jacente des populations stellaires.

---

## Dataset

**Galaxy10 DECaLS** — 17 736 images de galaxies, 10 classes morphologiques, 256×256 RGB.  
Source : [Walmsley et al. 2021](https://arxiv.org/abs/2102.08414) × [Zenodo 10845026](https://zenodo.org/records/10845026) (~2.7 GB)

```python
# Téléchargement
wget https://zenodo.org/records/10845026/files/Galaxy10_DECals.h5
```

Le dataset est à placer dans `data/Galaxy10_DECals.h5` (répertoire gitignored).

---

## Structure du projet

```
AstroVision/
│
├── src/astrovision/              # Package principal
│   ├── __init__.py
│   ├── data_loader.py            # Galaxy10 HDF5 + DataLoaders PyTorch
│   ├── models.py                 # SimpleCNN + EfficientNetGalaxy
│   ├── trainer.py                # Boucle entraînement AMP FP16 + W&B
│   ├── gradcam.py                # Grad-CAM + visualisation overlay
│   ├── crossmatcher.py           # Cross-match CDS XMatch (SDSS DR16, NSA)
│   ├── morphometry.py            # CAS, Gini, M20, Sérsic (Conselice 2003)
│   ├── segmenter.py              # Otsu + DINOv2 patch segmentation
│   ├── unet.py                   # U-Net sémantique (Ronneberger 2015)
│   └── synthesis.py              # Synthèse finale : morpho × photo × spectro
│
├── notebooks/
│   ├── 00_eda_galaxy10.ipynb           # EDA — distributions, galerie, PCA pixels
│   ├── 01_galaxy10_cnn.ipynb           # SimpleCNN + EfficientNet + Grad-CAM
│   ├── 02_results_analysis.ipynb       # Matrices confusion, F1, erreurs types
│   ├── 03_dinov2_features.ipynb        # Features zero-shot, UMAP, NN visuels
│   ├── 04_dinov2_finetune.ipynb        # DINOv2 FT, attention avant/après
│   ├── 05_crossmatch_analysis.ipynb    # Galaxy10 × SDSS DR16, CMD, populations
│   ├── 06_segmentation.ipynb           # CAS/Gini/M20 + DINOv2 patches + U-Net
│   └── 07_synthesis.ipynb              # Synthèse finale — triple croisement
│
├── data/                         # (gitignored) Galaxy10_DECals.h5 + caches
├── checkpoints/                  # (gitignored) Poids entraînés (.pt)
├── figures/                      # Figures exportées (PNG)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Pipeline en 7 notebooks

```
00_eda          → Comprendre les données (distributions, galerie, PCA)
     │
01_cnn          → Premier modèle (SimpleCNN + EfficientNet + Grad-CAM)
     │
02_results      → Analyse complète des erreurs et métriques
     │
03_dino_feat    → Espace latent DINOv2 zero-shot (UMAP, voisins visuels)
     │
04_dino_ft      → Fine-tuning DINOv2, attention maps, SOTA (0.839)
     │
05_crossmatch   → Bridge photométrique (SDSS, CMD, Blue Cloud / Red Sequence)
     │
06_segmentation → Morphométrie + U-Net + corrélation DINOv2 × CAS/Gini
     │
07_synthesis    → Synthèse : frac_bulbe × g-r × [Fe/H] + table publication
```

---

## Installation

### Prérequis

- Python 3.11+
- CUDA 12.x (recommandé ; CPU supporté mais lent)
- 16 GB RAM minimum ; 64 GB recommandé pour le dataset complet

```bash
git clone https://github.com/PhD-Brown/AstroVision.git
cd AstroVision
pip install -r requirements.txt
```

### GPU Blackwell (RTX 50xx — sm_120)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Note Windows

```python
# DataLoader : num_workers=0 obligatoire (OSError multiprocessing)
DataLoader(dataset, batch_size=64, num_workers=0)
```

---

## Utilisation rapide

### Classification

```python
from src.astrovision.data_loader import load_galaxy10_splits
from src.astrovision.models import build_model
from src.astrovision.trainer import train

# Données
train_loader, val_loader, test_loader = load_galaxy10_splits(
    h5_path='data/Galaxy10_DECals.h5', batch_size=64
)

# Modèle DINOv2 fine-tuned (meilleur résultat)
model = build_model('dinov2')
model = train(model, train_loader, val_loader, epochs=10,
              lr=5e-6, device='cuda', use_wandb=True)
```

### Grad-CAM

```python
from src.astrovision.gradcam import GradCAM

gcam  = GradCAM(model, target_layer=model.backbone.blocks[-1].norm1)
cams  = gcam.compute(images)          # (N, H, W) heatmaps
fig   = gcam.plot_gallery(images, cams, labels, n=8)
```

### Segmentation U-Net

```python
from src.astrovision.unet import UNet
from src.astrovision.segmenter import OtsuSegmenter
import torch, os

# Charger le modèle entraîné
model_unet = UNet(n_classes=3, bilinear=True, dropout=0.3)
model_unet.load_state_dict(
    torch.load(os.path.join('checkpoints', 'unet_galaxy10_best.pt'),
               map_location='cuda')
)

# Segmentation : {0: fond, 1: disque galactique, 2: bulbe}
model_unet.eval()
with torch.no_grad():
    logits = model_unet(images.to('cuda'))
    seg    = logits.argmax(1)         # (N, H, W)
```

### Morphométrie CAS/Gini/M20

```python
from src.astrovision.morphometry import batch_morphometry

df = batch_morphometry(images, labels)
# Colonnes : C (Concentration), A (Asymétrie), S (Smoothness),
#            Gini, M20, sersic_n
print(df.groupby('class_name')[['C', 'Gini', 'M20']].median())
```

### Cross-match SDSS

```python
from src.astrovision.crossmatcher import Galaxy10Crossmatcher

xm = Galaxy10Crossmatcher(h5_path='data/Galaxy10_DECals.h5',
                           cache_dir='data/')
df = xm.run_sdss()     # Cross-match CDS XMatch — SDSS DR16
xm.stats(df)
# ✓ 5 651 correspondances | g-r médian : 0.71
```

### Synthèse finale

```python
from src.astrovision.synthesis import AstroVisionSynthesis

synth  = AstroVisionSynthesis(data_dir='data/', figures_dir='figures/')
master = synth.build_master_dataframe()      # Fusionne tout automatiquement
figs   = synth.run_all_figures(master)       # 5 figures de synthèse
table  = synth.build_summary_table(master)  # Table publication-ready
```

---

## Écosystème AstroAI

AstroVision fait partie d'un écosystème plus large visant à **relier imagerie et spectroscopie galactique** :

```
┌─────────────────────────────┐     RA/Dec      ┌─────────────────────┐
│  AstroSpectro               │ ────────────► │  Cross-match        │
│  LAMOST DR5 × Gaia DR3      │               │  CDS XMatch 1"      │
│  43 019 spectres stellaires  │               └────────┬────────────┘
│  XGBoost 0.87 bal. acc      │                        │ Features DINOv2
└─────────────────────────────┘                        ▼
                                              ┌─────────────────────┐
                                              │  AstroVision        │
                                              │  Galaxy10 DECaLS    │
                                              │  17 736 images      │
                                              │  DINOv2 0.839       │
                                              └─────────────────────┘

         → Corrélation [Fe/H] spectroscopique × morphologie visuelle ←
```

Le finding central : les features **DINOv2 encodent spontanément la physique stellaire** (populations Blue Cloud / Red Sequence) sans avoir jamais vu de données spectroscopiques.

---

## Roadmap

- [x] **Phase 0** — Fondations, structure du projet, CI
- [x] **Phase 1** — SimpleCNN + EfficientNet-B0 + Grad-CAM (0.827)
- [x] **Phase 2** — DINOv2 zero-shot + fine-tuning, attention maps (0.839)
- [x] **Phase 3A** — Cross-match Galaxy10 × SDSS DR16 (5 651 matches, R²=0.536)
- [x] **Phase 3B** — Morphométrie CAS/Gini/M20 + Segmentation U-Net (mIoU=0.540)
- [x] **Phase 4** — Synthèse finale : fraction_bulbe × g-r × [Fe/H]
- [ ] **Publication** — A&A ou MNRAS (en préparation)
- [ ] **Bridge LAMOST** — Cross-match LAMOST DR7 × Galaxy10 pour [Fe/H] direct

---

## Hardware

| Composant | Spec |
|---|---|
| CPU | AMD Ryzen 9 5950X (16c/32t) |
| GPU | NVIDIA RTX 5060 Ti 16 GB (Blackwell, sm_120) |
| RAM | 64 GB DDR4 |
| OS | Windows 11 + venv Python 3.11 |
| Stack | PyTorch 2.7 + CUDA 12.8 + AMP FP16 |

**Temps d'entraînement indicatifs :**

| Modèle | Epochs | Temps |
|---|---|---|
| SimpleCNN | 30 | ~12 min |
| EfficientNet-B0 FT | 30 | ~25 min |
| DINOv2 ViT-B/14 FT | 10 | ~45 min |
| U-Net segmentation | 20 | ~3h 25min |

---

## Références

```bibtex
@article{walmsley2022galaxy10,
  title   = {Galaxy Zoo DECaLS: Detailed Visual Morphology Measurements from
             Volunteers and Deep Learning for 314 000 Galaxies},
  author  = {Walmsley, M. and others},
  journal = {MNRAS},
  year    = {2022},
  doi     = {10.1093/mnras/stac525}
}

@inproceedings{selvaraju2017gradcam,
  title     = {Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization},
  author    = {Selvaraju, R. R. and others},
  booktitle = {ICCV},
  year      = {2017}
}

@article{oquab2023dinov2,
  title   = {DINOv2: Learning Robust Visual Features without Supervision},
  author  = {Oquab, M. and others},
  journal = {TMLR},
  year    = {2024}
}

@article{tan2019efficientnet,
  title   = {EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
  author  = {Tan, M. and Le, Q.},
  journal = {ICML},
  year    = {2019}
}

@article{conselice2003cas,
  title   = {A Relationship between Galaxy Morphology and Environment in the Pisces-Perseus Cluster},
  author  = {Conselice, C. J.},
  journal = {ApJS},
  volume  = {147},
  pages   = {1--28},
  year    = {2003}
}

@article{lotz2004gini,
  title   = {A New Nonparametric Approach to Galaxy Morphological Classification},
  author  = {Lotz, J. M. and Primack, J. and Madau, P.},
  journal = {AJ},
  volume  = {128},
  pages   = {163--182},
  year    = {2004}
}

@article{ronneberger2015unet,
  title   = {U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author  = {Ronneberger, O. and Fischer, P. and Brox, T.},
  booktitle = {MICCAI},
  year    = {2015}
}
```

---

<div align="center">

**Développé à l'Université Laval · Département de physique · 2025–2026**

Compagnon : [AstroSpectro](https://github.com/PhD-Brown/AstroSpectro) — classification spectrale stellaire (LAMOST DR5 × XGBoost)

*"Ce que les spectres révèlent en longueurs d'onde, les images le trahissent en formes."*

</div>
