# 🔭 AstroVision

**Classification morphologique de galaxies par deep learning.**
Projet compagnon de [AstroSpectro](https://github.com/PhD-Brown/AstroSpectro) — même philosophie, domaine 2D.

---

## Résultats

| Méthode | Balanced Acc | Notes |
|---------|:-----------:|-------|
| Random baseline | 0.100 | 10 classes équilibrées |
| SimpleCNN from scratch | 0.595 | CNN 4-blocs, 651k params |
| DINOv2 zero-shot probe | 0.555 | Features pré-calculées |
| EfficientNet-B0 fine-tuned | 0.827 | Transfer learning ImageNet |
| **DINOv2 ViT-B/14 fine-tuned** | **0.839** | ViT 86M params, 10 epochs FP16 |

---

## Dataset

**Galaxy10 DECaLS** — 17 736 images de galaxies, 10 classes morphologiques, 256×256 RGB.
Source : [Walmsley et al. 2021](https://arxiv.org/abs/2102.08414) × [Zenodo 10845026](https://zenodo.org/records/10845026) (~2.7 GB)

| # | Classe | Recall DINOv2 FT |
|---|--------|:---------:|
| 0 | Disturbed | 0.53 |
| 1 | Merging | 0.86 |
| 2 | Round Smooth | 0.94 |
| 3 | In-between Round Smooth | 0.95 |
| 4 | Cigar Shaped Smooth | 0.72 |
| 5 | Barred Spiral | 0.87 |
| 6 | Unbarred Tight Spiral | 0.83 |
| 7 | Unbarred Loose Spiral | 0.78 |
| 8 | Edge-on without Bulge | 0.96 |
| 9 | Edge-on with Bulge | 0.93 |

---

## Structure

```
AstroVision/
├── src/astrovision/
│   ├── data_loader.py      # Galaxy10 HDF5 + DataLoaders
│   ├── models.py           # SimpleCNN + EfficientNetGalaxy
│   ├── trainer.py          # Boucle entraînement + W&B
│   └── gradcam.py          # Grad-CAM + visualisation
├── notebooks/
│   ├── 00_eda_galaxy10.ipynb         # EDA — 10 figures
│   ├── 01_galaxy10_cnn.ipynb         # SimpleCNN + EfficientNet
│   ├── 02_results_analysis.ipynb     # Analyse complète
│   ├── 03_dinov2_features.ipynb      # DINOv2 zero-shot
│   └── 04_dinov2_finetune.ipynb      # DINOv2 fine-tuning
├── data/                  # (gitignored)
├── checkpoints/           # (gitignored)
└── figures/
```

---

## Installation

```bash
pip install -r requirements.txt
```

**GPU RTX 50xx (Blackwell) :**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

---

## Utilisation rapide

```python
from astrovision import load_galaxy10_splits, build_model, train, GradCAM

# Données
train_loader, val_loader, test_loader = load_galaxy10_splits(batch_size=64)

# Modèle
model = build_model('efficientnet')
model = train(model, train_loader, val_loader, epochs=30, device='cuda')

# Interprétabilité
gcam = GradCAM(model, model.gradcam_target)
```

**DINOv2 fine-tuning :**
Voir `notebooks/04_dinov2_finetune.ipynb` — Mixed Precision FP16, init depuis probe linéaire.

---

## Findings Scientifiques

**1. Domain gap DINOv2 zero-shot**
Les features DINOv2 non-adaptées (0.555) sont inférieures au SimpleCNN from scratch (0.595).
Le modèle fixe les étoiles de fond brillantes plutôt que les structures galactiques.

**2. Fine-tuning restaure et dépasse**
Après 10 epochs (lr=5e-6, FP16), DINOv2 atteint 0.839 > EfficientNet (0.827).
Les attention maps post-FT couvrent toute la structure galactique.

**3. Classe Disturbed — plafond physique**
Recall plafonné à 0.53. Une galaxie perturbée peut ressembler à n'importe quelle autre
classe selon l'angle et la distance — ambiguïté physique intrinsèque.

---

## Roadmap

- [x] Phase 0 — Fondations + CI
- [x] Phase 1 — SimpleCNN + EfficientNet + Grad-CAM
- [x] Phase 2 — DINOv2 zero-shot + fine-tuning
- [ ] Phase 3 — Segmentation + Cross-match AstroSpectro × AstroVision
- [ ] Phase 4 — Publication A&A / MNRAS

---

## Références

- Walmsley M. et al. 2021 — *Galaxy Zoo DECaLS* — [arXiv:2102.08414](https://arxiv.org/abs/2102.08414)
- Selvaraju R. et al. 2017 — *Grad-CAM* — [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
- Oquab M. et al. 2023 — *DINOv2* — [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
- Tan M. & Le Q. 2019 — *EfficientNet* — [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)

---

*Projet développé dans le cadre du programme de physique à l'Université Laval.*
*Compagnon : [AstroSpectro](https://github.com/PhD-Brown/AstroSpectro) — classification spectrale stellaire.*
