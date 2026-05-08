# 🔭 AstroVision

**Classification morphologique de galaxies par deep learning.**  
Projet compagnon de [AstroSpectro](https://github.com/PhD-Brown/AstroSpectro) — même philosophie, domaine 2D.

---

## Objectif

Développer des pipelines de classification et segmentation d'objets célestes (galaxies, nébuleuses)
à partir d'images astronomiques, en explorant CNNs, transfer learning, et self-supervised learning (DINOv2).

## Dataset

**Galaxy10 DECaLS** — 17 736 images de galaxies, 10 classes morphologiques, 256×256 RGB.  
Source : [Zenodo 5764502](https://zenodo.org/record/5764502) (~2.7 GB)

| Classe | Description |
|--------|-------------|
| 0 | Disturbed |
| 1 | Merging |
| 2 | Round Smooth |
| 3 | In-between Round Smooth |
| 4 | Cigar Shaped Smooth |
| 5 | Barred Spiral |
| 6 | Unbarred Tight Spiral |
| 7 | Unbarred Loose Spiral |
| 8 | Edge-on without Bulge |
| 9 | Edge-on with Bulge |

## Structure

```
AstroVision/
├── src/astrovision/
│   ├── data_loader.py     # Téléchargement + DataLoaders Galaxy10
│   ├── models.py          # SimpleCNN, EfficientNetGalaxy
│   ├── trainer.py         # Boucle d'entraînement + W&B
│   └── gradcam.py         # Grad-CAM + visualisation
├── notebooks/
│   └── 01_galaxy10_cnn.ipynb
├── data/                  # (gitignored)
├── checkpoints/           # (gitignored)
└── figures/
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation rapide

```python
from astrovision import load_galaxy10_splits, build_model, train, GradCAM, plot_gradcam_grid

train_loader, val_loader, test_loader = load_galaxy10_splits(batch_size=64)

model = build_model('efficientnet')
model = train(model, train_loader, val_loader, epochs=30, device='cuda')

gcam = GradCAM(model, model.gradcam_target)
plot_gradcam_grid(model, gcam, images, labels, device='cuda')
```

## Roadmap

- [x] SimpleCNN baseline (from scratch)
- [x] EfficientNet-B0 transfer learning
- [x] Grad-CAM interpretabilité
- [ ] DINOv2 self-supervised features
- [ ] Segmentation (U-Net / SAM)
