"""
AstroVision - Galaxy10 DECaLS Data Loader
Télécharge et prépare le dataset depuis Zenodo (~2.7 GB HDF5).
"""

import urllib.request
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ── Constantes ────────────────────────────────────────────────────────────────
ZENODO_URL = "https://zenodo.org/records/10845026/files/Galaxy10_DECals.h5"

CLASS_NAMES = [
    "Disturbed",
    "Merging",
    "Round Smooth",
    "In-between Round Smooth",
    "Cigar Shaped Smooth",
    "Barred Spiral",
    "Unbarred Tight Spiral",
    "Unbarred Loose Spiral",
    "Edge-on without Bulge",
    "Edge-on with Bulge",
]

_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = _ROOT / "data"
H5_PATH = DATA_DIR / "Galaxy10_DECals.h5"

# ImageNet stats (utilisées pour le transfer learning EfficientNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ── Téléchargement ────────────────────────────────────────────────────────────
def _progress_hook(count, block_size, total_size):
    pct = min(count * block_size * 100 // total_size, 100)
    print(f"\r  Téléchargement : {pct}%", end="", flush=True)


def download_galaxy10(force: bool = False) -> Path:
    """Télécharge Galaxy10 DECaLS si absent."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if H5_PATH.exists() and not force:
        print(f"✓ Dataset déjà présent : {H5_PATH}")
        return H5_PATH

    print(f"Téléchargement Galaxy10 DECaLS (~2.7 GB)…")
    urllib.request.urlretrieve(ZENODO_URL, H5_PATH, _progress_hook)
    print(f"\n✓ Enregistré : {H5_PATH}")
    return H5_PATH


# ── Dataset PyTorch ───────────────────────────────────────────────────────────
class Galaxy10Dataset(Dataset):
    """Dataset PyTorch pour Galaxy10 DECaLS.

    Args:
        images: array (N, 256, 256, 3) uint8
        labels: array (N,) int
        transform: torchvision transform appliqué sur le tensor (C, H, W) float [0, 1]
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]  # (H, W, C) uint8
        label = int(self.labels[idx])

        # → tensor (C, H, W) float [0, 1]
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        if self.transform:
            img = self.transform(img)

        return img, label


# ── Transforms ────────────────────────────────────────────────────────────────
def get_transforms(train: bool = True) -> transforms.Compose:
    """Retourne les transforms train ou val/test."""
    base = [transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]

    if train:
        augment = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ]
        return transforms.Compose(augment + base)

    return transforms.Compose(base)


# ── Chargement complet ────────────────────────────────────────────────────────
def load_galaxy10_splits(
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    batch_size: int = 64,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Charge Galaxy10 et retourne (train_loader, val_loader, test_loader)."""
    download_galaxy10()

    with h5py.File(H5_PATH, "r") as f:
        images = f["images"][:]  # (N, 256, 256, 3) uint8
        labels = f["ans"][:]     # (N,) int

    n = len(labels)
    print(f"✓ {n} images chargées — {len(CLASS_NAMES)} classes")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)

    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test

    train_idx, val_idx, test_idx = (
        idx[:n_train],
        idx[n_train:n_train + n_val],
        idx[n_train + n_val:],
    )

    train_ds = Galaxy10Dataset(images[train_idx], labels[train_idx], get_transforms(True))
    val_ds   = Galaxy10Dataset(images[val_idx],   labels[val_idx],   get_transforms(False))
    test_ds  = Galaxy10Dataset(images[test_idx],  labels[test_idx],  get_transforms(False))

    kw = dict(num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw)

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader
