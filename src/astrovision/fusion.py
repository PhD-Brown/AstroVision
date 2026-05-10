"""
AstroVision — fusion.py
Classificateur multi-modal : DINOv2 visual features + morphométrie CAS/Gini/M20
+ photométrie SDSS (g-r, u-r, Mr).

Motivation scientifique :
    DINOv2 seul capture ~54% de la variance de g-r et corrèle avec Gini (ρ=0.53).
    En fusionnant explicitement ces modalités, on peut potentiellement dépasser
    la balanced_acc de 0.839 obtenue par DINOv2 fine-tuné seul.

Architecture : Late Fusion à 3 branches
    ┌─── Branch A : DINOv2 features (768) → MLP → 256 ────┐
    ├─── Branch B : Morphométrie (5)      → MLP → 32  ────┤ → Concat → MLP → 10
    └─── Branch C : Photométrie (3)       → MLP → 16  ────┘
                                          Modalités B et C optionnelles.

Usage :
    from astrovision.fusion import GalaxyFusionClassifier, FusionDataset, train_fusion
    model = GalaxyFusionClassifier(n_morph=5, n_photo=3)
    model = train_fusion(model, train_loader, val_loader, epochs=30)
"""

import os
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

NUM_CLASSES = 10
CLASS_NAMES = [
    "Disturbed", "Merging", "Round Smooth", "In-between Round Smooth",
    "Cigar Shaped Smooth", "Barred Spiral", "Unbarred Tight Spiral",
    "Unbarred Loose Spiral", "Edge-on without Bulge", "Edge-on with Bulge",
]


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class FusionDataset(Dataset):
    """Dataset multi-modal pour la fusion tardive.

    Gère les modalités manquantes (morphométrie ou photométrie optionnelles).
    Les NaN sont remplacés par 0 après standardisation — le masque `valid_*`
    permet au modèle d'ignorer ces dimensions si nécessaire.

    Args:
        dino_features  : (N, 768) features DINOv2 (obligatoire)
        labels         : (N,) labels entiers 0-9
        morph_features : (N, n_morph) CAS/Gini/M20 (optionnel)
        photo_features : (N, n_photo) g-r, u-r, Mr (optionnel)
        scaler_dino    : StandardScaler pré-fitté (si None → fit sur ce split)
        scaler_morph   : idem pour morphométrie
        scaler_photo   : idem pour photométrie
    """

    def __init__(self,
                 dino_features:  np.ndarray,
                 labels:         np.ndarray,
                 morph_features: Optional[np.ndarray] = None,
                 photo_features: Optional[np.ndarray] = None,
                 scaler_dino:    Optional[StandardScaler] = None,
                 scaler_morph:   Optional[StandardScaler] = None,
                 scaler_photo:   Optional[StandardScaler] = None):

        # Standardisation DINOv2
        if scaler_dino is None:
            scaler_dino = StandardScaler()
            scaler_dino.fit(dino_features)
        self.scaler_dino   = scaler_dino
        self.dino          = torch.tensor(
            scaler_dino.transform(dino_features), dtype=torch.float32)
        self.labels        = torch.tensor(labels, dtype=torch.long)

        # Morphométrie (optionnelle)
        self.has_morph = morph_features is not None
        if self.has_morph:
            morph_arr = morph_features.astype(np.float32).copy()
            valid_morph = ~np.isnan(morph_arr).any(axis=1)
            if scaler_morph is None:
                scaler_morph = StandardScaler()
                scaler_morph.fit(morph_arr[valid_morph])
            self.scaler_morph = scaler_morph
            morph_scaled      = np.zeros_like(morph_arr)
            morph_scaled[valid_morph] = scaler_morph.transform(morph_arr[valid_morph])
            self.morph       = torch.tensor(morph_scaled, dtype=torch.float32)
            self.valid_morph = torch.tensor(valid_morph, dtype=torch.bool)
        else:
            self.scaler_morph = None
            self.morph        = None
            self.valid_morph  = None

        # Photométrie (optionnelle)
        self.has_photo = photo_features is not None
        if self.has_photo:
            photo_arr  = photo_features.astype(np.float32).copy()
            valid_photo = ~np.isnan(photo_arr).any(axis=1)
            if scaler_photo is None:
                scaler_photo = StandardScaler()
                scaler_photo.fit(photo_arr[valid_photo])
            self.scaler_photo = scaler_photo
            photo_scaled      = np.zeros_like(photo_arr)
            photo_scaled[valid_photo] = scaler_photo.transform(photo_arr[valid_photo])
            self.photo        = torch.tensor(photo_scaled, dtype=torch.float32)
            self.valid_photo  = torch.tensor(valid_photo, dtype=torch.bool)
        else:
            self.scaler_photo = None
            self.photo        = None
            self.valid_photo  = None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "dino":  self.dino[idx],
            "label": self.labels[idx],
        }
        if self.has_morph:
            item["morph"]       = self.morph[idx]
            item["valid_morph"] = self.valid_morph[idx]
        if self.has_photo:
            item["photo"]       = self.photo[idx]
            item["valid_photo"] = self.valid_photo[idx]
        return item

    def get_scalers(self) -> dict:
        return {
            "dino":  self.scaler_dino,
            "morph": self.scaler_morph,
            "photo": self.scaler_photo,
        }


def build_fusion_datasets(
    dino_features:  np.ndarray,
    labels:         np.ndarray,
    train_idx:      np.ndarray,
    val_idx:        np.ndarray,
    test_idx:       np.ndarray,
    morph_features: Optional[np.ndarray] = None,
    photo_features: Optional[np.ndarray] = None,
) -> Tuple["FusionDataset", "FusionDataset", "FusionDataset"]:
    """Crée les trois splits avec standardisation fit uniquement sur train."""

    # Fit sur train uniquement
    train_ds = FusionDataset(
        dino_features[train_idx],
        labels[train_idx],
        morph_features[train_idx] if morph_features is not None else None,
        photo_features[train_idx] if photo_features is not None else None,
    )
    scalers = train_ds.get_scalers()

    val_ds = FusionDataset(
        dino_features[val_idx],
        labels[val_idx],
        morph_features[val_idx] if morph_features is not None else None,
        photo_features[val_idx] if photo_features is not None else None,
        scaler_dino=scalers["dino"],
        scaler_morph=scalers["morph"],
        scaler_photo=scalers["photo"],
    )
    test_ds = FusionDataset(
        dino_features[test_idx],
        labels[test_idx],
        morph_features[test_idx] if morph_features is not None else None,
        photo_features[test_idx] if photo_features is not None else None,
        scaler_dino=scalers["dino"],
        scaler_morph=scalers["morph"],
        scaler_photo=scalers["photo"],
    )
    return train_ds, val_ds, test_ds


# ═══════════════════════════════════════════════════════════════════════════════
# MODÈLE
# ═══════════════════════════════════════════════════════════════════════════════

class _MLP(nn.Module):
    """Bloc MLP avec BN + GELU + Dropout."""

    def __init__(self, in_dim: int, hidden_dims: list, dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.net     = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GalaxyFusionClassifier(nn.Module):
    """Classificateur multi-modal à fusion tardive.

    Architecture :
        Branch A (DINOv2 768-dim) : 768 → 512 → 256
        Branch B (Morphométrie)   : n_morph → 64 → 32    [optionnel]
        Branch C (Photométrie)    : n_photo → 32 → 16    [optionnel]
        Tête de fusion            : concat → 256 → 128 → n_classes

    Le masquage des modalités manquantes se fait en zérant les embeddings
    correspondants pendant le forward (plutôt qu'en supprimant les branches),
    ce qui permet au modèle de fonctionner même sans toutes les données.

    Args:
        n_morph  : nombre de features morphométriques (0 = désactivé)
        n_photo  : nombre de features photométriques  (0 = désactivé)
        dropout  : taux de dropout global
        n_classes: nombre de classes (défaut 10)
    """

    def __init__(self, n_morph: int = 5, n_photo: int = 3,
                 dropout: float = 0.3, n_classes: int = NUM_CLASSES):
        super().__init__()
        self.n_morph  = n_morph
        self.n_photo  = n_photo
        self.n_classes = n_classes

        # Branch A — DINOv2 (obligatoire)
        self.branch_dino  = _MLP(768, [512, 256], dropout=dropout)
        fusion_dim        = self.branch_dino.out_dim  # 256

        # Branch B — Morphométrie
        if n_morph > 0:
            self.branch_morph = _MLP(n_morph, [64, 32], dropout=dropout)
            fusion_dim       += self.branch_morph.out_dim  # 32
        else:
            self.branch_morph = None

        # Branch C — Photométrie
        if n_photo > 0:
            self.branch_photo = _MLP(n_photo, [32, 16], dropout=dropout)
            fusion_dim       += self.branch_photo.out_dim  # 16
        else:
            self.branch_photo = None

        # Tête de fusion
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.BatchNorm1d(256),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128),
            nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        """Forward pass multi-modal.

        Args:
            batch : dict avec clés 'dino', optionnellement 'morph' / 'valid_morph'
                    'photo' / 'valid_photo'.

        Returns:
            logits (B, n_classes)
        """
        # Branch A — toujours présente
        emb = [self.branch_dino(batch["dino"])]

        # Branch B — masquer si données invalides
        if self.branch_morph is not None and "morph" in batch:
            morph_emb = self.branch_morph(batch["morph"])
            if "valid_morph" in batch:
                mask = batch["valid_morph"].unsqueeze(1).float()
                morph_emb = morph_emb * mask
            emb.append(morph_emb)

        # Branch C — masquer si données invalides
        if self.branch_photo is not None and "photo" in batch:
            photo_emb = self.branch_photo(batch["photo"])
            if "valid_photo" in batch:
                mask = batch["valid_photo"].unsqueeze(1).float()
                photo_emb = photo_emb * mask
            emb.append(photo_emb)

        fused = torch.cat(emb, dim=1)
        return self.fusion_head(fused)

    def predict_proba(self, batch: dict) -> np.ndarray:
        """Probabilités softmax (numpy) — pratique pour sklearn."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(batch)
        return F.softmax(logits, dim=1).cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRAÎNEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def _run_fusion_epoch(model: GalaxyFusionClassifier,
                      loader: DataLoader,
                      optimizer, criterion,
                      device: torch.device,
                      train: bool) -> dict:
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch  = {k: v.to(device) for k, v in batch.items()
                      if isinstance(v, torch.Tensor)}
            labels = batch.pop("label")

            if train:
                optimizer.zero_grad()

            logits = model(batch)
            loss   = criterion(logits, labels)

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * len(labels)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    preds, targets = np.array(all_preds), np.array(all_labels)
    return {
        "loss":         total_loss / len(targets),
        "balanced_acc": balanced_accuracy_score(targets, preds),
        "acc":          float((preds == targets).mean()),
    }


def train_fusion(model:       GalaxyFusionClassifier,
                 train_loader: DataLoader,
                 val_loader:   DataLoader,
                 epochs:       int = 40,
                 lr:           float = 3e-4,
                 weight_decay: float = 1e-4,
                 ckpt_dir:     str = "../checkpoints/",
                 model_name:   str = "fusion_galaxy10",
                 use_wandb:    bool = True,
                 device:       str = "cuda") -> GalaxyFusionClassifier:
    """Entraîne le GalaxyFusionClassifier avec label smoothing + cosine schedule.

    Returns:
        Modèle avec les meilleurs poids (val balanced_acc max).
    """
    import wandb as wb

    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    model    = model.to(device_t)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{model_name}_best.pt")

    # Label smoothing (aide sur les classes ambiguës comme Disturbed)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    modal_str = "DINOv2"
    if model.n_morph > 0: modal_str += " + Morphométrie"
    if model.n_photo > 0: modal_str += " + Photométrie"

    print(f"{'═'*60}")
    print(f"  GalaxyFusionClassifier — {modal_str}")
    print(f"  Params : {n_params:,}   Device : {device_t}")
    print(f"{'═'*60}\n")

    if use_wandb:
        try:
            wb.init(project="AstroVision", name=model_name,
                    config={"epochs": epochs, "lr": lr,
                            "modalites": modal_str, "model": "FusionMLP"})
        except Exception:
            use_wandb = False

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        tr = _run_fusion_epoch(model, train_loader, optimizer, criterion,
                               device_t, train=True)
        vl = _run_fusion_epoch(model, val_loader,   optimizer, criterion,
                               device_t, train=False)
        scheduler.step()

        is_best = vl["balanced_acc"] > best_acc
        tag = "★" if is_best else " "
        if is_best:
            best_acc = vl["balanced_acc"]
            torch.save(model.state_dict(), ckpt_path)

        print(f"[{tag}] Epoch {epoch:02d}/{epochs}  "
              f"Train bal={tr['balanced_acc']:.4f} loss={tr['loss']:.4f}  |  "
              f"Val bal={vl['balanced_acc']:.4f} loss={vl['loss']:.4f}")

        if use_wandb:
            try:
                wb.log({"epoch": epoch,
                        "train/balanced_acc": tr["balanced_acc"],
                        "train/loss": tr["loss"],
                        "val/balanced_acc": vl["balanced_acc"],
                        "val/loss": vl["loss"],
                        "lr": scheduler.get_last_lr()[0]})
            except Exception:
                pass

    print(f"\n✓ Meilleure val balanced_acc : {best_acc:.4f}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device_t))
    if use_wandb:
        try: wb.finish()
        except Exception: pass
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# ÉVALUATION & ANALYSE D'ABLATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_fusion(model: GalaxyFusionClassifier,
                    loader: DataLoader,
                    device: str = "cuda") -> dict:
    """Évalue le modèle et retourne les métriques détaillées."""
    from sklearn.metrics import (balanced_accuracy_score,
                                 classification_report,
                                 confusion_matrix)
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    model    = model.to(device_t).eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch  = {k: v.to(device_t) for k, v in batch.items()
                      if isinstance(v, torch.Tensor)}
            labels = batch.pop("label")
            logits = model(batch)
            probs  = F.softmax(logits, dim=1)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    preds, targets = np.array(all_preds), np.array(all_labels)
    probs_arr = np.array(all_probs)

    return {
        "balanced_acc": balanced_accuracy_score(targets, preds),
        "accuracy":     float((preds == targets).mean()),
        "report":       classification_report(targets, preds,
                                              target_names=CLASS_NAMES,
                                              zero_division=0),
        "confusion":    confusion_matrix(targets, preds),
        "probs":        probs_arr,
        "preds":        preds,
        "targets":      targets,
    }


def ablation_study(dino_features:  np.ndarray,
                   labels:         np.ndarray,
                   train_idx:      np.ndarray,
                   val_idx:        np.ndarray,
                   test_idx:       np.ndarray,
                   morph_features: Optional[np.ndarray] = None,
                   photo_features: Optional[np.ndarray] = None,
                   epochs:         int = 30,
                   device:         str = "cuda") -> dict:
    """Lance une ablation sur les combinaisons de modalités.

    Teste les combinaisons :
        (1) DINOv2 seul
        (2) DINOv2 + Morphométrie
        (3) DINOv2 + Photométrie         (si disponible)
        (4) DINOv2 + Morphométrie + Photo (si disponible)

    Returns:
        dict : {config_name → balanced_acc_test}
    """
    configs = [
        ("DINOv2 seul",           None,           None),
        ("DINOv2 + Morpho",       morph_features, None),
    ]
    if photo_features is not None:
        configs.append(("DINOv2 + Photo",         None,           photo_features))
        configs.append(("DINOv2 + Morpho + Photo", morph_features, photo_features))

    results = {}
    for name, morph, photo in configs:
        print(f"\n{'─'*50}")
        print(f"  Ablation : {name}")
        print(f"{'─'*50}")

        n_morph = morph.shape[1] if morph is not None else 0
        n_photo = photo.shape[1] if photo is not None else 0

        train_ds, val_ds, test_ds = build_fusion_datasets(
            dino_features, labels, train_idx, val_idx, test_idx,
            morph_features=morph, photo_features=photo,
        )
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,
                                  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False,
                                  num_workers=0)
        test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False,
                                  num_workers=0)

        model = GalaxyFusionClassifier(n_morph=n_morph, n_photo=n_photo)
        model = train_fusion(model, train_loader, val_loader,
                             epochs=epochs, model_name=f"ablation_{name.replace(' ', '_')}",
                             use_wandb=False, device=device)

        eval_res     = evaluate_fusion(model, test_loader, device=device)
        results[name] = eval_res["balanced_acc"]
        print(f"  Test balanced_acc : {eval_res['balanced_acc']:.4f}")

    print(f"\n{'═'*50}")
    print("  RÉSUMÉ ABLATION")
    print(f"{'═'*50}")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:<35} {acc:.4f}")
    print(f"{'═'*50}")
    return results
