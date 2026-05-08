"""
AstroVision - Boucle d'entraînement avec suivi W&B.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import balanced_accuracy_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

CHECKPOINTS_DIR = Path(__file__).parent.parent.parent / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Epoch helpers ─────────────────────────────────────────────────────────────
def _run_epoch(model, loader, optimizer, criterion, device, train: bool):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(imgs)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    preds, targets = np.array(all_preds), np.array(all_labels)
    return {
        "loss":         total_loss / n,
        "acc":          float((preds == targets).mean()),
        "balanced_acc": balanced_accuracy_score(targets, preds),
    }


# ── Entraînement principal ────────────────────────────────────────────────────
def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str = "astrovision",
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    use_wandb: bool = True,
    device: str = "cuda",
) -> nn.Module:
    """Entraîne le modèle et sauvegarde le meilleur checkpoint.

    Returns:
        Modèle avec les poids du meilleur epoch (val balanced_acc).
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    if use_wandb:
        wandb.init(
            project="AstroVision",
            name=model_name,
            config={"epochs": epochs, "lr": lr, "weight_decay": weight_decay, "model": model_name},
        )

    best_val_acc = 0.0
    ckpt_path = CHECKPOINTS_DIR / f"{model_name}_best.pt"

    for epoch in range(1, epochs + 1):
        train_m = _run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_m   = _run_epoch(model, val_loader,   optimizer, criterion, device, train=False)
        scheduler.step()

        tag = "★" if val_m["balanced_acc"] > best_val_acc else " "
        print(
            f"[{tag}] Epoch {epoch:03d}/{epochs}  "
            f"Train  loss={train_m['loss']:.4f}  bal={train_m['balanced_acc']:.3f}  |  "
            f"Val    loss={val_m['loss']:.4f}  bal={val_m['balanced_acc']:.3f}"
        )

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_m["loss"],
                "train/acc": train_m["acc"],
                "train/balanced_acc": train_m["balanced_acc"],
                "val/loss": val_m["loss"],
                "val/acc": val_m["acc"],
                "val/balanced_acc": val_m["balanced_acc"],
                "lr": scheduler.get_last_lr()[0],
            })

        if val_m["balanced_acc"] > best_val_acc:
            best_val_acc = val_m["balanced_acc"]
            torch.save(model.state_dict(), ckpt_path)
            print(f"    → Nouveau best : {best_val_acc:.3f}  —  sauvegardé dans {ckpt_path}")

    if use_wandb:
        wandb.finish()

    # Recharge les meilleurs poids avant de retourner
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"\n✓ Entraînement terminé. Meilleur val balanced_acc : {best_val_acc:.3f}")
    return model
