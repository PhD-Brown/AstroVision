"""
AstroVision — unet.py
U-Net pour la segmentation sémantique de galaxies.

Architecture : Ronneberger et al. 2015 — U-Net: Convolutional Networks for
Biomedical Image Segmentation (arXiv:1505.04597), adapté pour les galaxies.

Entraînement sur pseudo-labels générés par OtsuSegmenter :
    - Classe 0 : fond de ciel
    - Classe 1 : disque galactique
    - Classe 2 : bulbe / noyau

Usage :
    from astrovision.unet import UNet, GalaxySegDataset, train_unet
    from astrovision.segmenter import OtsuSegmenter

    # Générer pseudo-labels
    otsu = OtsuSegmenter()
    pseudo_masks = otsu.batch_segment(images_train)

    # Dataset + Entraînement
    train_ds = GalaxySegDataset(images_train, pseudo_masks, train=True)
    model    = UNet(n_classes=3)
    model    = train_unet(model, train_loader, val_loader, epochs=20)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score

# ── Blocs U-Net ────────────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    """(Conv → BN → ReLU) × 2."""
    def __init__(self, in_ch: int, out_ch: int, mid_ch: int = None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class Down(nn.Module):
    """MaxPool2d + DoubleConv."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.block(x)


class Up(nn.Module):
    """ConvTranspose2d + skip connection + DoubleConv."""
    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up   = nn.Upsample(scale_factor=2, mode="bilinear",
                                    align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2)
        else:
            self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Padding si tailles différentes (images non-multiples de 16)
        dy = skip.shape[2] - x.shape[2]
        dx = skip.shape[3] - x.shape[3]
        x  = F.pad(x, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        return self.conv(torch.cat([skip, x], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x): return self.conv(x)


# ── U-Net complet ──────────────────────────────────────────────────────────────
class UNet(nn.Module):
    """U-Net pour segmentation sémantique de galaxies (3 classes par défaut).

    Architecture :
        Encoder  : 4 niveaux, 64 → 128 → 256 → 512 filtres
        Bottleneck : 1024 filtres
        Decoder  : 4 niveaux avec skip connections

    Input  : (B, 3, H, W) normalisé ImageNet
    Output : (B, n_classes, H, W) logits

    Args:
        n_classes : nombre de classes de sortie (défaut 3 : fond/disque/bulbe)
        bilinear  : True = upsampling bilinéaire, False = ConvTranspose2d
        dropout   : dropout dans le bottleneck
    """

    def __init__(self, n_classes: int = 3, bilinear: bool = True,
                 dropout: float = 0.3):
        super().__init__()
        self.n_classes = n_classes

        # Encoder
        self.enc1 = DoubleConv(3,   64)
        self.enc2 = Down(64,  128)
        self.enc3 = Down(128, 256)
        self.enc4 = Down(256, 512)

        # Bottleneck
        factor = 2 if bilinear else 1
        self.bottleneck = nn.Sequential(
            Down(512, 1024 // factor),
            nn.Dropout2d(dropout),
        )

        # Decoder
        self.dec4 = Up(1024, 512  // factor, bilinear)
        self.dec3 = Up(512,  256  // factor, bilinear)
        self.dec2 = Up(256,  128  // factor, bilinear)
        self.dec1 = Up(128,  64,             bilinear)
        self.out  = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1  = self.enc1(x)
        e2  = self.enc2(e1)
        e3  = self.enc3(e2)
        e4  = self.enc4(e3)
        bot = self.bottleneck(e4)
        d4  = self.dec4(bot, e4)
        d3  = self.dec3(d4,  e3)
        d2  = self.dec2(d3,  e2)
        d1  = self.dec1(d2,  e1)
        return self.out(d1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne les classes prédites (argmax)."""
        return self.forward(x).argmax(dim=1)


# ── Loss functions ─────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    """Dice Loss multi-classe.

    Meilleur que la Cross-Entropy pour les datasets de segmentation déséquilibrés
    (fond >> galaxie >> bulbe).
    """
    def __init__(self, smooth: float = 1.0, n_classes: int = 3):
        super().__init__()
        self.smooth    = smooth
        self.n_classes = n_classes

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        total_loss = 0.0
        for c in range(self.n_classes):
            p = probs[:, c]
            t = (targets == c).float()
            intersection = (p * t).sum()
            total_loss += 1.0 - (2 * intersection + self.smooth) / (
                p.sum() + t.sum() + self.smooth)
        return total_loss / self.n_classes


class CombinedLoss(nn.Module):
    """Dice + CrossEntropy (0.5 chacun) — meilleure convergence."""
    def __init__(self, n_classes: int = 3, weight: torch.Tensor = None):
        super().__init__()
        self.dice = DiceLoss(n_classes=n_classes)
        self.ce   = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.05)

    def forward(self, logits, targets):
        return 0.5 * self.dice(logits, targets) + 0.5 * self.ce(logits, targets)


# ── Dataset ────────────────────────────────────────────────────────────────────
class GalaxySegDataset(Dataset):
    """Dataset pour U-Net avec pseudo-labels Otsu.

    Args:
        images : (N, H, W, 3) uint8
        masks  : (N, H, W) uint8 — labels 0/1/2
        train  : si True, applique les augmentations
    """

    TRANSFORM_IMG_TRAIN = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])
    TRANSFORM_IMG_EVAL = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])

    def __init__(self, images: np.ndarray, masks: np.ndarray,
                 train: bool = True):
        assert len(images) == len(masks), "images et masks doivent avoir la même longueur"
        self.images = images
        self.masks  = masks
        self.train  = train

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        import PIL.Image as PILImage

        img  = self.images[idx]       # (H, W, 3) uint8
        mask = self.masks[idx]        # (H, W) uint8

        # Image → tensor normalisé
        if self.train:
            img_t = self.TRANSFORM_IMG_TRAIN(img)
        else:
            img_t = self.TRANSFORM_IMG_EVAL(img)

        # Mask → tensor (même dimensions que l'image)
        mask_pil = PILImage.fromarray(mask)
        mask_pil = mask_pil.resize((256, 256), PILImage.NEAREST)

        # Augmentation synchronisée image/mask
        if self.train and np.random.random() > 0.5:
            img_t    = torch.flip(img_t, dims=[2])         # flip horizontal
            mask_pil = mask_pil.transpose(PILImage.FLIP_LEFT_RIGHT)
        if self.train and np.random.random() > 0.5:
            img_t    = torch.flip(img_t, dims=[1])         # flip vertical
            mask_pil = mask_pil.transpose(PILImage.FLIP_TOP_BOTTOM)

        mask_t = torch.from_numpy(np.array(mask_pil)).long()
        return img_t, mask_t


# ── Entraînement ───────────────────────────────────────────────────────────────
def _run_epoch_seg(model, loader, optimizer, scaler, criterion,
                   device, train: bool):
    """Epoch de segmentation avec AMP FP16."""
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            if train:
                optimizer.zero_grad()

            with torch.autocast(device_type="cuda",
                                 dtype=torch.float16,
                                 enabled=(device.type == "cuda")):
                logits = model(imgs)
                loss   = criterion(logits, masks)

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * len(imgs)
            preds = logits.argmax(1).cpu().numpy().flatten()
            tgts  = masks.cpu().numpy().flatten()
            all_preds.extend(preds)
            all_labels.extend(tgts)

    n = len(loader.dataset)
    preds_arr = np.array(all_preds)
    tgts_arr  = np.array(all_labels)

    # IoU moyen (Jaccard)
    iou = jaccard_score(tgts_arr, preds_arr,
                        average="macro",
                        zero_division=0)
    return {"loss": total_loss / n, "mean_iou": float(iou)}


def train_unet(model: UNet,
               train_loader: DataLoader,
               val_loader: DataLoader,
               model_name: str = "unet_galaxy10",
               epochs: int = 20,
               lr: float = 1e-4,
               weight_decay: float = 1e-5,
               ckpt_dir: str = "../checkpoints/",
               use_wandb: bool = True,
               device: str = "cuda") -> UNet:
    """Entraîne le U-Net avec Dice+CE loss et AMP FP16.

    Returns:
        Modèle chargé avec les meilleurs poids (val mean_iou max).
    """
    import wandb

    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    model    = model.to(device_t)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{model_name}_best.pt")

    # Pondérer les classes : fond très fréquent, bulbe rare
    weights = torch.tensor([0.3, 1.5, 3.0]).to(device_t)
    criterion = CombinedLoss(n_classes=model.n_classes, weight=weights)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    scaler    = torch.amp.GradScaler("cuda", enabled=(device_t.type == "cuda"))

    if use_wandb:
        try:
            wandb.init(project="AstroVision", name=model_name,
                       config={"epochs":epochs,"lr":lr,"model":"UNet","amp":True})
        except Exception:
            use_wandb = False

    best_iou = 0.0
    print(f"Device  : {device_t}")
    print(f"Model   : UNet({model.n_classes} classes)")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params  : {n_params/1e6:.2f}M\n")

    for epoch in range(1, epochs + 1):
        tr = _run_epoch_seg(model, train_loader, optimizer, scaler,
                            criterion, device_t, train=True)
        vl = _run_epoch_seg(model, val_loader,   optimizer, scaler,
                            criterion, device_t, train=False)
        scheduler.step()

        tag = "★" if vl["mean_iou"] > best_iou else " "
        print(f"[{tag}] Epoch {epoch:02d}/{epochs}  "
              f"Train loss={tr['loss']:.4f} mIoU={tr['mean_iou']:.3f}  |  "
              f"Val   loss={vl['loss']:.4f} mIoU={vl['mean_iou']:.3f}")

        if use_wandb:
            try:
                wandb.log({"epoch": epoch,
                           "train/loss": tr["loss"], "train/miou": tr["mean_iou"],
                           "val/loss":   vl["loss"], "val/miou":   vl["mean_iou"],
                           "lr": scheduler.get_last_lr()[0]})
            except Exception:
                pass

        if vl["mean_iou"] > best_iou:
            best_iou = vl["mean_iou"]
            torch.save(model.state_dict(), ckpt_path)
            print(f"    → Nouveau best mIoU : {best_iou:.4f} — sauvegardé")

    if use_wandb:
        try:
            wandb.finish()
        except Exception:
            pass

    model.load_state_dict(torch.load(ckpt_path, map_location=device_t))
    print(f"\n✓ Entraînement terminé. Meilleur val mIoU : {best_iou:.4f}")
    return model


# ── Évaluation & Visualisation ─────────────────────────────────────────────────
DARK = {
    "figure.facecolor":"#0d1117","axes.facecolor":"#161b22",
    "axes.edgecolor":"#30363d","axes.labelcolor":"#c9d1d9",
    "axes.titlecolor":"#f0f6fc","xtick.color":"#8b949e",
    "ytick.color":"#8b949e","text.color":"#c9d1d9",
    "grid.color":"#21262d","figure.dpi":150,
    "savefig.facecolor":"#0d1117","savefig.bbox":"tight",
}
PALETTE = [
    "#6e40aa","#4c6edb","#23abd8","#1ac7c2","#1ddfa3",
    "#52f667","#aff05b","#e2b72f","#fb8a27","#f83e4b",
]

SEG_COLORS_MAP = {
    0: np.array([13,  17,  23],  dtype=np.uint8),   # fond
    1: np.array([78, 105, 219],  dtype=np.uint8),   # disque
    2: np.array([26, 199, 194],  dtype=np.uint8),   # bulbe
}

CLASS_NAMES = ["Fond", "Disque", "Bulbe"]

def segmap_to_rgb(seg: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*seg.shape, 3), dtype=np.uint8)
    for c, col in SEG_COLORS_MAP.items():
        rgb[seg == c] = col
    return rgb


def plot_unet_gallery(model: UNet, images: np.ndarray, labels: np.ndarray,
                      pseudo_masks: np.ndarray, device: str = "cuda",
                      n_per_class: int = 2, save_path: str = None) -> plt.Figure:
    """Galerie U-Net : original | pseudo-label | prédiction U-Net | overlay.

    Permet de voir si U-Net apprend quelque chose au-delà du simple Otsu.
    """
    import PIL.Image as PILImage
    plt.rcParams.update(DARK)

    MEAN = torch.tensor([0.485,0.456,0.406])
    STD  = torch.tensor([0.229,0.224,0.225])

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    n_cls = len(CLASS_NAMES)
    cols  = n_per_class * 4
    fig, axes = plt.subplots(n_cls, cols,
                              figsize=(cols * 2.5, n_cls * 3), dpi=100)
    fig.patch.set_facecolor("#0d1117")

    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device_t).eval()

    np.random.seed(42)
    for row, cls_idx in enumerate(range(n_cls)):
        cls_imgs  = images[labels == cls_idx]
        cls_masks = pseudo_masks[labels == cls_idx]
        n_samp    = min(n_per_class, len(cls_imgs))
        samp_idx  = np.random.choice(len(cls_imgs), n_samp, replace=False)

        for s, si in enumerate(samp_idx):
            img  = cls_imgs[si]
            pmsk = cls_masks[si]
            img_224 = np.array(PILImage.fromarray(img).resize((256,256)))

            # Prédiction U-Net
            img_t = transform(img).unsqueeze(0).to(device_t)
            with torch.no_grad():
                with torch.autocast("cuda", enabled=(device_t.type=="cuda")):
                    pred = model.predict(img_t)[0].cpu().numpy()

            # Désnormalisation pour affichage
            img_disp = img_t[0].cpu()
            for c in range(3):
                img_disp[c] = img_disp[c] * STD[c] + MEAN[c]
            img_disp = img_disp.permute(1,2,0).clamp(0,1).numpy()

            pmsk_rgb = segmap_to_rgb(np.array(
                PILImage.fromarray(pmsk).resize((256,256), PILImage.NEAREST)))
            pred_rgb = segmap_to_rgb(pred)
            overlay  = np.clip(0.6*img_disp + 0.4*pred_rgb.astype(np.float32)/255., 0,1)

            base = s * 4
            for j, (data, ttl) in enumerate([
                (img_disp, "Original"),
                (pmsk_rgb, "Pseudo-label"),
                (pred_rgb, "U-Net préd."),
                (overlay,  "Overlay"),
            ]):
                ax = axes[row, base + j]
                ax.imshow(data, vmin=0, vmax=1 if data.dtype == float else 255)
                ax.axis("off")
                if row == 0:
                    ax.set_title(ttl, fontsize=7.5, color="#8b949e", pad=3)
            if base == 0:
                axes[row, 0].set_ylabel(f"{cls_idx}.{CLASS_NAMES[cls_idx][:14]}",
                                         fontsize=7, color=PALETTE[cls_idx],
                                         rotation=0, labelpad=90, va="center")

    fig.suptitle("U-Net — Original | Pseudo-label | Prédiction | Overlay",
                 fontsize=12, fontweight="bold", color="#f0f6fc", y=1.005)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"✓ Galerie U-Net sauvegardée : {save_path}")
    return fig
