{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# AstroVision — 01 · Galaxy10 DECaLS : CNN + Grad-CAM\n",
    "\n",
    "**Objectif :** classification morphologique de galaxies sur Galaxy10 DECaLS.  \n",
    "**Pipeline :** téléchargement → SimpleCNN from scratch → EfficientNet fine-tuning → Grad-CAM\n",
    "\n",
    "| Étape | Module |\n",
    "|-------|--------|\n",
    "| Données | `astrovision.data_loader` |\n",
    "| Modèles | `astrovision.models` |\n",
    "| Entraînement | `astrovision.trainer` |\n",
    "| Visualisation | `astrovision.gradcam` |"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.insert(0, '../src')\n",
    "\n",
    "import torch\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from astrovision import (\n",
    "    load_galaxy10_splits, CLASS_NAMES,\n",
    "    build_model,\n",
    "    train,\n",
    "    GradCAM, plot_gradcam_grid,\n",
    ")\n",
    "\n",
    "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
    "print(f'Device : {DEVICE}')\n",
    "if DEVICE == 'cuda':\n",
    "    print(f'GPU    : {torch.cuda.get_device_name(0)}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 1 — Chargement des données"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_loader, val_loader, test_loader = load_galaxy10_splits(\n",
    "    batch_size=64,\n",
    "    num_workers=4,\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Aperçu rapide — grille 5×5 d'images\n",
    "imgs, labels = next(iter(train_loader))\n",
    "\n",
    "mean = torch.tensor([0.485, 0.456, 0.406])\n",
    "std  = torch.tensor([0.229, 0.224, 0.225])\n",
    "\n",
    "fig, axes = plt.subplots(5, 5, figsize=(12, 12), dpi=100)\n",
    "for i, ax in enumerate(axes.flat):\n",
    "    img = imgs[i].clone()\n",
    "    for c in range(3):\n",
    "        img[c] = img[c] * std[c] + mean[c]\n",
    "    ax.imshow(img.permute(1, 2, 0).clamp(0, 1).numpy())\n",
    "    ax.set_title(CLASS_NAMES[labels[i]], fontsize=7)\n",
    "    ax.axis('off')\n",
    "plt.suptitle('Galaxy10 DECaLS — exemples', fontsize=13, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.savefig('../figures/galaxy10_samples.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 2 — Étape A : SimpleCNN (baseline from scratch)"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "simple_cnn = build_model('simplecnn')\n",
    "print(simple_cnn)\n",
    "\n",
    "n_params = sum(p.numel() for p in simple_cnn.parameters() if p.requires_grad)\n",
    "print(f'\\nParamètres entraînables : {n_params:,}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "simple_cnn = train(\n",
    "    simple_cnn,\n",
    "    train_loader,\n",
    "    val_loader,\n",
    "    model_name='simplecnn_galaxy10',\n",
    "    epochs=30,\n",
    "    lr=1e-3,\n",
    "    use_wandb=True,\n",
    "    device=DEVICE,\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 3 — Étape B : EfficientNet-B0 (transfer learning)"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Phase 1 : backbone gelé — entraîner seulement la tête (5 epochs)\n",
    "effnet = build_model('efficientnet', freeze_backbone=True)\n",
    "effnet = train(\n",
    "    effnet,\n",
    "    train_loader, val_loader,\n",
    "    model_name='efficientnet_galaxy10_warmup',\n",
    "    epochs=5,\n",
    "    lr=1e-3,\n",
    "    use_wandb=True,\n",
    "    device=DEVICE,\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Phase 2 : fine-tuning complet (dégeler tout)\n",
    "for param in effnet.parameters():\n",
    "    param.requires_grad = True\n",
    "\n",
    "effnet = train(\n",
    "    effnet,\n",
    "    train_loader, val_loader,\n",
    "    model_name='efficientnet_galaxy10_finetune',\n",
    "    epochs=25,\n",
    "    lr=1e-4,   # LR plus bas pour le fine-tuning\n",
    "    use_wandb=True,\n",
    "    device=DEVICE,\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 4 — Grad-CAM : ce que le modèle regarde"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Grad-CAM sur SimpleCNN\n",
    "gcam_simple = GradCAM(simple_cnn, simple_cnn.gradcam_target)\n",
    "\n",
    "imgs_test, labels_test = next(iter(test_loader))\n",
    "\n",
    "fig = plot_gradcam_grid(\n",
    "    simple_cnn, gcam_simple,\n",
    "    imgs_test, labels_test,\n",
    "    device=DEVICE,\n",
    "    n_samples=8,\n",
    "    filename='gradcam_simplecnn.png',\n",
    ")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Grad-CAM sur EfficientNet — comparer les régions d'attention\n",
    "gcam_eff = GradCAM(effnet, effnet.gradcam_target)\n",
    "\n",
    "fig = plot_gradcam_grid(\n",
    "    effnet, gcam_eff,\n",
    "    imgs_test, labels_test,\n",
    "    device=DEVICE,\n",
    "    n_samples=8,\n",
    "    filename='gradcam_efficientnet.png',\n",
    ")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5 — Évaluation finale sur le test set\n",
    "\n",
    "Comparer SimpleCNN vs EfficientNet sur les métriques finales."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from astrovision.trainer import _run_epoch\n",
    "import torch.nn as nn\n",
    "\n",
    "criterion = nn.CrossEntropyLoss()\n",
    "device_t = torch.device(DEVICE)\n",
    "\n",
    "for name, model in [('SimpleCNN', simple_cnn), ('EfficientNet-B0', effnet)]:\n",
    "    metrics = _run_epoch(model, test_loader, None, criterion, device_t, train=False)\n",
    "    print(f'{name:20s}  acc={metrics[\"acc\"]:.3f}  balanced_acc={metrics[\"balanced_acc\"]:.3f}')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
