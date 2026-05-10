"""
AstroVision — uncertainty.py
Quantification d'incertitude pour la classification morphologique de galaxies.

Trois approches complementaires :
    1. MC Dropout       — T passes stochastiques -> incertitude epistemique
    2. Temperature Scaling — calibration post-hoc (ECE)
    3. Conformal Prediction — couverture garantie (1-alpha)

Corrections v2 :
    - MCDropoutPredictor.predict() : passage du dict complet au modele
      (compatible GalaxyFusionClassifier et modeles image classiques)
    - TemperatureScaler.fit()      : temperature forcee sur CPU,
      suppression du self.to(device_t) qui causait le mismatch device
    - plot_uncertainty_analysis()  : np.median() remplace .median() (pandas)
"""

import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import softmax
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

CLASS_NAMES = [
    "Disturbed", "Merging", "Round Smooth", "In-between Round Smooth",
    "Cigar Shaped Smooth", "Barred Spiral", "Unbarred Tight Spiral",
    "Unbarred Loose Spiral", "Edge-on without Bulge", "Edge-on with Bulge",
]
DARK = {
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",   "axes.labelcolor": "#c9d1d9",
    "axes.titlecolor": "#f0f6fc",  "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",      "text.color": "#c9d1d9",
    "grid.color": "#21262d",       "figure.dpi": 150,
    "savefig.facecolor": "#0d1117","savefig.bbox": "tight",
}
PALETTE = [
    "#6e40aa","#4c6edb","#23abd8","#1ac7c2","#1ddfa3",
    "#52f667","#aff05b","#e2b72f","#fb8a27","#f83e4b",
]


# ==============================================================================
# 1 — MC DROPOUT
# ==============================================================================

class MCDropoutPredictor:
    """Predictions par Monte Carlo Dropout (Gal & Ghahramani 2016).

    Reactive le dropout pendant l'inference pour T echantillons stochastiques.

    Compatible avec :
        - Modeles image : batch = (imgs_tensor, labels)
        - GalaxyFusionClassifier : batch = dict{"dino", "morph", "photo", "label"}

    Args:
        model  : modele PyTorch avec couches Dropout
        T      : nombre de passes stochastiques (defaut 30)
        device : device PyTorch
    """

    def __init__(self, model: nn.Module, T: int = 30, device: str = "cuda"):
        self.model  = model
        self.T      = T
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model  = self.model.to(self.device)

    def _enable_dropout(self):
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    @torch.no_grad()
    def predict(self, loader: DataLoader,
                return_samples: bool = False) -> dict:
        """T passes stochastiques sur tout le loader.

        Returns dict avec : mean_probs, variance, entropy, mutual_info,
                            preds, targets, [samples si return_samples=True]
        """
        self._enable_dropout()
        all_probs_runs = []
        all_labels     = []

        for t in range(self.T):
            probs_t  = []
            labels_t = []

            for batch in loader:

                # FIX v2 : gestion correcte des deux formats de batch
                if isinstance(batch, (list, tuple)):
                    # Format classique : (imgs_tensor, labels)
                    imgs, labels = batch
                    imgs   = imgs.to(self.device)
                    logits = self.model(imgs)

                elif isinstance(batch, dict):
                    # Format FusionDataset : extraire label, passer le reste au modele
                    labels      = batch["label"]
                    model_input = {
                        k: v.to(self.device)
                        for k, v in batch.items()
                        if k != "label" and isinstance(v, torch.Tensor)
                    }
                    logits = self.model(model_input)

                else:
                    raise ValueError(f"Format de batch non reconnu : {type(batch)}")

                probs_t.append(F.softmax(logits, dim=1).cpu().numpy())
                if t == 0:
                    labels_t.extend(
                        labels.numpy() if isinstance(labels, torch.Tensor)
                        else labels
                    )

            all_probs_runs.append(np.concatenate(probs_t, axis=0))
            if t == 0:
                all_labels = np.array(labels_t)
            print(f"  MC Dropout pass {t+1:02d}/{self.T}", end="\r")

        print()
        samples    = np.stack(all_probs_runs, axis=0)   # (T, N, C)
        mean_probs = samples.mean(axis=0)                # (N, C)
        variance   = samples.var(axis=0)                 # (N, C)

        pred_entropy       = scipy_entropy(mean_probs, axis=1)
        expected_entropy   = scipy_entropy(samples, axis=2).mean(axis=0)
        mutual_info        = np.clip(pred_entropy - expected_entropy, 0, None)

        result = {
            "mean_probs":  mean_probs,
            "variance":    variance,
            "entropy":     pred_entropy,
            "mutual_info": mutual_info,
            "preds":       mean_probs.argmax(axis=1),
            "targets":     all_labels,
        }
        if return_samples:
            result["samples"] = samples
        return result


# ==============================================================================
# 2 — TEMPERATURE SCALING
# ==============================================================================

class TemperatureScaler(nn.Module):
    """Calibration post-hoc par temperature (Guo et al. 2017).

    FIX v2 : tout le calcul est force sur CPU pour eviter les conflits de device.
    La temperature n'est jamais deplacee sur GPU.
    """

    def __init__(self):
        super().__init__()
        # CPU uniquement — pas de .to(device) sur cet objet
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.fitted      = False
        self.T_opt       = None
        self.ece_before  = None
        self.ece_after   = None

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # logits et temperature sont tous les deux sur CPU
        return logits / self.temperature

    def fit(self, model: nn.Module, val_loader: DataLoader,
            device: str = "cuda", max_iter: int = 50) -> float:
        """Optimise T* sur le val set. Tout calcul sur CPU apres collecte."""
        from torch.optim import LBFGS

        device_t = torch.device(device if torch.cuda.is_available() else "cpu")
        model    = model.to(device_t).eval()

        # FIX v2 : NE PAS faire self.to(device_t)
        # self.temperature reste sur CPU, les logits seront ramenes sur CPU

        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    imgs, labels = batch
                    logits = model(imgs.to(device_t))
                elif isinstance(batch, dict):
                    labels      = batch["label"]
                    model_input = {
                        k: v.to(device_t)
                        for k, v in batch.items()
                        if k != "label" and isinstance(v, torch.Tensor)
                    }
                    logits = model(model_input)
                else:
                    continue
                all_logits.append(logits.detach().cpu())   # force CPU
                all_labels.append(
                    labels.cpu() if isinstance(labels, torch.Tensor)
                    else torch.tensor(labels)
                )

        logits_all = torch.cat(all_logits)   # CPU
        labels_all = torch.cat(all_labels)   # CPU
        # self.temperature est deja sur CPU -> aucun mismatch possible

        optimizer = LBFGS([self.temperature], max_iter=max_iter, lr=0.01)
        criterion = nn.CrossEntropyLoss()

        def eval_fn():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits_all), labels_all)
            loss.backward()
            return loss

        optimizer.step(eval_fn)
        self.temperature.data.clamp_(0.01, 10.0)
        self.T_opt = self.temperature.item()

        self.ece_before = self._ece(
            softmax(logits_all.numpy(), axis=1), labels_all.numpy())
        self.ece_after  = self._ece(
            softmax(self.forward(logits_all).detach().numpy(), axis=1),
            labels_all.numpy())

        print(f"Temperature Scaling")
        print(f"  T*         = {self.T_opt:.4f}")
        print(f"  ECE avant  : {self.ece_before:.4f}")
        print(f"  ECE apres  : {self.ece_after:.4f}")
        self.fitted = True
        return self.T_opt

    def calibrate(self, probs: np.ndarray,
                  logits: Optional[np.ndarray] = None) -> np.ndarray:
        """Applique la calibration. Necessite .fit() au prealable."""
        if not self.fitted:
            raise RuntimeError(
                "Appeler .fit() avant .calibrate(). "
                "Alternative directe : softmax(log(probs) / T_opt, axis=1)")
        T = self.temperature.item()
        if logits is not None:
            return softmax(logits / T, axis=1)
        return softmax(np.log(np.clip(probs, 1e-10, 1.0)) / T, axis=1)

    @staticmethod
    def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
        confidences = probs.max(axis=1)
        accuracies  = (probs.argmax(axis=1) == labels).astype(float)
        bins = np.linspace(0, 1, n_bins + 1)
        ece  = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (confidences >= lo) & (confidences < hi)
            if mask.sum() == 0:
                continue
            ece += mask.sum() / len(labels) * abs(
                accuracies[mask].mean() - confidences[mask].mean())
        return float(ece)

    @staticmethod
    def plot_reliability_diagram(probs: np.ndarray, labels: np.ndarray,
                                  title: str = "Reliability Diagram",
                                  n_bins: int = 15,
                                  save_path: Optional[str] = None) -> plt.Figure:
        plt.rcParams.update(DARK)
        confidences = probs.max(axis=1)
        accuracies  = (probs.argmax(axis=1) == labels).astype(float)
        bins = np.linspace(0, 1, n_bins + 1)
        bin_accs, bin_confs = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (confidences >= lo) & (confidences < hi)
            if mask.sum() == 0:
                continue
            bin_accs.append(accuracies[mask].mean())
            bin_confs.append(confidences[mask].mean())
        ece = TemperatureScaler._ece(probs, labels, n_bins)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(title, fontsize=13, fontweight="bold")

        ax = axes[0]
        ax.plot([0,1],[0,1],"--",color="#8b949e",lw=1.5,label="Parfait")
        ax.bar(bin_confs, bin_accs, width=0.05, alpha=0.75,
               color="#23abd8", label=f"Modele (ECE={ece:.4f})")
        ax.set_xlabel("Confiance"); ax.set_ylabel("Precision")
        ax.set_title("Reliability Diagram"); ax.legend()
        ax.set_xlim(0,1); ax.set_ylim(0,1)

        ax = axes[1]
        ax.hist(confidences, bins=30, color="#1ddfa3", alpha=0.75)
        ax.axvline(confidences.mean(), color="#f83e4b", ls="--",
                   label=f"Moy. = {confidences.mean():.3f}")
        ax.set_xlabel("Confiance (max softmax)")
        ax.set_ylabel("Nombre de galaxies")
        ax.set_title("Distribution de confiance"); ax.legend()

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig


# ==============================================================================
# 3 — CONFORMAL PREDICTION
# ==============================================================================

class ConformalPredictor:
    """Conformal Prediction — couverture garantie a (1-alpha).

    P(y* in C(x)) >= 1 - alpha sans hypothese sur le modele.
    Score LAC (Angelopoulos et al. 2021).
    """

    def __init__(self, alpha: float = 0.1):
        if not 0 < alpha < 1:
            raise ValueError("alpha doit etre dans (0, 1)")
        self.alpha  = alpha
        self.q_hat  = None
        self.fitted = False

    def fit(self, cal_probs: np.ndarray, cal_labels: np.ndarray) -> float:
        n = len(cal_labels)
        scores         = 1.0 - cal_probs[np.arange(n), cal_labels]
        q_level        = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.q_hat     = float(np.quantile(scores, q_level))
        self.fitted    = True
        coverage       = (cal_probs[np.arange(n), cal_labels] >= 1 - self.q_hat).mean()
        print(f"Conformal Prediction (alpha={self.alpha})")
        print(f"  q_hat              = {self.q_hat:.4f}")
        print(f"  Couverture cal set : {coverage:.4f}  (cible >= {1-self.alpha:.2f})")
        return self.q_hat

    def predict(self, probs: np.ndarray) -> list:
        if not self.fitted:
            raise RuntimeError("Appeler .fit() avant .predict()")
        threshold = 1.0 - self.q_hat
        return [
            list(np.where(p >= threshold)[0]) or [int(p.argmax())]
            for p in probs
        ]

    def coverage_and_size(self, pred_sets: list,
                           true_labels: np.ndarray) -> dict:
        coverage = np.mean([true_labels[i] in pred_sets[i]
                            for i in range(len(true_labels))])
        sizes    = np.array([len(s) for s in pred_sets])
        return {
            "coverage":          float(coverage),
            "mean_size":         float(sizes.mean()),
            "size_distribution": {k: int((sizes == k).sum()) for k in range(1, 11)},
            "singleton_rate":    float((sizes == 1).mean()),
        }

    def plot_prediction_sets(self, pred_sets: list,
                              probs: np.ndarray,
                              true_labels: np.ndarray,
                              save_path: Optional[str] = None) -> plt.Figure:
        plt.rcParams.update(DARK)
        stats = self.coverage_and_size(pred_sets, true_labels)
        sizes = np.array([len(s) for s in pred_sets])

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            f"Conformal Prediction (alpha={self.alpha}) — "
            f"Couverture={stats['coverage']:.3f} | "
            f"Taille moy.={stats['mean_size']:.2f}",
            fontsize=13, fontweight="bold",
        )

        ax = axes[0]
        sv = list(stats["size_distribution"].keys())
        sc = list(stats["size_distribution"].values())
        ax.bar(sv, sc, color="#23abd8", alpha=0.8)
        ax.axvline(stats["mean_size"], color="#f83e4b", ls="--",
                   label=f"Moy. = {stats['mean_size']:.2f}")
        ax.set_xlabel("Taille de l'ensemble"); ax.set_ylabel("Nb galaxies")
        ax.set_title("Distribution des tailles"); ax.legend()

        ax = axes[1]
        class_cov = []
        for i in range(len(CLASS_NAMES)):
            mask = true_labels == i
            if mask.sum() == 0:
                class_cov.append(0.0); continue
            class_cov.append(float(np.mean(
                [true_labels[j] in pred_sets[j] for j in np.where(mask)[0]])))
        ax.barh(range(len(CLASS_NAMES)), class_cov, color=PALETTE, alpha=0.85)
        ax.axvline(1-self.alpha, color="#f0f6fc", ls="--", lw=1.5,
                   label=f"Cible {1-self.alpha:.2f}")
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_yticklabels([c[:12] for c in CLASS_NAMES], fontsize=8)
        ax.set_xlabel("Couverture"); ax.set_title("Couverture par classe")
        ax.set_xlim(0, 1.05); ax.legend()

        ax = axes[2]
        ax.scatter(probs.max(axis=1),
                   sizes + np.random.uniform(-0.15, 0.15, len(sizes)),
                   c=true_labels, cmap="turbo", s=8, alpha=0.4, rasterized=True)
        ax.set_xlabel("Confiance max (softmax)")
        ax.set_ylabel("Taille de l'ensemble")
        ax.set_title("Confiance vs Taille")

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig


# ==============================================================================
# VISUALISATION INCERTITUDE
# ==============================================================================

def plot_uncertainty_analysis(mc_results: dict,
                               save_path: Optional[str] = None) -> plt.Figure:
    """Figure complete d'analyse MC Dropout. 4 panels."""
    plt.rcParams.update(DARK)
    targets = mc_results["targets"]
    preds   = mc_results["preds"]
    entropy = mc_results["entropy"]
    mi      = mc_results["mutual_info"]
    correct = (preds == targets)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Analyse d'incertitude — MC Dropout",
                 fontsize=14, fontweight="bold")

    # A — Entropie par classe
    ax = axes[0, 0]
    vp = ax.violinplot([entropy[targets == i] for i in range(len(CLASS_NAMES))],
                       showmedians=True, showextrema=False)
    for body, col in zip(vp["bodies"], PALETTE):
        body.set_facecolor(col); body.set_alpha(0.7)
    vp["cmedians"].set_color("#f0f6fc"); vp["cmedians"].set_linewidth(2)
    ax.set_xticks(range(1, len(CLASS_NAMES)+1))
    ax.set_xticklabels([c[:8] for c in CLASS_NAMES], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Entropie H[y|x]"); ax.set_title("Incertitude totale par classe")

    # B — Info mutuelle par classe
    ax = axes[0, 1]
    vp2 = ax.violinplot([mi[targets == i] for i in range(len(CLASS_NAMES))],
                        showmedians=True, showextrema=False)
    for body, col in zip(vp2["bodies"], PALETTE):
        body.set_facecolor(col); body.set_alpha(0.7)
    vp2["cmedians"].set_color("#f0f6fc"); vp2["cmedians"].set_linewidth(2)
    ax.set_xticks(range(1, len(CLASS_NAMES)+1))
    ax.set_xticklabels([c[:8] for c in CLASS_NAMES], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Info mutuelle MI[y,w|x]"); ax.set_title("Incertitude epistemique par classe")

    # C — Entropie correct vs incorrect
    ax = axes[1, 0]
    ax.hist(entropy[correct],  bins=40, color="#1ddfa3", alpha=0.6, label="Correct",   density=True)
    ax.hist(entropy[~correct], bins=40, color="#f83e4b", alpha=0.6, label="Incorrect", density=True)
    ax.set_xlabel("Entropie predictive"); ax.set_ylabel("Densite")
    ax.set_title("Entropie : bien vs mal classees"); ax.legend()

    # D — Mediane entropie vs taux erreur par classe
    # FIX v2 : np.median() au lieu de .median() (attribut pandas inexistant sur ndarray)
    ax = axes[1, 1]
    med_entropy = [float(np.median(entropy[targets == i]))
                   if (targets == i).sum() > 0 else 0.0
                   for i in range(len(CLASS_NAMES))]
    error_rate  = [1 - float((preds[targets == i] == targets[targets == i]).mean())
                   if (targets == i).sum() > 0 else 0.0
                   for i in range(len(CLASS_NAMES))]
    ax.scatter(error_rate, med_entropy, c=PALETTE, s=120, zorder=5)
    for i, cls in enumerate(CLASS_NAMES):
        ax.annotate(cls[:10], (error_rate[i], med_entropy[i]),
                    fontsize=7, xytext=(3,3), textcoords="offset points")
    ax.set_xlabel("Taux d'erreur par classe"); ax.set_ylabel("Entropie mediane")
    ax.set_title("Incertitude vs Difficulte par classe")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def find_most_uncertain(mc_results: dict, n: int = 20) -> dict:
    """Retourne les N galaxies les plus incertaines (par entropie)."""
    order = np.argsort(-mc_results["entropy"])[:n]
    return {
        "indices":     order,
        "entropy":     mc_results["entropy"][order],
        "mutual_info": mc_results["mutual_info"][order],
        "pred":        mc_results["preds"][order],
        "true_label":  mc_results["targets"][order],
        "correct":     mc_results["preds"][order] == mc_results["targets"][order],
    }