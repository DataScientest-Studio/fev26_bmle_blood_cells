# -*- coding: utf-8 -*-
"""
Confusion matrices + classification reports pour les meilleurs modèles.
  - Best ConvNeXt-Tiny  : A2_ep20_b32
  - Best DenseNet-121   : A2_ep20_b32
"""
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
import timm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
from PIL import Image
from pathlib import Path
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import warnings
from dotenv import load_dotenv
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')
ROOT = Path(__file__).parents[2]
load_dotenv(ROOT / ".env")

matplotlib.use('Agg')


# ── Configuration ────────────────────────────────────────────────────────
if not os.getenv("ACEVEDO_DATA_DIR"):
    raise EnvironmentError(
        "ACEVEDO_DATA_DIR doit être défini dans ton .env local (chemin personnel)."
    )
DATA_DIR = Path(os.environ["ACEVEDO_DATA_DIR"])
OUTPUT_DIR = ROOT / "reports" / "Sara_DL_convnext_densenet_hyperparam"
SUBSAMPLE_N = 2000
TEST_SIZE = 0.15
VAL_SIZE = 0.15
SEED = 42
DEVICE = ("cuda" if torch.cuda.is_available() else
          "mps" if torch.backends.mps.is_available() else "cpu")
IGNORED = {"output"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BEST_MODELS = [
    {"exp": "A2_ep20_b32", "model_name": "convnext_tiny", "batch": 32, "activation": "default"},
    {"exp": "A2_ep20_b32", "model_name": "densenet121", "batch": 32, "activation": "default"},
]

# ── Seed ─────────────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

# ── Dataset ───────────────────────────────────────────────────────────────
CLASS_NAMES = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name not in IGNORED])
NUM_CLASSES = len(CLASS_NAMES)
print(f"Classes ({NUM_CLASSES}) : {CLASS_NAMES}")

paths, labels = [], []
for label, cls in enumerate(CLASS_NAMES):
    for p in (DATA_DIR / cls).iterdir():
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]:
            paths.append(p)
            labels.append(label)

rng = np.random.default_rng(SEED)
n_per = SUBSAMPLE_N // NUM_CLASSES
sel = []
for c in range(NUM_CLASSES):
    idx = [i for i, l in enumerate(labels) if l == c]
    sel.extend(rng.choice(idx, size=min(n_per, len(idx)), replace=False).tolist())
sel.sort()
paths = [paths[i] for i in sel]
labels = [labels[i] for i in sel]

idx = np.arange(len(paths))
idx_tv, idx_test = train_test_split(idx, test_size=TEST_SIZE, stratify=labels, random_state=SEED)
labels_tv = [labels[i] for i in idx_tv]
idx_train, idx_val = train_test_split(idx_tv, test_size=VAL_SIZE / (1 - TEST_SIZE),
                                      stratify=labels_tv, random_state=SEED)
print(f"Test set : {len(idx_test)} images")

tf = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class DS(Dataset):
    def __init__(self, paths, labels, indices, tf):
        self.p = [paths[i] for i in indices]
        self.labels = [labels[i] for i in indices]
        self.tf = tf

    def __len__(self): return len(self.p)

    def __getitem__(self, i):
        return self.tf(Image.open(self.p[i]).convert("RGB")), self.labels[i]

# ── Modèle ────────────────────────────────────────────────────────────────


class _PoolFlatten(nn.Module):
    def forward(self, x):
        return x.mean([-2, -1]) if x.dim() == 4 else x


def build_model(model_name, activation):
    if activation == "default":
        return timm.create_model(model_name, pretrained=False, num_classes=NUM_CLASSES)
    model = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool='')
    act_fn = {"gelu": nn.GELU(), "silu": nn.SiLU(), "mish": nn.Mish()}[activation]
    head = nn.Sequential(
        _PoolFlatten(),
        nn.Linear(model.num_features, 256), act_fn, nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES),
    )
    if hasattr(model, 'head'):
        model.head = head
    elif hasattr(model, 'classifier'):
        model.classifier = head
    return model

# ── Inférence ─────────────────────────────────────────────────────────────


@torch.no_grad()
def predict(model, loader):
    model.eval()
    yt, yp, ys = [], [], []
    for imgs, lbls in tqdm(loader, leave=False):
        imgs = imgs.to(DEVICE)
        probs = torch.softmax(model(imgs).float(), dim=1).cpu().numpy()
        yt.extend(lbls.numpy())
        yp.extend(probs.argmax(1))
        ys.extend(probs)
    return np.array(yt), np.array(yp), np.array(ys)


# ── Boucle principale ──────────────────────────────────────────────────────
fig_cm, axes_cm = plt.subplots(1, 2, figsize=(18, 7))

for ax, cfg in zip(axes_cm, BEST_MODELS):
    safe_key = f"{cfg['exp']}__{cfg['model_name']}"
    pth = OUTPUT_DIR / f"best_{safe_key}.pth"

    model = build_model(cfg["model_name"], cfg["activation"])
    model.load_state_dict(torch.load(pth, map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE)

    loader = DataLoader(DS(paths, labels, idx_test, tf),
                        batch_size=cfg["batch"], shuffle=False, num_workers=0)
    y_true, y_pred, y_scores = predict(model, loader)
    del model

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    auc = roc_auc_score(y_true.astype(int), y_scores, multi_class="ovr",
                        average="macro", labels=list(range(NUM_CLASSES)))

    # ── Confusion matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    labels_short = [c[:4] for c in CLASS_NAMES]
    sns.heatmap(
        cm_norm, annot=cm, fmt="d", ax=ax,
        cmap="Blues", vmin=0, vmax=1,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 10}
    )
    ax.set_title(
        f"{cfg['model_name']}  —  {cfg['exp']}\n"
        f"acc={acc:.4f}  |  F1={f1:.4f}  |  AUC={auc:.4f}",
        fontsize=11, fontweight="bold", pad=12
    )
    ax.set_xlabel("Prediction", fontsize=10)
    ax.set_ylabel("Vraie classe", fontsize=10)
    ax.tick_params(axis='x', rotation=35, labelsize=9)
    ax.tick_params(axis='y', rotation=0, labelsize=9)

    # ── Classification report — texte + PNG ──────────────────────────────
    report_str = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
    header = (f"{'='*65}\n"
              f"  {cfg['model_name'].upper()}  |  {cfg['exp']}\n"
              f"  acc={acc:.4f}  f1_macro={f1:.4f}  auc={auc:.4f}\n"
              f"{'='*65}\n")
    print(header + report_str)

    # Sauvegarde .txt
    txt_path = OUTPUT_DIR / f"classif_report_{safe_key}.txt"
    txt_path.write_text(header + report_str, encoding="utf-8")

    # Sauvegarde .png (tableau matplotlib)
    import pandas as pd
    from sklearn.metrics import classification_report as cr
    report_dict = cr(y_true, y_pred, target_names=CLASS_NAMES,
                     digits=4, zero_division=0, output_dict=True)
    df_report = pd.DataFrame(report_dict).T.round(4)
    df_report = df_report.drop(index=["accuracy"], errors="ignore")

    fig_r, ax_r = plt.subplots(figsize=(10, 4))
    ax_r.axis("off")
    tbl = ax_r.table(
        cellText=df_report.values,
        rowLabels=df_report.index,
        colLabels=df_report.columns,
        cellLoc="center", loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.6)
    # Couleur header et index
    for (row, col), cell in tbl.get_celld().items():
        if row == 0 or col == -1:
            cell.set_facecolor("#4C72B0")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#EEF2FF")
    ax_r.set_title(
        f"Classification Report — {cfg['model_name']} | {cfg['exp']}\n"
        f"acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}",
        fontsize=11, fontweight="bold", pad=12
    )
    fig_r.tight_layout()
    png_path = OUTPUT_DIR / f"classif_report_{safe_key}.png"
    fig_r.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig_r)
    print(f"  -> Rapport sauvegarde : {txt_path.name}  +  {png_path.name}")

plt.suptitle(
    "Matrices de confusion — meilleurs modeles (A2_ep20_b32)\n"
    "Valeurs : nombre de predictions | Couleur : taux par classe reelle",
    fontsize=13,
    fontweight="bold",
    y=1.01)
plt.tight_layout()
out_path = OUTPUT_DIR / "confusion_best_models.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nMatrices sauvegardees : {out_path.name}")
