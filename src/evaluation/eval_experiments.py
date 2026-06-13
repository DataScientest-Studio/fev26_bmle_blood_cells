"""
Évaluation autonome de tous les modèles entraînés.
Calcule accuracy, F1-macro et AUC-ROC sur le test set pour chaque expérience.
"""
import os, json, random, warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

# ── Configuration (doit correspondre exactement au notebook) ─────────────
DATA_DIR    = Path(r"C:\Users\julie\Downloads\datasets\Acevedo\Acevedo")
OUTPUT_DIR  = Path(r"C:\Users\julie\MAR26-BDS-BLOODCELLS-1\reports\Sara_DL_convnext_densenet_hyperparam")
SUBSAMPLE_N = 2000
TEST_SIZE   = 0.15
VAL_SIZE    = 0.15
SEED        = 42
DEVICE      = ("cuda" if torch.cuda.is_available() else
               "mps"  if torch.backends.mps.is_available() else "cpu")
IGNORED     = {"output"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

EXPERIMENTS = [
    {"name": "A1_baseline",   "phase1_epochs": 5,  "phase2_epochs": 10, "batch_size": 32,  "lr1": 1e-3, "lr2": 1e-4,  "activation": "default"},
    {"name": "A2_ep20_b32",   "phase1_epochs": 5,  "phase2_epochs": 20, "batch_size": 32,  "lr1": 1e-3, "lr2": 1e-4,  "activation": "default"},
    {"name": "A3_ep25_b64",   "phase1_epochs": 5,  "phase2_epochs": 25, "batch_size": 64,  "lr1": 1e-3, "lr2": 1e-4,  "activation": "default"},
    {"name": "A4_ep20_b64",   "phase1_epochs": 5,  "phase2_epochs": 20, "batch_size": 64,  "lr1": 1e-3, "lr2": 1e-4,  "activation": "default"},
    {"name": "B1_lr_prudent", "phase1_epochs": 5,  "phase2_epochs": 20, "batch_size": 32,  "lr1": 1e-3, "lr2": 5e-5,  "activation": "default"},
    {"name": "B2_lr_mid",     "phase1_epochs": 5,  "phase2_epochs": 20, "batch_size": 32,  "lr1": 5e-4, "lr2": 1e-4,  "activation": "default"},
    {"name": "B3_lr_agg",     "phase1_epochs": 5,  "phase2_epochs": 20, "batch_size": 32,  "lr1": 1e-3, "lr2": 2e-4,  "activation": "default"},
    {"name": "C1_gelu",       "phase1_epochs": 5,  "phase2_epochs": 20, "batch_size": 32,  "lr1": 1e-3, "lr2": 1e-4,  "activation": "gelu"},
    {"name": "C2_silu",       "phase1_epochs": 5,  "phase2_epochs": 20, "batch_size": 32,  "lr1": 1e-3, "lr2": 1e-4,  "activation": "silu"},
    {"name": "C3_mish",       "phase1_epochs": 5,  "phase2_epochs": 20, "batch_size": 32,  "lr1": 1e-3, "lr2": 1e-4,  "activation": "mish"},
]

# ── Reproductibilité ─────────────────────────────────────────────────────
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ── Chargement du dataset ─────────────────────────────────────────────────
CLASS_NAMES = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name not in IGNORED])
NUM_CLASSES = len(CLASS_NAMES)
print(f"Classes ({NUM_CLASSES}) : {CLASS_NAMES}")

paths, labels = [], []
for label, cls in enumerate(CLASS_NAMES):
    for ext in [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]:
        paths.extend(list((DATA_DIR / cls).glob(f"*{ext}")))
        labels.extend([label] * len(list((DATA_DIR / cls).glob(f"*{ext}"))))

# Sous-échantillonnage stratifié identique au notebook
rng = np.random.default_rng(SEED)
n_per = SUBSAMPLE_N // NUM_CLASSES
sel = []
for c in range(NUM_CLASSES):
    idx = [i for i, l in enumerate(labels) if l == c]
    sel.extend(rng.choice(idx, size=min(n_per, len(idx)), replace=False).tolist())
sel.sort()
paths  = [paths[i]  for i in sel]
labels = [labels[i] for i in sel]

# Split identique au notebook
idx = np.arange(len(paths))
idx_tv, idx_test = train_test_split(idx, test_size=TEST_SIZE, stratify=labels, random_state=SEED)
labels_tv = [labels[i] for i in idx_tv]
val_ratio = VAL_SIZE / (1 - TEST_SIZE)
idx_train, idx_val = train_test_split(idx_tv, test_size=val_ratio, stratify=labels_tv, random_state=SEED)
print(f"Test set : {len(idx_test)} images")

# ── Dataset & DataLoader ──────────────────────────────────────────────────
class DS(Dataset):
    def __init__(self, paths, labels, indices, tf):
        self.p = [paths[i] for i in indices]
        self.l = [labels[i] for i in indices]
        self.tf = tf
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        img = Image.open(self.p[i]).convert("RGB")
        return self.tf(img), self.l[i]

tf = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ── Construction du modèle ────────────────────────────────────────────────
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
        nn.Linear(model.num_features, 256),
        act_fn,
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES),
    )
    if hasattr(model, 'head'): model.head = head
    elif hasattr(model, 'classifier'): model.classifier = head
    return model

# ── Inférence ─────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model, loader):
    model.eval()
    yt, yp, ys = [], [], []
    for imgs, lbls in tqdm(loader, leave=False):
        imgs = imgs.to(DEVICE)
        probs = torch.softmax(model(imgs).float(), dim=1).cpu().numpy()
        yt.extend(lbls.numpy()); yp.extend(probs.argmax(1)); ys.extend(probs)
    return np.array(yt), np.array(yp), np.array(ys)

# ── Boucle d'évaluation ───────────────────────────────────────────────────
rows = []
for exp in EXPERIMENTS:
    for model_name in ["convnext_tiny", "densenet121"]:
        safe_key = f"{exp['name']}__{model_name}"
        pth  = OUTPUT_DIR / f"best_{safe_key}.pth"
        hist = OUTPUT_DIR / f"history_{safe_key}.json"
        if not (pth.exists() and hist.exists()):
            continue

        with open(hist) as f:
            h = json.load(f)

        model = build_model(model_name, exp["activation"])
        model.load_state_dict(torch.load(pth, map_location=DEVICE, weights_only=True))
        model = model.to(DEVICE)

        loader = DataLoader(DS(paths, labels, idx_test, tf), batch_size=exp["batch_size"],
                            shuffle=False, num_workers=0)
        y_true, y_pred, y_scores = predict(model, loader)
        del model; torch.cuda.empty_cache() if DEVICE == "cuda" else None

        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)
        try:
            auc = roc_auc_score(y_true.astype(int), y_scores, multi_class="ovr",
                                average="macro", labels=list(range(NUM_CLASSES)))
        except ValueError:
            auc = float("nan")

        rows.append({
            "Expérience"  : exp["name"],
            "Modèle"      : model_name,
            "Batch"       : exp["batch_size"],
            "lr1"         : exp["lr1"],
            "lr2"         : exp["lr2"],
            "Activation"  : exp["activation"],
            "P2 epochs max": exp["phase2_epochs"],
            "Epochs run"  : len(h["val_acc"]),
            "Best val_acc": round(max(h["val_acc"]), 4),
            "Test acc"    : round(acc, 4),
            "F1 macro"    : round(f1,  4),
            "AUC-ROC"     : round(auc, 4) if not np.isnan(auc) else "NaN",
        })
        print(f"  OK {exp['name']:18} {model_name:15} acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}")

df = pd.DataFrame(rows)
df = df.sort_values("Test acc", ascending=False).reset_index(drop=True)
df.index += 1

csv_path = OUTPUT_DIR / "results_comparatif.csv"
df.to_csv(csv_path, index=False)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.4f}".format)
print("\n" + "="*120)
print("  TABLEAU COMPARATIF — accuracy / F1-macro / AUC-ROC")
print("="*120)
print(df.to_string())
print(f"\nSauvegardé : {csv_path}")
