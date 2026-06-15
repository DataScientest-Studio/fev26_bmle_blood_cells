"""
Entraînement DenseNet-121 + ConvNeXt-Tiny sur le dataset PBC complet.
Sauvegarde les .pth dans reports/Fred_DL_pipeline_report_full/
"""

import json
import os
import random
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── Config ────────────────────────────────────────────────────────────[...]
DATA_DIR   = Path("/Users/fredericdelabot/Documents/DataScientest/Projet CHU Lyon"
                  "/mendeley/1/PBC_dataset_normal_DIB")
OUTPUT_DIR = Path(__file__).parent.parent.parent / "reports/Fred_DL_pipeline_report_full"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_CLASSES = {
    "basophil", "eosinophil", "erythroblast", "ig",
    "lymphocyte", "monocyte", "neutrophil", "platelet",
}
CLASS_NAMES = sorted([d.name for d in DATA_DIR.iterdir()
                      if d.is_dir() and d.name in EXPECTED_CLASSES])
NUM_CLASSES = len(CLASS_NAMES)
SEED        = 42
DEVICE      = ("cuda" if torch.cuda.is_available() else
               "mps"  if torch.backends.mps.is_available() else "cpu")

MODELS_TO_TRAIN = {
    "DenseNet-121":  {"timm_name": "densenet121",   "input_size": 224},
    "ConvNeXt-Tiny": {"timm_name": "convnext_tiny", "input_size": 224},
}

CFG = {
    "batch_size"  : 32,
    "lr_head"     : 1e-3,
    "lr_full"     : 1e-4,
    "weight_decay": 1e-4,
    "epochs_head" : 5,
    "epochs_full" : 10,
    "patience"    : 3,
    "num_workers" : 0,
    "test_size"   : 0.15,
    "val_size"    : 0.15,
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


set_seed(SEED)
print(f"Device     : {DEVICE}")
print(f"Classes    : {CLASS_NAMES}")
print(f"Output dir : {OUTPUT_DIR}")


# ── Dataset ───────────────────────────────────────────────────────────��[...]

def load_paths():
    paths, labels = [], []
    exts = {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}
    for label, cls in enumerate(CLASS_NAMES):
        for p in (DATA_DIR / cls).iterdir():
            if p.suffix.lower() in exts:
                paths.append(p); labels.append(label)
    return paths, labels


class CellDataset(Dataset):
    def __init__(self, paths, labels, tf):
        self.paths = paths; self.labels = labels; self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tf(img), self.labels[idx]


def get_transforms(input_size):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ── Split ────────────────────────────────────────────────────────────[...]
all_paths, all_labels = load_paths()
print(f"Total images : {len(all_paths)}")

idx = list(range(len(all_paths)))
idx_trainval, idx_test = train_test_split(idx, test_size=CFG["test_size"],
                                          stratify=all_labels, random_state=SEED)
trainval_lbls = [all_labels[i] for i in idx_trainval]
idx_train, idx_val = train_test_split(idx_trainval,
                                      test_size=CFG["val_size"] / (1 - CFG["test_size"]),
                                      stratify=trainval_lbls, random_state=SEED)
print(f"Train: {len(idx_train)}  Val: {len(idx_val)}  Test: {len(idx_test)}")


# ── Training ───────────────────────────────────────────────────────────[...]

def train_one_epoch(model, loader, opt, crit):
    model.train()
    total_loss = correct = total = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        out  = model(imgs)
        loss = crit(out, lbls)
        loss.backward(); opt.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == lbls).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, crit):
    model.eval()
    total_loss = correct = total = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        out  = model(imgs)
        loss = crit(out, lbls)
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == lbls).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


def train_model(model_key: str):
    cfg       = MODELS_TO_TRAIN[model_key]
    safe_key  = model_key.replace("-", "_")
    save_path = OUTPUT_DIR / f"best_{safe_key}.pth"
    hist_path = OUTPUT_DIR / f"history_{safe_key}.json"

    if save_path.exists() and hist_path.exists():
        print(f"\n{model_key} déjà entraîné → {save_path.name}")
        return

    print(f"\n{'='*55}")
    print(f"  Entraînement : {model_key}")
    print(f"{'='*55}")

    tf = get_transforms(cfg["input_size"])
    train_paths = [all_paths[i] for i in idx_train]
    val_paths   = [all_paths[i] for i in idx_val]
    train_lbls  = [all_labels[i] for i in idx_train]
    val_lbls    = [all_labels[i] for i in idx_val]

    train_dl = DataLoader(CellDataset(train_paths, train_lbls, tf),
                          batch_size=CFG["batch_size"], shuffle=True,
                          num_workers=CFG["num_workers"])
    val_dl   = DataLoader(CellDataset(val_paths, val_lbls, tf),
                          batch_size=CFG["batch_size"], shuffle=False,
                          num_workers=CFG["num_workers"])

    model = timm.create_model(cfg["timm_name"], pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    crit  = nn.CrossEntropyLoss()

    # ── Phase 1 : backbone gelé ────────────────────────────────────────────────
    HEAD_NAMES = {"classifier", "head", "fc"}
    for name, p in model.named_parameters():
        p.requires_grad = name.split(".")[0] in HEAD_NAMES

    opt  = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=CFG["lr_head"], weight_decay=CFG["weight_decay"])
    sch  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs_head"])

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    print(f"\n  Phase 1 — backbone gelé ({CFG['epochs_head']} epochs)")
    for epoch in range(CFG["epochs_head"]):
        t0 = time.time()
        tl, ta = train_one_epoch(model, train_dl, opt, crit)
        vl, va = evaluate(model, val_dl, crit)
        sch.step()
        history["train_loss"].append(tl); history["val_loss"].append(vl)
        history["train_acc"].append(ta);  history["val_acc"].append(va)
        if va > best_val_acc:
            best_val_acc = va
            torch.save(model.state_dict(), save_path)
        print(f"    Ep {epoch+1:02d}  train={ta:.3f}  val={va:.3f}  ({time.time()-t0:.0f}s)")

    # ── Phase 2 : fine-tuning complet ────────────────────────────────────────
    for p in model.parameters():
        p.requires_grad = True

    opt = optim.AdamW(model.parameters(), lr=CFG["lr_full"], weight_decay=CFG["weight_decay"])
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs_full"])

    patience_cnt = 0
    print(f"\n  Phase 2 — fine-tuning ({CFG['epochs_full']} epochs, patience={CFG['patience']})")
    for epoch in range(CFG["epochs_full"]):
        t0 = time.time()
        tl, ta = train_one_epoch(model, train_dl, opt, crit)
        vl, va = evaluate(model, val_dl, crit)
        sch.step()
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(ta)
        history["val_acc"].append(va)
        improved = va > best_val_acc
        if improved:
            best_val_acc = va
            torch.save(model.state_dict(), save_path)
            patience_cnt = 0
        else:
            patience_cnt += 1
        tag = " ✓" if improved else f" (patience {patience_cnt}/{CFG['patience']})"
        print(f"    Ep {epoch+1:02d}  train={ta:.3f}  val={va:.3f}  ({time.time()-t0:.0f}s){tag}")
        if patience_cnt >= CFG["patience"]:
            print("    Early stopping.")
            break

    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  Meilleur val_acc : {best_val_acc:.4f}")
    print(f"  Sauvegardé      : {save_path}")

    # ── Eval test ─────────────────────────────────────────────────────────�[...]
    test_paths = [all_paths[i] for i in idx_test]
    test_lbls = [all_labels[i] for i in idx_test]
    test_dl = DataLoader(CellDataset(test_paths, test_lbls, tf),
                         batch_size=CFG["batch_size"], shuffle=False,
                         num_workers=CFG["num_workers"])
    model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
    _, test_acc = evaluate(model, test_dl, crit)
    print(f"  Test accuracy   : {test_acc:.4f}")


# ── Main ────────────────────────────────────────────────────────────�[...]
if __name__ == "__main__":
    for key in MODELS_TO_TRAIN:
        train_model(key)

    print("\n\nEntraînement terminé.")
    print(f"Modèles disponibles dans : {OUTPUT_DIR}")
    for f in sorted(OUTPUT_DIR.glob("best_*.pth")):
        print(f"  {f.name}")

    # Sauvegarde class_names pour l'inférence
    with open(OUTPUT_DIR / "class_names.json", "w") as f:
        json.dump(CLASS_NAMES, f)
    print("  class_names.json")
