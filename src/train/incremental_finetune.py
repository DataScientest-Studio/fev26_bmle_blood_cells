#!/usr/bin/env python
# coding: utf-8
"""
Fine-tuning incrémental — continue l'entraînement du modèle @production
actuel sur un nouveau lot de données (ex : un lot TIFF "autre instrument",
200 images), au lieu de repartir d'un entraînement from scratch.

Buffer de replay (--replay-per-class, 25/classe par défaut) : mélange un
échantillon Mendeley (distribution d'origine) avec le nouveau lot pendant
le fine-tuning. Sans lui, un lot TIFF peut faire s'effondrer une classe
absente ou sous-représentée (ex : recall_monocyte 0.99 -> 0.42 observé en
pratique sur le premier lot testé, avant l'ajout du replay).

Le garde-fou de promotion (macro_f1 + recall par classe sur les 8 classes,
pas seulement 2) reste le filet de sécurité final : si une génération
dégrade quand même une classe, elle reste @challenger, @production n'est
jamais touché.

Évaluation de référence sur test_cropped (Mendeley JPG, jamais vu pendant
l'entraînement initial ni le fine-tuning) : permet de détecter un oubli sur
les classes absentes du nouveau lot (ex: platelet, absente des archives
TCIA TIFF).

Usage :
    python -m src.train.incremental_finetune \
        --batch-dir data/tiff_batches/batch_001 --generation v2
"""

import argparse
import json
import os
import random
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from mlflow.tracking import MlflowClient
from PIL import Image
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_recall_fscore_support, recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.monitoring.resource_monitor import ResourceMonitor
from src.monitoring.supabase_logger import log_class_metrics_and_confusion, log_training_run

warnings.filterwarnings("ignore")

MLFLOW_MODEL_NAME = "blood-cell-densenet121"
RECALL_TOLERANCE = 0.02
INPUT_SIZE = 224
CLASS_NAMES = [
    "basophil", "eosinophil", "erythroblast", "ig",
    "lymphocyte", "monocyte", "neutrophil", "platelet",
]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class BloodCellDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


def get_transform():
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def sample_replay_paths(replay_dir: Path, per_class: int, seed: int, class_names=CLASS_NAMES):
    """Échantillonne `per_class` images par classe depuis le dataset d'origine
    (Mendeley), pour les mélanger avec le nouveau lot pendant le fine-tuning —
    évite que le modèle n'écrase la distribution d'origine avec la nouvelle."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    rng = random.Random(seed)
    paths, labels = [], []
    for label, cls in enumerate(class_names):
        cls_dir = replay_dir / cls
        if not cls_dir.is_dir():
            continue
        imgs = [p for p in cls_dir.iterdir() if p.suffix.lower() in exts]
        chosen = rng.sample(imgs, min(per_class, len(imgs)))
        paths.extend(chosen)
        labels.extend([label] * len(chosen))
    return paths, labels


def load_dataset_paths(data_dir: Path, class_names=CLASS_NAMES):
    exts = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}
    paths, labels = [], []
    for label, cls in enumerate(class_names):
        cls_dir = data_dir / cls
        if not cls_dir.is_dir():
            continue
        imgs = [p for p in cls_dir.iterdir() if p.suffix.lower() in exts]
        paths.extend(imgs)
        labels.extend([label] * len(imgs))
    return paths, labels


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, leave=False, desc="  Train"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for imgs, labels in tqdm(loader, leave=False, desc="  Eval réf."):
        imgs = imgs.to(device)
        preds = model(imgs).argmax(1).cpu().numpy()
        y_true.extend(labels.numpy())
        y_pred.extend(preds)
    return np.array(y_true), np.array(y_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", required=True, help="Dossier du lot (ex: data/tiff_batches/batch_001)")
    parser.add_argument("--generation", required=True, help="Tag de génération (ex: v2)")
    parser.add_argument(
        "--reference-dir", default=None,
        help="Dossier d'évaluation de référence (défaut: data/crossval_source/test_cropped)",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5, help="LR bas — fine-tuning sur un petit lot")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--replay-dir", default=None,
        help="Dossier source pour le buffer de replay (défaut: data/crossval_source/train_final). "
             "Mélangé avec le lot pour éviter l'oubli catastrophique sur la distribution d'origine.",
    )
    parser.add_argument(
        "--replay-per-class", type=int, default=25,
        help="Nombre d'images Mendeley par classe à mélanger avec le lot (0 pour désactiver)",
    )
    args = parser.parse_args()

    _cwd = Path().resolve()
    PROJECT_ROOT = _cwd
    for _p in [_cwd] + list(_cwd.parents):
        if (_p / "data").is_dir() and (_p / "src").is_dir():
            PROJECT_ROOT = _p
            break

    batch_dir = Path(args.batch_dir)
    if not batch_dir.is_absolute():
        batch_dir = PROJECT_ROOT / batch_dir
    reference_dir = Path(args.reference_dir) if args.reference_dir else PROJECT_ROOT / "data" / "crossval_source" / "test_cropped"
    replay_dir = Path(args.replay_dir) if args.replay_dir else PROJECT_ROOT / "data" / "crossval_source" / "train_final"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"PyTorch  : {torch.__version__}")
    print(f"Device   : {device}")
    print(f"Lot      : {batch_dir}")
    print(f"Référence: {reference_dir}")

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    mlflow.set_experiment("blood_cell_incremental_finetune")
    client = MlflowClient()

    # ── Chargement du champion actuel ────────────────────────────────────────
    prod_mv = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, "production")
    prod_run = client.get_run(prod_mv.run_id)
    prod_metrics = prod_run.data.metrics
    print(f"Champion actuel : v{prod_mv.version} (generation={prod_mv.tags.get('generation')}) "
          f"macro_f1={prod_metrics.get('macro_f1', 0):.4f}")

    model = mlflow.pytorch.load_model(f"models:/{MLFLOW_MODEL_NAME}@production", map_location="cpu")
    model = model.to(device)

    # ── Données du lot (train/val internes pour l'early stopping) ───────────
    paths, labels = load_dataset_paths(batch_dir)
    print(f"Lot : {len(paths)} images, {len(set(labels))} classes présentes")
    if len(paths) < 10:
        raise ValueError(f"Lot trop petit ({len(paths)} images) — abandon.")

    # ── Buffer de replay (Mendeley, distribution d'origine) ──────────────────
    if args.replay_per_class > 0:
        replay_paths, replay_labels = sample_replay_paths(
            replay_dir, args.replay_per_class, args.seed,
        )
        print(f"Replay : {len(replay_paths)} images Mendeley ajoutées "
              f"({args.replay_per_class}/classe depuis {replay_dir.name})")
        paths = paths + replay_paths
        labels = labels + replay_labels

    try:
        idx_train, idx_val = train_test_split(
            range(len(paths)), test_size=0.15, stratify=labels, random_state=args.seed,
        )
    except ValueError:
        # Une classe du lot n'a qu'1 membre — la stratification est impossible
        print("  [info] Split non stratifié (au moins une classe a < 2 images dans ce lot)")
        idx_train, idx_val = train_test_split(
            range(len(paths)), test_size=0.15, random_state=args.seed,
        )
    tf = get_transform()

    def make_loader(idxs, shuffle):
        ds = BloodCellDataset([paths[i] for i in idxs], [labels[i] for i in idxs], tf)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=2)

    train_loader = make_loader(idx_train, True)
    val_loader = make_loader(idx_val, False)

    # ── Données de référence (test_cropped, 8 classes) ───────────────────────
    ref_paths, ref_labels = load_dataset_paths(reference_dir)
    print(f"Référence : {len(ref_paths)} images, {len(set(ref_labels))} classes")
    ref_loader = DataLoader(
        BloodCellDataset(ref_paths, ref_labels, tf), batch_size=32, shuffle=False, num_workers=2,
    )

    # ── Fine-tuning ───────────────────────────────────────────────────────────
    started_at = datetime.now(timezone.utc)
    monitor = ResourceMonitor().start()
    t_start = time.time()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    patience_counter = 0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    y_true = y_pred = None
    status = "failed"

    with mlflow.start_run(run_name=f"finetune_{args.generation}_{batch_dir.name}") as run:
        mlflow.log_params({
            "generation": args.generation, "batch": batch_dir.name,
            "base_version": prod_mv.version, "lr": args.lr, "epochs": args.epochs,
            "batch_size": args.batch_size, "n_total_images": len(paths),
            "replay_per_class": args.replay_per_class,
        })
        mlflow.set_tags({"generation": args.generation, "source": f"tiff_batch_{batch_dir.name}"})

        try:
            for epoch in range(args.epochs):
                t0 = time.time()
                tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
                vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
                scheduler.step()
                mlflow.log_metrics({"train_loss": tr_loss, "val_loss": vl_loss,
                                     "train_acc": tr_acc, "val_acc": vl_acc}, step=epoch)
                improved = vl_acc > best_val_acc
                if improved:
                    best_val_acc = vl_acc
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                tag = " [ok]" if improved else f" (patience {patience_counter}/{args.patience})"
                print(f"  E{epoch+1:02d}/{args.epochs} | loss {tr_loss:.4f}->{vl_loss:.4f} | "
                      f"acc {tr_acc:.3f}->{vl_acc:.3f} | {time.time()-t0:.1f}s{tag}")
                if patience_counter >= args.patience:
                    print("  Early stopping (sur le lot).")
                    break

            model.load_state_dict(best_state)
            model.eval()

            # ── Évaluation de référence (détection oubli catastrophique) ────
            y_true, y_pred = predict_all(model, ref_loader, device)
            macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            acc = accuracy_score(y_true, y_pred)
            recalls = recall_score(y_true, y_pred, average=None, labels=list(range(len(CLASS_NAMES))), zero_division=0)
            recall_per_class = dict(zip(CLASS_NAMES, recalls.tolist()))

            print(f"\nRéférence — accuracy={acc:.4f}  macro_f1={macro_f1:.4f}")
            for cls, r in recall_per_class.items():
                print(f"  recall_{cls:<15} {r:.4f}")

            mlflow.log_metrics({
                "accuracy": acc, "macro_f1": macro_f1, "best_batch_val_acc": best_val_acc,
                **{f"recall_{c}": r for c, r in recall_per_class.items()},
            })

            input_size = INPUT_SIZE
            model_info = mlflow.pytorch.log_model(
                model, name="densenet121",
                input_example=torch.zeros(1, 3, input_size, input_size).numpy(),
                serialization_format="pickle",
            )
            mv = mlflow.register_model(model_uri=model_info.model_uri, name=MLFLOW_MODEL_NAME)
            client.set_model_version_tag(MLFLOW_MODEL_NAME, mv.version, "generation", args.generation)
            client.set_model_version_tag(MLFLOW_MODEL_NAME, mv.version, "source", f"tiff_batch_{batch_dir.name}")

            # ── Garde-fou de promotion (8 classes) ──────────────────────────
            prod_f1 = float(prod_metrics.get("macro_f1", 0.0))
            regressions = []
            for cls in CLASS_NAMES:
                prod_recall = float(prod_metrics.get(f"recall_{cls}", 0.0))
                new_recall = recall_per_class[cls]
                if new_recall < prod_recall - RECALL_TOLERANCE:
                    regressions.append(f"recall_{cls} {new_recall:.4f} < {prod_recall:.4f} - {RECALL_TOLERANCE}")

            if macro_f1 >= prod_f1 and not regressions:
                client.set_registered_model_alias(MLFLOW_MODEL_NAME, "production", mv.version)
                print(f"  [ok] v{mv.version} -> @production (macro_f1 {macro_f1:.4f} >= {prod_f1:.4f})")
                status = "promoted"
            else:
                client.set_registered_model_alias(MLFLOW_MODEL_NAME, "challenger", mv.version)
                reasons = regressions[:]
                if macro_f1 < prod_f1:
                    reasons.insert(0, f"macro_f1 {macro_f1:.4f} < {prod_f1:.4f}")
                print(f"  [KO] v{mv.version} reste @challenger : {' | '.join(reasons)}")
                status = "challenger"

        finally:
            resource_summary = monitor.stop()
            log_training_run(
                mlflow_run_id=run.info.run_id, model_name=MLFLOW_MODEL_NAME,
                generation=args.generation, fold=None, device=device,
                started_at=started_at, ended_at=datetime.now(timezone.utc),
                resource_summary=resource_summary,
            )
            if y_true is not None:
                log_class_metrics_and_confusion(
                    mlflow_run_id=run.info.run_id, model_name=MLFLOW_MODEL_NAME,
                    generation=args.generation, fold=None,
                    class_names=CLASS_NAMES, y_true=y_true, y_pred=y_pred,
                )

    print(f"\nTemps total : {(time.time() - t_start) / 60:.1f} min")
    print(f"Résultat : {status}")


if __name__ == "__main__":
    main()
