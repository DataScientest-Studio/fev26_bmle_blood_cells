#!/usr/bin/env python
# coding: utf-8

# ## Cross-validation 5 folds — images améliorées (Cellpose crop + augmentation)
# 4 modèles × 5 folds = 20 entraînements
# Dataset : train_final + val_cropped + test_cropped (Mendeley PBC 17k)

import argparse
from PIL import Image
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, recall_score
)
from sklearn.model_selection import train_test_split, StratifiedKFold
import timm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mlflow.pytorch
import mlflow
from mlflow.tracking import MlflowClient
import os
import json
import random
import time
import warnings
from datetime import datetime, timezone

from src.monitoring.resource_monitor import ResourceMonitor
from src.monitoring.supabase_logger import log_class_metrics_and_confusion, log_training_run
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────

class BloodCellDataset(Dataset):
    def __init__(self, paths, labels, indices, transform=None):
        self.paths = [paths[i] for i in indices]
        self.labels = [labels[i] for i in indices]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


if __name__ == "__main__":
    print(f"PyTorch  : {torch.__version__}")
    print(f"CUDA     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU      : {torch.cuda.get_device_name(0)}")

    # ─────────────────────────────────────────────
    #  Configuration
    # ─────────────────────────────────────────────

    _cwd = Path().resolve()
    PROJECT_ROOT = _cwd
    for _p in [_cwd] + list(_cwd.parents):
        if (_p / "data").is_dir() and (_p / "src").is_dir():
            PROJECT_ROOT = _p
            break
    print(f"Racine du projet : {PROJECT_ROOT}")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=str,
        default=os.environ.get("CROSSVAL_DATA_DIR", str(PROJECT_ROOT / "data" / "crossval_source")),
        help="Dossier contenant train_final/, val_cropped/, test_cropped/",
    )
    parser.add_argument(
        "--models", type=str, default="DenseNet-121",
        help="Liste d'architectures séparées par des virgules (clés de MODELS_CONFIG)",
    )
    parser.add_argument(
        "--generation", type=str, default="v1",
        help="Tag de génération appliqué aux versions enregistrées dans le Model Registry",
    )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Nombre max d'epochs par fold (réduire pour un smoke test rapide)",
    )
    parser.add_argument(
        "--folds", type=int, default=5,
        help="Nombre de folds de la cross-validation (réduire pour un smoke test rapide)",
    )
    args = parser.parse_args()

    # Dossiers des images améliorées (Cellpose crop + augmentation)
    _ML_BASE = Path(args.data_dir)
    TRAIN_DIR = _ML_BASE / "train_final"
    VAL_DIR = _ML_BASE / "val_cropped"
    TEST_DIR = _ML_BASE / "test_cropped"

    EXPECTED_CLASSES = [
        "basophil", "eosinophil", "erythroblast", "ig",
        "lymphocyte", "monocyte", "neutrophil", "platelet",
    ]

    MLFLOW_MODEL_NAME = "blood-cell-densenet121"
    RECALL_TOLERANCE = 0.02

    N_FOLDS = args.folds

    CFG = {
        # Le nombre de epochs n'entre PAS dans ce chemin : un run interrompu doit
        # pouvoir reprendre via le mécanisme de checkpoint même si --epochs diffère
        # légèrement. N_FOLDS, en revanche, change la composition même des folds
        # (StratifiedKFold(n_splits=N) ne découpe pas pareil selon N) — un cache
        # partagé entre deux valeurs de N_FOLDS rechargerait par erreur un modèle
        # entraîné sur un découpage différent, juste parce que le numéro de fold
        # coïncide.
        "output_dir": str(PROJECT_ROOT / "reports" / f"Romane_DL_crossval_ameliorees_{args.folds}folds"),
        "num_epochs": args.epochs,
        "batch_size": 32,
        "lr_head": 1e-3,
        "lr_full": 1e-4,
        "weight_decay": 1e-4,
        "patience": 3,
        "num_workers": 4,
        "seed": 42,
        "device": (
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        ),
    }

    MODELS_CONFIG = {
        "EfficientNet-B3": {"name": "tf_efficientnet_b3", "input_size": 300},
        "ConvNeXt-Tiny": {"name": "convnext_tiny", "input_size": 224},
        "DenseNet-121": {"name": "densenet121", "input_size": 224},
        "ResNet-50": {"name": "resnet50", "input_size": 224},
    }

    MODEL_COLORS = {
        "EfficientNet-B3": "#D85A30",
        "ConvNeXt-Tiny": "#7F77DD",
        "DenseNet-121": "#1D9E75",
        "ResNet-50": "#378ADD",
    }

    requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in requested_models if m not in MODELS_CONFIG]
    if unknown:
        raise ValueError(f"Modèles inconnus dans --models : {unknown} (dispo : {list(MODELS_CONFIG)})")
    MODELS_CONFIG = {k: v for k, v in MODELS_CONFIG.items() if k in requested_models}

    os.makedirs(CFG["output_dir"], exist_ok=True)

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(CFG["seed"])

    # ─────────────────────────────────────────────
    #  Chargement des images depuis les 3 dossiers
    # ─────────────────────────────────────────────

    def load_dataset_paths(data_dir, class_names):
        data_dir = Path(data_dir)
        paths, labels = [], []
        for label, cls in enumerate(class_names):
            cls_dir = data_dir / cls
            if not cls_dir.is_dir():
                continue
            exts = {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}
            imgs = [p for p in cls_dir.iterdir() if p.suffix.lower() in exts]
            paths.extend(imgs)
            labels.extend([label] * len(imgs))
        return paths, labels

    CLASS_NAMES = EXPECTED_CLASSES
    NUM_CLASSES = len(CLASS_NAMES)

    print("\nChargement des datasets...")
    paths_train, labels_train = load_dataset_paths(TRAIN_DIR, CLASS_NAMES)
    paths_val, labels_val = load_dataset_paths(VAL_DIR, CLASS_NAMES)
    paths_test, labels_test = load_dataset_paths(TEST_DIR, CLASS_NAMES)

    all_paths = paths_train + paths_val + paths_test
    all_labels = labels_train + labels_val + labels_test

    print(f"  train_final  : {len(paths_train)} images")
    print(f"  val_cropped  : {len(paths_val)}   images")
    print(f"  test_cropped : {len(paths_test)}  images")
    print(f"  TOTAL        : {len(all_paths)} images | {NUM_CLASSES} classes")

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        n = all_labels.count(cls_idx)
        print(f"    {cls_name:<15} : {n}")

    # Vérification
    for d, name in [(TRAIN_DIR, "train_final"), (VAL_DIR, "val_cropped"), (TEST_DIR, "test_cropped")]:
        status = "OK" if Path(d).is_dir() else "MANQUANT"
        print(f"  {status}  {d}")

    print(f"\nDevice   : {CFG['device']}")
    print(f"Modèles  : {list(MODELS_CONFIG.keys())}")
    print(f"Folds    : {N_FOLDS}")
    print(f"Runs tot : {N_FOLDS * len(MODELS_CONFIG)}")

    # ─────────────────────────────────────────────
    #  Transforms
    # ─────────────────────────────────────────────

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def get_transform(input_size):
        return transforms.Compose([
            transforms.Resize((input_size, input_size),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def get_dataloaders_fold(input_size, idx_train, idx_val, idx_test):
        tf = get_transform(input_size)
        pin = (CFG["device"] == "cuda")

        def _make(idx, shuffle):
            if not idx:
                return None
            ds = BloodCellDataset(all_paths, all_labels, idx, transform=tf)
            return DataLoader(ds, batch_size=CFG["batch_size"],
                              shuffle=shuffle, num_workers=CFG["num_workers"], pin_memory=pin)

        return _make(idx_train, True), _make(idx_val, False), _make(idx_test, False)

    # ─────────────────────────────────────────────
    #  Modèles
    # ─────────────────────────────────────────────

    def build_model(model_key, num_classes, pretrained=True):
        cfg = MODELS_CONFIG[model_key]
        return timm.create_model(cfg["name"], pretrained=pretrained, num_classes=num_classes)

    def freeze_backbone(model):
        HEAD_NAMES = {"classifier", "head", "fc"}
        for name, param in model.named_parameters():
            top_level = name.split(".")[0]
            param.requires_grad = top_level in HEAD_NAMES

    def unfreeze_backbone(model):
        for param in model.parameters():
            param.requires_grad = True

    # ─────────────────────────────────────────────
    #  Fonctions d'entraînement
    # ─────────────────────────────────────────────

    def train_one_epoch(model, loader, optimizer, criterion, device, use_amp=False):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels in tqdm(loader, leave=False, desc="  Train"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=use_amp):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += imgs.size(0)
        return running_loss / total, correct / total

    @torch.no_grad()
    def evaluate(model, loader, criterion, device, use_amp=False):
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.autocast(device_type=device, enabled=use_amp):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += imgs.size(0)
        return running_loss / total, correct / total

    @torch.no_grad()
    def get_predictions(model, loader, device):
        model.eval()
        y_true, y_pred, y_scores = [], [], []
        for imgs, labels in tqdm(loader, leave=False, desc="  Inference"):
            imgs = imgs.to(device)
            logits = model(imgs).float()
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            y_scores.extend(probs)
        return np.array(y_true), np.array(y_pred), np.array(y_scores)

    def train_model_fold(model_key, fold_num, fold_dir, idx_train, idx_val):
        """Entraîne un modèle pour un fold donné — 2 phases + early stopping."""
        safe_key = model_key.replace("-", "_")
        save_path = fold_dir / f"best_fold{fold_num}_{safe_key}.pth"
        hist_path = fold_dir / f"history_fold{fold_num}_{safe_key}.json"
        ckpt_path = fold_dir / f"ckpt_fold{fold_num}_{safe_key}.pth"

        # Skip si déjà fait
        if save_path.exists() and hist_path.exists() and not ckpt_path.exists():
            print(f"  -> {model_key} fold {fold_num} déjà entraîné — rechargement")
            m = build_model(model_key, NUM_CLASSES, pretrained=False)
            m.load_state_dict(torch.load(save_path, map_location=CFG["device"], weights_only=True))
            m = m.to(CFG["device"]).eval()
            with open(hist_path) as f:
                history = json.load(f)
            return m, history, None, None

        print(f"\n  {'='*50}")
        print(f"  {model_key}  —  Fold {fold_num}/{N_FOLDS}")
        print(f"  {'='*50}")
        set_seed(CFG["seed"] + fold_num * 100)

        cfg_m = MODELS_CONFIG[model_key]
        device = CFG["device"]
        use_amp = device in ("cuda", "mps")

        train_loader, val_loader, _ = get_dataloaders_fold(
            cfg_m["input_size"], idx_train, idx_val, []
        )

        model = build_model(model_key, NUM_CLASSES, pretrained=True).to(device)

        train_labels_list = [all_labels[i] for i in idx_train]
        cls_counts = np.bincount(train_labels_list, minlength=NUM_CLASSES).astype(float)
        cls_weights = 1.0 / np.where(cls_counts > 0, cls_counts, 1.0)
        cls_weights = torch.tensor(
            cls_weights / cls_weights.sum() * NUM_CLASSES, dtype=torch.float
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=cls_weights)

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_val_acc = 0.0
        patience_counter = 0

        resume_phase, resume_epoch = 1, 0
        ckpt_opt_state = ckpt_sched_state = None

        if ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model"])
                history = ckpt["history"]
                best_val_acc = ckpt["best_val_acc"]
                patience_counter = ckpt["patience_counter"]
                resume_phase = ckpt["phase"]
                resume_epoch = ckpt["epoch"] + 1
                ckpt_opt_state = ckpt["optimizer"]
                ckpt_sched_state = ckpt["scheduler"]
                print(f"  Reprise checkpoint — Phase {resume_phase}, epoch {resume_epoch + 1}")
            except Exception as e:
                print(f"  Checkpoint corrompu ({e}) — réentraînement depuis zéro")
                ckpt_path.unlink(missing_ok=True)

        def _save_ckpt(phase, epoch, optimizer, scheduler):
            torch.save({
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "history": history,
                "best_val_acc": best_val_acc, "patience_counter": patience_counter,
                "phase": phase, "epoch": epoch,
            }, ckpt_path)

        PHASE1_EPOCHS = 5
        remaining = CFG["num_epochs"] - PHASE1_EPOCHS

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"))
        mlflow.set_experiment("blood_cell_crossval_ameliorees")
        mlflow.start_run(run_name=f"{model_key}_fold{fold_num}")
        run_id = mlflow.active_run().info.run_id
        started_at = datetime.now(timezone.utc)
        resource_monitor = ResourceMonitor().start()
        mlflow.log_params({
            "model": model_key, "fold": fold_num,
            "input_size": cfg_m["input_size"],
            "num_epochs": CFG["num_epochs"], "batch_size": CFG["batch_size"],
            "lr_head": CFG["lr_head"], "lr_full": CFG["lr_full"],
            "weight_decay": CFG["weight_decay"], "patience": CFG["patience"],
            "seed": CFG["seed"],
        })

        try:
            # Phase 1 : backbone gelé
            if resume_phase == 1:
                print("  Phase 1 — feature extraction")
                freeze_backbone(model)
                optimizer = optim.AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=CFG["lr_head"], weight_decay=CFG["weight_decay"]
                )
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE1_EPOCHS)
                if ckpt_opt_state:
                    optimizer.load_state_dict(ckpt_opt_state)
                    scheduler.load_state_dict(ckpt_sched_state)

                for epoch in range(resume_epoch, PHASE1_EPOCHS):
                    t0 = time.time()
                    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, use_amp)
                    vl_loss, vl_acc = evaluate(model, val_loader, criterion, device, use_amp)
                    scheduler.step()
                    history["train_loss"].append(tr_loss)
                    history["train_acc"].append(tr_acc)
                    history["val_loss"].append(vl_loss)
                    history["val_acc"].append(vl_acc)
                    _save_ckpt(1, epoch, optimizer, scheduler)
                    mlflow.log_metrics({
                        "p1_train_loss": tr_loss, "p1_val_loss": vl_loss,
                        "p1_train_acc": tr_acc, "p1_val_acc": vl_acc,
                    }, step=epoch)
                    print(f"  P1 E{epoch+1:02d}/{PHASE1_EPOCHS} | "
                          f"loss {tr_loss:.4f}->{vl_loss:.4f} | "
                          f"acc {tr_acc:.3f}->{vl_acc:.3f} | {time.time()-t0:.1f}s")

                resume_epoch = 0
                ckpt_opt_state = ckpt_sched_state = None

            # Phase 2 : fine-tuning complet
            print("  Phase 2 — fine-tuning complet")
            unfreeze_backbone(model)
            optimizer = optim.AdamW(model.parameters(), lr=CFG["lr_full"], weight_decay=CFG["weight_decay"])
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)
            if ckpt_opt_state:
                optimizer.load_state_dict(ckpt_opt_state)
                scheduler.load_state_dict(ckpt_sched_state)

            for epoch in range(resume_epoch, remaining):
                t0 = time.time()
                tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, use_amp)
                vl_loss, vl_acc = evaluate(model, val_loader, criterion, device, use_amp)
                scheduler.step()
                history["train_loss"].append(tr_loss)
                history["train_acc"].append(tr_acc)
                history["val_loss"].append(vl_loss)
                history["val_acc"].append(vl_acc)

                if vl_acc > best_val_acc:
                    best_val_acc = vl_acc
                    patience_counter = 0
                    torch.save(model.state_dict(), save_path)
                else:
                    patience_counter += 1

                _save_ckpt(2, epoch, optimizer, scheduler)
                mlflow.log_metrics({
                    "p2_train_loss": tr_loss, "p2_val_loss": vl_loss,
                    "p2_train_acc": tr_acc, "p2_val_acc": vl_acc,
                }, step=epoch)
                print(f"  P2 E{epoch+1:02d}/{remaining} | "
                      f"loss {tr_loss:.4f}->{vl_loss:.4f} | "
                      f"acc {tr_acc:.3f}->{vl_acc:.3f} | "
                      f"best {best_val_acc:.3f} | {time.time()-t0:.1f}s")

                if patience_counter >= CFG["patience"]:
                    print(f"  Early stopping à l'epoch {len(history['val_acc'])}")
                    break

            if ckpt_path.exists():
                ckpt_path.unlink()

            with open(hist_path, "w") as f:
                json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f)

            if save_path.exists():
                model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
            model.eval()
            print(f"  Best val acc : {best_val_acc:.4f}")
            mlflow.log_metric("best_val_acc", best_val_acc)
            mlflow.log_metric("final_epoch", len(history["val_acc"]))

        finally:
            resource_summary = resource_monitor.stop()
            log_training_run(
                mlflow_run_id=run_id, model_name=model_key, generation=args.generation, fold=fold_num,
                device=CFG["device"], started_at=started_at, ended_at=datetime.now(timezone.utc),
                resource_summary=resource_summary,
            )
            mlflow.end_run()

        return model, history, run_id, resource_summary

    def register_and_promote_best_fold(model_key, df_model_results, generation):
        """Enregistre le meilleur fold (macro_f1 max) dans le Model Registry,
        tagué generation=<generation>. Promeut @production seulement si
        macro_f1 ne régresse pas ET aucun recall par classe ne régresse
        au-delà de RECALL_TOLERANCE (les 8 classes, pas seulement 2)."""
        if model_key != "DenseNet-121":
            print(f"  [SKIP registry] {model_key} non enregistré (registry réservé à DenseNet-121)")
            return

        best_row = df_model_results.loc[df_model_results["macro_f1"].idxmax()]
        fold_num = int(best_row["fold"])
        weights_path = Path(best_row["weights_path"])
        recall_per_class = best_row["recall_per_class"]

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"))
        mlflow.set_experiment("blood_cell_crossval_ameliorees")

        loaded = build_model(model_key, NUM_CLASSES, pretrained=False)
        loaded.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        loaded.eval()

        client = MlflowClient()

        with mlflow.start_run(run_name=f"{model_key}_best_fold{fold_num}_{generation}") as run:
            mlflow.log_params({
                "model": model_key, "fold": fold_num, "generation": generation,
                "n_train": int(best_row["n_train"]), "n_val": int(best_row["n_val"]),
                "n_test": int(best_row["n_test"]),
            })
            mlflow.set_tags({"generation": generation, "source": "dl_crossval_train"})
            mlflow.log_metrics({
                "accuracy": float(best_row["accuracy"]),
                "macro_f1": float(best_row["macro_f1"]),
                "weighted_f1": float(best_row["weighted_f1"]),
                "auc_roc": float(best_row["auc_roc"]),
                **{f"recall_{cls}": float(r) for cls, r in recall_per_class.items()},
            })
            input_size = MODELS_CONFIG[model_key]["input_size"]
            model_info = mlflow.pytorch.log_model(
                loaded, name="densenet121",
                input_example=torch.zeros(1, 3, input_size, input_size).numpy(),
                serialization_format="pickle",
            )

            mv = mlflow.register_model(model_uri=model_info.model_uri, name=MLFLOW_MODEL_NAME)
            client.set_model_version_tag(MLFLOW_MODEL_NAME, mv.version, "generation", generation)
            client.set_model_version_tag(MLFLOW_MODEL_NAME, mv.version, "fold", str(fold_num))

            new_macro_f1 = float(best_row["macro_f1"])
            try:
                prod_mv = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, "production")
                prod_metrics = client.get_run(prod_mv.run_id).data.metrics
                prod_macro_f1 = float(prod_metrics.get("macro_f1", 0.0))

                regressions = []
                for cls in CLASS_NAMES:
                    prod_recall = float(prod_metrics.get(f"recall_{cls}", 0.0))
                    new_recall = float(recall_per_class.get(cls, 0.0))
                    if new_recall < prod_recall - RECALL_TOLERANCE:
                        regressions.append(
                            f"recall_{cls} {new_recall:.4f} < {prod_recall:.4f} - {RECALL_TOLERANCE}"
                        )

                if new_macro_f1 >= prod_macro_f1 and not regressions:
                    client.set_registered_model_alias(MLFLOW_MODEL_NAME, "production", mv.version)
                    print(f"  [ok] v{mv.version} -> @production "
                          f"(macro_f1 {new_macro_f1:.4f} >= {prod_macro_f1:.4f})")
                else:
                    client.set_registered_model_alias(MLFLOW_MODEL_NAME, "challenger", mv.version)
                    reasons = regressions[:]
                    if new_macro_f1 < prod_macro_f1:
                        reasons.insert(0, f"macro_f1 {new_macro_f1:.4f} < {prod_macro_f1:.4f}")
                    print(f"  [KO] v{mv.version} reste @challenger : {' | '.join(reasons)}")

            except mlflow.exceptions.MlflowException:
                client.set_registered_model_alias(MLFLOW_MODEL_NAME, "production", mv.version)
                print(f"  Premier modèle de production -> v{mv.version}")

    # ─────────────────────────────────────────────
    #  Cross-validation 5 folds
    # ─────────────────────────────────────────────

    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CFG["seed"])
    all_fold_results = []

    t_total_start = time.time()

    for fold_idx, (idx_trainval_arr, idx_test_arr) in enumerate(kf.split(all_paths, all_labels)):
        fold_num = fold_idx + 1
        fold_dir = Path(CFG["output_dir"]) / f"fold_{fold_num}"
        fold_dir.mkdir(exist_ok=True)

        idx_trainval = idx_trainval_arr.tolist()
        idx_test = idx_test_arr.tolist()

        # Split trainval → train (85%) + val (15%)
        labels_trainval = [all_labels[i] for i in idx_trainval]
        idx_tv_arr = np.arange(len(idx_trainval))
        idx_tr_rel, idx_vl_rel = train_test_split(
            idx_tv_arr, test_size=0.15, stratify=labels_trainval, random_state=CFG["seed"]
        )
        idx_train = [idx_trainval[i] for i in idx_tr_rel]
        idx_val = [idx_trainval[i] for i in idx_vl_rel]

        print(f"\n{'#'*60}")
        print(f"  FOLD {fold_num}/{N_FOLDS}")
        print(f"  Train:{len(idx_train)}  Val:{len(idx_val)}  Test:{len(idx_test)}")
        print(f"{'#'*60}")

        for model_key in MODELS_CONFIG:
            t_run_start = time.time()

            model, history, fold_run_id, _ = train_model_fold(
                model_key, fold_num, fold_dir, idx_train, idx_val
            )

            # Évaluation sur le test fold
            cfg_m = MODELS_CONFIG[model_key]
            _, _, test_loader = get_dataloaders_fold(cfg_m["input_size"], [], [], idx_test)
            y_true, y_pred, y_scores = get_predictions(model, test_loader, CFG["device"])

            acc = accuracy_score(y_true, y_pred)
            macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            wt_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            recall_per_class = recall_score(
                y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)), zero_division=0
            )

            if fold_run_id is not None:
                log_class_metrics_and_confusion(
                    mlflow_run_id=fold_run_id, model_name=model_key, generation=args.generation,
                    fold=fold_num, class_names=CLASS_NAMES, y_true=y_true, y_pred=y_pred,
                )

            y_true_1d = y_true.ravel()
            try:
                auc = roc_auc_score(y_true_1d, y_scores, multi_class="ovr", average="macro")
            except ValueError:
                auc = float("nan")

            safe_key = model_key.replace("-", "_")
            all_fold_results.append({
                "fold": fold_num,
                "model": model_key,
                "accuracy": acc,
                "macro_f1": macro_f1,
                "weighted_f1": wt_f1,
                "auc_roc": auc,
                "recall_per_class": dict(zip(CLASS_NAMES, recall_per_class.tolist())),
                "n_train": len(idx_train),
                "n_val": len(idx_val),
                "n_test": len(idx_test),
                "epochs": len(history["val_acc"]),
                "elapsed_min": (time.time() - t_run_start) / 60,
                "weights_path": str(fold_dir / f"best_fold{fold_num}_{safe_key}.pth"),
            })

            print(f"  FOLD {fold_num} | {model_key:<18} | "
                  f"Acc={acc:.4f}  MacF1={macro_f1:.4f}  AUC={auc:.4f}")

            # Libération mémoire GPU entre les modèles
            del model
            torch.cuda.empty_cache()

        # Sauvegarde intermédiaire après chaque fold
        df_interim = pd.DataFrame(all_fold_results)
        df_interim.to_csv(Path(CFG["output_dir"]) / "crossval_results_per_fold.csv", index=False)
        print(f"\n  Fold {fold_num} sauvegardé.")

    # ─────────────────────────────────────────────
    #  Résumé mean ± std
    # ─────────────────────────────────────────────

    df_folds = pd.DataFrame(all_fold_results)
    df_folds.to_csv(Path(CFG["output_dir"]) / "crossval_results_per_fold.csv", index=False)

    print(f"\nEnregistrement MLflow — generation={args.generation}")
    for model_key in MODELS_CONFIG:
        register_and_promote_best_fold(
            model_key, df_folds[df_folds["model"] == model_key], args.generation
        )

    metrics_cols = ["accuracy", "macro_f1", "weighted_f1", "auc_roc"]
    summary_rows = []

    for model_key in MODELS_CONFIG:
        sub = df_folds[df_folds["model"] == model_key][metrics_cols]
        row = {"model": model_key}
        for col in metrics_cols:
            row[f"{col}_mean"] = sub[col].mean()
            row[f"{col}_std"] = sub[col].std()
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows).set_index("model")
    df_summary.to_csv(Path(CFG["output_dir"]) / "crossval_summary.csv")

    print(f"\n{'='*65}")
    print(f"  RÉSUMÉ CROSS-VALIDATION — {N_FOLDS} FOLDS — IMAGES AMÉLIORÉES")
    print(f"{'='*65}")
    print(f"\n{'Modèle':<20} {'Accuracy':>12} {'Macro F1':>12} {'AUC-ROC':>12}")
    print("  " + "─" * 58)
    for model_key, row in df_summary.iterrows():
        print(f"  {model_key:<18} "
              f"  {row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f}"
              f"  {row['macro_f1_mean']:.4f}±{row['macro_f1_std']:.4f}"
              f"  {row['auc_roc_mean']:.4f}±{row['auc_roc_std']:.4f}")
    print(f"{'='*65}")

    total_min = (time.time() - t_total_start) / 60
    print(f"\nTemps total : {total_min:.1f} min")

    # ─────────────────────────────────────────────
    #  Visualisation boxplots
    # ─────────────────────────────────────────────

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    for ax, metric, title in zip(
        axes,
        ["accuracy", "macro_f1", "auc_roc"],
        ["Accuracy", "Macro F1", "AUC-ROC"]
    ):
        data_plot = [
            df_folds[df_folds["model"] == mk][metric].values
            for mk in MODELS_CONFIG
        ]
        bp = ax.boxplot(data_plot, patch_artist=True, notch=False, vert=True)
        for patch, mk in zip(bp['boxes'], MODELS_CONFIG):
            patch.set_facecolor(MODEL_COLORS[mk])
            patch.set_alpha(0.7)

        ax.set_xticklabels(list(MODELS_CONFIG.keys()), rotation=20, fontsize=9)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(title)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle(
        f"Cross-validation {N_FOLDS} folds — images améliorées (Cellpose + augmentation)\n"
        f"4 modèles × {N_FOLDS} folds = {N_FOLDS * len(MODELS_CONFIG)} entraînements",
        fontsize=13, fontweight='bold', y=1.03
    )
    plt.tight_layout()
    plt.savefig(Path(CFG["output_dir"]) / "crossval_boxplots.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ─────────────────────────────────────────────
    #  Heatmap par fold
    # ─────────────────────────────────────────────

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, metric, title in zip(
        axes,
        ["accuracy", "macro_f1", "auc_roc"],
        ["Accuracy", "Macro F1", "AUC-ROC"]
    ):
        pivot = df_folds.pivot(index="model", columns="fold", values=metric)
        sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlGn",
                    vmin=pivot.values.min() - 0.01, vmax=1.0,
                    ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8})
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel("Fold")
        ax.set_ylabel("")

    plt.suptitle("Métriques par fold et par modèle — images améliorées",
                 fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(Path(CFG["output_dir"]) / "crossval_heatmap_folds.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("\nFichiers générés :")
    for f in sorted(Path(CFG["output_dir"]).rglob("*.*")):
        print(f"  {f.relative_to(CFG['output_dir'])}")
    print(f"\nDossier : {CFG['output_dir']}")
