"""
Entraînement DenseNet-121 — classification de cellules sanguines (8 classes).

Usage :
    python -m src.train.training
    python -m src.train.training --data-dir data/raw --epochs-head 3 --epochs-full 5
"""

import argparse
import json
import os
import random
import time
import warnings
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

MLFLOW_MODEL_NAME = "blood-cell-densenet121"

CLASSES = [
    "basophil", "eosinophil", "erythroblast", "ig",
    "lymphocyte", "monocyte", "neutrophil", "platelet",
]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

DEFAULT_CFG = {
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
    "seed"        : 42,
    "input_size"  : 224,
}


class CellDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


def get_transform(input_size: int):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_dataset(data_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}
    paths, labels = [], []
    for label, cls in enumerate(CLASSES):
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            continue
        for p in cls_dir.iterdir():
            if p.suffix.lower() in exts:
                paths.append(p)
                labels.append(label)
    if not paths:
        raise FileNotFoundError(
            f"Aucune image trouvée dans {data_dir}.\n"
            f"Vérifiez DATA_RAW_DIR dans .env (dossier attendu : {data_dir})"
        )
    return paths, labels


def _train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = correct = total = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == lbls).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def _evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        out = model(imgs)
        loss = criterion(out, lbls)
        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == lbls).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


def _setup_mlflow():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT}/mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("blood-cell-classification")


def _register_and_promote(model, run_id: str, test_acc: float) -> None:
    """Enregistre le modèle dans le Registry et promeut si meilleur que la version en Production."""
    client = MlflowClient()

    mv = mlflow.register_model(
        model_uri=f"runs:/{run_id}/densenet121",
        name=MLFLOW_MODEL_NAME,
    )
    new_version = mv.version

    try:
        prod_versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["Production"])
        if prod_versions:
            prev_run = client.get_run(prod_versions[0].run_id)
            prev_acc = float(prev_run.data.metrics.get("test_acc", 0.0))
            if test_acc > prev_acc:
                client.transition_model_version_stage(
                    name=MLFLOW_MODEL_NAME, version=prod_versions[0].version,
                    stage="Archived",
                )
                client.transition_model_version_stage(
                    name=MLFLOW_MODEL_NAME, version=new_version, stage="Production",
                )
                print(f"  Nouveau modèle promu en Production (test_acc {test_acc:.4f} > {prev_acc:.4f})")
            else:
                client.transition_model_version_stage(
                    name=MLFLOW_MODEL_NAME, version=new_version, stage="Staging",
                )
                print(f"  Modèle précédent conservé en Production (test_acc {prev_acc:.4f} >= {test_acc:.4f})")
        else:
            client.transition_model_version_stage(
                name=MLFLOW_MODEL_NAME, version=new_version, stage="Production",
            )
            print("  Premier modèle enregistré → promu en Production")
    except Exception as e:
        print(f"  Avertissement Registry : {e}")


def train(data_dir: Path, output_dir: Path, cfg: dict) -> dict:
    _setup_mlflow()

    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device  : {device}")

    paths, labels = load_dataset(data_dir)
    print(f"Images  : {len(paths)} dans {len(set(labels))} classes")

    idx = list(range(len(paths)))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=cfg["test_size"], stratify=labels, random_state=seed,
    )
    trainval_lbls = [labels[i] for i in idx_trainval]
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=cfg["val_size"] / (1 - cfg["test_size"]),
        stratify=trainval_lbls,
        random_state=seed,
    )
    print(f"Split   : train={len(idx_train)}  val={len(idx_val)}  test={len(idx_test)}")

    tf = get_transform(cfg["input_size"])

    def make_dl(idxs, shuffle):
        return DataLoader(
            CellDataset([paths[i] for i in idxs], [labels[i] for i in idxs], tf),
            batch_size=cfg["batch_size"], shuffle=shuffle, num_workers=cfg["num_workers"],
        )

    train_dl = make_dl(idx_train, shuffle=True)
    val_dl   = make_dl(idx_val,   shuffle=False)
    test_dl  = make_dl(idx_test,  shuffle=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "best_densenet121.pth"

    model = timm.create_model("densenet121", pretrained=True, num_classes=len(CLASSES))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    with mlflow.start_run() as run:
        mlflow.log_params(cfg)
        mlflow.log_param("model", "densenet121")
        mlflow.log_param("num_classes", len(CLASSES))
        mlflow.log_param("device", device)

        # ── Phase 1 : backbone gelé ──────────────────────────────────────────
        HEAD_NAMES = {"classifier", "head", "fc"}
        for name, p in model.named_parameters():
            p.requires_grad = name.split(".")[0] in HEAD_NAMES

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg["lr_head"], weight_decay=cfg["weight_decay"],
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs_head"])

        print(f"\nPhase 1 — backbone gelé ({cfg['epochs_head']} epochs)")
        for epoch in range(cfg["epochs_head"]):
            t0 = time.time()
            tl, ta = _train_epoch(model, train_dl, optimizer, criterion, device)
            vl, va = _evaluate(model, val_dl, criterion, device)
            scheduler.step()
            history["train_loss"].append(tl)
            history["val_loss"].append(vl)
            history["train_acc"].append(ta)
            history["val_acc"].append(va)
            mlflow.log_metrics(
                {"train_loss": tl, "val_loss": vl, "train_acc": ta, "val_acc": va},
                step=epoch,
            )
            if va > best_val_acc:
                best_val_acc = va
                torch.save(model.state_dict(), model_path)
            print(f"  Ep {epoch+1:02d}  train={ta:.3f}  val={va:.3f}  ({time.time()-t0:.0f}s)")

        # ── Phase 2 : fine-tuning complet ────────────────────────────────────
        for p in model.parameters():
            p.requires_grad = True

        optimizer = optim.AdamW(
            model.parameters(), lr=cfg["lr_full"], weight_decay=cfg["weight_decay"],
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs_full"])

        patience_cnt = 0
        offset = cfg["epochs_head"]
        print(f"\nPhase 2 — fine-tuning ({cfg['epochs_full']} epochs, patience={cfg['patience']})")
        for epoch in range(cfg["epochs_full"]):
            t0 = time.time()
            tl, ta = _train_epoch(model, train_dl, optimizer, criterion, device)
            vl, va = _evaluate(model, val_dl, criterion, device)
            scheduler.step()
            history["train_loss"].append(tl)
            history["val_loss"].append(vl)
            history["train_acc"].append(ta)
            history["val_acc"].append(va)
            mlflow.log_metrics(
                {"train_loss": tl, "val_loss": vl, "train_acc": ta, "val_acc": va},
                step=offset + epoch,
            )
            improved = va > best_val_acc
            if improved:
                best_val_acc = va
                torch.save(model.state_dict(), model_path)
                patience_cnt = 0
            else:
                patience_cnt += 1
            tag = " ✓" if improved else f" (patience {patience_cnt}/{cfg['patience']})"
            print(f"  Ep {epoch+1:02d}  train={ta:.3f}  val={va:.3f}  ({time.time()-t0:.0f}s){tag}")
            if patience_cnt >= cfg["patience"]:
                print("  Early stopping.")
                break

        # ── Évaluation test ──────────────────────────────────────────────────
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        _, test_acc = _evaluate(model, test_dl, criterion, device)

        mlflow.log_metrics({"best_val_acc": best_val_acc, "test_acc": test_acc})
        mlflow.pytorch.log_model(
            model,
            name="densenet121",
        )

        print(f"\nMeilleur val_acc : {best_val_acc:.4f}")
        print(f"Test accuracy    : {test_acc:.4f}")
        print(f"Modèle           : {model_path}")
        print(f"MLflow run ID    : {run.info.run_id}")

        # ── Registry : enregistrement + comparaison ──────────────────────────
        print("\nMLflow Registry :")
        _register_and_promote(model, run.info.run_id, test_acc)

    metrics = {
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "history": history,
        "config": {k: v for k, v in cfg.items()},
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "class_names.json").write_text(json.dumps(CLASSES))

    return metrics


def _parse_args():
    parser = argparse.ArgumentParser(description="Entraîne DenseNet-121 sur le dataset PBC")
    parser.add_argument("--data-dir",    default=os.getenv("DATA_RAW_DIR", "data/raw"),
                        help="Dossier contenant les sous-dossiers par classe")
    parser.add_argument("--output-dir",  default=os.getenv("MODELS_DIR", "models"))
    parser.add_argument("--epochs-head", type=int, default=DEFAULT_CFG["epochs_head"])
    parser.add_argument("--epochs-full", type=int, default=DEFAULT_CFG["epochs_full"])
    parser.add_argument("--batch-size",  type=int, default=DEFAULT_CFG["batch_size"])
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    def _resolve(p):
        p = Path(p)
        return p if p.is_absolute() else ROOT / p

    cfg = {**DEFAULT_CFG,
           "epochs_head": args.epochs_head,
           "epochs_full": args.epochs_full,
           "batch_size" : args.batch_size}

    metrics = train(_resolve(args.data_dir), _resolve(args.output_dir), cfg)
    print(f"\nRésumé : val_acc={metrics['best_val_acc']:.4f}  test_acc={metrics['test_acc']:.4f}")
