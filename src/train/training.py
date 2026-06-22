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
import subprocess
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
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
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

warnings.filterwarnings("ignore")
plt.switch_backend("Agg")

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

MLFLOW_MODEL_NAME = "blood-cell-densenet121"
RECALL_TOLERANCE = 0.02

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
    "run_type"    : "base",
    "part"        : 0,
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
            if p.suffix.lower() in exts and not p.name.startswith("."):
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


@torch.no_grad()
def _predict_all(model, loader, device):
    """Retourne prédictions, vraies étiquettes et probabilités sur un DataLoader."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        probs = torch.softmax(model(imgs), dim=1).cpu().numpy()
        all_preds.extend(probs.argmax(axis=1).tolist())
        all_labels.extend(lbls.tolist())
        all_probs.append(probs)
    return np.array(all_preds), np.array(all_labels), np.vstack(all_probs)


def _compute_test_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    per_class_recall = recall_score(labels, preds, average=None, zero_division=0)
    return {
        "macro_f1":            f1_score(labels, preds, average="macro",    zero_division=0),
        "weighted_f1":         f1_score(labels, preds, average="weighted",  zero_division=0),
        "precision_macro":     precision_score(labels, preds, average="macro",   zero_division=0),
        "recall_macro":        recall_score(labels, preds, average="macro",      zero_division=0),
        "recall_erythroblast": per_class_recall[CLASSES.index("erythroblast")],
        "recall_ig":           per_class_recall[CLASSES.index("ig")],
    }


def _log_artifacts(preds: np.ndarray, labels: np.ndarray, output_dir: Path) -> None:
    """Génère et logue confusion_matrix.png, classification_report.txt, label_mapping.json."""
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(CLASSES)), yticks=range(len(CLASSES)),
        xticklabels=CLASSES, yticklabels=CLASSES,
        xlabel="Prédit", ylabel="Réel",
        title="Matrice de confusion — test set",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)
    mlflow.log_artifact(str(cm_path))

    # Classification report
    report = classification_report(labels, preds, target_names=CLASSES, digits=3)
    report_path = output_dir / "classification_report.txt"
    report_path.write_text(report)
    mlflow.log_artifact(str(report_path))

    # Label mapping
    mapping_path = output_dir / "label_mapping.json"
    mapping_path.write_text(json.dumps({i: c for i, c in enumerate(CLASSES)}, indent=2))
    mlflow.log_artifact(str(mapping_path))


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _get_dvc_hash(data_dir: Path) -> str:
    """Lit le hash MD5 DVC du dataset depuis le fichier .dvc correspondant."""
    import yaml
    candidates = list(ROOT.glob("data/*.dvc")) + list(ROOT.glob("data/runs/*.dvc"))
    for dvc_file in candidates:
        try:
            manifest = yaml.safe_load(dvc_file.read_text())
            out = manifest["outs"][0]
            dvc_path = (ROOT / "data" / out["path"]).resolve()
            if dvc_path == data_dir.resolve():
                return out["md5"]
        except Exception:
            continue
    return "unknown"


def _log_dataset_input(data_dir: Path, paths: list, labels: list, dvc_hash: str) -> None:
    """Lie la version DVC du dataset au run MLflow via mlflow.log_input()."""
    try:
        from mlflow.data.filesystem_dataset_source import FileSystemDatasetSource
        from mlflow.data.meta_dataset import MetaDataset
        source = FileSystemDatasetSource(path=str(data_dir))
        dataset = MetaDataset(source=source, name=data_dir.name, digest=dvc_hash)
        mlflow.log_input(dataset, context="training")
    except Exception:
        pass  # log_input optionnel — tags dvc_* suffisent pour la traçabilité


def _setup_mlflow():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    exp_name = "bloodcells-densenet121"
    if client.get_experiment_by_name(exp_name) is None:
        # mlflow-artifacts:/ force le mode proxy (--serve-artifacts côté serveur)
        # nécessaire pour logger des artifacts depuis l'hôte vers le container Docker
        client.create_experiment(exp_name, artifact_location="mlflow-artifacts:/")
    mlflow.set_experiment(exp_name)


def _register_and_promote(run_id: str, metrics: dict) -> None:
    """Enregistre le modèle et promeut via aliases MLflow 3.x (@production / @challenger).

    Garde-fous de promotion :
      - macro_f1 >= prod (strict)
      - recall_erythroblast et recall_ig non régressifs (tolérance RECALL_TOLERANCE)
    """
    client = MlflowClient()

    mv = mlflow.register_model(
        model_uri=f"runs:/{run_id}/densenet121",
        name=MLFLOW_MODEL_NAME,
    )
    new_version = mv.version
    client.set_registered_model_alias(MLFLOW_MODEL_NAME, "challenger", new_version)
    print(f"  Version {new_version} enregistrée -> @challenger")

    try:
        prod_mv = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, "production")
        prod_run = client.get_run(prod_mv.run_id)
        pm = prod_run.data.metrics

        prod_f1          = float(pm.get("macro_f1", 0.0))
        prod_recall_ery  = float(pm.get("recall_erythroblast", 0.0))
        prod_recall_ig   = float(pm.get("recall_ig", 0.0))

        new_f1          = float(metrics.get("macro_f1", 0.0))
        new_recall_ery  = float(metrics.get("recall_erythroblast", 0.0))
        new_recall_ig   = float(metrics.get("recall_ig", 0.0))

        f1_ok          = new_f1 >= prod_f1
        recall_ery_ok  = new_recall_ery >= prod_recall_ery - RECALL_TOLERANCE
        recall_ig_ok   = new_recall_ig  >= prod_recall_ig  - RECALL_TOLERANCE

        if f1_ok and recall_ery_ok and recall_ig_ok:
            client.set_registered_model_alias(MLFLOW_MODEL_NAME, "production", new_version)
            client.delete_registered_model_alias(MLFLOW_MODEL_NAME, "challenger")
            print(f"  [ok] Promu @production  macro_f1 {new_f1:.4f} >= {prod_f1:.4f}")
        else:
            reasons = []
            if not f1_ok:
                reasons.append(f"macro_f1 {new_f1:.4f} < {prod_f1:.4f}")
            if not recall_ery_ok:
                reasons.append(
                    f"recall_erythroblast {new_recall_ery:.4f} < {prod_recall_ery:.4f} - {RECALL_TOLERANCE}"
                )
            if not recall_ig_ok:
                reasons.append(
                    f"recall_ig {new_recall_ig:.4f} < {prod_recall_ig:.4f} - {RECALL_TOLERANCE}"
                )
            print(f"  [KO] Garde-fou — reste @challenger : {' | '.join(reasons)}")

    except mlflow.exceptions.MlflowException:
        client.set_registered_model_alias(MLFLOW_MODEL_NAME, "production", new_version)
        client.delete_registered_model_alias(MLFLOW_MODEL_NAME, "challenger")
        print("  Premier modèle -> @production")


def train(data_dir: Path, output_dir: Path, cfg: dict) -> dict:
    _setup_mlflow()
    t_start = time.time()

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
    n_train, n_val, n_test = len(idx_train), len(idx_val), len(idx_test)
    print(f"Split   : train={n_train}  val={n_val}  test={n_test}")

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

        # ── Params ───────────────────────────────────────────────────────────
        mlflow.log_params({
            **cfg,
            "model":       "densenet121",
            "num_classes": len(CLASSES),
            "device":      device,
            "optimizer":   "AdamW",
            "n_train":     n_train,
            "n_val":       n_val,
            "n_test":      n_test,
        })

        # ── Tags ─────────────────────────────────────────────────────────────
        dvc_hash = _get_dvc_hash(data_dir)
        mlflow.set_tags({
            "git_commit":       _get_git_commit(),
            "run_type":         cfg.get("run_type", "base"),
            "dvc_dataset_hash": dvc_hash,
            "dataset_name":     data_dir.name,
        })

        # ── Dataset versioning DVC → MLflow ──────────────────────────────────
        _log_dataset_input(data_dir, paths, labels, dvc_hash)

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
            tag = " [ok]" if improved else f" (patience {patience_cnt}/{cfg['patience']})"
            print(f"  Ep {epoch+1:02d}  train={ta:.3f}  val={va:.3f}  ({time.time()-t0:.0f}s){tag}")
            if patience_cnt >= cfg["patience"]:
                print("  Early stopping.")
                break

        # ── Évaluation test ──────────────────────────────────────────────────
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        _, test_acc = _evaluate(model, test_dl, criterion, device)
        preds, trues, _ = _predict_all(model, test_dl, device)

        test_metrics = _compute_test_metrics(preds, trues)
        train_time_s = time.time() - t_start
        n_params = sum(p.numel() for p in model.parameters())

        mlflow.log_metrics({
            "best_val_acc": best_val_acc,
            "test_acc":     test_acc,
            "train_time_s": round(train_time_s, 1),
            "n_params":     n_params,
            **test_metrics,
        })

        # ── Artifacts ────────────────────────────────────────────────────────
        _log_artifacts(preds, trues, output_dir)

        # ── Modèle ───────────────────────────────────────────────────────────
        mlflow.pytorch.log_model(model, name="densenet121")

        print(f"\nMeilleur val_acc    : {best_val_acc:.4f}")
        print(f"Test accuracy       : {test_acc:.4f}")
        print(f"macro_f1            : {test_metrics['macro_f1']:.4f}")
        print(f"recall_erythroblast : {test_metrics['recall_erythroblast']:.4f}")
        print(f"recall_ig           : {test_metrics['recall_ig']:.4f}")
        print(f"Modèle              : {model_path}")
        print(f"MLflow run ID       : {run.info.run_id}")

        # ── Registry ─────────────────────────────────────────────────────────
        print("\nMLflow Registry :")
        _register_and_promote(run.info.run_id, {**test_metrics, "test_acc": test_acc})

    metrics = {
        "best_val_acc": best_val_acc,
        "test_acc":     test_acc,
        "run_id":       run.info.run_id,
        **test_metrics,
        "history": history,
        "config":  {k: v for k, v in cfg.items()},
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
    parser.add_argument("--run-type",    default="base", choices=["base", "retrain"],
                        help="Type de run MLflow : base ou retrain")
    parser.add_argument("--part",        type=int, default=0,
                        help="Numéro de la partie du dataset (0 = tout)")
    parser.add_argument("--seed",        type=int, default=DEFAULT_CFG["seed"],
                        help="Graine aléatoire (défaut : 42)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    def _resolve(p):
        p = Path(p)
        return p if p.is_absolute() else ROOT / p

    cfg = {
        **DEFAULT_CFG,
        "epochs_head": args.epochs_head,
        "epochs_full": args.epochs_full,
        "batch_size":  args.batch_size,
        "run_type":    args.run_type,
        "part":        args.part,
        "seed":        args.seed,
    }

    metrics = train(_resolve(args.data_dir), _resolve(args.output_dir), cfg)
    print(
        f"\nRésumé : val_acc={metrics['best_val_acc']:.4f}"
        f"  test_acc={metrics['test_acc']:.4f}"
        f"  macro_f1={metrics['macro_f1']:.4f}"
    )
