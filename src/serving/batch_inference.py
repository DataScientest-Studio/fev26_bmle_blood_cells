"""
Blood Cell — Folder Report
Analyse un dossier complet : ML SVC + 4 modèles DL (ensemble 5 folds) + Cellpose.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.Morphology.morphology_cellpose_v2 import morphology_cellpose_v2, CLASS_CONFIGS
    CELLPOSE_AVAILABLE = True
except Exception:
    CELLPOSE_AVAILABLE = False

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent

DEFAULT_CROSSVAL_DIR = PROJECT_ROOT / "reports" / "DL_crossval_models"
ML_BUNDLE_PATH       = PROJECT_ROOT / "reports" / "ML_reports_validation" / "best_ml_model.pkl"
_ONEDRIVE_CACHE      = PROJECT_ROOT / "reports"
N_FOLDS = 5

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_SIZE       = (128, 128)
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]
IMAGE_EXTS     = {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}

DL_CLASS_NAMES = [
    "basophil", "eosinophil", "erythroblast", "ig",
    "lymphocyte", "monocyte", "neutrophil", "platelet",
]
DL_VALID_CLASSES = set(DL_CLASS_NAMES)
DL_MODELS_CFG = {
    "DenseNet-121":    {"timm_name": "densenet121",        "input_size": 224, "file": "best_DenseNet_121.pth"},
    "ConvNeXt-Tiny":   {"timm_name": "convnext_tiny",      "input_size": 224, "file": "best_ConvNeXt_Tiny.pth"},
    "EfficientNet-B3": {"timm_name": "tf_efficientnet_b3", "input_size": 300, "file": "best_EfficientNet_B3.pth"},
    "ResNet-50":       {"timm_name": "resnet50",            "input_size": 224, "file": "best_ResNet_50.pth"},
}
MODEL_COLORS = {
    "ML SVC":          "#378ADD",
    "DenseNet-121":    "#1D9E75",
    "ConvNeXt-Tiny":   "#7F77DD",
    "EfficientNet-B3": "#D85A30",
    "ResNet-50":       "#888780",
}

# ── Model loaders ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_cellpose_cache() -> dict:
    """Charge le cache Cellpose OneDrive → dict {image_name: morph_dict}."""
    if not CELLPOSE_AVAILABLE:
        return {}
    cache_base = _ONEDRIVE_CACHE / "Cellpose" / "morpho_cellpose_v2"
    if not cache_base.with_suffix(".parquet").exists():
        return {}
    try:
        results = morphology_cellpose_v2.load(cache_base)
        return {r["image_name"]: r for r in results}
    except Exception:
        return {}



@st.cache_resource
def load_ml_artifacts(ml_bundle_path: str):
    """Charge le bundle ML SVM (format {model, scaler, classes})."""
    bundle_path = Path(ml_bundle_path)
    if not bundle_path.exists():
        return None, None, None, f"{bundle_path.name} introuvable"
    bundle = joblib.load(bundle_path)
    return bundle["model"], bundle["scaler"], list(bundle["classes"]), None


@st.cache_resource
def load_dl_model(model_key: str, dl_dir: str):
    """Charge un seul modèle depuis dl_dir (fallback mode)."""
    pth = Path(dl_dir) / DL_MODELS_CFG[model_key]["file"]
    if not pth.exists():
        return None, str(pth)
    import timm, torch
    cfg       = DL_MODELS_CFG[model_key]
    state     = torch.load(pth, map_location="cpu", weights_only=True)
    head_keys = [k for k in state
                 if k.startswith(("classifier", "head", "fc"))
                 and k.endswith(".weight")
                 and state[k].ndim == 2]
    num_classes = state[head_keys[0]].shape[0] if head_keys else len(DL_CLASS_NAMES)
    model = timm.create_model(cfg["timm_name"], pretrained=False, num_classes=num_classes)
    model.load_state_dict(state)
    model.eval()
    return model, str(pth)


@st.cache_resource
def load_dl_fold_ensemble(model_key: str, crossval_dir: str, n_folds: int = 5):
    """
    Charge les n_folds modèles d'un même modèle pour faire l'ensemble.
    Cherche dans :
      - crossval_dir/fold_{i}/best_fold{i}_{model_key}.pth   (structure crossval)
      - crossval_dir/best_fold{i}_{model_key}.pth            (structure pour_mac à plat)
    Retourne (liste de modèles, liste de chemins trouvés, liste de chemins manquants).
    """
    cfg = DL_MODELS_CFG[model_key]
    base = Path(crossval_dir)
    key_slug = model_key.replace("-", "_").replace(" ", "_")

    models, found, missing = [], [], []
    for i in range(1, n_folds + 1):
        candidates = [
            base / f"fold_{i}" / f"best_fold{i}_{key_slug}.pth",
            base / f"best_fold{i}_{key_slug}.pth",
        ]
        pth = next((p for p in candidates if p.exists()), None)
        if pth is None:
            missing.append(f"fold {i}")
            continue
        try:
            import timm, torch
            ckpt  = torch.load(pth, map_location="cpu", weights_only=False)
            state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            head_keys = [k for k in state
                         if k.startswith(("classifier", "head", "fc"))
                         and k.endswith(".weight")
                         and state[k].ndim == 2]
            num_classes = state[head_keys[0]].shape[0] if head_keys else len(DL_CLASS_NAMES)
            m = timm.create_model(cfg["timm_name"], pretrained=False, num_classes=num_classes)
            m.load_state_dict(state)
            m.eval()
            models.append(m)
            found.append(f"fold {i}")
        except Exception as e:
            missing.append(f"fold {i} (err: {e})")

    return models, found, missing


# ── GradCAM helpers ───────────────────────────────────────────────────────────

def _gradcam_target_layer(model, model_key: str):
    if "EfficientNet" in model_key: return [model.conv_head]
    if "ConvNeXt"     in model_key: return [model.stages[-1].blocks[-1].conv_dw]
    if "DenseNet"     in model_key: return [model.features.denseblock4.denselayer16.conv2]
    if "ResNet"       in model_key: return [model.layer4[-1].conv3]
    raise ValueError(f"Architecture non reconnue : {model_key}")


def compute_gradcam(model, img_path: Path, input_size: int,
                    model_key: str, class_idx: int) -> tuple[np.ndarray | None, str | None]:
    """Retourne (overlay RGB uint8, None) ou (None, message_erreur)."""
    try:
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from torchvision import transforms

        tf = transforms.Compose([
            transforms.Resize((input_size, input_size),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        img_pil = Image.open(img_path).convert("RGB")
        tensor  = tf(img_pil).unsqueeze(0)

        target_layers = _gradcam_target_layer(model, model_key)
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
        grayscale_cam = cam(
            input_tensor=tensor,
            targets=[ClassifierOutputTarget(class_idx)],
        )[0]

        img_resized = np.array(
            img_pil.resize((input_size, input_size), Image.BICUBIC)
        ).astype(np.float32) / 255.0
        overlay = show_cam_on_image(img_resized, grayscale_cam, use_rgb=True)
        return overlay, None
    except Exception as e:
        return None, str(e)


# ── Feature / inference helpers ────────────────────────────────────────────────

def extract_ml_features(img_path: Path) -> np.ndarray:
    """92 features identiques à l'entraînement SVM : RGB/HSV/LAB mean+std (18) + histos (64) + LBP (10)."""
    N_BINS   = 16
    LBP_BINS = 10
    img_rgb  = np.array(Image.open(img_path).convert("RGB").resize(IMG_SIZE))
    feats: list[float] = list(img_rgb.mean(axis=(0, 1))) + list(img_rgb.std(axis=(0, 1)))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(float)
    feats += list(hsv.mean(axis=(0, 1))) + list(hsv.std(axis=(0, 1)))
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(float)
    feats += list(lab.mean(axis=(0, 1))) + list(lab.std(axis=(0, 1)))
    for c in range(3):
        h, _ = np.histogram(img_rgb[:, :, c], bins=N_BINS, range=(0, 256), density=True)
        feats += list(h)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h_gray, _ = np.histogram(gray, bins=N_BINS, range=(0, 256), density=True)
    feats += list(h_gray)
    try:
        from skimage.feature import local_binary_pattern
        lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
        h_lbp, _ = np.histogram(lbp, bins=LBP_BINS, range=(0, LBP_BINS), density=True)
        feats += list(h_lbp)
    except Exception:
        feats += [0.0] * LBP_BINS
    return np.array(feats, dtype=np.float32)


def _infer_one(model, tensor) -> np.ndarray:
    import torch
    with torch.no_grad():
        return torch.softmax(model(tensor), dim=1)[0].cpu().numpy()


def predict_dl(models, img_path: Path, input_size: int):
    """
    Inférence DL — accepte un modèle unique ou une liste (ensemble 5 folds).
    La proba finale est la moyenne sur tous les folds disponibles.
    """
    from torchvision import transforms
    import torch
    tf = transforms.Compose([
        transforms.Resize((input_size, input_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tensor = tf(Image.open(img_path).convert("RGB")).unsqueeze(0)

    model_list = models if isinstance(models, list) else [models]
    probas = [_infer_one(m, tensor) for m in model_list]
    proba  = np.mean(probas, axis=0)   # moyenne des folds

    n           = len(proba)
    class_names = DL_CLASS_NAMES[:n]
    valid_mask  = np.array([c in DL_VALID_CLASSES for c in class_names])
    proba_valid = proba * valid_mask
    total = proba_valid.sum()
    if total > 0:
        proba_valid /= total
    idx = int(proba_valid.argmax())
    return class_names[idx], float(proba_valid[idx]), proba_valid, class_names


def run_cellpose_for(img_path: Path, class_name: str | None):
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    item    = {"image": img_rgb.astype(np.float32) / 255.0,
               "class_name": class_name, "image_name": img_path.name, "path": str(img_path)}
    r       = morphology_cellpose_v2([item], target_size=(360, 360)).run()[0]

    vis     = cv2.resize(img_rgb, (360, 360)).copy()
    mask    = r.get("mask")
    contour = r.get("contour")
    if mask is not None and mask.any():
        green = np.zeros_like(vis)
        green[mask > 0] = [0, 200, 80]
        vis = cv2.addWeighted(vis, 0.75, green, 0.25, 0)
    if contour is not None:
        cv2.drawContours(vis, [contour], -1, (255, 80, 0), 2)
    return cv2.resize(img_rgb, (360, 360)), vis, r


# ── Folder picker (macOS osascript — pas besoin de tkinter) ───────────────────

def _pick_folder(initial_dir: str = "") -> str:
    import platform
    if platform.system() == "Darwin":
        import subprocess
        if initial_dir and Path(initial_dir).exists():
            loc = f' default location POSIX file "{initial_dir}"'
        else:
            loc = ""
        script = (f'tell app "Finder" to set f to '
                  f'(choose folder with prompt "Choisir un dossier"{loc})\n'
                  f'return POSIX path of f')
        try:
            r = subprocess.run(["osascript", "-e", script],
                               capture_output=True, text=True, timeout=60)
            return r.stdout.strip()
        except Exception:
            return ""
    else:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.wm_attributes("-topmost", True)
            folder = filedialog.askdirectory(
                title="Choisir un dossier",
                initialdir=initial_dir if initial_dir and Path(initial_dir).exists() else str(Path.home()),
            )
            root.destroy()
            return folder
        except Exception:
            return ""


def _pick_image_file(initial_dir: str = "") -> str:
    import platform
    if platform.system() == "Darwin":
        import subprocess
        script = ('tell app "Finder" to set f to '
                  '(choose file with prompt "Choisir une image")\n'
                  'return POSIX path of f')
        try:
            r = subprocess.run(["osascript", "-e", script],
                               capture_output=True, text=True, timeout=60)
            return r.stdout.strip()
        except Exception:
            return ""
    else:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.wm_attributes("-topmost", True)
            file = filedialog.askopenfilename(
                title="Choisir une image",
                initialdir=initial_dir or str(Path.home()),
                filetypes=[("Images", "*.jpg *.jpeg *.png *.tiff *.bmp"), ("Tous", "*.*")],
            )
            root.destroy()
            return file
        except Exception:
            return ""


def _pick_ml_file(initial_dir: str = "") -> str:
    import platform
    if platform.system() == "Darwin":
        import subprocess
        if initial_dir and Path(initial_dir).exists():
            loc = f' default location POSIX file "{initial_dir}"'
        else:
            loc = ""
        script = (f'tell app "Finder" to set f to '
                  f'(choose file with prompt "Choisir le fichier .pkl"{loc} of type {{"pkl"}})\n'
                  f'return POSIX path of f')
        try:
            r = subprocess.run(["osascript", "-e", script],
                               capture_output=True, text=True, timeout=60)
            return r.stdout.strip()
        except Exception:
            return ""
    else:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.wm_attributes("-topmost", True)
            file = filedialog.askopenfilename(
                title="Choisir le fichier .pkl",
                initialdir=initial_dir if initial_dir and Path(initial_dir).exists() else str(Path.home()),
                filetypes=[("Pickle", "*.pkl"), ("Tous", "*.*")],
            )
            root.destroy()
            return file
        except Exception:
            return ""


# ── Batch processing ───────────────────────────────────────────────────────────

def process_folder(images: list[Path], clf, scaler, dl_models: dict) -> list[dict]:
    rows = []
    bar  = st.progress(0, text="Initialisation…")

    for i, img_path in enumerate(images):
        bar.progress((i + 1) / len(images), text=f"{img_path.name}  ({i + 1}/{len(images)})")
        row: dict = {"path": img_path, "name": img_path.name}

        # ML SVC
        if clf is not None and scaler is not None:
            try:
                feats          = extract_ml_features(img_path).reshape(1, -1)
                feats_sc       = scaler.transform(feats)
                raw_pred = clf.predict(feats_sc)[0]
                proba    = clf.predict_proba(feats_sc)[0]
                # SVC entraine sur indices entiers -> convertit en nom de classe
                if isinstance(raw_pred, (int, np.integer, float)):
                    row["ml_pred"]    = DL_CLASS_NAMES[int(raw_pred)]
                    row["ml_classes"] = DL_CLASS_NAMES
                else:
                    row["ml_pred"]    = str(raw_pred)
                    row["ml_classes"] = list(clf.classes_)
                row["ml_conf"]  = float(proba.max())
                row["ml_proba"] = proba.tolist()
            except Exception as exc:
                row["ml_pred"] = f"err: {exc}"
                row["ml_conf"] = 0.0
        else:
            row["ml_pred"] = "N/A"
            row["ml_conf"] = 0.0

        # DL models
        for key, model in dl_models.items():
            col = key.replace("-", "_").replace(" ", "_")
            if model is None:
                row[f"{col}_pred"]  = "N/A"
                row[f"{col}_conf"]  = 0.0
                row[f"{col}_proba"] = None
            else:
                try:
                    pred, conf, proba, cls_names = predict_dl(model, img_path, DL_MODELS_CFG[key]["input_size"])
                    row[f"{col}_pred"]       = pred
                    row[f"{col}_conf"]       = conf
                    row[f"{col}_proba"]      = proba.tolist()
                    row[f"{col}_cls_names"]  = cls_names
                except Exception as exc:
                    row[f"{col}_pred"]  = f"err: {exc}"
                    row[f"{col}_conf"]  = 0.0
                    row[f"{col}_proba"] = None

        rows.append(row)

    bar.empty()
    return rows


# ── Detail panel ───────────────────────────────────────────────────────────────

def render_detail(row: dict, dl_models: dict,
                  use_cellpose: bool = False, gradcam_model_key: str | None = None):
    st.markdown("---")
    st.subheader(f"Détail — {row['name']}")

    img_path = row["path"]
    orig     = np.array(Image.open(img_path).convert("RGB"))
    overlay  = None
    morph    = {}
    cp_class = None

    if use_cellpose and CELLPOSE_AVAILABLE:
        candidates = []
        if row.get("ml_pred") not in (None, "N/A") and not str(row["ml_pred"]).startswith("err"):
            candidates.append((row["ml_conf"], row["ml_pred"]))
        for key in dl_models:
            col  = key.replace("-", "_").replace(" ", "_")
            pred = row.get(f"{col}_pred")
            conf = row.get(f"{col}_conf", 0.0)
            if pred and pred not in ("N/A",) and not str(pred).startswith("err"):
                candidates.append((conf, pred))
        cp_class = max(candidates, key=lambda x: x[0])[1] if candidates else None
        if cp_class and cp_class not in CLASS_CONFIGS:
            cp_class = None

        cache_key = f"cp_{img_path}"
        if cache_key not in st.session_state:
            with st.spinner("Segmentation Cellpose…"):
                try:
                    orig, overlay, morph = run_cellpose_for(img_path, cp_class)
                    st.session_state[cache_key] = (orig, overlay, morph)
                except Exception as exc:
                    st.error(f"Cellpose : {exc}")
                    st.session_state[cache_key] = (orig, None, {})
        orig, overlay, morph = st.session_state[cache_key]

    # ── GradCAM ───────────────────────────────────────────────────────────────
    gradcam_img = None
    gradcam_err = None
    if gradcam_model_key:
        if gradcam_model_key in dl_models and dl_models[gradcam_model_key]:
            gc_cache_key = f"gradcam_{img_path}_{gradcam_model_key}"
            if gc_cache_key not in st.session_state:
                with st.spinner(f"GradCAM++ {gradcam_model_key}…"):
                    model_for_gc = dl_models[gradcam_model_key]
                    if isinstance(model_for_gc, list):
                        model_for_gc = model_for_gc[0]
                    col_slug = gradcam_model_key.replace("-", "_").replace(" ", "_")
                    pred_name = row.get(f"{col_slug}_pred")
                    cls_names = row.get(f"{col_slug}_cls_names", DL_CLASS_NAMES)
                    if pred_name and pred_name in cls_names:
                        class_idx = cls_names.index(pred_name)
                        gc, err = compute_gradcam(
                            model_for_gc, img_path,
                            DL_MODELS_CFG[gradcam_model_key]["input_size"],
                            gradcam_model_key, class_idx,
                        )
                        st.session_state[gc_cache_key] = (gc, err)
                    else:
                        st.session_state[gc_cache_key] = (None, "Pas de prédiction DL disponible")
            gradcam_img, gradcam_err = st.session_state[gc_cache_key]
        else:
            gradcam_err = f"Modèle {gradcam_model_key} non chargé"

    # ── Ligne images ───────────────────────────────────────────────────────────
    show_seg = use_cellpose and CELLPOSE_AVAILABLE
    show_gc  = gradcam_model_key is not None

    img_cols_n = 1 + int(show_seg) + int(show_gc)
    img_cols = st.columns([2, 1]) if img_cols_n == 1 else st.columns(img_cols_n)
    col_idx  = 0

    with img_cols[col_idx]:
        st.caption("Image originale")
        if orig is not None:
            st.image(orig, use_container_width=True)
    col_idx += 1

    if show_seg:
        with img_cols[col_idx]:
            st.caption(f"Cellpose — classe : **{cp_class or 'auto'}**")
            if overlay is not None:
                area    = morph.get("area_px")
                caption = f"Aire : {area:.0f} px²" if area else "Aucune cellule détectée"
                st.image(overlay, caption=caption, use_container_width=True)
            else:
                st.info("Segmentation indisponible")
        col_idx += 1

    if show_gc:
        with img_cols[col_idx]:
            col_slug   = gradcam_model_key.replace("-", "_").replace(" ", "_")
            pred_label = row.get(f"{col_slug}_pred", "?")
            st.caption(f"GradCAM++ — **{gradcam_model_key}** → `{pred_label}`")
            if gradcam_img is not None:
                st.image(gradcam_img, use_container_width=True)
            elif gradcam_err:
                st.warning(f"⚠️ {gradcam_err}")
            else:
                st.info("GradCAM++ indisponible")

    # ── Ligne résultats : 5 colonnes ──────────────────────────────────────────
    def _render_model_col(col, name: str, color: str, pred: str, conf: float,
                          proba, cls_names):
        with col:
            if pred not in ("N/A",) and not str(pred).startswith("err"):
                conf_color = "#ff6b6b" if conf < 0.75 else "white"
                st.markdown(
                    f'<div style="background:{color};color:white;padding:8px 10px;'
                    f'border-radius:8px 8px 0 0;font-weight:600;font-size:0.85em;">'
                    f'{name}<br>'
                    f'<code style="background:rgba(255,255,255,0.25);color:white;'
                    f'padding:1px 5px;border-radius:3px">{pred}</code>'
                    f'&nbsp;<span style="color:{conf_color}">{conf*100:.1f}%</span></div>',
                    unsafe_allow_html=True,
                )
                with st.container(border=True):
                    if proba:
                        pairs = sorted(zip(cls_names, proba), key=lambda x: -x[1])
                        for cls, p in pairs[:4]:
                            st.progress(float(p), text=f"{cls}  {p*100:.1f}%")
            else:
                st.markdown(
                    f'<div style="background:{color};color:white;padding:8px 10px;'
                    f'border-radius:8px 8px 0 0;font-weight:600;font-size:0.85em;">'
                    f'{name}</div>',
                    unsafe_allow_html=True,
                )
                with st.container(border=True):
                    st.caption(f"{pred}")

    res_cols = st.columns(5)

    # ML SVC
    _render_model_col(
        res_cols[0], "ML SVC",
        MODEL_COLORS.get("ML SVC", "#378ADD"),
        row.get("ml_pred", "N/A"), row.get("ml_conf", 0.0),
        row.get("ml_proba"), row.get("ml_classes", DL_CLASS_NAMES),
    )

    # 4 DL models
    for i, key in enumerate(dl_models, start=1):
        col_k = key.replace("-", "_").replace(" ", "_")
        _render_model_col(
            res_cols[i], key,
            MODEL_COLORS.get(key, "#888888"),
            row.get(f"{col_k}_pred", "N/A"), row.get(f"{col_k}_conf", 0.0),
            row.get(f"{col_k}_proba"), row.get(f"{col_k}_cls_names", DL_CLASS_NAMES),
        )


# ── Single image render ────────────────────────────────────────────────────────

def render_single_image(source, clf, scaler, dl_models: dict,
                        use_cellpose: bool = False,
                        gradcam_model_key: str | None = None):
    """source : Path (image locale) ou UploadedFile (Streamlit)."""
    import tempfile
    is_path  = isinstance(source, Path)
    img      = Image.open(source).convert("RGB")
    img_name = source.name

    if is_path:
        tmp_path = source
        cleanup  = False
    else:
        suffix = Path(source.name).suffix or ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(source.getvalue())
            tmp_path = Path(tmp.name)
        cleanup = True

    try:
        # ── Inférence ML+DL (cachée en session_state par image) ───────────────
        infer_key = f"infer_{tmp_path}"
        if infer_key not in st.session_state:
            all_preds: dict = {}
            errors:    dict = {}

            with st.spinner("Analyse en cours…"):
                if clf is not None and scaler is not None:
                    try:
                        feats    = extract_ml_features(tmp_path).reshape(1, -1)
                        feats_sc = scaler.transform(feats)
                        raw_pred = clf.predict(feats_sc)[0]
                        proba    = clf.predict_proba(feats_sc)[0]
                        if isinstance(raw_pred, (int, np.integer, float)):
                            pred_class = DL_CLASS_NAMES[int(raw_pred)]
                            classes    = DL_CLASS_NAMES
                        else:
                            pred_class = str(raw_pred)
                            classes    = list(clf.classes_)
                        all_preds["ML SVC"] = {"pred": pred_class, "conf": float(proba.max()),
                                               "proba": proba, "classes": classes}
                    except Exception as exc:
                        errors["ML SVC"] = str(exc)

                for key, model in dl_models.items():
                    if model is None:
                        continue
                    try:
                        pred, conf, proba, cls_names = predict_dl(
                            model, tmp_path, DL_MODELS_CFG[key]["input_size"])
                        all_preds[key] = {"pred": pred, "conf": conf,
                                          "proba": proba, "classes": cls_names}
                    except Exception as exc:
                        errors[key] = str(exc)

            st.session_state[infer_key] = (all_preds, errors)

        all_preds, errors = st.session_state[infer_key]

        # ── Cellpose live ──────────────────────────────────────────────────────
        orig     = np.array(img)
        overlay  = None
        morph    = {}
        cp_class = None

        if use_cellpose and CELLPOSE_AVAILABLE:
            candidates = [(d["conf"], d["pred"]) for d in all_preds.values()]
            if candidates:
                cp_class = max(candidates, key=lambda x: x[0])[1]
                if cp_class and cp_class not in CLASS_CONFIGS:
                    cp_class = None

            cache_key = f"cp_single_{tmp_path}"
            if cache_key not in st.session_state:
                with st.spinner("Segmentation Cellpose…"):
                    try:
                        o, ov, m = run_cellpose_for(tmp_path, cp_class)
                        st.session_state[cache_key] = (o, ov, m)
                    except Exception as exc:
                        st.warning(f"Cellpose : {exc}")
                        st.session_state[cache_key] = (orig, None, {})
            orig, overlay, morph = st.session_state[cache_key]

        # ── GradCAM ───────────────────────────────────────────────────────────
        gradcam_img = None
        gradcam_err = None
        if gradcam_model_key:
            if gradcam_model_key in dl_models and dl_models[gradcam_model_key]:
                gc_cache_key = f"gradcam_{tmp_path}_{gradcam_model_key}"
                if gc_cache_key not in st.session_state:
                    with st.spinner(f"GradCAM++ {gradcam_model_key}…"):
                        model_for_gc = dl_models[gradcam_model_key]
                        if isinstance(model_for_gc, list):
                            model_for_gc = model_for_gc[0]
                        pred_data = all_preds.get(gradcam_model_key)
                        if pred_data:
                            class_idx = pred_data["classes"].index(pred_data["pred"])
                            gc, err = compute_gradcam(
                                model_for_gc, tmp_path,
                                DL_MODELS_CFG[gradcam_model_key]["input_size"],
                                gradcam_model_key, class_idx,
                            )
                            st.session_state[gc_cache_key] = (gc, err)
                        else:
                            st.session_state[gc_cache_key] = (None, "Pas de prédiction DL disponible")
                gradcam_img, gradcam_err = st.session_state[gc_cache_key]
            else:
                gradcam_err = f"Modèle {gradcam_model_key} non chargé"

        # ── Ligne images ──────────────────────────────────────────────────────
        show_gc  = gradcam_model_key is not None
        show_seg = use_cellpose and CELLPOSE_AVAILABLE

        img_cols_n = 1 + int(show_seg) + int(show_gc)
        img_cols   = st.columns([2, 1]) if img_cols_n == 1 else st.columns(img_cols_n)
        col_idx    = 0

        with img_cols[col_idx]:
            st.caption(f"**{img_name}**")
            st.image(orig, use_container_width=True)
            st.caption(f"{img.size[0]}×{img.size[1]} px")
        col_idx += 1

        if show_seg:
            with img_cols[col_idx]:
                st.caption(f"Cellpose — classe : **{cp_class or 'auto'}**")
                if overlay is not None:
                    area    = morph.get("area_px")
                    caption = f"Aire : {area:.0f} px²" if area else "Aucune cellule détectée"
                    st.image(overlay, caption=caption, use_container_width=True)
                else:
                    st.info("Segmentation indisponible")
            col_idx += 1

        if show_gc:
            with img_cols[col_idx]:
                pred_label = all_preds.get(gradcam_model_key, {}).get("pred", "?")
                st.caption(f"GradCAM++ — **{gradcam_model_key}** → `{pred_label}`")
                if gradcam_img is not None:
                    st.image(gradcam_img, use_container_width=True)
                elif gradcam_err:
                    st.warning(f"⚠️ {gradcam_err}")
                else:
                    st.info("GradCAM++ indisponible")

        # ── Ligne résultats : 5 colonnes ──────────────────────────────────────
        if errors:
            for n, err in errors.items():
                st.warning(f"⚠️ {n} : {err}")
        if not all_preds:
            st.error("Aucun modèle n'a pu produire de prédiction.")
        else:
            res_cols      = st.columns(5)
            all_model_names = ["ML SVC"] + list(dl_models.keys())
            for i, model_name in enumerate(all_model_names):
                color = MODEL_COLORS.get(model_name, "#888888")
                data  = all_preds.get(model_name)
                with res_cols[i]:
                    if data:
                        pred, conf = data["pred"], data["conf"]
                        conf_color = "#ff6b6b" if conf < 0.75 else "white"
                        st.markdown(
                            f'<div style="background:{color};color:white;padding:8px 10px;'
                            f'border-radius:8px 8px 0 0;font-weight:600;font-size:0.85em;">'
                            f'{model_name}<br>'
                            f'<code style="background:rgba(255,255,255,0.25);color:white;'
                            f'padding:1px 5px;border-radius:3px">{pred}</code>'
                            f'&nbsp;<span style="color:{conf_color}">{conf*100:.1f}%</span></div>',
                            unsafe_allow_html=True,
                        )
                        with st.container(border=True):
                            pairs = sorted(zip(data["classes"], data["proba"]),
                                           key=lambda x: -x[1])
                            for cls, p in pairs[:4]:
                                st.progress(float(p), text=f"{cls}  {p*100:.1f}%")
                    else:
                        st.markdown(
                            f'<div style="background:{color};color:white;padding:8px 10px;'
                            f'border-radius:8px 8px 0 0;font-weight:600;font-size:0.85em;">'
                            f'{model_name}</div>',
                            unsafe_allow_html=True,
                        )
                        with st.container(border=True):
                            st.caption("Indisponible")

    finally:
        if cleanup:
            tmp_path.unlink(missing_ok=True)


# ── UI ─────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Blood Cell — Analyse", layout="wide")
st.title("Blood Cell — Classification")
st.caption("ML SVC + 4 modèles DL — ensemble 5 folds (moyenne des probabilités)")

use_folds_choice = st.radio(
    "Modèles DL",
    ["⚡ Fold 1 (rapide)", "🎯 5 folds (précis)"],
    horizontal=True,
)
use_folds = "5 folds" in use_folds_choice

use_cellpose = CELLPOSE_AVAILABLE and st.radio(
    "Segmentation Cellpose",
    ["Non", "Oui"],
    horizontal=True,
) == "Oui"

use_gradcam = st.radio("GradCAM++", ["Non", "Oui"], horizontal=True) == "Oui"
if use_gradcam:
    gradcam_model_key = st.selectbox(
        "Modèle GradCAM++",
        list(DL_MODELS_CFG.keys()),
        index=0,
    )
else:
    gradcam_model_key = None

mode = st.radio("Mode d'analyse", ["🖼️ Image unique", "📁 Dossier"], horizontal=True)

def _clear_infer_cache():
    """Supprime tous les résultats d'inférence mis en cache (infer_*, gradcam_*, results)."""
    keys = [k for k in st.session_state if k.startswith(("infer_", "gradcam_"))]
    for k in keys:
        del st.session_state[k]
    st.session_state.pop("results", None)


# ── Sidebar : configuration ────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    if not CELLPOSE_AVAILABLE:
        st.warning("⚠️ Cellpose non disponible — segmentation désactivée")

    # ML model path — clé unique = "ml_bundle_path" dans session_state
    st.markdown("**Modèle ML**")
    if "ml_bundle_path" not in st.session_state:
        st.session_state.ml_bundle_path = str(ML_BUNDLE_PATH)
    col_ml_btn, col_ml_path = st.columns([1, 3])
    with col_ml_btn:
        if st.button("📂", key="pick_ml", help="Choisir le fichier best_ml_model.pkl"):
            current = Path(st.session_state.ml_bundle_path).parent
            picked = _pick_ml_file(initial_dir=str(current))
            if picked:
                st.session_state.ml_bundle_path = picked
                _clear_infer_cache()
                st.rerun()
    with col_ml_path:
        typed = st.text_input(
            "best_ml_model.pkl", value=st.session_state.ml_bundle_path,
            label_visibility="collapsed",
        )
        if typed != st.session_state.ml_bundle_path:
            st.session_state.ml_bundle_path = typed
            _clear_infer_cache()
            st.rerun()

    # DL crossval path — clé unique = "crossval_dir" dans session_state
    st.markdown("**Modèles DL (cross-val)**")
    if "crossval_dir" not in st.session_state:
        st.session_state.crossval_dir = str(DEFAULT_CROSSVAL_DIR)
    col_dl_btn, col_dl_path = st.columns([1, 3])
    with col_dl_btn:
        if st.button("📂", key="pick_dl", help="Choisir le dossier des folds DL"):
            folder = _pick_folder(initial_dir=st.session_state.crossval_dir)
            if folder:
                st.session_state.crossval_dir = folder
                _clear_infer_cache()
                st.rerun()
    with col_dl_path:
        typed = st.text_input(
            "Dossier cross-val", value=st.session_state.crossval_dir,
            label_visibility="collapsed",
            help="Contient fold_1/.../fold_5/ avec best_fold{N}_*.pth",
        )
        if typed != st.session_state.crossval_dir:
            st.session_state.crossval_dir = typed
            _clear_infer_cache()
            st.rerun()

crossval_dir = st.session_state.crossval_dir

# ── Load models (once) ─────────────────────────────────────────────────────────
clf, scaler, ml_classes, ml_err = load_ml_artifacts(st.session_state.ml_bundle_path)
dl_models   = {}   # key → model ou liste de modèles
fold_status = {}   # key → (found_list, missing_list)

for key in DL_MODELS_CFG:
    n = N_FOLDS if use_folds else 1
    models, found, missing = load_dl_fold_ensemble(key, crossval_dir, n)
    dl_models[key]   = models if models else None
    fold_status[key] = (found, missing)

# ── Model status indicators ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**Statut des modèles**")
    _ml_path_ok = Path(st.session_state.ml_bundle_path).exists()
    if _ml_path_ok and clf:
        st.write("ML SVC :", "✅")
    elif not _ml_path_ok:
        st.write("ML SVC :", f"❌ {Path(st.session_state.ml_bundle_path).name} introuvable")
    else:
        st.write("ML SVC :", f"❌ {ml_err}")
    for key in DL_MODELS_CFG:
        found, missing = fold_status[key]
        if use_folds:
            label = f"{key} ({len(found)}/{N_FOLDS} folds)"
            st.write(label, "✅" if found else "❌")
            if missing:
                st.caption(f"  Manquants : {', '.join(missing)}")
        else:
            st.write(f"{key} :", "✅" if dl_models[key] else "❌ introuvable")

# ── Mode : Image unique ────────────────────────────────────────────────────────
if "🖼️" in mode:
    _IMG_DEFAULT_FOLDER = Path("/Users/fredericdelabot/Library/CloudStorage/OneDrive-Personnel/BloodCellCaches/ImgsDemo")

    if "image_path" not in st.session_state:
        st.session_state["image_path"] = ""

    col_btn, col_path = st.columns([1, 5])
    with col_btn:
        if st.button("🖼️ Choisir image", use_container_width=True):
            picked = _pick_image_file(initial_dir=str(_IMG_DEFAULT_FOLDER))
            if picked and picked != st.session_state["image_path"]:
                # vider les caches avec la clé normalisée (Path → str)
                old_path = Path(st.session_state["image_path"]) if st.session_state["image_path"] else None
                if old_path:
                    st.session_state.pop(f"infer_{old_path}", None)
                    st.session_state.pop(f"cp_single_{old_path}", None)
                st.session_state["image_path"] = picked
                st.rerun()
    with col_path:
        st.text_input("Chemin de l'image", key="image_path",
                      label_visibility="collapsed",
                      placeholder="/chemin/vers/image.jpg")

    image_path = Path(st.session_state["image_path"]) if st.session_state["image_path"] else None

    if image_path and image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
        dl_models_single = {
            key: (models[0] if isinstance(models, list) and models else models)
            for key, models in dl_models.items()
        }
        infer_key = f"infer_{image_path}"
        cp_key    = f"cp_single_{image_path}"
        gc_key = f"gradcam_{image_path}_{gradcam_model_key}" if gradcam_model_key else None
        needs_compute = (
            infer_key not in st.session_state
            or (use_cellpose and CELLPOSE_AVAILABLE and cp_key not in st.session_state)
            or (gc_key and gc_key not in st.session_state)
        )
        result_slot = st.empty()
        if needs_compute:
            result_slot.empty()   # efface les anciens résultats pendant le calcul
        with result_slot.container():
            render_single_image(image_path, clf, scaler, dl_models_single,
                                use_cellpose, gradcam_model_key)
    elif st.session_state["image_path"]:
        st.error("Fichier introuvable ou format non supporté.")
    else:
        st.info("⬆️ Choisissez une image pour lancer la classification.")

# ── Mode : Dossier ─────────────────────────────────────────────────────────────
else:
    _DEFAULT_FOLDER = Path("/Users/fredericdelabot/Library/CloudStorage/OneDrive-Personnel/BloodCellCaches/ImgsDemo")
    if "folder_path" not in st.session_state:
        st.session_state.folder_path = str(_DEFAULT_FOLDER) if _DEFAULT_FOLDER.exists() else ""

    col_btn, col_path = st.columns([1, 5])
    with col_btn:
        if st.button("📁 Choisir dossier", use_container_width=True):
            picked = _pick_folder()
            if picked:
                st.session_state.folder_path = picked
                st.session_state.pop("results", None)
                st.rerun()

    with col_path:
        typed = st.text_input("Chemin du dossier", value=st.session_state.folder_path,
                              label_visibility="collapsed",
                              placeholder="/chemin/vers/dossier/images")
        if typed != st.session_state.folder_path:
            st.session_state.folder_path = typed
            st.session_state.pop("results", None)

    folder = Path(st.session_state.folder_path) if st.session_state.folder_path else None

    if folder and folder.is_dir():
        images = sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS])
        st.caption(f"{len(images)} image(s) trouvée(s) dans `{folder}`")

        find_btn = st.button("🔍 Find Classes", type="primary",
                             disabled=len(images) == 0, use_container_width=False)

        if find_btn:
            st.session_state.pop("results", None)
            with st.spinner("Analyse en cours — ML + DL pour chaque image…"):
                st.session_state.results = process_folder(images, clf, scaler, dl_models)
            st.rerun()

    elif st.session_state.folder_path:
        st.error("Dossier introuvable.")

    if "results" in st.session_state and st.session_state.results:
        results = st.session_state.results
        st.markdown("---")

        import pandas as pd

        def fmt(pred, conf):
            if pred in ("N/A",) or str(pred).startswith("err"):
                return pred
            return f"{pred}  ({conf*100:.0f}%)"

        table_rows = []
        for r in results:
            row_dict = {
                "Fichier": r["name"],
                "ML SVC":  fmt(r.get("ml_pred", "N/A"), r.get("ml_conf", 0)),
            }
            for key in dl_models:
                col       = key.replace("-", "_").replace(" ", "_")
                n_found   = len(fold_status[key][0])
                col_label = f"{key} ({n_found}f)" if use_folds and n_found > 1 else key
                row_dict[col_label] = fmt(r.get(f"{col}_pred", "N/A"), r.get(f"{col}_conf", 0))
            table_rows.append(row_dict)

        df = pd.DataFrame(table_rows)

        col_title, col_export = st.columns([4, 1])
        with col_title:
            st.subheader(f"Résultats — {len(results)} images")
        with col_export:
            def _build_export_df(results, dl_models, fold_status, use_folds):
                rows = []
                for r in results:
                    row = {"Fichier": r["name"],
                           "ML_pred": r.get("ml_pred", "N/A"),
                           "ML_conf": round(r.get("ml_conf", 0) * 100, 1)}
                    for key in dl_models:
                        col_k   = key.replace("-", "_").replace(" ", "_")
                        n_found = len(fold_status[key][0])
                        label   = f"{key}_{n_found}f" if use_folds and n_found > 1 else key
                        row[f"{label}_pred"] = r.get(f"{col_k}_pred", "N/A")
                        row[f"{label}_conf"] = round(r.get(f"{col_k}_conf", 0) * 100, 1)
                    rows.append(row)
                return pd.DataFrame(rows)

            export_df = _build_export_df(results, dl_models, fold_status, use_folds)
            st.download_button(
                label="⬇️ Export CSV",
                data=export_df.to_csv(index=False).encode("utf-8"),
                file_name=f"predictions_{Path(st.session_state.folder_path).name}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        event = st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )

        sel = event.selection.rows
        if sel:
            render_detail(results[sel[0]], dl_models, use_cellpose, gradcam_model_key)
        else:
            st.info("Clique sur une ligne pour afficher les détails et la segmentation Cellpose.")

    elif folder and folder.is_dir():
        st.info("Appuie sur **Find Classes** pour lancer l'analyse.")
