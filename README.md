# BloodCells — Classification de cellules sanguines

Projet de Data Science appliqué à la classification automatique de cellules sanguines
par Machine Learning et Deep Learning, dans un contexte clinique d'aide au diagnostic.

**Dataset** : [Mendeley PBC](https://data.mendeley.com/datasets/snkd93bnjr/1) — 17 092 images, 8 classes  
**Équipe** : Sara, Frédéric, Romane — promotion MAR26 BDS, DataScientest

---

## Classes prédites

| Classe | Description |
|---|---|
| `basophil` | Basophile |
| `eosinophil` | Éosinophile |
| `erythroblast` | Érythroblaste |
| `ig` | Granulocyte immature |
| `lymphocyte` | Lymphocyte |
| `monocyte` | Monocyte |
| `neutrophil` | Neutrophile |
| `platelet` | Plaquette |

---

## Structure du projet

```
├── app/
│   └── streamlit/          # Application de démo interactive
│       ├── app.py
│       └── pages/          # 6 pages (Contexte → Conclusion)
├── configs/
│   └── densenet121.yaml    # Config d'entraînement DenseNet-121
├── models/
│   └── densenet121_crossval_v1/  # Modèle déployé (cross-val 5-fold)
├── notebooks/
│   ├── 01_EDA/             # Exploration, flou, standardisation, taille cellules
│   ├── 02_Preprocessing/   # Segmentation (CV2, Cellpose), vérification exports
│   ├── 03_Machine_Learning/# LazyPredict, pipeline, évaluation, SHAP
│   └── 04_Deep_Learning/   # Pipeline DL, augmentation, hyperparamètres
├── reports/                # Rapports PDF/Word et figures par expérience
├── scripts/                # Scripts utilitaires (push GitHub, validation)
├── src/
│   ├── streamlit/          # App Streamlit (classification image unique + dossier)
│   ├── exports/            # Génération de rapports Word automatiques
│   ├── exploration/        # Scripts d'exploration EDA
│   ├── independent_scripts/# Scripts autonomes (EDA, standardisation)
│   └── jupyter/            # Scripts .py correspondant aux notebooks
├── tests/                  # Tests unitaires (dataset, transforms, predict)
├── Makefile                # Commandes raccourcies
└── requirements/           # Dépendances Python
    ├── base.txt            # Dépendances principales
    ├── dev.txt             # Dépendances de développement
    ├── api.txt             # Dépendances FastAPI
    └── streamlit.txt       # Dépendances Streamlit
```

---

## Installation

```bash
git clone https://github.com/DataScientest-Studio/MAR26-BDS-BLOODCELLS-1.git
cd MAR26-BDS-BLOODCELLS-1

python -m venv .venv
source .venv/bin/activate   # Windows : .venv\Scripts\activate

make install
# ou
pip install -r requirements/base.txt -r requirements/dev.txt
```

Copier le fichier d'environnement :
```bash
cp .env.example .env
```

---

## Lancer l'application Streamlit

```bash
make streamlit-new
# ou
streamlit run app/streamlit/app.py
```

L'application propose :
- Classification d'une image unique (ML + DL)
- Analyse par dossier avec GradCAM++
- Résultats des expériences (cross-validation, métriques, calibration)

---

## Modèles

| Approche | Modèle | Méthode | Métriques |
|---|---|---|---|
| Machine Learning | SVM RBF | 37 / 92 features morphologiques | — |
| Deep Learning | ConvNeXt-Tiny | Fine-tuning, augmentation | — |
| Deep Learning | **DenseNet-121** ✅ | Cross-val 5-fold, 20 epochs | Modèle déployé |
| Deep Learning | EfficientNet-B3 | Fine-tuning | — |
| Deep Learning | ResNet-50 | Fine-tuning | — |

> Le modèle déployé est **DenseNet-121** (cross-validation 5-fold, dataset complet 17 092 images).

---

## Tests

```bash
make test
# ou
pytest tests/ -v
```

---

## Suivi des expériences

Les expériences DL sont tracées avec **MLflow** :

```bash
mlflow ui
```

---

## Principales dépendances

- `torch` / `torchvision` / `timm` — Deep Learning
- `scikit-learn` / `xgboost` / `lightgbm` — Machine Learning
- `streamlit` — Application interactive
- `grad-cam` — Visualisation GradCAM++
- `cellpose` — Segmentation cellulaire
- `mlflow` — Suivi des expériences
- `shap` — Explicabilité ML
