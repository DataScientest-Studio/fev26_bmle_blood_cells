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

## Architecture

Le projet repose sur une stack conteneurisée (Docker) avec plusieurs services interconnectés :

| Service | Rôle | Port |
|---|---|---|
| **Streamlit** | Interface utilisateur (classification, monitoring, recherche) | 8501 |
| **FastAPI** | Backend d'inférence (predict, GradCAM++, feedback) | 8000 |
| **MLflow** | Suivi des expériences et registre de modèles | 5001 |
| **Airflow** | Orchestration des pipelines ML (entraînement, fine-tuning) | 8080 |
| **Supabase / PostgreSQL** | Stockage des prédictions et logs | — |

---

## Structure du projet

```
├── airflow/
│   └── dags/                   # DAGs Airflow
│       ├── blood_cell_pipeline.py                       # Pipeline entraînement complet
│       ├── blood_cell_incremental_finetune_pipeline.py  # Fine-tuning incrémental
│       └── blood_cell_drift_monitoring.py               # Monitoring drift quotidien + alertes email
├── configs/
│   └── densenet121.yaml        # Config d'entraînement DenseNet-121
├── docker/
│   ├── docker-compose.dev.yml  # Stack complète (Streamlit, API, MLflow, preprocessing, training)
│   ├── api/                    # Dockerfile FastAPI
│   ├── mlflow/                 # Dockerfile MLflow
│   ├── preprocessing/          # Dockerfile preprocessing
│   ├── streamlit/              # Dockerfile Streamlit
│   └── training/               # Dockerfile entraînement
├── models/
│   └── densenet121_crossval_v1/ # Modèle déployé (cross-val 5-fold)
├── notebooks/
│   ├── 01_EDA/                 # Exploration, flou, standardisation, taille cellules
│   ├── 02_Preprocessing/       # Segmentation (CV2, Cellpose), vérification exports
│   ├── 03_Machine_Learning/    # LazyPredict, pipeline, évaluation, SHAP
│   └── 04_Deep_Learning/       # Pipeline DL, augmentation, hyperparamètres
├── reports/                    # Rapports PDF/Word et figures par expérience
├── scripts/                    # Scripts utilitaires
├── src/
│   ├── serving/
│   │   ├── app.py              # Application Streamlit (Classification, Logs, Monitoring, Recherche, Drift)
│   │   ├── api.py              # API FastAPI (predict, gradcam, feedback, training)
│   │   └── batch_inference.py  # Inférence par lot
│   ├── auth/
│   │   ├── users.py            # Authentification utilisateurs
│   │   └── db.py               # Connexion base de données
│   ├── monitoring/
│   │   ├── email_alert.py      # Alertes email automatiques
│   │   ├── resource_monitor.py # Monitoring ressources système
│   │   └── supabase_logger.py  # Logging vers Supabase/PostgreSQL
│   ├── evidently/
│   │   ├── drift_report.py             # Rapports de drift (IVDR 2017/746)
│   │   ├── export_reports.py           # Export des rapports HTML
│   │   ├── generate_showcase_report.py # Rapport HTML showcase (4 métriques clés)
│   │   └── reports_html/               # Rapports HTML générés
│   ├── evaluation/
│   │   ├── eval_best_models.py
│   │   └── eval_experiments.py
│   ├── train/
│   │   ├── dl_crossval_train.py    # Entraînement cross-validation
│   │   ├── incremental_finetune.py # Fine-tuning incrémental
│   │   └── training.py
│   ├── models/
│   │   └── predict_model.py
│   ├── data/                   # Chargement et transformation des données
│   ├── features/               # Extraction de features morphologiques
│   ├── mlflow/                 # Intégration MLflow
│   └── visualization/          # Visualisations (GradCAM++, métriques)
├── tests/                      # Tests unitaires (dataset, transforms, predict)
├── Makefile                    # Commandes raccourcies
└── requirements/               # Dépendances Python
    ├── base.txt
    ├── dev.txt
    ├── api.txt
    └── streamlit.txt
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

Copier le fichier d'environnement et renseigner les variables :
```bash
cp .env.example .env
```

Variables requises : `API_SECRET_KEY`, `SUPABASE_HOST`, `SUPABASE_PORT`, `SUPABASE_DB`, `SUPABASE_USER`, `SUPABASE_PASSWORD`, `DAGSHUB_USER`, `DAGSHUB_TOKEN`.

Pour les alertes email drift : `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `ALERT_EMAILS` (destinataires séparés par virgule).

---

## Lancer la stack (Docker)

```bash
cd docker
docker compose -f docker-compose.dev.yml up --build
```

Services disponibles :
- Streamlit : http://localhost:8501
- FastAPI (docs) : http://localhost:8000/docs
- MLflow : http://localhost:5001

---

## Lancer l'application seule (sans Docker)

```bash
make streamlit-new
# ou
streamlit run src/serving/app.py
```

L'API FastAPI doit être lancée séparément :
```bash
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload
```

---

## Application Streamlit

L'interface propose 5 onglets :

| Onglet | Fonctionnalité |
|---|---|
| **Classification** | Analyse image unique ou par lot (simulation patient), GradCAM++, feedback |
| **Logs** | Historique des entraînements MLflow |
| **Monitoring** | Rapports de monitoring du modèle |
| **Recherche** | Retrouver des prédictions par nom d'image ou par patient |
| **Drift** | Rapports de drift Evidently (conformité IVDR 2017/746) |

---

## Modèles

| Approche | Modèle | Méthode | Statut |
|---|---|---|---|
| Machine Learning | SVM RBF | 37 / 92 features morphologiques | — |
| Deep Learning | ConvNeXt-Tiny | Fine-tuning, augmentation | — |
| Deep Learning | **DenseNet-121** ✅ | Cross-val 5-fold, 20 epochs | Déployé |
| Deep Learning | EfficientNet-B3 | Fine-tuning | — |
| Deep Learning | ResNet-50 | Fine-tuning | — |

> Le modèle déployé est **DenseNet-121** (cross-validation 5-fold, dataset complet 17 092 images), enregistré dans le registre MLflow sous le nom `blood-cell-densenet121`.

---

## Monitoring Drift (Evidently)

Le monitoring post-marché repose sur **Evidently** et surveille 4 métriques clés, conformément à l'**IVDR 2017/746**, au **MDCG 2020-1** et à l'**ISO 14971**.

### Métriques surveillées

| # | Métrique | Signal détecté | Test statistique |
|---|---|---|---|
| 1 | **Data drift — features image** | Changement de microscope, colorant, protocole d'acquisition | Wasserstein / Jensen-Shannon |
| 2 | **Drift de distribution des classes** | Biais de sélection, changement épidémiologique, dérive comportement modèle | Jensen-Shannon divergence |
| 3 | **Drift de confidence** | Dégradation silencieuse (signal précoce avant erreur de classe) | Wasserstein distance |
| 4 | **Désaccord médecin** | Ground truth clinique réel — taux de désaccord sur feedbacks en base | Calcul direct |

**Features image surveillées (7)** : `mean_brightness`, `std_brightness`, `mean_r`, `mean_g`, `mean_b`, `image_width`, `image_height`  
**Référence** : 2 400 images `Source_full`  
**Classes critiques** : `Erythroblast`, `IG` (granulocyte immature)

### Seuils IVDR / ISO 14971

| Niveau | Score | Action |
|---|---|---|
| ✅ Normal | < 0.10 | Aucune action requise |
| ⚠️ Warning | 0.10 – 0.20 | Surveillance accrue |
| 🟠 Alerte | 0.20 – 0.30 | Investigation + envisager ré-entraînement (MDCG 2020-1) |
| 🔴 Critique | ≥ 0.30 | Action immédiate obligatoire (ISO 14971 §9) |

### Rapport HTML showcase

Le script `src/evidently/generate_showcase_report.py` produit un rapport HTML visuel (`reports_html/showcase_drift_report.html`) avec badges colorés, barres de score et légende IVDR — utilisable directement dans Streamlit via `st.components.v1.html()`.

```bash
python -m src.evidently.generate_showcase_report
```

### Alertes email automatiques

Le module `src/monitoring/email_alert.py` envoie un email HTML avec toutes les métriques dès que le niveau dépasse le seuil configuré (`min_level`, défaut : `warning`). Alerte additionnelle si le macro F1 baisse de plus de 5% entre générations.

---

## Pipelines Airflow

| DAG | Description |
|---|---|
| `blood_cell_pipeline` | Pipeline complet : preprocessing → entraînement → évaluation → enregistrement MLflow |
| `blood_cell_incremental_finetune_pipeline` | Fine-tuning incrémental automatique sur nouveaux lots de données |
| `blood_cell_drift_monitoring` | Rapport Evidently quotidien (7h) → vérification macro F1 → alerte email si drift détecté (IVDR 2017/746) |

```bash
# Lancer Airflow
cd airflow
docker compose -f docker-compose-airflow.yml up --build
```

Airflow UI : http://localhost:8080

---

## Tests

```bash
make test
# ou
pytest tests/ -v
```

---

## Suivi des expériences (MLflow)

```bash
mlflow ui
# ou via Docker : http://localhost:5001
```

---

## Principales dépendances

- `torch` / `torchvision` / `timm` — Deep Learning
- `fastapi` / `uvicorn` — API backend
- `scikit-learn` / `xgboost` / `lightgbm` — Machine Learning
- `streamlit` — Interface utilisateur
- `grad-cam` — Visualisation GradCAM++
- `cellpose` — Segmentation cellulaire
- `mlflow` — Suivi des expériences et registre de modèles
- `evidently` — Monitoring et détection de drift
- `apache-airflow` — Orchestration des pipelines
- `psycopg2` / `supabase` — Stockage des prédictions
- `shap` — Explicabilité ML
