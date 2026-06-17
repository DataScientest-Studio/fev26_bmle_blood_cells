# Rangement du dépôt — 17 juin 2026

**Auteure :** Romane  
**Date :** 17 juin 2026  
**Branche :** Romane → main

---

## Contexte

La racine du dépôt avait accumulé des fichiers éparpillés au fil des itérations.
Ce rangement vise à rendre la structure lisible et cohérente avant la Phase 2.

---

## Fichiers déplacés

### Fichiers DVC (`.dvc`)
| Avant | Après |
|---|---|
| `Source_100.dvc` | `data/Source_100.dvc` |
| `Source_full.dvc` | `data/Source_full.dvc` |
| `Models.dvc` | `models.dvc` |

Les pointeurs DVC sont regroupés avec ce qu'ils décrivent.

---

### Requirements Python
| Avant | Après |
|---|---|
| `requirements.txt` | `requirements/base.txt` |
| `requirements-api.txt` | `requirements/api.txt` |
| `requirements-dev.txt` | `requirements/dev.txt` |
| `requirements-streamlit.txt` | `requirements/streamlit.txt` |

Tous les fichiers de dépendances sont dans le dossier `requirements/`.

---

### Documentation
| Avant | Après |
|---|---|
| `model_card.md` | `docs/model_card.md` |
| `reports/Guide_Test_Phase1.md` | `docs/Guide_Test_Phase1.md` |

Les fichiers markdown de documentation rejoignent `docs/`.

---

### Dockerfiles et Docker Compose
| Avant | Après |
|---|---|
| `Dockerfile` | `docker/streamlit/Dockerfile` |
| `Dockerfile.api` | `docker/api/Dockerfile` |
| `Dockerfile.mlflow` | `docker/mlflow/Dockerfile` |
| `docker-compose.dev.yml` | `docker/docker-compose.dev.yml` |

Tous les fichiers Docker sont regroupés dans `docker/`, un sous-dossier par service.

> **Note :** `docker/mlflow/Dockerfile` a également été mis à jour avec le flag
> `--serve-artifacts` ajouté par Sara dans son commit Phase 2.

---

### Rapports Word (`.docx`)
Les rapports à la racine de `reports/` ont été déplacés dans les sous-dossiers par auteur.

**→ `reports/Sara/`**
- `Guide_Test_Phase1.docx`
- `MLOps_fichiers_transfert_DenseNet121.docx`
- `Phase1_etat_des_lieux.docx`
- `Guide_Test_Phase2.docx` *(commit Sara du 17/06)*
- `Guide_Test_Phases1_2.docx` *(commit Sara du 17/06)*
- `Phases1_2_etat_des_lieux.docx` *(commit Sara du 17/06)*

---

## Nouveaux dossiers créés

| Dossier | Contenu |
|---|---|
| `docker/` | Dockerfiles par service + docker-compose |
| `airflow/` | DAG Airflow + docker-compose Airflow |

---

## Structure résultante (racine)

```
.
├── airflow/                  # Orchestration Airflow
├── app/                      # Streamlit
├── configs/                  # Configs entraînement
├── data/                     # Données + pointeurs DVC
├── docker/                   # Tous les Dockerfiles
├── docs/                     # Documentation markdown
├── models/                   # Modèles entraînés
├── notebooks/                # Notebooks d'exploration
├── reports/
│   ├── Romane/
│   └── Sara/
├── requirements/             # Dépendances Python
├── scripts/                  # Scripts utilitaires
├── src/                      # Code source
├── tests/                    # Tests unitaires
├── MLproject                 # Entry point MLflow run
├── conda.yaml                # Env conda pour MLflow run
├── Makefile
└── README.md
```

---

## Impact sur les commandes existantes

Si vous utilisez des chemins en dur dans vos scripts locaux, mettez-les à jour :

- `requirements.txt` → `requirements/base.txt`
- `Dockerfile` → `docker/streamlit/Dockerfile`
- `docker-compose.dev.yml` → `docker/docker-compose.dev.yml`

Le `Makefile` et le CI (`.github/workflows/ci.yml`) ont été mis à jour en conséquence.
