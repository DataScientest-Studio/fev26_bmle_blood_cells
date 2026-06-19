# Preprocessing — téléchargement et préparation des données via DVC + DagsHub
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/preprocessing.txt requirements/preprocessing.txt
RUN pip install --upgrade pip && pip install -r requirements/preprocessing.txt

# Code source data + config DVC
COPY src/data/  ./src/data/
COPY configs/   ./configs/
COPY .dvc/      ./.dvc/

# Fichiers .dvc (manifestes légers — hash des datasets)
COPY data/Source_100.dvc  ./data/Source_100.dvc
COPY data/Source_full.dvc ./data/Source_full.dvc

RUN mkdir -p data models

# Lance dvc pull puis validation de la structure
CMD ["python", "-m", "src.data.make_dataset"]
