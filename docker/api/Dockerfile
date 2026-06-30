# FastAPI backend — prédictions DL + interface MLflow
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Dépendances système — libgl1 requis par cv2 (pytorch_grad_cam), même avec opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copier requirements
COPY requirements/api.txt requirements/api.txt
RUN pip install --upgrade pip && pip install -r requirements/api.txt

# Code source
COPY src/          ./src/
COPY configs/      ./configs/
COPY scripts/      ./scripts/
# Config DVC remote (URL uniquement, sans credentials — injectés à l'exécution)
COPY .dvc/config   ./.dvc/config
# Pointeur DVC du modèle (nécessaire pour dvc pull models.dvc au démarrage)
COPY models.dvc    ./models.dvc

# Créer les répertoires de travail
RUN mkdir -p mlruns data models logs \
    && git init \
    && git config user.email "api@bloodcells.local" \
    && git config user.name "API"

# Script de démarrage : dvc pull models.dvc puis uvicorn
COPY docker/api/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && pip install --no-cache-dir dvc==3.67.1

# Port FastAPI
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["/entrypoint.sh"]
