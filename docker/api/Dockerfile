# FastAPI backend — prédictions DL + interface MLflow
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copier requirements
COPY requirements/api.txt requirements/api.txt
RUN pip install --upgrade pip && pip install -r requirements/api.txt

# Code source
COPY src/          ./src/
COPY configs/      ./configs/
COPY scripts/      ./scripts/

# Créer les répertoires de travail
RUN mkdir -p mlruns data models logs

# Port FastAPI
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
