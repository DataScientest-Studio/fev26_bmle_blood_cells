# Image de base Python — compatible avec torch, timm, cellpose
FROM python:3.11-slim

# Métadonnées
LABEL maintainer="fev26_bmle_blood_cells"
LABEL description="Pipeline DenseNet-121 — classification cellules sanguines"

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Répertoire de travail
WORKDIR /app

# Dépendances système (OpenCV, PIL, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copier le code source
COPY configs/ ./configs/
COPY src/     ./src/
COPY scripts/ ./scripts/
COPY tests/   ./tests/
COPY Makefile .

# Dossiers créés à l'exécution (montés via volume en prod)
RUN mkdir -p data models reports mlruns

# Port Streamlit
EXPOSE 8501

# Commande par défaut : lancer l'app d'inférence
CMD ["python", "-m", "streamlit", "run", "src/serving/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
