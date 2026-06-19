# Training — entraînement DenseNet-121 one-shot avec tracking MLflow
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/training.txt requirements/training.txt
RUN pip install --upgrade pip && pip install -r requirements/training.txt

COPY src/     ./src/
COPY configs/ ./configs/

RUN mkdir -p data models mlruns logs

# Paramètres surchargeables via variables d'environnement ou docker compose run
ENV DATA_DIR=data/raw \
    OUTPUT_DIR=models \
    EPOCHS_HEAD=5 \
    EPOCHS_FULL=10 \
    BATCH_SIZE=32 \
    RUN_TYPE=base \
    PART=0

CMD python -m src.train.training \
    --data-dir  ${DATA_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --epochs-head ${EPOCHS_HEAD} \
    --epochs-full ${EPOCHS_FULL} \
    --batch-size  ${BATCH_SIZE} \
    --run-type    ${RUN_TYPE} \
    --part        ${PART}
