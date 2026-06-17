# MLflow server — tracking des expériences DL
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

RUN pip install mlflow==3.11.1 psycopg2-binary

EXPOSE 5000

# Stockage local (SQLite) — pas besoin de Postgres pour le dev
ENV MLFLOW_BACKEND_STORE_URI=sqlite:////mlflow/mlflow.db
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts

VOLUME /mlflow

CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--default-artifact-root", "/mlflow/artifacts", \
     "--serve-artifacts", \
     "--host", "0.0.0.0", "--port", "5000"]
