# MLflow server — tracking des expériences DL
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

RUN pip install mlflow==3.11.1 psycopg2-binary

EXPOSE 5000

# Stockage local (SQLite) — pas besoin de Postgres pour le dev
ENV MLFLOW_BACKEND_STORE_URI=sqlite:////mlflow/mlflow.db
ENV MLFLOW_ARTIFACTS_DESTINATION=/mlflow/artifacts

VOLUME /mlflow

# --artifacts-destination (pas --default-artifact-root) : avec --serve-artifacts,
# les expériences reçoivent un artifact_location en mlflow-artifacts:/..., proxié
# en HTTP par le serveur. --default-artifact-root expose un chemin filesystem brut
# que les clients distants (ou même le client local hors conteneur) ne peuvent pas lire.
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts", \
     "--serve-artifacts", \
     "--host", "0.0.0.0", "--port", "5000"]
