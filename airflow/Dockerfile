FROM apache/airflow:2.9.1-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
USER airflow

RUN pip install --no-cache-dir \
    apache-airflow-providers-ssh \
    mlflow==3.11.1 \
    dvc==3.67.1 \
    timm
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
