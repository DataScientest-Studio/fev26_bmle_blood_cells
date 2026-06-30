#!/bin/bash
set -e

# Récupère le dernier modèle @production depuis DagsHub au démarrage,
# en utilisant les credentials injectés par docker-compose (DAGSHUB_USER/TOKEN).
# Échec silencieux : si DagsHub est inaccessible, le fallback local est utilisé.
if [ -n "$DAGSHUB_USER" ] && [ -n "$DAGSHUB_TOKEN" ]; then
    dvc remote modify dagshub --local auth basic 2>/dev/null || true
    dvc remote modify dagshub --local user "$DAGSHUB_USER" 2>/dev/null || true
    dvc remote modify dagshub --local password "$DAGSHUB_TOKEN" 2>/dev/null || true
    dvc pull models.dvc 2>/dev/null && echo "[ok] models/ synchronisé depuis DagsHub" \
        || echo "[warn] dvc pull échoué — modèle local utilisé en fallback"
else
    echo "[warn] DAGSHUB_USER/TOKEN non définis — dvc pull ignoré"
fi

exec uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload
