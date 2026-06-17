# Guide de test — Phase 1 (Windows & Mac)

## Prérequis

Avoir un fichier `.env` à la racine du projet (copier `.env.example` et remplir les valeurs).

---

## Test 1 — Connexion DagsHub

### VSCode (Windows & Mac)
1. Ouvrir le panneau **Run & Debug** (`Ctrl+Shift+D` / `Cmd+Shift+D`)
2. Sélectionner **"Test DagsHub connection"**
3. Cliquer sur ▶️

**Résultat attendu dans le terminal :**
```
[1/2] Modèle DenseNet
  Statut  : déjà à jour (ou téléchargé)
  Présent : True
  Taille  : 27.2 MB

[2/2] Source_100 (images de test)
  Statut  : déjà à jour
  Images  : 100/100

Connexion DagsHub OK
```

---

## Test 2 — Entraînement (training.py seul, sans API)

### VSCode (Windows & Mac)
1. **Run & Debug** → sélectionner **"Test training (Source_100 - 1 epoch)"**
2. Cliquer sur ▶️ — durée ~1-2 minutes

**Résultat attendu :**
```
Device  : cpu (ou mps sur Mac M1/M2)
Images  : 100 dans 8 classes
Split   : train=69  val=16  test=15

Phase 1 — backbone gelé (1 epoch)
  Ep 01  train=0.xxx  val=0.xxx  (Xs)

Phase 2 — fine-tuning (1 epoch, patience=3)
  Ep 01  train=0.xxx  val=0.xxx  (Xs)

Meilleur val_acc : 0.xxxx
Test accuracy    : 0.xxxx
Modèle           : models/best_densenet121.pth
MLflow run ID    : xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

MLflow Registry :
  Premier modèle enregistré → promu en Production
  (ou : Nouveau modèle promu / Modèle précédent conservé)

Résumé : val_acc=0.xxxx  test_acc=0.xxxx
```

> ℹ️ Avec seulement 100 images et 1 epoch, les métriques seront faibles (~0.10-0.25) — c'est normal, ce test vérifie uniquement que le **pipeline fonctionne**, pas la qualité du modèle.

### Terminal (Mac)
```bash
cd /chemin/vers/fev26_bmle_blood_cells
source .venv/bin/activate
python -m src.train.training \
  --data-dir data/Source_100 \
  --output-dir models \
  --epochs-head 1 \
  --epochs-full 1 \
  --batch-size 8
```

### Terminal (Windows PowerShell)
```powershell
cd C:\Users\<votre-nom>\fev26_bmle_blood_cells
.venv\Scripts\Activate.ps1
python -m src.train.training `
  --data-dir data/Source_100 `
  --output-dir models `
  --epochs-head 1 `
  --epochs-full 1 `
  --batch-size 8
```

---

## Test 3 — Prédiction (predict_model.py seul, sans API)

> ⚠️ Nécessite que le test 2 ait été lancé une fois (pour générer `models/best_densenet121.pth`)

> ℹ️ Si le modèle vient du test rapide (1 epoch / 100 images), la prédiction peut être incorrecte — c'est attendu. La prédiction sera correcte avec le modèle complet entraîné sur le dataset entier.

### VSCode (Windows & Mac)
1. **Run & Debug** → sélectionner **"Test predict_model (une image)"**
2. Cliquer sur ▶️

**Résultat attendu :**
```
Prédiction : NEUTROPHIL  (ou autre classe si modèle test rapide)
Confiance  : xx.x%

Top 3 :
  neutrophil      xx.x%
  lymphocyte      xx.x%
  monocyte        xx.x%
```

### Terminal (Mac)
```bash
python -m src.models.predict_model \
  --image data/Source_100/neutrophil/BNE_100878.jpg \
  --model models/best_densenet121.pth
```

### Terminal (Windows PowerShell)
```powershell
python -m src.models.predict_model `
  --image data/Source_100/neutrophil/BNE_100878.jpg `
  --model models/best_densenet121.pth
```

---

## Test 4 — API FastAPI complète

### Étape 4.1 — Démarrer le serveur

#### VSCode (Windows & Mac)
1. **Run & Debug** → sélectionner **"Lancer API FastAPI"**
2. Cliquer sur ▶️
3. Attendre le message dans le terminal :
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

> ℹ️ Le message `Warning: Could not load model at startup` a été corrigé. Si le modèle `models/best_densenet121.pth` existe, il sera chargé sans erreur.

#### Terminal (Mac)
```bash
source .venv/bin/activate
uvicorn src.serving.api:app --reload --port 8000
```

#### Terminal (Windows PowerShell)
```powershell
.venv\Scripts\Activate.ps1
uvicorn src.serving.api:app --reload --port 8000
```

---

### Étape 4.2 — Tester via Swagger UI (Windows & Mac identique)

1. Ouvrir le navigateur : **http://localhost:8000/docs**
2. Les endpoints sont organisés en 3 groupes :

```
Inférence      →  POST /predict
Entraînement   →  POST /training
Info           →  GET /health  |  GET /  |  GET /classes  |  GET /model-info
```

#### Test GET /health (groupe Info)
1. Cliquer sur `GET /health`
2. "Try it out" → "Execute"
3. **Réponse attendue :** `{ "status": "ok" }`

#### Test POST /predict (groupe Inférence)
1. Cliquer sur `POST /predict`
2. "Try it out"
3. Cliquer sur "Choose File" → sélectionner une image dans `data/Source_100/neutrophil/`
4. "Execute"
5. **Réponse attendue :**
```json
{
  "class": "Neutrophil",
  "confidence": 0.876,
  "all_probas": {
    "Basophil": 0.001,
    "Neutrophil": 0.876,
    ...
  }
}
```

#### Test POST /training (groupe Entraînement)
1. Cliquer sur `POST /training`
2. "Try it out"
3. Remplacer le body par :
```json
{
  "data_dir": "data/Source_100",
  "epochs_head": 1,
  "epochs_full": 1,
  "batch_size": 8
}
```
4. "Execute" — attendre ~1-2 minutes
5. **Réponse attendue :**
```json
{
  "status": "ok",
  "val_acc": 0.xxxx,
  "test_acc": 0.xxxx,
  "model_path": "models/best_densenet121.pth"
}
```

---

### Étape 4.3 — Tester via terminal (optionnel)

#### Mac (curl)
```bash
# Health check
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/predict \
  -F "file=@data/Source_100/neutrophil/BNE_100878.jpg"

# Training
curl -X POST http://localhost:8000/training \
  -H "Content-Type: application/json" \
  -d '{"data_dir":"data/Source_100","epochs_head":1,"epochs_full":1,"batch_size":8}'
```

#### Windows (PowerShell)
```powershell
# Health check
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET

# Predict
$form = @{ file = Get-Item "data\Source_100\neutrophil\BNE_100878.jpg" }
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method POST -Form $form

# Training
$body = @{ data_dir="data/Source_100"; epochs_head=1; epochs_full=1; batch_size=8 } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/training" -Method POST -Body $body -ContentType "application/json"
```

---

## Récapitulatif des tests Phase 1

| Test | Commande VSCode | Succès si... |
|---|---|---|
| DagsHub | "Test DagsHub connection" | `Connexion DagsHub OK` |
| training.py | "Test training (Source_100 - 1 epoch)" | `Résumé : val_acc=...` + `MLflow run ID` affichés |
| predict_model.py | "Test predict_model (une image)" | Classe + confiance affichés (peu importe la classe) |
| API démarrage | "Lancer API FastAPI" | `Application startup complete` sans erreur |
| API /health | Swagger → GET /health | `{"status":"ok"}` |
| API /predict | Swagger → POST /predict | `class` + `confidence` retournés |
| API /training | Swagger → POST /training | `val_acc` + `test_acc` retournés |
