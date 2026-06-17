# Model Card — DenseNet-121 Blood Cell Classifier

## Résumé

Modèle de classification de cellules sanguines entraîné sur le dataset Mendeley PBC (17 000 images, 8 classes). Solution gagnante du projet DataScientest fev26_bmle_blood_cells.

| Métrique | DenseNet-121 (5 folds) |
|---|---|
| Accuracy moyenne | ~0.97+ |
| Macro F1 | ~0.97+ |
| AUC-ROC | ~0.99+ |
| Paramètres | 7.0M |
| Taille modèle (.pth) | ~28 MB |

## Dataset

- **Source** : Mendeley PBC dataset normal DIB (Acevedo et al., 2020)
- **Taille** : ~17 000 images TIFF (360×363 px)
- **Preprocessing** : crop Cellpose + resize 224×224 + normalisation ImageNet
- **Split** : cross-validation stratifiée 5 folds (train 72% / val 13% / test 15%)

## 8 Classes

| Classe | Description | Cliniquement critique |
|---|---|---|
| basophil | Basophile | Non |
| eosinophil | Éosinophile | Non |
| erythroblast | Érythroblaste | **Oui** |
| ig | Granulocytes immatures | **Oui** |
| lymphocyte | Lymphocyte | Non |
| monocyte | Monocyte | Non |
| neutrophil | Neutrophile | Non |
| platelet | Plaquette | Non |

## Architecture & Entraînement

- **Architecture** : DenseNet-121 (timm `densenet121`, pretrained ImageNet)
- **Stratégie** : 2 phases — backbone gelé (5 epochs, lr=1e-3) puis fine-tuning complet (15 epochs, lr=1e-4)
- **Optimizer** : AdamW, weight_decay=1e-4
- **Scheduler** : CosineAnnealingLR
- **Early stopping** : patience=3
- **Loss** : CrossEntropyLoss avec pondération inverse des classes

## Hyperparamètres

```yaml
input_size: 224
batch_size: 32
lr_head: 1e-3
lr_full: 1e-4
weight_decay: 1e-4
epochs_head: 5
epochs_full: 15
patience: 3
seed: 42
```

## Limitations

- Entraîné uniquement sur des cellules normales (pas de cellules pathologiques rares)
- Performances non garanties sur microscopes différents du dataset Mendeley
- Les classes `erythroblast` et `ig` sont critiques cliniquement — une vérification humaine est recommandée
- Pas testé sur populations pédiatriques

## Poids du modèle

Les fichiers `.pth` ne sont pas versionnés dans ce repo (trop lourds pour GitHub).
Ils sont stockés sur : [à compléter — HuggingFace Hub ou Google Drive partagé]

Pour charger le modèle :
```python
import timm, torch

model = timm.create_model("densenet121", pretrained=False, num_classes=8)
model.load_state_dict(torch.load("best_fold1_DenseNet_121.pth", map_location="cpu"))
model.eval()
```

## Références

- Acevedo, A. et al. (2020). A dataset of microscopic peripheral blood cell images for development of automatic recognition systems. *Data in Brief*.
- Huang, G. et al. (2017). Densely Connected Convolutional Networks. *CVPR*.
