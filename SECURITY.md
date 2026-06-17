# Politique de sécurité

## Versions supportées

Ce projet est en cours de développement actif. Seule la dernière version sur la branche `master` fait l'objet d'un suivi de sécurité.

## Signaler une vulnérabilité

Si vous découvrez une vulnérabilité de sécurité, **n'ouvrez pas d'issue publique GitHub**.

Signalez-la via une [GitHub Security Advisory](https://github.com/DataScientest-Studio/fev26_bmle_blood_cells/security/advisories/new) (rapport privé et confidentiel).

Merci d'inclure :
- Une description de la vulnérabilité
- Les étapes pour la reproduire
- L'impact potentiel

Vous pouvez vous attendre à un accusé de réception sous **5 jours ouvrés** et une mise à jour sous **15 jours ouvrés**.

## Limites de sécurité du projet

Ce projet est un **outil de recherche et d'aide à la visualisation**, et non un dispositif médical certifié.

- Les modèles de classification sont entraînés sur un jeu de données public (voir section Dataset ci-dessous) et n'ont pas été validés pour un usage clinique.
- Les prédictions des modèles **ne doivent pas** se substituer au diagnostic d'un professionnel de santé qualifié.
- Aucune donnée patient n'est collectée, stockée ou traitée par cette application. Toutes les images utilisées proviennent du jeu de données public.
- L'application ne se connecte à aucun système de dossier médical électronique (DME/EHR).

## Précautions liées aux données médicales

Bien que cette application ne traite pas de vraies données patient, les précautions suivantes s'appliquent :

- **Ne pas importer** d'images réelles de patients ni aucune donnée médicale à caractère personnel dans cette application.
- En cas de déploiement dans un environnement clinique ou hospitalier, s'assurer de la conformité avec les réglementations applicables (RGPD en Europe, HIPAA aux États-Unis).
- Les journaux de prédictions stockés en base de données (Supabase) ne contiennent que des noms de fichiers et les sorties du modèle — aucun identifiant patient.
- Les clés API, identifiants de base de données et tokens ne doivent jamais être commités dans le dépôt. Utiliser des fichiers `.env` (listés dans `.gitignore`) pour gérer les secrets en local.

## Dataset

**Dataset 1 — Acevedo et al. (Mendeley Data)**

| Attribut | Détail |
|---|---|
| Nom | A dataset of microscopic peripheral blood cell images for development of automatic recognition systems |
| Source | CHU de Barcelone |
| Images | 17 092 images JPG, 360×363 pixels |
| Classes | 8 : neutrophiles, éosinophiles, basophiles, lymphocytes, monocytes, granulocytes immatures (IG), érythroblastes, plaquettes |
| Annotation | Par des pathologistes cliniques experts |
| Exclusions | Sujets infectés, malades ou sous traitement pharmacologique exclus |
| Usage prévu | Modèles de Machine Learning et Deep Learning |
| DOI | [10.17632/snkd93bnjr/1](https://data.mendeley.com/datasets/snkd93bnjr/1) |
| Licence | CC BY 4.0 — Open access |

Ce jeu de données ne contient aucune information permettant d'identifier des patients. Son utilisation dans ce projet est conforme à la licence Creative Commons CC BY 4.0.

## Dépendances

Ce projet repose sur des bibliothèques tierces (PyTorch, MLflow, Streamlit, FastAPI, etc.). Les correctifs de sécurité de ces bibliothèques ne sont pas appliqués automatiquement. Mettre régulièrement à jour les dépendances avec :

```bash
pip install --upgrade -r requirements/base.txt
```
