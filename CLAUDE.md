# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Projet

Lead scoring pour campagnes marketing bancaires : prédire la probabilité (0–1) qu'un lead souscrive à un dépôt à terme.

**Dataset** : Bank Marketing UCI — fichier `data/bank.csv` déjà présent (11 162 lignes, 17 colonnes).  
**Variable cible** : colonne `deposit` (valeurs `yes`/`no`, à encoder en 1/0).

## Environnement

Un `.venv` Python est déjà créé à la racine. Pour l'activer et installer les dépendances :

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt   # une fois requirements.txt créé
```

Lancer Jupyter :

```powershell
jupyter notebook
```

Exécuter un script src directement :

```powershell
python -m lead_scoring.src.data_loader   # si __init__.py présents
# ou
python lead_scoring/src/data_loader.py
```

## Structure cible à créer

```
lead_scoring/
├── data/raw/          # copie/lien du dataset brut (bank.csv)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   └── evaluate.py
├── requirements.txt
└── README.md
```

Le fichier `sample.ipynb` à la racine est un brouillon exploratoire — ne pas le modifier ni le supprimer.

## Architecture

Les notebooks ne contiennent **pas** de logique métier : ils importent et appellent uniquement les fonctions exposées dans `src/`. Chaque module `src/` correspond à une semaine du plan de travail :

| Fichier | Responsabilité |
|---|---|
| `data_loader.py` | Chargement du CSV, gestion d'erreur si fichier absent |
| `preprocessing.py` | Valeurs manquantes, encodage catégoriel, StandardScaler, split X/y |
| `model.py` | Entraînement LogisticRegression et RandomForestClassifier, `predict_proba` |
| `evaluate.py` | AUC-ROC, matrice de confusion, `classification_report`, courbe ROC, courbe lift, top-10 leads |

Toutes les fonctions sont documentées en **français**.

## Plan d'implémentation par semaine

- **S1 (EDA)** — `01_eda.ipynb` + `data_loader.py` : chargement, statistiques descriptives, distribution cible, heatmap corrélation, boxplots
- **S2 (Preprocessing)** — `02_preprocessing.ipynb` + `preprocessing.py` : nettoyage, encodage, normalisation, séparation X/y
- **S3 (Modélisation)** — `03_modeling.ipynb` + `model.py` : split 80/20 `random_state=42`, LogisticRegression, scores 0–1
- **S4 (Évaluation)** — `04_evaluation.ipynb` + `evaluate.py` : RandomForest vs LogReg, top-10, courbes ROC & lift

## Contraintes de code

- `train_test_split` toujours avec `random_state=42`, ratio 80/20.
- Afficher les métriques clés (prints) **et** les graphiques à chaque étape notebook.
- Si `data/bank.csv` est introuvable, lever une `FileNotFoundError` explicite avec le chemin attendu.
- `deposit` est la seule variable cible ; l'encoder `yes→1, no→0` avant tout traitement.
