# GEMINI.md

Ce fichier sert de guide pour Gemini (Antigravity) lorsqu'il travaille sur ce dépôt. Il contient le contexte du projet, les commandes utiles, la structure attendue et les règles de codage.

---

## 🎯 Projet

**Objectif** : Lead scoring pour les campagnes marketing d'une banque. Il s'agit de prédire la probabilité (0-1) qu'un client ("lead") souscrive à un dépôt à terme (colonne `deposit`).

- **Dataset** : Bank Marketing UCI — situé dans [lead_scoring/data/raw/bank.csv](file:///c:/Users/Yohan/UQAC/python_UQAC/projet_optimisateur_marketing/lead_scoring/data/raw/bank.csv) (et doublé à la racine dans `data/bank.csv`).
- **Variable cible** : Colonne `deposit` (valeurs `yes`/`no` à encoder respectivement en `1`/`0`).

---

## 💻 Environnement et Commandes Utiles

L'environnement virtuel Python `.venv` est déjà créé à la racine du projet.

### Activation et Installation

Sur Windows (PowerShell) :
```powershell
.venv\Scripts\Activate.ps1
# Une fois requirements.txt créé :
pip install -r requirements.txt
```

### Lancement de Jupyter Notebooks

```powershell
jupyter notebook
```

### Exécution des Modules Python

Toujours exécuter les modules depuis la racine du projet pour préserver les résolutions de paquets :
```powershell
python -m lead_scoring.src.data_loader
# Ou directement :
python lead_scoring/src/data_loader.py
```

---

## 📂 Structure du Projet

```
projet_optimisateur_marketing/
├── data/                       # Copie du dataset brut à la racine
│   └── bank.csv
├── lead_scoring/
│   ├── __init__.py
│   ├── data/
│   │   └── raw/
│   │       └── bank.csv        # Dataset principal utilisé par data_loader.py
│   ├── notebooks/
│   │   ├── 01_eda.ipynb           # [FAIT] Exploration et analyse des données
│   │   ├── 02_preprocessing.ipynb # [À FAIRE] Nettoyage et préparation
│   │   ├── 03_modeling.ipynb      # [À FAIRE] Modélisation et scores
│   │   └── 04_evaluation.ipynb    # [À FAIRE] Évaluation finale et comparaison
│   └── src/
│       ├── __init__.py
│       ├── data_loader.py      # [FAIT] Chargement des données et encodage cible
│       ├── preprocessing.py    # [À FAIRE] Nettoyage, encodage catégoriel, scaling
│       ├── model.py            # [À FAIRE] Entraînement des modèles (LogReg, RandomForest)
│       └── evaluate.py         # [À FAIRE] Métriques, courbes (ROC, Lift) et top leads
├── CLAUDE.md                   # Guide pour Claude Code
├── gemini.md                   # Ce fichier (Guide pour Gemini)
├── notes_apprentissage.md      # Réponses aux questions clés et explications du projet
├── lead-scoring-checklist.html # Suivi graphique de la progression du projet
└── sample.ipynb                # Brouillon exploratoire (ne pas modifier ni supprimer)
```

---

## 🛠️ Plan d'Implémentation par Semaine

- **Semaine 1 (EDA)** : [01_eda.ipynb](file:///c:/Users/Yohan/UQAC/python_UQAC/projet_optimisateur_marketing/lead_scoring/notebooks/01_eda.ipynb) & [data_loader.py](file:///c:/Users/Yohan/UQAC/python_UQAC/projet_optimisateur_marketing/lead_scoring/src/data_loader.py)
  - Chargement du CSV.
  - Statistiques descriptives (`describe()`, formes, types).
  - Analyse de la variable cible `deposit`.
  - Visualisations : distribution, heatmap de corrélation, boxplots.
- **Semaine 2 (Preprocessing)** : [02_preprocessing.ipynb](file:///c:/Users/Yohan/UQAC/python_UQAC/projet_optimisateur_marketing/lead_scoring/notebooks/02_preprocessing.ipynb) & [preprocessing.py](file:///c:/Users/Yohan/UQAC/python_UQAC/projet_optimisateur_marketing/lead_scoring/src/preprocessing.py)
  - Nettoyage des valeurs manquantes et traitement des valeurs aberrantes.
  - Encodage des variables catégorielles (One-Hot Encoding).
  - Normalisation/Standardisation des variables numériques (`StandardScaler`).
- **Semaine 3 (Modélisation)** : [03_modeling.ipynb](file:///c:/Users/Yohan/UQAC/python_UQAC/projet_optimisateur_marketing/lead_scoring/notebooks/03_modeling.ipynb) & [model.py](file:///c:/Users/Yohan/UQAC/python_UQAC/projet_optimisateur_marketing/lead_scoring/src/model.py)
  - Séparation Train/Test (80/20, `random_state=42`).
  - Entraînement de la Régression Logistique.
  - Prédiction des probabilités (`predict_proba()`).
- **Semaine 4 (Évaluation)** : [04_evaluation.ipynb](file:///c:/Users/Yohan/UQAC/python_UQAC/projet_optimisateur_marketing/lead_scoring/notebooks/04_evaluation.ipynb) & [evaluate.py](file:///c:/Users/Yohan/UQAC/python_UQAC/projet_optimisateur_marketing/lead_scoring/src/evaluate.py)
  - Entraînement d'un second modèle (Random Forest).
  - Comparaison des modèles (AUC-ROC, Matrice de confusion, classification report).
  - Génération de la courbe ROC et de la courbe Lift.
  - Extraction du top 10 des leads les plus susceptibles de convertir.

---

## 📏 Contraintes et Règles de Codage

1. **Séparation des responsabilités** : Les notebooks ne doivent contenir aucun code métier/logique brut. Ils doivent uniquement importer et appeler les fonctions définies dans les modules de `lead_scoring/src/`.
2. **Langue** : Le code (noms de variables, fonctions) et la documentation (docstrings, commentaires, textes de notebooks) doivent être rédigés en **français**.
3. **Reproductibilité** : Toujours utiliser `random_state=42` et un découpage Train/Test 80/20.
4. **Gestion d'erreurs** : Dans `data_loader.py`, lever une erreur `FileNotFoundError` claire avec le chemin absolu si `bank.csv` est introuvable.
5. **Pré-traitement de la cible** : Encoder la colonne `deposit` (`yes` -> 1, `no` -> 0) dès le chargement dans `data_loader.py`.
6. **Affichage dans les notebooks** : Toujours afficher explicitement les métriques clés sous forme de texte (prints) et de visualisations graphiques à chaque étape importante.
