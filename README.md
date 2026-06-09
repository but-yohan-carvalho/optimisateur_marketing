# Optimisateur de Campagnes Marketing — Lead Scoring Bancaire

Ce projet a pour objectif de prédire la probabilité de conversion des clients ("leads") d'une banque pour la souscription d'un dépôt à terme (campagnes de télémarketing). Il s'agit d'un problème de classification binaire supervisée.

Le modèle développé classe les clients par score (de 0 à 1) afin d'optimiser les efforts des équipes commerciales en ciblant les prospects les plus chauds.

---

## 📂 Structure du Projet

```
projet_optimisateur_marketing/
├── data/                       # Dataset d'origine (bank.csv)
├── lead_scoring/
│   ├── data/
│   │   ├── raw/
│   │   │   └── bank.csv        # Dataset principal utilisé
│   │   └── processed/          # Datasets d'entraînement et test nettoyés/normalisés [NEW]
│   │       ├── X_train.csv
│   │       ├── X_test.csv
│   │       ├── y_train.csv
│   │       └── y_test.csv
│   ├── notebooks/
│   │   ├── 01_eda.ipynb           # Exploration et analyse descriptive
│   │   ├── 02_preprocessing.ipynb # Nettoyage, encodage et standardisation
│   │   ├── 03_modeling.ipynb      # Entraînement de la Régression Logistique et scores
│   │   └── 04_evaluation.ipynb    # [À FAIRE] Évaluation finale (Random Forest vs LogReg)
│   └── src/
│       ├── data_loader.py      # Chargement du dataset brut et encodage cible
│       ├── preprocessing.py    # Logique métier du pré-traitement des données
│       ├── model.py            # Fonctions d'entraînement et de sauvegarde de modèles
│       └── evaluate.py         # [À FAIRE] Fonctions d'évaluation et de courbes de performance
├── models/                     # Modèles entraînés et sauvegardés [NEW]
│   └── regression_logistique.joblib
├── CLAUDE.md                   # Guide pour Claude Code
├── gemini.md                   # Guide pour Gemini Code
├── requirements.txt            # Liste des dépendances Python [NEW]
├── notes_apprentissage.md      # Journal de bord pédagogique et réponses aux questions
├── lead-scoring-checklist.html # Suivi de progression interactif du projet
└── sample.ipynb                # Brouillon exploratoire
```

---

## Installation et Environnement virtuel

Le projet utilise un environnement virtuel Python `.venv` situé à la racine.

### 1. Activer l'environnement virtuel (Windows PowerShell) :
```powershell
.venv\Scripts\Activate.ps1
```

### 2. Installer les dépendances :
```powershell
pip install -r requirements.txt
```

---

## Utilisation des scripts et notebooks

Les modules dans `src/` peuvent être exécutés directement depuis la racine pour tester leur fonctionnement :

```powershell
# Exécuter le data loader
.venv\Scripts\python.exe -m lead_scoring.src.data_loader

# Exécuter le preprocessing (nettoie et enregistre dans data/processed/)
.venv\Scripts\python.exe -m lead_scoring.src.preprocessing

# Entraîner le modèle de base (sauvegarde dans models/)
.venv\Scripts\python.exe -m lead_scoring.src.model
```

Pour explorer de manière interactive, lancez Jupyter Notebook :
```powershell
jupyter notebook
```
Et ouvrez les notebooks dans l'ordre de `01_eda.ipynb` à `04_evaluation.ipynb`.

---

## Progression du projet

* **Semaine 1 (EDA)** : Chargement et exploration descriptive des variables du dataset. Analyse de la cible `deposit` (taux de souscription de 47.38%).
* **Semaine 2 (Preprocessing)** : Encodage des colonnes binaires (1/0), One-Hot Encoding des variables textuelles, gestion de `pdays = -1`, et standardisation (`StandardScaler`) ajustée sur l'ensemble Train.
* **Semaine 3 (Modélisation)** : Entraînement de la Régression Logistique et obtention des scores continus de probabilités.
* **Semaine 4 (Évaluation) [À FAIRE]** : Entraînement d'un second modèle (Random Forest), comparaison des métriques de classification et construction des courbes Lift et ROC pour identifier le Top 10 des leads.
