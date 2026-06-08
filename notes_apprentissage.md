# Notes d'apprentissage — Lead Scoring

Questions posées pendant le projet, avec explications.

---

**Q : À quoi sert le fichier `data_loader.py` ?**

Dans ce projet, les notebooks (`01_eda.ipynb`, etc.) ne contiennent pas de code "brut" — ils importent des fonctions depuis `src/`. Le `data_loader.py` est le seul endroit où on lit le fichier CSV.

Avantages de cette séparation :

- **Réutilisabilité** : tous les notebooks chargent les données de la même façon, avec une seule ligne `from lead_scoring.src.data_loader import charger_donnees`
- **Maintenabilité** : si le chemin du fichier change, on corrige à un seul endroit
- **Fiabilité** : la vérification du fichier manquant et l'encodage de `deposit` sont faits une seule fois, pas répétés dans chaque notebook

C'est le principe de **séparation des responsabilités** : chaque fichier a un rôle précis et limité.

---

## S1 — EDA & Chargement des données

**Q : À quoi ça sert de compter les valeurs manquantes ?**

Dans un dataset réel, certaines cellules peuvent être vides (`NaN`). La plupart des algorithmes ML plantent ou donnent de mauvais résultats si on leur passe des données incomplètes. On compte les valeurs manquantes dès l'EDA pour :

1. Vérifier qu'il y en a (sanity check)
2. Identifier quelles colonnes sont touchées
3. Décider quoi faire : écarter la colonne (si > ~50% manquant), imputer (remplacer par la moyenne/médiane), ou laisser tel quel si l'algorithme le gère

Sur le dataset Bank Marketing, il n'y a normalement pas de `NaN` — mais on vérifie toujours avant de toucher aux données.

---

**Q : Pourquoi faire `df.describe()` en EDA ?**

`df.describe()` donne un résumé statistique de toutes les colonnes numériques en une seule ligne de code. On l'utilise pour :

1. **Repérer les valeurs aberrantes** — un min ou max anormal (ex: `balance` à -6847) signale un outlier
2. **Détecter les variables codées** — ex: `pdays = -1` n'est pas une vraie valeur numérique, c'est un code pour "jamais contacté"
3. **Voir la dispersion** — si la moyenne (mean) et la médiane (50%) sont très différentes, la distribution est asymétrique → certains algorithmes ML sont sensibles à ça
4. **Vérifier la cohérence** — les valeurs min/max doivent avoir du sens métier (ex: `age` entre 18 et 95 = cohérent)

C'est le premier regard qu'on pose sur les données numériques avant tout traitement.

---

**Q : À quoi correspondent les colonnes du dataset Bank Marketing ?**

| Colonne | Description |
|---|---|
| `age` | Âge du client |
| `job` | Profession |
| `marital` | Situation maritale |
| `education` | Niveau d'études |
| `default` | A-t-il un crédit en défaut ? |
| `balance` | Solde bancaire moyen (en €) |
| `housing` | A-t-il un prêt immobilier ? |
| `loan` | A-t-il un prêt personnel ? |
| `contact` | Type de contact (téléphone fixe/mobile/inconnu) |
| `day` | Jour du mois du dernier contact |
| `month` | Mois du dernier contact |
| `duration` | Durée du dernier appel (en secondes) |
| `campaign` | Nombre de contacts pendant cette campagne |
| `pdays` | Jours depuis le dernier contact d'une campagne précédente (-1 = jamais contacté) |
| `previous` | Nombre de contacts lors de campagnes précédentes |
| `poutcome` | Résultat de la campagne précédente |
| `deposit` | **Cible** — a-t-il souscrit au dépôt à terme ? (1=oui, 0=non) |

**Points importants :**
- `pdays = -1` signifie que le client n'a jamais été contacté avant → valeur codée à traiter en S2
- `duration` est problématique pour un vrai modèle prédictif : on ne connaît pas la durée de l'appel *avant* de l'avoir passé

---

**Q : Pourquoi utiliser des boxplots en EDA ?**

On fait des boxplots de variables numériques (`age`, `balance`) **séparés par deposit=0 et deposit=1** pour comparer visuellement les deux groupes.

- Si les boîtes sont **décalées** → la variable distingue bien les souscripteurs des non-souscripteurs → utile pour le modèle
- Si les boîtes **se superposent** → la variable ne fait pas la différence → moins utile

C'est une validation visuelle intuitive **avant** d'entraîner le modèle, pour mieux comprendre les données.

---

**Q : Pourquoi utiliser une heatmap de corrélation en EDA ?**

La heatmap sert à détecter les variables redondantes et les variables utiles pour le modèle.

- **Variables redondantes** : si deux colonnes sont très corrélées entre elles (ex: +0.95), elles portent la même information — garder les deux n'apporte rien et peut perturber certains modèles (notamment la régression logistique). On peut en supprimer une.
- **Variables potentiellement prédictives** : si une colonne est corrélée avec `deposit`, c'est un signal qu'elle sera utile pour prédire la souscription.

En résumé : la heatmap aide à **choisir quelles variables garder** avant d'entraîner le modèle.

---

**Q : Que peut-on conclure des résultats de `apercu()` ?**

On a obtenu :
- **11 162 lignes, 17 colonnes** → dataset de taille correcte pour du ML
- **0 valeurs manquantes** → pas besoin d'imputation, on peut passer directement au preprocessing
- **5 873 "no" vs 5 289 "yes"** → taux de souscription de **47.38%**

Ce dernier point est important : le dataset est **quasi-équilibré** (presque 50/50). C'est une bonne nouvelle — beaucoup de datasets réels sont très déséquilibrés (ex: 95% "no", 5% "yes"), ce qui oblige à utiliser des techniques spéciales (SMOTE, class_weight, etc.). Ici on n'en aura pas besoin.

---