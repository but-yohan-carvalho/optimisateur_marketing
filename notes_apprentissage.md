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

## Résumé S1

Le dataset est **propre et équilibré** — 0 valeurs manquantes, 47% yes / 53% no. Pas de nettoyage majeur requis.

**Variables les plus intéressantes pour la prédiction :**
- `duration` — corrélation 0.45 avec deposit (mais inconnue avant l'appel)
- `balance` — les souscripteurs ont un solde légèrement plus élevé
- `previous`, `pdays` — l'historique de campagnes a un léger impact positif

**3 choses à régler en S2 :**
1. `balance` a des outliers extrêmes à plafonner
2. `pdays = -1` est une valeur codée à transformer
3. Les colonnes texte (`job`, `marital`, `education`…) doivent être encodées en chiffres

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

**Q : À quoi servent les points au-dessus des boîtes dans un boxplot ?**

Ce sont les **outliers** (valeurs aberrantes) — des valeurs qui s'éloignent trop du reste des données.

**Règle de détection :**
- Outlier haut : valeur > `Q3 + 1.5 × (Q3 - Q1)`
- Outlier bas : valeur < `Q1 - 1.5 × (Q3 - Q1)`

Dans ce dataset :
- `age` : quelques clients très âgés (~75-95 ans), peu nombreux
- `balance` : clients avec des soldes très élevés (jusqu'à 80 000€), très nombreux

**Impact en ML :** les outliers peuvent fausser l'entraînement, surtout pour la régression logistique qui est sensible aux valeurs extrêmes. En S2 on devra décider : garder, plafonner (cap), ou supprimer ces valeurs.

---

**Q : Corrélation faible = variable inutile pour le modèle ?**

Non. La corrélation ne mesure que les relations **linéaires** entre deux variables. Une variable avec une faible corrélation avec la cible peut quand même être très utile pour un algorithme non-linéaire comme Random Forest.

Exemple dans ce dataset : `age` et `balance` ont une corrélation quasi nulle avec `deposit` (~0.03 et 0.08), mais un Random Forest pourra détecter des patterns complexes comme "les clients entre 30 et 40 ans avec un solde > 2000€ souscrivent plus souvent".

**Règle** : ne jamais écarter une variable sur la seule base d'une faible corrélation linéaire.

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

## S2 — Preprocessing (Nettoyage, Encodage et Normalisation)

**Q : Pourquoi est-il indispensable d'appliquer l'ajustement (`fit`) du `StandardScaler` uniquement sur l'ensemble d'entraînement (`X_train`) ?**

C'est une règle d'or pour éviter les **fuites de données (data leakage)**. 
Si on ajustait le scaler sur l'ensemble du dataset (ou sur `X_test`), le modèle aurait indirectement accès à des informations statistiques du test set (comme sa moyenne et son écart-type). L'ensemble de test doit rester totalement invisible lors de la phase de préparation pour garantir une évaluation honnête et réaliste du modèle. 
On fait donc :
1. `scaler.fit(X_train)`
2. `X_train_scaled = scaler.transform(X_train)`
3. `X_test_scaled = scaler.transform(X_test)`

---

**Q : Comment a-t-on traité la valeur `-1` dans la colonne `pdays` ?**

La colonne `pdays` indique le nombre de jours écoulés depuis le dernier contact. La valeur `-1` signifie "jamais contacté auparavant".
Laisser `-1` tel quel dans une variable numérique pose problème, surtout pour les modèles linéaires comme la régression logistique, car l'algorithme va interpréter ce `-1` mathématiquement (comme étant inférieur à 0, 10 ou 100 jours, ce qui n'a pas de sens métier).

Pour résoudre cela proprement :
1. On a créé une nouvelle variable binaire indicatrice : **`pdays_jamais_contacte`** (vaut 1 si `pdays == -1`, sinon 0).
2. On a remplacé la valeur `-1` par **`0`** dans la colonne `pdays` (puisque le fait de ne jamais avoir été contacté est désormais capturé par notre nouvelle colonne binaire, 0 devient une valeur neutre qui ne perturbe pas l'apprentissage).

---

**Q : Comment fonctionne la fonction d'encodage One-Hot `encoder_variables_cat` et pourquoi est-elle indispensable ?**

Les modèles mathématiques ne comprennent pas le texte (ex: `"married"`, `"single"`, `"divorced"`). Si on remplaçait ces catégories par `0`, `1`, `2`, le modèle croirait qu'il y a une hiérarchie ou un ordre de grandeur (ex: célibataire est "plus grand" que marié).

Pour éviter cela, on utilise le **One-Hot Encoding** via la fonction `pd.get_dummies()` de pandas :
1. On crée une nouvelle colonne binaire (0 ou 1) pour chaque valeur de texte possible (ex: `marital_married` et `marital_single`).
2. Si un client est célibataire (`single`), la colonne `marital_single` vaudra `1` et `marital_married` vaudra `0`.
3. L'argument `drop_first=True` permet d'enlever la première colonne (ex: `marital_divorced`) pour éviter la multicolinéarité.
4. L'argument `dtype=int` permet d'obtenir des `1` et des `0` au lieu de `True`/`False` pour une meilleure lisibilité dans nos fichiers de données.

---

**Q : Pourquoi utiliser `pd.get_dummies(..., drop_first=True)` pour l'encodage One-Hot ?**

L'argument `drop_first=True` permet de supprimer la première modalité (colonne) de chaque variable catégorielle encodée. Par exemple, si la variable `marital` a 3 modalités ("married", "single", "divorced"), elle sera encodée sous forme de 2 colonnes binaires au lieu de 3.

C'est indispensable pour les modèles comme la régression logistique pour éviter le **piège de la multicolinéarité** (ou "dummy variable trap"). Si on gardait les 3 colonnes, leur somme vaudrait toujours 1 pour chaque ligne, ce qui est parfaitement corrélé avec le terme constant (l'ordonnée à l'origine ou *intercept*), empêchant le solveur mathématique de converger correctement vers une solution unique pour les coefficients.

---

## S3 — Modélisation (Régression Logistique)

**Q : Quelle est la différence entre `predict()` et `predict_proba()` pour un modèle ?**

- **`predict(X)`** : renvoie directement la classe prédite (`0` ou `1`) en appliquant un seuil par défaut (généralement 0.5).
- **`predict_proba(X)`** : renvoie les probabilités d'appartenance à chaque classe (ex: `[0.18, 0.82]`).

Pour le **lead scoring**, nous avons besoin des probabilités issues de `predict_proba(X)[:, 1]`. Cela permet d'attribuer un score continu de 0 à 1 à chaque client. Ainsi, le service marketing peut classer et cibler les prospects par ordre de priorité, du plus chaud au plus froid, plutôt que d'obtenir une simple réponse binaire oui/non.

---

**Q : À quoi sert le paramètre `max_iter` dans la Régression Logistique ?**

Le modèle de régression logistique n'a pas de solution analytique directe ; il utilise un solveur d'optimisation numérique (comme L-BFGS) pour ajuster les coefficients pas à pas.
`max_iter` définit le nombre maximum de pas (d'itérations) que le solveur est autorisé à effectuer. Par défaut, scikit-learn utilise `max_iter=100`. Cependant, comme nous avons 43 variables après One-Hot encoding, le solveur a besoin de plus d'étapes pour converger vers la solution optimale. Nous l'avons donc fixé à `1000` pour éviter les avertissements de non-convergence.

---

## S4 — Évaluation et Comparaison (Random Forest & Graphiques de performance)

**Q : Pourquoi avoir entraîné un modèle Random Forest en plus de la Régression Logistique ?**

La Régression Logistique est un modèle linéaire simple qui suppose que l'effet de chaque variable est additif et linéaire. 
En marketing bancaire, les comportements sont souvent plus complexes et interactifs (par exemple: un solde bancaire élevé peut être un bon indicateur de souscription, mais seulement si le client a un certain niveau d'études ou d'âge). 

Le Random Forest (Forêt Aléatoire) est un modèle non-linéaire basé sur un ensemble d'arbres de décision. Il est capable de détecter automatiquement ces interactions fines sans qu'on ait besoin de les spécifier à la main. C'est ce qui lui a permis de surclasser la Régression Logistique sur presque toutes les métriques de test, et en particulier sur le **Rappel (89.60% contre 79.77%)**, ce qui signifie qu'il détecte près de 10% de souscripteurs réels de plus.

---

**Q : Qu'est-ce que la courbe de gain cumulé (Lift) et quelle est son utilité marketing ?**

La courbe de gain cumulé montre le pourcentage de souscripteurs réels que l'on parvient à toucher en fonction du pourcentage de prospects contactés (lorsqu'ils sont triés du score le plus élevé au score le plus faible).

**Utilité marketing :**
Si on appelait les clients au hasard (ligne diagonale grise), pour trouver 50% des souscripteurs réels, il faudrait appeler 50% de la base de données.
Avec notre modèle Random Forest, la courbe grimpe très vite : en n'appelant que le **top 20%** des clients ayant les meilleurs scores, on trouve près de **60%** de l'ensemble des souscripteurs réels. C'est un gain d'efficacité et de coût gigantesque pour le centre d'appels de la banque.

---

**Q : Comment a-t-on extrait le "Top 10 leads" de manière commercialement exploitable ?**

Les variables de notre jeu de test `X_test` sont standardisées (moyenne à 0, écart-type à 1) pour les besoins de la régression logistique. Transmettre ces données aux commerciaux (ex: âge = `-0.76` et solde = `-0.36`) n'aurait aucun sens pour eux.

Pour résoudre cela :
1. Nous avons tiré parti du fait que `train_test_split` conserve les index d'origine du DataFrame.
2. Nous avons chargé le dataset brut non normalisé (`bank.csv`) et filtré les lignes correspondant aux index de `X_test` via `df_brut.loc[X_test.index]`.
3. Nous avons ajouté la colonne `score_conversion` calculée par le modèle.
4. Nous avons trié le résultat pour obtenir un tableau contenant les valeurs réelles (âge en années, solde en euros, profession en texte) directement lisibles par les commerciaux pour préparer leurs appels.

---