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

## Résumé Étape 1

Le dataset est **propre et équilibré** — 0 valeurs manquantes, 47% yes / 53% no. Pas de nettoyage majeur requis.

**Variables les plus intéressantes pour la prédiction :**
- `duration` — corrélation 0.45 avec deposit (mais inconnue avant l'appel)
- `balance` — les souscripteurs ont un solde légèrement plus élevé
- `previous`, `pdays` — l'historique de campagnes a un léger impact positif

**3 choses à régler en Étape 2 :**
1. `balance` a des outliers extrêmes à plafonner
2. `pdays = -1` est une valeur codée à transformer
3. Les colonnes texte (`job`, `marital`, `education`…) doivent être encodées en chiffres

---

## Étape 1 — EDA & Chargement des données

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

## Étape 2 — Preprocessing (Nettoyage, Encodage et Normalisation)

**Q : Pourquoi est-il indispensable d'appliquer l'ajustement (`fit`) du `StandardScaler` uniquement sur l'ensemble d'entraînement (`X_train`) ?**

C'est une règle d'or pour éviter les **fuites de données (data leakage)**. 
Si on ajustait le scaler sur l'ensemble du dataset (ou sur `X_test`), le modèle aurait indirectement accès à des informations statistiques du test set (comme sa moyenne et son écart-type). L'ensemble de test doit rester totalement invisible lors de la phase de préparation pour garantir une évaluation honnête et réaliste du modèle. 
On fait donc :
1. `scaler.fit(X_train)`
2. `X_train_scaled = scaler.transform(X_train)`
3. `X_test_scaled = scaler.transform(X_test)`

---

**Q : Comment fonctionne concrètement la fonction [normaliser_donnees](file:///c:/Users/pyrog/PROG/Uqac/ML/Projet_ml/lead_scoring/src/preprocessing.py#L104-L130) et que retourne-t-elle ?**

La fonction [normaliser_donnees](file:///c:/Users/pyrog/PROG/Uqac/ML/Projet_ml/lead_scoring/src/preprocessing.py#L104-L130) automatise la mise à l'échelle (normalisation/standardisation) des colonnes numériques. 

Voici ses étapes clés :
1. **Copie des données** : `X_train_scaled = X_train.copy()` et `X_test_scaled = X_test.copy()` évitent de modifier les variables originales par effet de bord.
2. **Filtrage des colonnes** : Elle vérifie que les colonnes numériques spécifiées sont bien présentes dans le jeu d'entraînement.
3. **Apprentissage et mise à l'échelle (`fit_transform`) sur le Train** : Elle calcule la moyenne et l'écart-type pour chaque colonne de `X_train` et les applique directement pour centrer et réduire ces données.
4. **Mise à l'échelle seule (`transform`) sur le Test** : Elle applique exactement les mêmes statistiques (moyenne/écart-type du Train) aux colonnes du jeu de test `X_test`.

**Pourquoi retourne-t-elle aussi l'objet `scaler` ?**
L'objet `scaler` (qui contient en mémoire les moyennes et écarts-types calculés sur les données d'entraînement) est renvoyé en plus des données normalisées. C'est essentiel pour pouvoir normaliser de nouveaux prospects individuels de la même façon en production avant de les soumettre au modèle pour prédire leur score.

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

## Étape 3 — Modélisation (Régression Logistique)

**Q : Quelle est la différence entre `predict()` et `predict_proba()` pour un modèle ?**

- **`predict(X)`** : renvoie directement la classe prédite (`0` ou `1`) en appliquant un seuil par défaut (généralement 0.5).
- **`predict_proba(X)`** : renvoie les probabilités d'appartenance à chaque classe (ex: `[0.18, 0.82]`).

Pour le **lead scoring**, nous avons besoin des probabilités issues de `predict_proba(X)[:, 1]`. Cela permet d'attribuer un score continu de 0 à 1 à chaque client. Ainsi, le service marketing peut classer et cibler les prospects par ordre de priorité, du plus chaud au plus froid, plutôt que d'obtenir une simple réponse binaire oui/non.

---

**Q : Pourquoi dans [entrainer_regression_logistique](file:///c:/Users/pyrog/PROG/Uqac/ML/Projet_ml/lead_scoring/src/model.py#L7-L28) la graine est-elle fixée à 42 et le nombre d'itérations maximum à 1000 ?**

* **`random_state=42` (graine) :** Les algorithmes de Machine Learning utilisent parfois de l'aléa pour initialiser leurs paramètres ou mélanger les données. Définir une graine de départ fixe (42 par convention) garantit la reproductibilité des résultats. Chaque exécution du code produira ainsi exactement les mêmes coefficients de régression logistique, les mêmes scores de leads et les mêmes performances.
* **`max_iter=1000` (itérations) :** La régression logistique utilise un solveur d'optimisation numérique (L-BFGS par défaut) qui ajuste pas à pas les coefficients pour minimiser les erreurs. Par défaut, Scikit-Learn s'arrête à 100 itérations. Comme notre dataset contient 43 variables après encodage One-Hot, le solveur a besoin de plus d'étapes pour converger. Augmenter ce paramètre à 1000 permet au modèle de trouver la solution optimale globale sans générer d'avertissement de non-convergence.

---

**Q : À quoi sert concrètement la prédiction des scores pour la banque ?**

Dans le cadre du lead scoring, obtenir une probabilité précise (entre 0% et 100%) d'achat via la méthode [predire_probabilites](file:///c:/Users/pyrog/PROG/Uqac/ML/Projet_ml/lead_scoring/src/model.py#L31-L45) permet d'optimiser l'efficacité commerciale :
1. **Priorisation des actions** : Les conseillers peuvent appeler en priorité les clients ayant les probabilités de conversion les plus élevées (ex: > 80%).
2. **Choix des canaux marketing** : On cible différemment les clients selon leur score (ex: appel téléphonique personnalisé pour les scores très élevés, email promotionnel automatisé à bas coût pour les scores moyens, pas d'action pour les scores faibles).
3. **Optimisation du budget** : Au lieu de démarcher l'ensemble de la base de clients au hasard, on ne cible que les meilleurs prospects, réduisant drastiquement les coûts tout en capturant une grande partie des conversions réelles.

---

**Q : Quelle est la différence entre la prédiction des scores et l'évaluation du modèle ?**

La différence essentielle réside dans l'**objectif** et l'**échelle** de l'analyse :
* **La prédiction des scores (Lead Scoring)** s'intéresse à l'**échelle individuelle** (client par client). Son objectif est opérationnel : on cherche à attribuer une probabilité de conversion à chaque prospect pour guider le travail quotidien des équipes de vente.
* **L'évaluation du modèle** s'intéresse à l'**échelle globale** (sur tout le dataset de test). Son objectif est analytique : on mesure la qualité globale du modèle à l'aide de statistiques de performance (comme l'AUC-ROC, la précision ou le rappel). Elle permet de décider si le modèle est robuste et prêt pour un déploiement réel.

---

## Étape 4 — Évaluation et Comparaison (Random Forest & Graphiques de performance)

**Q : Que signifie la métrique AUC-ROC et comment l'interpréter ?**

L'**AUC-ROC** est une métrique d'évaluation globale de la performance du modèle.
* **ROC** (*Receiver Operating Characteristic*) : C'est la courbe de performance qui trace le taux de vrais positifs en fonction du taux de faux positifs pour tous les seuils de décision possibles.
* **AUC** (*Area Under the Curve*) : C'est l'aire sous cette courbe ROC, comprise entre 0.5 (modèle équivalent au hasard) et 1.0 (modèle parfait).

**Interprétation marketing :**
Si l'AUC-ROC de votre modèle est de **0.91** (comme c'est le cas ici), cela signifie que si l'on prend au hasard deux clients dans la base de données (un qui a souscrit et un qui a refusé), le modèle attribuera un score de conversion plus élevé au premier dans **91% des cas**. C'est donc un excellent indicateur de la capacité de classement du modèle.

---

**Q : Qu'est-ce qu'un Random Forest Classifier (Forêt Aléatoire) et comment fonctionne-t-il ?**

Le **Random Forest Classifier** est un algorithme de Machine Learning puissant utilisé pour la classification binaire (ex: souscription ou non). Il fonctionne selon les principes suivants :
1. **L'arbre de décision (la base)** : Il prend des décisions en posant une suite de questions binaires (Oui/Non) sur les variables du prospect. Un seul arbre a tendance à apprendre par cœur et à faire du sur-apprentissage (*overfitting*).
2. **Le vote majoritaire (la forêt)** : Le Random Forest entraîne une multitude d'arbres de décision indépendants (souvent plus de 100). Pour prédire le comportement d'un nouveau lead, chaque arbre donne sa prédiction et le modèle final retient le choix majoritaire.
3. **Le double niveau d'aléa (*Random*)** : Pour s'assurer que les arbres soient différents et complémentaires, on introduit de l'aléa lors de leur création :
   * **Sur les lignes (*Bagging*)** : Chaque arbre est entraîné sur un échantillon aléatoire de clients.
   * **Sur les colonnes (*Feature Selection*)** : À chaque nœud d'un arbre, on ne cherche la meilleure question à poser que parmi un sous-ensemble aléatoire de variables.

C'est un modèle robuste, non-linéaire (qui capture les interactions complexes) et qui permet d'évaluer l'importance de chaque variable dans la décision d'achat.

---

**Q : Quels sont les indicateurs clés (métriques) pour évaluer un modèle de lead scoring et comment les interpréter ?**

Pour évaluer la performance de classification binaire, on utilise 5 indicateurs clés :
* **L'Accuracy (Justesse globale)** : C'est le pourcentage de prédictions correctes (vrais acheteurs et vrais non-acheteurs) sur l'ensemble de la base de prospects. Attention, elle peut être trompeuse si les classes sont très déséquilibrées.
* **La Precision (Précision)** : C'est la proportion de vrais acheteurs parmi tous les clients prédits comme acheteurs par le modèle. Elle mesure la fiabilité du ciblage. *(Ex : une précision de 82% signifie que sur 100 clients appelés car suggérés par le modèle, 82 vont réellement souscrire).*
* **Le Recall (Rappel / Sensibilité)** : C'est la proportion de vrais acheteurs détectés par le modèle parmi tous les vrais acheteurs réels de la base. Elle mesure la capacité de détection. *(Ex : un rappel de 89% signifie qu'on capture près de 9 acheteurs réels sur 10).*
* **Le F1-Score** : C'est la moyenne synthétique de la Précision et du Rappel. Il permet de mesurer l'équilibre général du modèle (un F1-score élevé garantit que le modèle n'a pas sacrifié le rappel au profit de la précision ou inversement).
* **L'AUC-ROC** : Elle mesure l'aptitude globale du modèle à trier et classer les clients par niveau de probabilité (score).

---

**Q : Pourquoi avoir entraîné un modèle Random Forest en plus de la Régression Logistique ?**

La Régression Logistique est un modèle linéaire simple qui suppose que l'effet de chaque variable est additif et linéaire. 
En marketing bancaire, les comportements sont souvent plus complexes et interactifs (par exemple: un solde bancaire élevé peut être un bon indicateur de souscription, mais seulement si le client a un certain niveau d'études ou d'âge). 

Le Random Forest (Forêt Aléatoire) est un modèle non-linéaire basé sur un ensemble d'arbres de décision. Il est capable de détecter automatiquement ces interactions fines sans qu'on ait besoin de les spécifier à la main. C'est ce qui lui a permis de surclasser la Régression Logistique sur presque toutes les métriques de test, et en particulier sur le **Rappel (89.60% contre 79.77%)**, ce qui signifie qu'il détecte près de 10% de souscripteurs réels de plus.

---

**Q : Si le Random Forest est plus performant, à quoi sert-il d'avoir entraîné un modèle de Régression Logistique ?**

Dans un projet de Data Science, entraîner un modèle simple comme la Régression Logistique en premier présente plusieurs avantages fondamentaux :
* **Modèle de référence (*Baseline*)** : Elle sert de point de départ. Elle nous donne un score minimum à battre pour justifier l'usage d'algorithmes plus complexes. Si le Random Forest n'avait pas fait mieux qu'elle, nous aurions conservé la Régression Logistique.
* **Explicabilité** : C'est un modèle linéaire transparent. Grâce à ses coefficients directs, on comprend immédiatement quelles variables ont le plus d'influence (ex: chaque seconde d'appel passée au téléphone augmente le score de $X$). Le Random Forest est en comparaison une « boîte noire » de centaines d'arbres difficile à expliquer simplement.
* **Rapidité et légèreté** : Elle nécessite extrêmement peu de ressources de calcul et de mémoire pour s'entraîner et s'exécuter, ce qui la rend idéale pour du temps réel à très grande échelle.
* **Stabilité** : Elle est moins sujette au sur-apprentissage (*overfitting*), en particulier sur de très petits jeux de données où les modèles complexes ont tendance à apprendre par cœur.

---

**Q : Qu'est-ce que la courbe de gain cumulé (Lift) et quelle est son utilité marketing ?**

La courbe de gain cumulé montre le pourcentage de souscripteurs réels que l'on parvient à toucher en fonction du pourcentage de prospects contactés (lorsqu'ils sont triés du score le plus élevé au score le plus faible).

**Utilité marketing :**
Si on appelait les clients au hasard (ligne diagonale grise), pour trouver 50% des souscripteurs réels, il faudrait appeler 50% de la base de données.
Avec notre modèle Random Forest, la courbe grimpe très vite : en n'appelant que le **top 30%** des clients ayant les meilleurs scores, on trouve près de **56%** de l'ensemble des souscripteurs réels (et en ciblant le **top 40%**, on capte près de **73%**). C'est un gain d'efficacité et de coût gigantesque pour le centre d'appels de la banque.

---

**Q : Comment a-t-on extrait le "Top 10 leads" de manière commercialement exploitable ?**

Les variables de notre jeu de test `X_test` sont standardisées (moyenne à 0, écart-type à 1) pour les besoins de la régression logistique. Transmettre ces données aux commerciaux (ex: âge = `-0.76` et solde = `-0.36`) n'aurait aucun sens pour eux.

Pour résoudre cela :
1. Nous avons tiré parti du fait que `train_test_split` conserve les index d'origine du DataFrame.
2. Nous avons chargé le dataset brut non normalisé (`bank.csv`) et filtré les lignes correspondant aux index de `X_test` via `df_brut.loc[X_test.index]`.
3. Nous avons ajouté la colonne `score_conversion` calculée par le modèle.
4. Nous avons trié le résultat pour obtenir un tableau contenant les valeurs réelles (âge en années, solde en euros, profession en texte) directement lisibles par les commerciaux pour préparer leurs appels.

---