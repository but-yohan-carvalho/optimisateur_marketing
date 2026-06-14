import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from lead_scoring.src.data_loader import charger_donnees
from lead_scoring.src.model import entrainer_random_forest


def calculer_metriques(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Calcule les principales métriques de performance de classification.
    
    Args:
        y_true (pd.Series): Les vraies étiquettes de classe.
        y_pred (np.ndarray): Les classes prédites (0 ou 1).
        y_prob (np.ndarray): Les probabilités prédites (score continu).
        
    Returns:
        dict: Dictionnaire contenant les scores d'accuracy, precision, recall, f1 et roc_auc.
    """
    metriques = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "AUC-ROC": roc_auc_score(y_true, y_prob)
    }
    return metriques


def tracer_courbes_roc(y_true: pd.Series, modeles_probs: dict) -> None:
    """
    Trace les courbes ROC pour comparer visuellement différents modèles.
    
    Args:
        y_true (pd.Series): Les vraies étiquettes de classe.
        modeles_probs (dict): Un dictionnaire de la forme {Nom_Modèle: Probabilités_Prédites}.
    """
    plt.figure(figsize=(8, 6))
    
    # Ligne diagonale de référence (modèle aléatoire)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Aléatoire (AUC = 0.5)")
    
    for nom_modele, y_prob in modeles_probs.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f"{nom_modele} (AUC = {auc_score:.4f})", lw=2)
        
    plt.title("Comparaison des courbes ROC")
    plt.xlabel("Taux de Faux Positifs (FPR)")
    plt.ylabel("Taux de Vrais Positifs (TPR)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def tracer_courbes_lift(y_true: pd.Series, modeles_probs: dict) -> None:
    """
    Trace les courbes de gain cumulé (Lift / Cumulative Gains) pour comparer les performances marketing.
    
    Args:
        y_true (pd.Series): Les vraies étiquettes de classe.
        modeles_probs (dict): Un dictionnaire de la forme {Nom_Modèle: Probabilités_Prédites}.
    """
    plt.figure(figsize=(8, 6))
    
    # Ligne diagonale de référence (modèle aléatoire)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ciblage aléatoire")
    
    total_positives = y_true.sum()
    n_total = len(y_true)
    
    for nom_modele, y_prob in modeles_probs.items():
        # Création d'un dataframe trié par probabilités descendantes
        df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
        df = df.sort_values(by="y_prob", ascending=False).reset_index(drop=True)
        
        # Calcul des gains cumulés
        df["pop_pct"] = (df.index + 1) / n_total
        df["gain_cumule"] = df["y_true"].cumsum() / total_positives
        
        # Ajout du point initial (0, 0)
        pop_pct = [0.0] + df["pop_pct"].tolist()
        gain_cumule = [0.0] + df["gain_cumule"].tolist()
        
        plt.plot(pop_pct, gain_cumule, label=f"{nom_modele}", lw=2)
        
    plt.title("Comparaison des courbes de gain cumulé (Lift)")
    plt.xlabel("Pourcentage de population ciblée (triée par score)")
    plt.ylabel("Pourcentage des souscripteurs réels trouvés")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def extraire_top_leads(X_test: pd.DataFrame, y_prob: np.ndarray, top_n: int = 10) -> pd.DataFrame:
    """
    Extrait les top N leads les plus susceptibles de souscrire, en affichant
    leurs caractéristiques d'origine (non normalisées) pour lisibilité commerciale.
    
    Args:
        X_test (pd.DataFrame): Les variables explicatives de test normalisées.
        y_prob (np.ndarray): Les probabilités de souscription prédites.
        top_n (int): Le nombre de prospects à extraire. Par défaut, 10.
        
    Returns:
        pd.DataFrame: Les top N prospects avec leurs caractéristiques réelles et leur score.
    """
    # 1. Charger les données brutes d'origine pour récupérer les vraies valeurs non normalisées
    df_brut = charger_donnees()
    
    # 2. Filtrer le dataset brut sur la base des index du jeu de test X_test
    df_test_original = df_brut.loc[X_test.index].copy()
    
    # 3. Ajouter la colonne des scores prédits
    df_test_original["score_conversion"] = y_prob
    
    # 4. Trier et extraire les meilleurs leads
    top_leads = df_test_original.sort_values(by="score_conversion", ascending=False).head(top_n)
    
    return top_leads


if __name__ == "__main__":
    print("--- Lancement du test local du module evaluate.py ---")
    
    # Chargement des données pré-traitées
    dossier_data = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
    
    try:
        X_train = pd.read_csv(os.path.join(dossier_data, "X_train.csv"))
        y_train = pd.read_csv(os.path.join(dossier_data, "y_train.csv")).squeeze()
        X_test = pd.read_csv(os.path.join(dossier_data, "X_test.csv"))
        y_test = pd.read_csv(os.path.join(dossier_data, "y_test.csv")).squeeze()
        
        # 1. Entraîner le Random Forest
        rf_modele = entrainer_random_forest(X_train, y_train)
        
        # 2. Obtenir les prédictions
        y_pred = rf_modele.predict(X_test)
        y_prob = rf_modele.predict_proba(X_test)[:, 1]
        
        # 3. Calculer les métriques
        metriques = calculer_metriques(y_test, y_pred, y_prob)
        print("\nMétriques obtenues sur le test set pour le Random Forest :")
        for k, v in metriques.items():
            print(f"- {k} : {v:.4f}")
            
        # 4. Tester l'extraction des top leads
        top_leads = extraire_top_leads(X_test, y_prob, top_n=5)
        print("\nTop 5 des leads les plus chauds (caractéristiques réelles) :")
        print(top_leads[["age", "job", "balance", "duration", "score_conversion"]])
        
        # 5. Sauvegarde du modèle Random Forest
        chemin_model = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "random_forest.joblib"))
        joblib.dump(rf_modele, chemin_model)
        print(f"\nModèle Random Forest sauvegardé avec succès dans : {chemin_model}")
        
    except FileNotFoundError as e:
        print(f"Erreur lors du test : fichiers de données introuvables. Détails : {e}")
        
    print("--- Fin du test local du module evaluate.py ---")
