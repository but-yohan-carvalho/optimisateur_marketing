import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def entrainer_regression_logistique(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42, max_iter: int = 1000) -> LogisticRegression:
    """
    Entraîne un modèle de Régression Logistique sur les données d'entraînement.
    
    Args:
        X_train (pd.DataFrame): Variables explicatives d'entraînement.
        y_train (pd.Series): Variable cible d'entraînement.
        random_state (int): Graine de reproductibilité. Par défaut, 42.
        max_iter (int): Nombre maximum d'itérations pour la convergence. Par défaut, 1000.
        
    Returns:
        LogisticRegression: Le modèle de régression logistique entraîné.
    """
    # Instanciation de la régression logistique
    modele = LogisticRegression(random_state=random_state, max_iter=max_iter)
    
    # Entraînement du modèle
    print("Entraînement de la Régression Logistique en cours...")
    modele.fit(X_train, y_train)
    print("Modèle entraîné avec succès.")
    
    return modele


def predire_probabilites(modele: LogisticRegression, X: pd.DataFrame) -> np.ndarray:
    """
    Calcule les probabilités de souscription (classe 1) pour les données fournies.
    
    Args:
        modele (LogisticRegression): Le modèle entraîné.
        X (pd.DataFrame): Les variables explicatives pour lesquelles effectuer les prédictions.
        
    Returns:
        np.ndarray: Tableau 1D contenant les probabilités de conversion (comprises entre 0 et 1).
    """
    # predict_proba renvoie [probabilité_classe_0, probabilité_classe_1]
    # On ne s'intéresse qu'à la classe 1 (le client souscrit)
    probabilites = modele.predict_proba(X)[:, 1]
    return probabilites


def sauvegarder_modele(modele: LogisticRegression, chemin_fichier: str) -> None:
    """
    Sérialise et sauvegarde le modèle entraîné sur le disque au format .joblib.
    
    Args:
        modele (LogisticRegression): Le modèle entraîné à sauvegarder.
        chemin_fichier (str): Chemin d'accès complet du fichier de sauvegarde.
    """
    # Création du dossier parent s'il n'existe pas
    dossier_parent = os.path.dirname(chemin_fichier)
    if dossier_parent:
        os.makedirs(dossier_parent, exist_ok=True)
        
    # Sauvegarde
    joblib.dump(modele, chemin_fichier)
    print(f"Modèle sauvegardé avec succès à l'emplacement : {chemin_fichier}")


if __name__ == "__main__":
    # Test local de l'ensemble du pipeline de modélisation
    print("--- Lancement du test du module model.py ---")
    
    # Définition des chemins
    dossier_data = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
    chemin_model = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "regression_logistique.joblib"))
    
    # 1. Chargement des données pré-traitées
    try:
        X_train = pd.read_csv(os.path.join(dossier_data, "X_train.csv"))
        y_train = pd.read_csv(os.path.join(dossier_data, "y_train.csv")).squeeze() # squeeze convertit en Series
        X_test = pd.read_csv(os.path.join(dossier_data, "X_test.csv"))
        
        print(f"Données chargées avec succès depuis {dossier_data}.")
        print(f"Dimensions X_train: {X_train.shape}, y_train: {y_train.shape}")
        
        # 2. Entraînement
        modele = entrainer_regression_logistique(X_train, y_train)
        
        # 3. Prédiction (test de la fonction de probabilité)
        probabilites = predire_probabilites(modele, X_test)
        print(f"Exemple de probabilités prédites (5 premières) : {probabilites[:5]}")
        
        # 4. Sauvegarde
        sauvegarder_modele(modele, chemin_model)
        
    except FileNotFoundError as e:
        print(f"Erreur lors du test : les fichiers de données pré-traitées sont introuvables. Détails : {e}")
        print("Assure-toi d'avoir exécuté le preprocessing d'abord.")
        
    print("--- Fin du test du module model.py ---")
