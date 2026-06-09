import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lead_scoring.src.data_loader import charger_donnees

def nettoyer_donnees(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les données brutes de Bank Marketing :
    - Vérifie et signale la présence de valeurs manquantes (NaN).
    - Traite la valeur 'pdays = -1' en créant une variable indicatrice binaire
      et en remplaçant la valeur d'origine par 0.
    - Encode les variables binaires textuelles (default, housing, loan) en 1/0.
    
    Args:
        df (pd.DataFrame): Le DataFrame brut de départ.
        
    Returns:
        pd.DataFrame: Le DataFrame nettoyé.
    """
    df_copie = df.copy()
    
    # 1. Vérification des valeurs manquantes
    nb_nan = df_copie.isna().sum().sum()
    if nb_nan > 0:
        print(f"[Sanity Check] Attention : {nb_nan} valeurs manquantes détectées dans le dataset.")
    else:
        print("[Sanity Check] Aucune valeur manquante détectée.")
        
    # 2. Traitement de la variable pdays = -1 (jamais contacté)
    if "pdays" in df_copie.columns:
        # Création de la variable indicatrice (1 si pdays == -1, 0 sinon)
        df_copie["pdays_jamais_contacte"] = (df_copie["pdays"] == -1).astype(int)
        # Remplacement de -1 par 0
        df_copie["pdays"] = df_copie["pdays"].replace(-1, 0)
        print("Traitement de 'pdays = -1' effectué : création de la variable binaire 'pdays_jamais_contacte'.")
        
    # 3. Encodage des colonnes binaires (yes/no -> 1/0)
    colonnes_binaires = ["default", "housing", "loan"]
    mapping_binaire = {"yes": 1, "no": 0}
    
    for col in colonnes_binaires:
        if col in df_copie.columns:
            df_copie[col] = df_copie[col].map(mapping_binaire)
            print(f"Encodage de la colonne binaire '{col}' effectué (yes -> 1, no -> 0).")
            
    return df_copie


def encoder_variables_cat(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode les variables catégorielles multi-classes restantes en One-Hot Encoding.
    
    Args:
        df (pd.DataFrame): Le DataFrame nettoyé.
        
    Returns:
        pd.DataFrame: Le DataFrame avec encodage catégoriel appliqué.
    """
    df_copie = df.copy()
    
    # Liste des variables catégorielles multi-classes à encoder
    colonnes_cat = ["job", "marital", "education", "contact", "month", "poutcome"]
    
    # Filtrer uniquement les colonnes existantes dans le DataFrame
    colonnes_existantes = [col for col in colonnes_cat if col in df_copie.columns]
    
    # Application du One-Hot Encoding en éliminant la première modalité pour éviter la multicolinéarité
    df_copie = pd.get_dummies(df_copie, columns=colonnes_existantes, drop_first=True, dtype=int)
    
    print(f"Encodage catégoriel effectué sur les colonnes : {colonnes_existantes}")
    return df_copie


def separer_train_test(df: pd.DataFrame, cible: str = "deposit", test_size: float = 0.2, random_state: int = 42):
    """
    Sépare les variables explicatives X de la variable cible y, et divise en ensembles Train/Test.
    
    Args:
        df (pd.DataFrame): Le DataFrame complet et encodé.
        cible (str): Le nom de la colonne cible. Par défaut, 'deposit'.
        test_size (float): La proportion de données pour l'ensemble de test. Par défaut, 0.2.
        random_state (int): Graine de reproductibilité. Par défaut, 42.
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    if cible not in df.columns:
        raise KeyError(f"La colonne cible '{cible}' est introuvable dans le DataFrame.")
        
    X = df.drop(columns=[cible])
    y = df[cible]
    
    # Utilisation de stratify=y pour conserver la distribution des classes dans les splits Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Découpage Train/Test effectué (ratio {(1-test_size):.0%}/{test_size:.0%}).")
    print(f"Dimensions - X_train : {X_train.shape}, X_test : {X_test.shape}")
    return X_train, X_test, y_train, y_test


def normaliser_donnees(X_train: pd.DataFrame, X_test: pd.DataFrame, cols_a_normaliser: list) -> tuple:
    """
    Normalise (centre et réduit) les colonnes numériques spécifiées à l'aide de StandardScaler.
    Le scaler est entraîné (fit) uniquement sur le Train et appliqué (transform) sur le Train et le Test.
    
    Args:
        X_train (pd.DataFrame): Variables explicatives d'entraînement.
        X_test (pd.DataFrame): Variables explicatives de test.
        cols_a_normaliser (list): Liste des noms de colonnes numériques à mettre à l'échelle.
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler_entraine)
    """
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Filtrer uniquement les colonnes à normaliser qui sont bien présentes
    cols_existantes = [col for col in cols_a_normaliser if col in X_train.columns]
    
    # Instanciation et ajustement du scaler uniquement sur l'ensemble d'entraînement
    scaler = StandardScaler()
    X_train_scaled[cols_existantes] = scaler.fit_transform(X_train[cols_existantes])
    X_test_scaled[cols_existantes] = scaler.transform(X_test[cols_existantes])
    
    print(f"Normalisation StandardScaler effectuée sur les colonnes : {cols_existantes}")
    return X_train_scaled, X_test_scaled, scaler


if __name__ == "__main__":
    # Test local de l'ensemble du pipeline
    print("--- Lancement du test du module de preprocessing ---")
    
    # 1. Chargement des données brutes
    df_brut = charger_donnees()
    
    # 2. Nettoyage
    df_nettoye = nettoyer_donnees(df_brut)
    
    # 3. Encodage catégoriel
    df_encode = encoder_variables_cat(df_nettoye)
    
    # 4. Séparation Train/Test
    X_train, X_test, y_train, y_test = separer_train_test(df_encode, cible="deposit")
    
    # 5. Normalisation des variables numériques
    colonnes_numeriques = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
    X_train_scaled, X_test_scaled, scaler = normaliser_donnees(X_train, X_test, colonnes_numeriques)
    
    # 6. Sauvegarde des datasets préparés
    dossier_sortie = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
    os.makedirs(dossier_sortie, exist_ok=True)
    
    X_train_scaled.to_csv(os.path.join(dossier_sortie, "X_train.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(dossier_sortie, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(dossier_sortie, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(dossier_sortie, "y_test.csv"), index=False)
    
    print(f"\nFichiers sauvegardés avec succès dans : {dossier_sortie}")
    print("--- Fin du test du module de preprocessing ---")
