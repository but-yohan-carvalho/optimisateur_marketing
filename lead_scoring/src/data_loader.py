import os
import pandas as pd

_DEFAULT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "bank.csv"
)

def charger_donnees(chemin: str = _DEFAULT_PATH) -> pd.DataFrame:
    chemin = os.path.abspath(chemin)
    if not os.path.isfile(chemin):
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")
    df = pd.read_csv(chemin)
    df["deposit"] = df["deposit"].map({"yes": 1, "no": 0})
    return df


def apercu(df: pd.DataFrame) -> None:   
    print(f"Dimensions : {df.shape[0]} lignes x {df.shape[1]} colonnes")
    print(f"\nValeurs manquantes :\n{df.isnull().sum()}")
    print(f"\nDistribution cible :\n{df['deposit'].value_counts()}")
    print(f"\nTaux de souscription : {df['deposit'].mean():.2%}")


if __name__ == "__main__":
    df = charger_donnees()
    apercu(df)