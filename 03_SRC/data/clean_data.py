import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "03_SRC"))

from utils.config import DATA_RAW_PATH, DATA_PROCESSED_PATH, TARGET_COLUMN, ID_COLUMN
from utils.logger import get_logger

logger = get_logger(__name__)

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    for col in df.select_dtypes(include=["number"]).columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include=["object"]).columns:
        if col != ID_COLUMN:
            df[col] = df[col].fillna(df[col].mode()[0])

    if TARGET_COLUMN in df.columns and df[TARGET_COLUMN].dtype == "object":
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"No": 0, "Yes": 1}).astype(int)

    return df

def main():
    df = pd.read_csv(DATA_RAW_PATH)
    clean_df = clean_dataframe(df)
    DATA_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(DATA_PROCESSED_PATH, index=False)
    logger.info(f"Données nettoyées sauvegardées : {DATA_PROCESSED_PATH}")

if __name__ == "__main__":
    main()
