"""
Pipeline d'ingestion des nouvelles données.

En production, ce script est déclenché par Airflow chaque semaine.
Il surveille le dossier 01_Data/incoming/ où arrive le nouvel export
de la base de données client (via ETL, SFTP, S3, etc.).

Si un nouveau fichier est présent :
  1. Validation du schéma
  2. Nettoyage + feature engineering via le pipeline existant
  3. Sauvegarde comme nouvelles données courantes (churn_features.csv)
  4. Archivage du fichier brut traité

Si aucun nouveau fichier : sortie propre sans erreur.
"""

import sys
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "03_SRC"))

from data.clean_data import clean_dataframe
from features.build_features import build_features
from features.select_features import remove_useless_columns
from utils.config import DATA_PROCESSED_PATH, ID_COLUMN
from utils.logger import get_logger

logger = get_logger(__name__)

INCOMING_DIR = ROOT / "01_Data" / "incoming"
ARCHIVE_DIR  = INCOMING_DIR / "processed"
INCOMING_FILE = INCOMING_DIR / "data_new.csv"

REQUIRED_COLUMNS = {
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
}


def has_new_data() -> bool:
    return INCOMING_FILE.exists() and INCOMING_FILE.stat().st_size > 0


def validate_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le fichier entrant : {missing}")


def archive_incoming(path: Path) -> None:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = ARCHIVE_DIR / f"data_raw_{ts}.csv"
    shutil.move(str(path), str(dest))
    logger.info(f"Fichier archivé : {dest.name}")


def run() -> bool:
    """Retourne True si de nouvelles données ont été ingérées, False sinon."""
    if not has_new_data():
        logger.info("Aucune nouvelle donnée dans 01_Data/incoming/ : pipeline ignoré")
        return False

    logger.info(f"Nouveau fichier détecté : {INCOMING_FILE.name}")

    df = pd.read_csv(INCOMING_FILE)
    logger.info(f"Données chargées : {len(df)} lignes")

    validate_schema(df)
    logger.info("Schéma validé")

    df = clean_dataframe(df)
    df = build_features(df)
    df = remove_useless_columns(df, [ID_COLUMN])

    DATA_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PROCESSED_PATH, index=False)
    logger.info(f"Nouvelles données sauvegardées : {DATA_PROCESSED_PATH.name} ({len(df)} lignes)")

    archive_incoming(INCOMING_FILE)
    return True


if __name__ == "__main__":
    ingested = run()
    sys.exit(0 if ingested else 0)
