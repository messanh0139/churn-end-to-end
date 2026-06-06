from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_RAW_PATH = ROOT_DIR / "01_Data" / "raw" / "data.csv"
DATA_PROCESSED_PATH = ROOT_DIR / "01_Data" / "processed" / "churn_features.csv"

MODEL_DIR = ROOT_DIR / "04_Models"
BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

RANDOM_STATE = 42
TEST_SIZE = 0.2
