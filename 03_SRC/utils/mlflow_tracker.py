import hashlib
import os
from contextlib import contextmanager
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.data
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TRACKING_URI = f"sqlite:///{ROOT}/06_MLOps/mlruns.db"


def get_tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def setup(experiment_name: str):
    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_experiment(experiment_name)


@contextmanager
def start_run(run_name: str):
    with mlflow.start_run(run_name=run_name) as active_run:
        yield active_run


def log_dataset(df: pd.DataFrame, source_path: Path, target_column: str):
    """Enregistre la traçabilité des données utilisées pour l'entraînement."""
    file_hash = hashlib.md5(source_path.read_bytes()).hexdigest()

    mlflow.log_params({
        "data_source": str(source_path),
        "data_rows": len(df),
        "data_columns": len(df.columns),
        "data_hash_md5": file_hash,
        "target_column": target_column,
    })

    dataset = mlflow.data.from_pandas(df, source=str(source_path), targets=target_column)
    mlflow.log_input(dataset, context="training")


def log_sklearn_model(model, artifact_path: str = "model"):
    mlflow.sklearn.log_model(model, artifact_path=artifact_path)


def log_params(params: dict):
    mlflow.log_params(params)


def log_metrics(metrics: dict):
    filtered = {k: v for k, v in metrics.items() if v is not None and k != "confusion_matrix"}
    mlflow.log_metrics(filtered)
