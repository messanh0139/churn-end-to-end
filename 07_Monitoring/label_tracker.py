import sys
import csv
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "03_SRC"))

from utils.logger import get_logger

logger = get_logger(__name__)

LABELS_LOG = ROOT / "07_Monitoring" / "actual_labels.csv"
PREDICTIONS_LOG = ROOT / "07_Monitoring" / "predictions_log.csv"
REAL_METRICS_PATH = ROOT / "07_Monitoring" / "performance_reports"


def log_actual_label(prediction_id: str, actual_churn: int):
    LABELS_LOG.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now().isoformat(),
        "prediction_id": prediction_id,
        "actual_churn": int(actual_churn)
    }
    file_exists = LABELS_LOG.exists()
    with open(LABELS_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    logger.info(f"Label réel enregistré pour prediction_id={prediction_id}")


def compute_real_metrics() -> dict | None:
    if not PREDICTIONS_LOG.exists() or not LABELS_LOG.exists():
        logger.warning("Logs de prédictions ou étiquettes réelles introuvables.")
        return None

    predictions = pd.read_csv(PREDICTIONS_LOG)
    labels = pd.read_csv(LABELS_LOG)

    merged = predictions.merge(labels, on="prediction_id", how="inner")

    if merged.empty:
        logger.warning("Aucune prédiction avec étiquette réelle disponible.")
        return None

    y_true = merged["actual_churn"]
    y_pred = merged["churn_prediction"]
    y_proba = merged["churn_probability"]

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "n_evaluated": len(merged),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4) if y_true.nunique() > 1 else None,
        "real_churn_rate": round(float(y_true.mean()), 4),
        "predicted_churn_rate": round(float(y_pred.mean()), 4),
    }

    REAL_METRICS_PATH.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REAL_METRICS_PATH / f"real_metrics_{ts}.json"
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Métriques réelles sauvegardées : {path}")

    return metrics


if __name__ == "__main__":
    metrics = compute_real_metrics()
    if metrics:
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
