import sys
import csv
import json
from pathlib import Path
from datetime import datetime

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "03_SRC"))

from utils.logger import get_logger

logger = get_logger(__name__)

PREDICTIONS_LOG = ROOT / "07_Monitoring" / "predictions_log.csv"
PERFORMANCE_REPORTS_PATH = ROOT / "07_Monitoring" / "performance_reports"


def log_prediction(input_data: dict, prediction: dict):
    PREDICTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "prediction_id": prediction.get("prediction_id"),
        "timestamp": datetime.now().isoformat(),
        **input_data,
        "churn_prediction": prediction.get("churn_prediction"),
        "churn_probability": prediction.get("churn_probability"),
        "risk_segment": prediction.get("risk_segment"),
        "recommended_action": prediction.get("recommended_action"),
    }
    file_exists = PREDICTIONS_LOG.exists()
    with open(PREDICTIONS_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def generate_performance_report() -> dict | None:
    if not PREDICTIONS_LOG.exists():
        logger.warning("Aucun log de prédictions trouvé.")
        return None

    df = pd.read_csv(PREDICTIONS_LOG)

    report = {
        "timestamp": datetime.now().isoformat(),
        "total_predictions": len(df),
        "churn_rate_predicted": round(float(df["churn_prediction"].mean()), 4),
        "avg_churn_probability": round(float(df["churn_probability"].mean()), 4),
        "risk_distribution": df["risk_segment"].value_counts().to_dict(),
        "action_distribution": df["recommended_action"].value_counts().to_dict(),
        "period": {
            "start": df["timestamp"].min(),
            "end": df["timestamp"].max()
        }
    }

    PERFORMANCE_REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = PERFORMANCE_REPORTS_PATH / f"performance_report_{ts}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Rapport généré : {report_path}")

    return report


if __name__ == "__main__":
    report = generate_performance_report()
    if report:
        print(json.dumps(report, indent=2, ensure_ascii=False))
