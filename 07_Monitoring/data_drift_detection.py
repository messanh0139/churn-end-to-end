import sys
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "03_SRC"))

from utils.logger import get_logger
from utils.config import DATA_PROCESSED_PATH, DATA_REFERENCE_PATH

logger = get_logger(__name__)

DRIFT_REPORTS_PATH = ROOT / "07_Monitoring" / "drift_reports"


def detect_numeric_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05) -> dict:
    results = {}
    for col in reference_df.select_dtypes(include=["number"]).columns:
        if col in current_df.columns:
            stat, p_value = ks_2samp(reference_df[col].dropna(), current_df[col].dropna())
            results[col] = {
                "test": "kolmogorov-smirnov",
                "statistic": round(float(stat), 4),
                "p_value": round(float(p_value), 4),
                "drift_detected": bool(p_value < threshold)
            }
    return results


def detect_categorical_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05) -> dict:
    results = {}
    for col in reference_df.select_dtypes(include=["object", "category"]).columns:
        if col in current_df.columns:
            categories = set(reference_df[col].dropna()) | set(current_df[col].dropna())
            ref_counts = reference_df[col].value_counts().reindex(categories, fill_value=0)
            cur_counts = current_df[col].value_counts().reindex(categories, fill_value=0)
            chi2, p_value, _, _ = chi2_contingency(pd.DataFrame({"ref": ref_counts, "cur": cur_counts}).T)
            results[col] = {
                "test": "chi2",
                "statistic": round(float(chi2), 4),
                "p_value": round(float(p_value), 4),
                "drift_detected": bool(p_value < threshold)
            }
    return results


def run_drift_detection(reference_path: Path, current_path: Path, threshold: float = 0.05) -> dict:
    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(current_path)

    results = {
        **detect_numeric_drift(reference_df, current_df, threshold),
        **detect_categorical_drift(reference_df, current_df, threshold)
    }

    n_drifted = sum(v["drift_detected"] for v in results.values())

    report = {
        "timestamp": datetime.now().isoformat(),
        "reference": str(reference_path),
        "current": str(current_path),
        "threshold": threshold,
        "total_features": len(results),
        "features_with_drift": n_drifted,
        "drift_alert": n_drifted > 0,
        "results": results
    }

    if n_drifted > 0:
        logger.warning(f"Drift détecté sur {n_drifted}/{len(results)} variable(s)")
    else:
        logger.info("Aucun drift détecté")

    DRIFT_REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = DRIFT_REPORTS_PATH / f"drift_report_{ts}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Rapport sauvegardé : {report_path}")

    return report


if __name__ == "__main__":
    reference = DATA_REFERENCE_PATH if DATA_REFERENCE_PATH.exists() else DATA_PROCESSED_PATH
    report = run_drift_detection(reference, DATA_PROCESSED_PATH)
    print(json.dumps(report, indent=2, ensure_ascii=False))
