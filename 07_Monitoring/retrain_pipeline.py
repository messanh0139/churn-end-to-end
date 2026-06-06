import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "03_SRC"))

from utils.logger import get_logger
from utils.config import DATA_PROCESSED_PATH

logger = get_logger(__name__)

DRIFT_RATIO_THRESHOLD = 0.3


def should_retrain(drift_report: dict) -> bool:
    total = drift_report.get("total_features", 0)
    if total == 0:
        return False
    return (drift_report.get("features_with_drift", 0) / total) >= DRIFT_RATIO_THRESHOLD


def run(current_path: Path = DATA_PROCESSED_PATH, force: bool = False):
    from data_drift_detection import run_drift_detection
    from models.optimize_model import optimize

    logger.info("--- Lancement du pipeline de réentraînement ---")

    logger.info("Étape 1 : détection du drift")
    drift_report = run_drift_detection(DATA_PROCESSED_PATH, current_path)

    if force:
        logger.info("Réentraînement forcé (--force)")
        optimize()
    elif should_retrain(drift_report):
        n = drift_report["features_with_drift"]
        total = drift_report["total_features"]
        logger.warning(f"Drift sur {n}/{total} variables — réentraînement lancé")
        from alerts import send_drift_alert
        send_drift_alert(drift_report)
        optimize()
    else:
        logger.info("Drift insuffisant — pas de réentraînement nécessaire")

    return drift_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de réentraînement automatique")
    parser.add_argument("--current-data", type=Path, default=DATA_PROCESSED_PATH,
                        help="Chemin vers les nouvelles données")
    parser.add_argument("--force", action="store_true", help="Forcer le réentraînement")
    args = parser.parse_args()

    report = run(current_path=args.current_data, force=args.force)
    print(json.dumps(report, indent=2, ensure_ascii=False))
