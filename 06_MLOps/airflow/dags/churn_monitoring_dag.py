"""
DAGs de monitoring MLOps pour le modèle de prédiction de churn.

- churn_daily_monitoring   : rapport de performance + métriques réelles (chaque jour à 8h)
- churn_weekly_drift_check : ingestion → drift detection → réentraînement conditionnel (chaque lundi à 8h)
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator, ShortCircuitOperator

PROJECT = Path("/opt/airflow/project")
MONITORING = PROJECT / "07_Monitoring"
DRIFT_REPORTS_DIR = MONITORING / "drift_reports"
DRIFT_RATIO_THRESHOLD = 0.3

PYTHON_ENV = {
    "PYTHONPATH": f"{PROJECT}/03_SRC:{PROJECT}/07_Monitoring",
}

default_args = {
    "owner": "churn-mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


# DAG 1 : Monitoring quotidien
with DAG(
    dag_id="churn_daily_monitoring",
    description="Rapport de performance + métriques réelles du modèle de churn",
    schedule="0 8 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["churn", "monitoring", "daily"],
) as daily_dag:

    generate_performance_report = BashOperator(
        task_id="generate_performance_report",
        bash_command=f"python {MONITORING}/monitor_performance.py || true",
        env=PYTHON_ENV,
        append_env=True,
    )

    compute_real_metrics = BashOperator(
        task_id="compute_real_metrics",
        bash_command=f"python {MONITORING}/label_tracker.py || true",
        env=PYTHON_ENV,
        append_env=True,
    )

    generate_performance_report >> compute_real_metrics


# DAG 2 : Ingestion, drift detection et réentraînement conditionnel (hebdomadaire)
def _ingest_new_data(**context) -> bool:
    """Ingère les nouvelles données depuis 01_Data/incoming/data_new.csv.
    Retourne True si des données ont été ingérées (pipeline continue),
    False sinon (ShortCircuitOperator arrête le pipeline proprement).
    """
    sys.path.insert(0, str(PROJECT / "03_SRC"))
    from data.ingest_data import run as ingest_run  # noqa: PLC0415
    return ingest_run()


def _decide_retrain(**context):
    """Lit le dernier rapport de drift et décide si on réentraîne."""
    reports = sorted(DRIFT_REPORTS_DIR.glob("drift_report_*.json"))
    if not reports:
        return "drift_ok"

    report = json.loads(reports[-1].read_text(encoding="utf-8"))
    total = report.get("total_features", 0)
    drifted = report.get("features_with_drift", 0)

    if total > 0 and (drifted / total) >= DRIFT_RATIO_THRESHOLD:
        context["ti"].xcom_push(key="drift_report_path", value=str(reports[-1]))
        return "retrain_model"
    return "drift_ok"


def _send_drift_alert(**context):
    """Log le rapport de drift (Slack/email optionnels : ne bloque jamais le pipeline)."""
    try:
        sys.path.insert(0, str(PROJECT / "07_Monitoring"))
        sys.path.insert(0, str(PROJECT / "03_SRC"))
        from alerts import send_drift_alert  # noqa: PLC0415
        from utils.logger import get_logger  # noqa: PLC0415
        log = get_logger("send_drift_alert")

        ti = context["ti"]
        report_path = ti.xcom_pull(task_ids="decide_retrain", key="drift_report_path")
        if not report_path:
            return
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        send_drift_alert(report)
        log.info("Alerte drift traitée")
    except Exception as exc:  # noqa: BLE001
        import logging
        logging.getLogger("send_drift_alert").warning(
            "send_drift_alert non critique : ignoré : %s", exc
        )


with DAG(
    dag_id="churn_weekly_drift_check",
    description="Détection de drift des données + réentraînement automatique du modèle",
    schedule="0 8 * * 1",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["churn", "drift", "retraining", "weekly"],
) as weekly_dag:

    fetch_new_data = ShortCircuitOperator(
        task_id="fetch_new_data",
        python_callable=_ingest_new_data,
        ignore_downstream_trigger_rules=True,
    )

    detect_data_drift = BashOperator(
        task_id="detect_data_drift",
        bash_command=f"python {MONITORING}/data_drift_detection.py",
        env=PYTHON_ENV,
        append_env=True,
    )

    decide_retrain = BranchPythonOperator(
        task_id="decide_retrain",
        python_callable=_decide_retrain,
    )

    retrain_model = BashOperator(
        task_id="retrain_model",
        bash_command=f"python {PROJECT}/03_SRC/models/optimize_model.py",
        env=PYTHON_ENV,
        append_env=True,
    )

    send_alert = PythonOperator(
        task_id="send_drift_alert",
        python_callable=_send_drift_alert,
    )

    def _log_no_drift(**context):
        import logging
        reports = sorted(DRIFT_REPORTS_DIR.glob("drift_report_*.json"))
        if reports:
            report = json.loads(reports[-1].read_text(encoding="utf-8"))
            drifted = report.get("features_with_drift", 0)
            total = report.get("total_features", 0)
            logging.getLogger("drift_ok").info(
                "Aucun réentraînement nécessaire : drift %d/%d variables (seuil 30%%)",
                drifted, total,
            )

    drift_ok = PythonOperator(
        task_id="drift_ok",
        python_callable=_log_no_drift,
    )

    fetch_new_data >> detect_data_drift >> decide_retrain >> [retrain_model, drift_ok]
    retrain_model >> send_alert
