"""
DAGs de monitoring MLOps pour le modèle de prédiction de churn.

- churn_daily_monitoring : rapport de performance + métriques réelles (chaque jour à 8h)
- churn_weekly_drift_check : détection de drift + réentraînement si nécessaire (chaque lundi à 8h)
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

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


# ---------------------------------------------------------------------------
# DAG 1 — Monitoring quotidien
# ---------------------------------------------------------------------------
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
        bash_command=(
            f"python {MONITORING}/monitor_performance.py"
        ),
        env=PYTHON_ENV,
        append_env=True,
    )

    compute_real_metrics = BashOperator(
        task_id="compute_real_metrics",
        bash_command=(
            f"python {MONITORING}/label_tracker.py"
        ),
        env=PYTHON_ENV,
        append_env=True,
    )

    generate_performance_report >> compute_real_metrics


# ---------------------------------------------------------------------------
# DAG 2 — Détection de drift hebdomadaire + réentraînement conditionnel
# ---------------------------------------------------------------------------
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
    """Envoie une alerte Slack + email avec le rapport de drift."""
    sys.path.insert(0, str(PROJECT / "07_Monitoring"))
    sys.path.insert(0, str(PROJECT / "03_SRC"))
    from alerts import send_drift_alert  # noqa: PLC0415

    ti = context["ti"]
    report_path = ti.xcom_pull(task_ids="decide_retrain", key="drift_report_path")
    if report_path:
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        send_drift_alert(report)


with DAG(
    dag_id="churn_weekly_drift_check",
    description="Détection de drift des données + réentraînement automatique du modèle",
    schedule="0 8 * * 1",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["churn", "drift", "retraining", "weekly"],
) as weekly_dag:

    detect_data_drift = BashOperator(
        task_id="detect_data_drift",
        bash_command=(
            f"python {MONITORING}/data_drift_detection.py"
        ),
        env=PYTHON_ENV,
        append_env=True,
    )

    decide_retrain = BranchPythonOperator(
        task_id="decide_retrain",
        python_callable=_decide_retrain,
    )

    retrain_model = BashOperator(
        task_id="retrain_model",
        bash_command=(
            f"python {PROJECT}/03_SRC/models/optimize_model.py"
        ),
        env=PYTHON_ENV,
        append_env=True,
    )

    send_alert = PythonOperator(
        task_id="send_drift_alert",
        python_callable=_send_drift_alert,
    )

    drift_ok = EmptyOperator(task_id="drift_ok")

    detect_data_drift >> decide_retrain >> [retrain_model, drift_ok]
    retrain_model >> send_alert
