#!/bin/bash
# Fix all directory permissions before docker-compose up
# Airflow runs as host UID (1000) via user: "${AIRFLOW_UID:-1000}:0"

set -e
echo "Fixing permissions for Docker containers..."

# Airflow logs (may be owned by root from previous runs)
sudo chown -R "$(id -u):$(id -g)" 06_MLOps/airflow/logs/ 2>/dev/null || true

# Directories that containers need to write to
chmod 777 06_MLOps/airflow/logs/
chmod o+w 06_MLOps/
chmod 666 06_MLOps/mlruns.db 2>/dev/null || true

chmod 777 07_Monitoring/drift_reports/
chmod 777 07_Monitoring/performance_reports/
sudo chown "$(id -u):$(id -g)" 07_Monitoring/predictions_log.csv 07_Monitoring/actual_labels.csv 2>/dev/null || true

chmod 777 01_Data/processed/
chmod 777 01_Data/incoming/

chmod 777 04_Models/

echo "Done. You can now run: docker-compose up"
