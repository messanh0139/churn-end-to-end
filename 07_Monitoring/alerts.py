import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import requests

from utils.logger import get_logger

logger = get_logger(__name__)


def send_slack_alert(message: str) -> bool:
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        logger.warning("SLACK_WEBHOOK_URL non configuré : alerte Slack ignorée")
        return False
    try:
        response = requests.post(webhook_url, json={"text": message}, timeout=10)
        response.raise_for_status()
        logger.info("Alerte Slack envoyée")
        return True
    except Exception as e:
        logger.error(f"Échec alerte Slack : {e}")
        return False


def send_email_alert(subject: str, body: str) -> bool:
    sender = os.environ.get("ALERT_EMAIL_SENDER")
    password = os.environ.get("ALERT_EMAIL_PASSWORD")
    receiver = os.environ.get("ALERT_EMAIL_RECEIVER")

    if not all([sender, password, receiver]):
        logger.warning("Credentials email non configurés : alerte email ignorée")
        return False
    try:
        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = receiver
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())

        logger.info(f"Email envoyé à {receiver}")
        return True
    except Exception as e:
        logger.error(f"Échec alerte email : {e}")
        return False


def send_drift_alert(drift_report: dict):
    n = drift_report["features_with_drift"]
    total = drift_report["total_features"]
    ts = drift_report["timestamp"]

    drifted_features = [
        f"  - {col} (p={v['p_value']})"
        for col, v in drift_report["results"].items()
        if v["drift_detected"]
    ]

    message = (
        f"Drift détecté : {ts}\n"
        f"{n}/{total} variables ont dérivé :\n"
        + "\n".join(drifted_features)
    )

    send_slack_alert(message)
    send_email_alert(
        subject=f"[Churn API] Drift détecté : {n}/{total} variables",
        body=message,
    )
