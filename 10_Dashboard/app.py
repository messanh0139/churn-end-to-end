import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MONITORING_DIR = ROOT / "07_Monitoring"
MODELS_DIR = ROOT / "04_Models"
# URL de l'API, peut être surchargée via variable d'environnement en production
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Customer Churn Dashboard",
    layout="wide",
)

st.title("Customer Churn Prediction — Dashboard Métier")

tab1, tab2, tab3, tab4 = st.tabs(["Prédiction client", "Monitoring", "Comparaison des modèles", "Ingestion données"])


with tab1:
    st.header("Prédire le risque de churn d'un client")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Profil client")
        gender = st.selectbox("Genre", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Oui" if x else "Non")
        partner = st.selectbox("Partenaire", ["Yes", "No"])
        dependents = st.selectbox("Personnes à charge", ["Yes", "No"])
        tenure = st.slider("Ancienneté (mois)", 0, 72, 12)
        monthly_charges = st.number_input("Charges mensuelles (€)", 0.0, 200.0, 65.0, step=0.5)
        total_charges = st.number_input("Charges totales (€)", 0.0, 10000.0, float(monthly_charges * tenure), step=1.0)

    with col2:
        st.subheader("Services")
        phone_service = st.selectbox("Téléphonie", ["Yes", "No"])
        multiple_lines = st.selectbox("Lignes multiples", ["Yes", "No", "No phone service"])
        internet = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Sécurité en ligne", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Sauvegarde en ligne", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Protection appareil", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Support technique", ["Yes", "No", "No internet service"])

    with col3:
        st.subheader("Contrat & Paiement")
        streaming_tv = st.selectbox("TV streaming", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Films streaming", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Type de contrat", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Facturation dématérialisée", ["Yes", "No"])
        payment = st.selectbox("Mode de paiement", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ])

    st.divider()

    if st.button("Lancer la prédiction", type="primary", use_container_width=True):
        payload = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone_service,
            "MultipleLines": multiple_lines, "InternetService": internet,
            "OnlineSecurity": online_security, "OnlineBackup": online_backup,
            "DeviceProtection": device_protection, "TechSupport": tech_support,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            result = response.json()

            prob = result["churn_probability"]
            segment = result["risk_segment"]
            action = result["recommended_action"]
            prediction_id = result["prediction_id"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Probabilité de churn", f"{prob * 100:.1f}%")
            c2.metric("Prédiction", "Churn" if result["churn_prediction"] else "Fidèle")
            c3.metric("Segment de risque", segment)
            c4.metric("Action recommandée", action)

            st.progress(prob, text=f"Score de risque : {prob * 100:.1f}%")

            # Seuils définis avec l'équipe métier pour catégoriser le niveau de risque
            if prob >= 0.80:
                st.error(f"Risque très élevé — {action}")
            elif prob >= 0.60:
                st.warning(f"Risque élevé — {action}")
            elif prob >= 0.30:
                st.info(f"Risque moyen — {action}")
            else:
                st.success(f"Risque faible — {action}")

            st.caption(f"ID prédiction : `{prediction_id}`")

            with st.expander("Enregistrer le vrai résultat (feedback)"):
                actual = st.radio("Ce client a-t-il finalement churné ?", ["Oui", "Non"], horizontal=True)
                if st.button("Enregistrer"):
                    fb = requests.post(f"{API_URL}/feedback", json={
                        "prediction_id": prediction_id,
                        "actual_churn": 1 if actual == "Oui" else 0,
                    }, timeout=10)
                    st.success(f"Feedback enregistré — ID : `{prediction_id[:12]}...`")

        except Exception as e:
            st.error(f"Erreur API : {e}. Vérifiez que l'API tourne sur {API_URL}")


with tab2:
    st.header("Surveillance du modèle en production")

    log_path = MONITORING_DIR / "predictions_log.csv"

    if log_path.exists():
        df = pd.read_csv(log_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        st.subheader("Indicateurs clés")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total prédictions", len(df))
        c2.metric("Taux de churn prédit", f"{df['churn_prediction'].mean() * 100:.1f}%")
        c3.metric("Probabilité moyenne", f"{df['churn_probability'].mean() * 100:.1f}%")
        c4.metric("Dernière prédiction", df["timestamp"].max().strftime("%d/%m %H:%M"))

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Répartition par segment de risque")
            risk_counts = df["risk_segment"].value_counts().rename_axis("Segment").reset_index(name="Nombre")
            st.bar_chart(risk_counts.set_index("Segment"))

        with col_b:
            st.subheader("Évolution du taux de churn prédit")
            daily = df.set_index("timestamp").resample("D")["churn_prediction"].mean() * 100
            daily.name = "Taux de churn (%)"
            st.line_chart(daily)
    else:
        st.info("Aucune prédiction enregistrée. Lancez l'API et faites des prédictions.")

    st.divider()

    col_drift, col_metrics = st.columns(2)

    with col_drift:
        st.subheader("Détection de drift (dernier rapport)")
        drift_dir = MONITORING_DIR / "drift_reports"
        reports = sorted(drift_dir.glob("drift_report_*.json")) if drift_dir.exists() else []
        if reports:
            report = json.loads(reports[-1].read_text(encoding="utf-8"))
            n_drifted = report["features_with_drift"]
            total = report["total_features"]
            ratio = n_drifted / total * 100 if total > 0 else 0

            st.metric("Variables en drift", f"{n_drifted} / {total}", f"{ratio:.0f}%")

            if n_drifted > 0:
                st.warning("Drift détecté — réentraînement recommandé si ratio >= 30%")
                drifted_rows = [
                    {"Variable": col, "Test": v["test"], "p-value": v["p_value"]}
                    for col, v in report["results"].items()
                    if v["drift_detected"]
                ]
                st.dataframe(pd.DataFrame(drifted_rows), hide_index=True)
            else:
                st.success("Aucun drift détecté")
        else:
            st.info("Aucun rapport de drift disponible.")

    with col_metrics:
        st.subheader("Métriques réelles (feedbacks clients)")
        perf_dir = MONITORING_DIR / "performance_reports"
        real_files = sorted(perf_dir.glob("real_metrics_*.json")) if perf_dir.exists() else []
        if real_files:
            m = json.loads(real_files[-1].read_text(encoding="utf-8"))
            st.metric("Évaluations disponibles", m.get("n_evaluated", 0))
            c1, c2, c3 = st.columns(3)
            c1.metric("Précision", f"{m['precision'] * 100:.1f}%" if m.get("precision") is not None else "N/A")
            c2.metric("Rappel", f"{m['recall'] * 100:.1f}%" if m.get("recall") is not None else "N/A")
            c3.metric("F1-Score", f"{m['f1_score'] * 100:.1f}%" if m.get("f1_score") is not None else "N/A")
        else:
            st.info("Envoyez des feedbacks via /feedback pour calculer les métriques réelles.")


with tab3:
    st.header("Comparaison des modèles entraînés")

    meta_path = MODELS_DIR / "model_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        best = meta.get("best_model", "svc")
        st.success(f"Modèle en production : {best.upper()}")

    comparison_path = MODELS_DIR / "model_comparison_results.csv"
    if comparison_path.exists():
        df = pd.read_csv(comparison_path, index_col=0)
        display_cols = [c for c in ["accuracy", "precision", "recall", "f1_score", "roc_auc"] if c in df.columns]

        st.subheader("Tableau comparatif")
        st.dataframe(
            df[display_cols].style.highlight_max(axis=0, color="#d4edda").format("{:.4f}"),
            use_container_width=True,
        )

        if "f1_score" in df.columns:
            st.subheader("F1-Score par modèle")
            st.bar_chart(df["f1_score"].sort_values())

        if "roc_auc" in df.columns:
            st.subheader("ROC-AUC par modèle")
            st.bar_chart(df["roc_auc"].sort_values())
    else:
        st.info("Lancez train_model.py pour générer les résultats de comparaison.")


with tab4:
    st.header("Déposer de nouvelles données")
    st.write(
        "Déposez un fichier CSV client pour déclencher la détection de drift "
        "et le réentraînement automatique du modèle via Airflow."
    )

    INCOMING_DIR = ROOT / "01_Data" / "incoming"
    incoming_file = INCOMING_DIR / "data_new.csv"

    uploaded = st.file_uploader("Fichier CSV (même format que les données d'entraînement)", type="csv")

    if uploaded is not None:
        try:
            df_preview = pd.read_csv(uploaded)
            st.write(f"{len(df_preview)} lignes, {len(df_preview.columns)} colonnes")
            st.dataframe(df_preview.head(3), hide_index=True)

            if st.button("Valider et déposer dans le pipeline", type="primary"):
                INCOMING_DIR.mkdir(parents=True, exist_ok=True)
                uploaded.seek(0)
                incoming_file.write_bytes(uploaded.read())
                st.success(
                    f"Fichier déposé dans incoming/. "
                    "Airflow le traitera au prochain run (lundi 8h) ou via un trigger manuel."
                )
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier : {e}")

    st.divider()
    st.subheader("Statut du pipeline")

    if incoming_file.exists():
        size_kb = incoming_file.stat().st_size // 1024
        st.warning(f"Fichier en attente de traitement : data_new.csv ({size_kb} Ko) — déclenchez le DAG dans Airflow.")
    else:
        st.success("Aucun fichier en attente — pipeline disponible.")

    # Résultat du dernier traitement
    drift_dir = MONITORING_DIR / "drift_reports"
    reports = sorted(drift_dir.glob("drift_report_*.json")) if drift_dir.exists() else []
    if reports:
        last = json.loads(reports[-1].read_text(encoding="utf-8"))
        n = last["features_with_drift"]
        total = last["total_features"]
        ts = last["timestamp"][:19].replace("T", " ")
        ratio = round(n / total * 100) if total > 0 else 0

        st.subheader("Résultat du dernier pipeline")
        c1, c2, c3 = st.columns(3)
        c1.metric("Date du traitement", ts)
        c2.metric("Variables en drift", f"{n} / {total}")
        c3.metric("Taux de drift", f"{ratio}%")

        if ratio >= 30:
            st.success(f"Drift détecté ({ratio}%) — modèle ré-entraîné automatiquement.")
        else:
            st.info(f"Drift faible ({ratio}%) — modèle inchangé.")

    if st.button("Rafraîchir le statut"):
        st.rerun()
