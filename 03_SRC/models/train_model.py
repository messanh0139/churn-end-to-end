import sys
import json
from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "03_SRC"))

from data.clean_data import clean_dataframe
from features.build_features import build_features
from features.select_features import remove_useless_columns, get_feature_types
from models.evaluate_model import evaluate_classification_model
from utils.config import DATA_RAW_PATH, BEST_MODEL_PATH, METADATA_PATH, TARGET_COLUMN, ID_COLUMN, RANDOM_STATE, TEST_SIZE
from utils.logger import get_logger
from utils.mlflow_tracker import setup, start_run, log_sklearn_model, log_params, log_metrics, log_dataset

logger = get_logger(__name__)

MLFLOW_EXPERIMENT = "churn_model_comparison"


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols, categorical_cols = get_feature_types(X)
    return ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])


def train():
    setup(MLFLOW_EXPERIMENT)

    df = pd.read_csv(DATA_RAW_PATH)
    df = clean_dataframe(df)
    df = build_features(df)
    df = remove_useless_columns(df, [ID_COLUMN])

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
        ),
        "svc": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
    }

    if XGBOOST_AVAILABLE:
        models["xgboost"] = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1
        )

    results = {}
    best_model_name = None
    best_pipeline = None
    best_f1 = -1

    for name, model in models.items():
        logger.info(f"Entraînement : {name}")

        with start_run(run_name=name):
            log_dataset(df, DATA_RAW_PATH, TARGET_COLUMN)
            pipeline = Pipeline([
                ("preprocessor", build_preprocessor(X_train)),
                ("model", model)
            ])
            pipeline.fit(X_train, y_train)
            metrics = evaluate_classification_model(pipeline, X_test, y_test)
            results[name] = metrics

            log_params({"model_type": name, "test_size": TEST_SIZE, "random_state": RANDOM_STATE})
            log_metrics(metrics)
            log_sklearn_model(pipeline)

            if metrics["f1_score"] > best_f1:
                best_f1 = metrics["f1_score"]
                best_model_name = name
                best_pipeline = pipeline

    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, BEST_MODEL_PATH)

    metadata = {
        "best_model": best_model_name,
        "models_compared": list(models.keys()),
        "target": TARGET_COLUMN,
        "metrics": results,
        "model_path": str(BEST_MODEL_PATH),
        "xgboost_available": XGBOOST_AVAILABLE,
    }

    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(results).T.to_csv(ROOT / "04_Models" / "model_comparison_results.csv")

    logger.info(f"Meilleur modèle : {best_model_name} (F1={best_f1})")
    logger.info(f"Modèle sauvegardé : {BEST_MODEL_PATH}")


if __name__ == "__main__":
    train()
