import sys
import json
from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "03_SRC"))

from data.clean_data import clean_dataframe
from features.build_features import build_features
from features.select_features import remove_useless_columns, get_feature_types
from utils.config import DATA_RAW_PATH, BEST_MODEL_PATH, METADATA_PATH, TARGET_COLUMN, ID_COLUMN, RANDOM_STATE, TEST_SIZE
from utils.mlflow_tracker import setup, start_run, log_sklearn_model, log_params, log_metrics, log_dataset

import mlflow

MLFLOW_EXPERIMENT = "churn_svc_optimization"


def get_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }


def optimize():
    setup(MLFLOW_EXPERIMENT)

    df = pd.read_csv(DATA_RAW_PATH)
    df = clean_dataframe(df)
    df = build_features(df)
    df = remove_useless_columns(df, [ID_COLUMN])

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    numeric_cols, categorical_cols = get_feature_types(X)

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = SVC(class_weight="balanced", probability=True, random_state=RANDOM_STATE)
    param_distributions = {
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
        "model__kernel": ["rbf", "poly", "sigmoid"],
    }

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    search = RandomizedSearchCV(
        pipeline, param_distributions=param_distributions,
        n_iter=20, scoring="f1", cv=5,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=1,
    )

    with start_run(run_name="svc_randomizedsearch"):
        log_dataset(df, DATA_RAW_PATH, TARGET_COLUMN)
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        metrics = get_metrics(best_model, X_test, y_test)

        log_params({**search.best_params_, "n_iter": 20, "cv": 5, "scoring": "f1"})
        mlflow.log_metric("best_cv_f1", round(search.best_score_, 4))
        log_metrics(metrics)
        log_sklearn_model(best_model)

        BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, BEST_MODEL_PATH)

        metadata = {
            "optimized_model": "svc",
            "best_params": search.best_params_,
            "best_cv_f1": round(search.best_score_, 4),
            "test_metrics": metrics,
            "model_path": str(BEST_MODEL_PATH),
        }

        METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    optimize()
