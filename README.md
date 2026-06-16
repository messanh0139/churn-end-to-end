# Customer Churn Prediction

Projet complet de prédiction et d'optimisation de l'attrition client — Bloc 5 (MLOps).

## Stack technique

| Composant | Technologie |
|-----------|-------------|
| Modèle | SVC (kernel=rbf, class_weight=balanced) |
| API | FastAPI + Uvicorn |
| Orchestration | Apache Airflow |
| Tracking | MLflow |
| Conteneurisation | Docker / Docker Compose |
| CI/CD | GitHub Actions |

## Dataset

- Fichier : `01_Data/raw/data.csv`
- Lignes : 7043 — Colonnes : 21 — Cible : `Churn`

## Déploiement (Docker)

### Lancer tous les services

```bash
docker-compose up --build
```

### Services disponibles

| Service | URL | Identifiants |
|---------|-----|--------------|
| API REST | http://localhost:8000 | — |
| Documentation API | http://localhost:8000/docs | — |
| Dashboard Streamlit | http://localhost:8501 | — |
| MLflow UI | http://localhost:5000 | — |
| Airflow | http://localhost:8080 | admin / admin |

### Arrêter les services

```bash
docker-compose down
```

## Modèles inclus

- Régression Logistique
- Random Forest
- SVC *(meilleur modèle)*
- XGBoost

## Développement local (sans Docker)

### Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows
pip install -r 06_MLOps/requirements.txt
```

### Exécution

```bash
python 03_SRC/data/clean_data.py
python 03_SRC/models/train_model.py
python 03_SRC/models/optimize_model.py
uvicorn 05_API.main:app --reload
```

## CI/CD (GitHub Actions)

Le pipeline se déclenche automatiquement à chaque push sur `main` :

1. **Tests** — `pytest tests/`
2. **Entraînement** — `train_model.py` + upload de l'artefact `best_model.pkl`
3. **Build Docker** — construction de l'image `churn-api:latest`

Un déclenchement manuel (**workflow_dispatch**) est également disponible pour ré-entraîner le modèle à la demande.

## Notebooks

1. `01_data_understanding.ipynb`
2. `02_feature_engineering.ipynb`
3. `03_feature_selection.ipynb`
4. `04_model_training.ipynb`
5. `05_model_optimization.ipynb`
6. `06_model_interpretability.ipynb`

## Architecture

```
Customer_Churn_Prediction/
├── 01_Data/          # Données brutes et traitées
├── 02_Notebooks/     # Exploration et analyse
├── 03_SRC/           # Code source (data, features, models)
├── 04_Models/        # Modèle entraîné (best_model.pkl)
├── 05_API/           # FastAPI (main.py, predict.py)
├── 06_MLOps/         # Docker, Airflow, MLflow, requirements
├── 07_Monitoring/    # Drift, performance, alertes, retraining
├── 10_Dashboard/     # Streamlit dashboard
└── tests/            # Tests unitaires pytest
```
