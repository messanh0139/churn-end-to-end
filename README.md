# Customer_Churn_Prediction

Projet complet de prédiction et d'optimisation de l'attrition client pour la validation du Bloc 5.

## Modèles inclus
- Régression Logistique
- Random Forest
- SVC
- XGBoost

## Dataset
- Fichier : `01_Data/raw/data.csv`
- Lignes : 7043
- Colonnes : 21
- Cible : `Churn`

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r 06_MLOps/requirements.txt
```

## Exécution

```bash
python 03_SRC/data/clean_data.py
python 03_SRC/models/train_model.py
python 03_SRC/models/optimize_model.py
uvicorn 05_API.main:app --reload
```

## Notebooks
1. `01_data_understanding.ipynb`
2. `02_feature_engineering.ipynb`
3. `03_feature_selection.ipynb`
4. `04_model_training.ipynb`
5. `05_model_optimization.ipynb`
6. `06_model_interpretability.ipynb`
