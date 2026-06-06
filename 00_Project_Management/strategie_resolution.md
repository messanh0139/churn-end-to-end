# Stratégie de résolution

## Type de problème
Classification binaire supervisée.

## Cible
- 0 : client non churn
- 1 : client churn

## Modèles retenus
- Régression Logistique : baseline interprétable.
- Random Forest : modèle robuste sur données tabulaires.
- SVC : modèle à marge adapté aux séparations complexes.
- XGBoost : modèle de boosting performant.

## Métriques
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Le recall et le F1-score sont prioritaires pour identifier les clients à risque.
