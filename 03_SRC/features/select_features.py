import pandas as pd

def remove_useless_columns(df: pd.DataFrame, columns_to_remove: list[str]) -> pd.DataFrame:
    existing_columns = [col for col in columns_to_remove if col in df.columns]
    return df.drop(columns=existing_columns)

def get_feature_types(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return numeric_cols, categorical_cols
