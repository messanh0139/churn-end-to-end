import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "tenure" in df.columns and "MonthlyCharges" in df.columns:
        df["avg_charge_per_tenure"] = df["MonthlyCharges"] / df["tenure"].replace(0, 1)

    if "InternetService" in df.columns:
        df["has_fiber"] = (df["InternetService"] == "Fiber optic").astype(int)

    if "Contract" in df.columns:
        df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)

    if "OnlineSecurity" in df.columns:
        df["has_online_security"] = (df["OnlineSecurity"] == "Yes").astype(int)

    if "TechSupport" in df.columns:
        df["has_tech_support"] = (df["TechSupport"] == "Yes").astype(int)

    return df
