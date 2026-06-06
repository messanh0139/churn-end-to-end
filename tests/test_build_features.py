import pandas as pd

from features.build_features import build_features

def test_build_features():
    df = pd.DataFrame({
        "tenure": [10],
        "MonthlyCharges": [50.0],
        "InternetService": ["Fiber optic"],
        "Contract": ["Month-to-month"],
        "OnlineSecurity": ["No"],
        "TechSupport": ["Yes"]
    })

    result = build_features(df)

    assert "avg_charge_per_tenure" in result.columns
    assert "has_fiber" in result.columns
    assert "is_month_to_month" in result.columns
    assert "has_online_security" in result.columns
    assert "has_tech_support" in result.columns
