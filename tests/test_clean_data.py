import pandas as pd

from data.clean_data import clean_dataframe

def test_clean_dataframe():
    df = pd.DataFrame({
        "TotalCharges": ["10.5", " "],
        "Churn": ["Yes", "No"]
    })

    cleaned = clean_dataframe(df)

    assert cleaned["TotalCharges"].isna().sum() == 0
    assert set(cleaned["Churn"].unique()) == {0, 1}
