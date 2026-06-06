import sys
import uuid
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "03_SRC"))
sys.path.append(str(ROOT / "07_Monitoring"))

from models.predict import predict_churn
from monitor_performance import log_prediction
from label_tracker import log_actual_label

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API de prédiction de l'attrition client",
    version="1.0.0"
)

class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

class FeedbackInput(BaseModel):
    prediction_id: str
    actual_churn: int

@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running"}

@app.post("/predict")
def predict(customer: CustomerInput):
    prediction_id = str(uuid.uuid4())
    input_data = customer.model_dump()
    result = predict_churn(input_data)
    result["prediction_id"] = prediction_id
    log_prediction(input_data, result)
    return result

@app.post("/feedback")
def feedback(data: FeedbackInput):
    log_actual_label(data.prediction_id, data.actual_churn)
    return {"status": "ok", "prediction_id": data.prediction_id}
