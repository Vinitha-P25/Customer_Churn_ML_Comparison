import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()

model = joblib.load("churn_model.joblib")

class CustomerData(BaseModel):
    Age: float
    Tenure: float
    MonthlyCharges: float
    SupportCalls: float
    ContractLength: float
    TotalCharges: float
    InternetService: str
    PaymentMethod: str
    NumServices: float

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}


@app.post("/predict")
def predict_churn(customer: CustomerData):

    input_data = pd.DataFrame([customer.model_dump()])

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    churn_probability = probability[0][1] * 100

    if prediction[0] == 1:
        result = "Churn"
    else:
        result = "No Churn"

    return {
    "churn_prediction": result,
    "churn_probability": round(churn_probability, 2)
}