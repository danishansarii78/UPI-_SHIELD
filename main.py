# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Create the FastAPI app
app = FastAPI()

# --- Data Model for Input ---
# This tells FastAPI what kind of data to expect in a request.
# It should have all the features our model was trained on.
# We are creating a class that inherits from Pydantic's BaseModel
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

# --- Load the Model ---
# Load the pre-trained model from the file
model = joblib.load('fraud_model.pkl')

# --- API Endpoint ---
# This is the main endpoint that will receive transaction data and return a prediction.
@app.post("/predict")
def predict_fraud(transaction: Transaction):
    # Convert the input data into a pandas DataFrame
    input_data = pd.DataFrame([transaction.dict()])

    # Make a prediction
    prediction = model.predict(input_data)[0]
    
    # Get the probability score for fraud
    # predict_proba returns [[prob_class_0, prob_class_1]]
    probability = model.predict_proba(input_data)[0][1]

    # Determine the result message
    result = "Fraud" if prediction == 1 else "Not Fraud"
    
    # Return the result in a JSON format
    return {
        "prediction": result,
        "fraud_probability_score": f"{probability:.4f}"
    }

# A simple root endpoint to check if the API is running
@app.get("/")
def read_root():
    return {"message": "UPI Shield Fraud Detection API is running!"}