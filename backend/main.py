from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle
import json
import os

app = FastAPI(title="Credit Card Fraud Detection API")

# Allow CORS for local HTML file testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model pipeline globally
MODEL_PATH = 'models/best_model.pkl'
METRICS_PATH = 'models/metrics.json'

preprocessor = None
model = None

@app.on_event("startup")
def load_model():
    global preprocessor, model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            data = pickle.load(f)
            preprocessor = data['preprocessor']
            model = data['model']
            print(f"Loaded {data['model_name']} model successfully.")
    else:
        print(f"WARNING: Model file not found at {MODEL_PATH}")

class TransactionRecord(BaseModel):
    amount: float
    transaction_hour: int
    merchant_type: str
    location: str
    account_age: int

@app.get("/metrics")
def get_metrics():
    """Serves the model metrics output to the frontend for charting"""
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            return json.load(f)
    return {"error": "Metrics file not found. Train the model first."}

@app.post("/predict")
def predict_fraud(record: TransactionRecord):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    try:
        # 1. Convert to DataFrame mimicking training structure
        input_data = pd.DataFrame([record.model_dump()])
        
        # 2. Preprocess
        processed_data = preprocessor.transform(input_data)
        
        # 3. Predict
        prob = model.predict_proba(processed_data)[0][1]
        is_fraud = bool(prob > 0.5) 
        
        # 4. Generate basic explanation rules based on feature inputs
        explanation = []
        if record.amount > 1000:
            explanation.append("High transaction amount.")
        if record.transaction_hour >= 2 and record.transaction_hour <= 5:
            explanation.append("Unusual transaction hour.")
        if record.location in ['International', 'DarkWeb']:
            explanation.append("High risk location.")
        if record.merchant_type == 'Crypto':
            explanation.append("High risk merchant category.")
        if record.account_age < 30:
            explanation.append("New account.")
            
        if not explanation:
            explanation.append("Transaction characteristics appear normal.")

        return {
            "fraud_probability": round(prob * 100, 2),
            "prediction": "Fraud" if is_fraud else "Safe",
            "explanation": " ".join(explanation)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # When running directly `python main.py`
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
