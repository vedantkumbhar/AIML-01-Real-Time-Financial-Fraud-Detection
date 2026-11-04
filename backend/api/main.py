from pathlib import Path
import time

import joblib
import numpy as np
import xgboost as xgb

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from backend.features.feast_service import get_transaction_features

# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "realtime_fraud_model.json"
FEATURE_PATH = ROOT / "models" / "realtime_feature_columns.pkl"

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

FEATURES = joblib.load(FEATURE_PATH)

app = FastAPI(
    title="Nexvia Financial Fraud Detection API",
    description="Real-time transaction fraud scoring using XGBoost",
    version="1.0.0"
)

# --------------------------------------------------
# INPUT SCHEMA
# --------------------------------------------------

class Transaction(BaseModel):
    step: int = Field(ge=1)
    transaction_type: str
    amount: float = Field(ge=0)
    oldbalanceOrg: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)


# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

def prepare_features(transaction: Transaction):

    transaction_type = transaction.transaction_type.upper()

    valid_types = [
        "PAYMENT",
        "TRANSFER",
        "CASH_OUT",
        "CASH_IN",
        "DEBIT"
    ]

    if transaction_type not in valid_types:
        raise ValueError("Invalid transaction type")

    amount = transaction.amount
    sender_balance = transaction.oldbalanceOrg
    receiver_balance = transaction.oldbalanceDest

    data = {
        "step": transaction.step,
        "amount": amount,
        "log_amount": np.log1p(amount),
        "oldbalanceOrg": sender_balance,
        "oldbalanceDest": receiver_balance,
        "amount_to_sender_balance": amount / (sender_balance + 1),
        "amount_to_receiver_balance": amount / (receiver_balance + 1),
    }

    for txn_type in valid_types:
        data[f"type_{txn_type}"] = int(
            transaction_type == txn_type
        )

    # Ensure exactly the training feature order
    row = {
        feature: data[feature]
        for feature in FEATURES
    }

    return np.array(
        [[row[feature] for feature in FEATURES]],
        dtype=np.float32
    )


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "project": "Nexvia Financial Fraud Detection",
        "status": "running",
        "model": "XGBoost Real-Time Model"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


# --------------------------------------------------
# FRAUD PREDICTION
# --------------------------------------------------

@app.post("/predict")
def predict(transaction: Transaction):

    start_time = time.perf_counter()

    try:
        features = prepare_features(transaction)

        probability = float(
            model.predict_proba(features)[0][1]
        )

        threshold = 0.6539098024368286

        prediction = int(probability >= threshold)

        if probability >= 0.85:
            risk = "HIGH"
        elif probability >= threshold:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        return {
            "transaction_type": transaction.transaction_type.upper(),
            "fraud_probability": round(probability, 6),
            "prediction": prediction,
            "result": (
                "FRAUDULENT"
                if prediction == 1
                else "LEGITIMATE"
            ),
            "risk_level": risk,
            "response_time_ms": round(elapsed_ms, 3)
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

# --------------------------------------------------
# FEAST FEATURE RETRIEVAL
# --------------------------------------------------

@app.get("/features/{transaction_id}")
def get_features(transaction_id: str):

    try:
        features = get_transaction_features(transaction_id)

        if not features or all(value is None for value in features.values()):
            raise HTTPException(
                status_code=404,
                detail=f"No online features found for {transaction_id}"
            )

        return {
            "transaction_id": transaction_id,
            "feature_source": "Feast Online Store",
            "features": features
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

# --------------------------------------------------
# RUN WITH UVICORN
# --------------------------------------------------
