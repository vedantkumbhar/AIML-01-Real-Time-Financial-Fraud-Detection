
# AIML-01 – API Documentation

## 1. Overview

The AIML-01 Fraud Detection Platform uses FastAPI to provide HTTP endpoints for transaction fraud prediction and Feast feature retrieval.

**Base URL (local):**

`http://127.0.0.1:8000`

Interactive API documentation:

`http://127.0.0.1:8000/docs`

## 2. Authentication

Authentication is not currently implemented. The API is intended for local academic demonstration.

Do not expose it publicly with sensitive financial data without adding appropriate security controls.

## 3. Endpoints

### 3.1 POST /predict

**Purpose:** Predict whether a financial transaction is potentially fraudulent.

**Request Content-Type:** `application/json`

**Request body:**

```json
{
  "step": 632,
  "transaction_type": "TRANSFER",
  "amount": 480404.03,
  "oldbalanceOrg": 480404.03,
  "oldbalanceDest": 0.0
}
```

**Request parameters:**

| Field | Type | Description |
|---|---|---|
| step | Integer | Transaction time step |
| transaction_type | String | PAYMENT, TRANSFER, CASH_OUT, CASH_IN or DEBIT |
| amount | Float | Transaction amount |
| oldbalanceOrg | Float | Sender's previous balance |
| oldbalanceDest | Float | Receiver's previous balance |

**Successful response example:**

```json
{
  "transaction_type": "TRANSFER",
  "fraud_probability": 0.999861,
  "prediction": 1,
  "result": "FRAUDULENT",
  "risk_level": "HIGH",
  "response_time_ms": 5.643
}
```

The response values above illustrate a previously tested transaction; response time and prediction values can vary.

**Response fields:**

| Field | Description |
|---|---|
| transaction_type | Transaction category |
| fraud_probability | Model-estimated fraud probability |
| prediction | 1 = Fraud, 0 = Legitimate |
| result | Human-readable prediction |
| risk_level | LOW, MEDIUM or HIGH |
| response_time_ms | Prediction processing time |

### 3.2 GET /features/{transaction_id}

**Purpose:** Retrieve transaction features from the Feast Online Store.

**Example URL:**

`http://127.0.0.1:8000/features/txn_0`

**Successful response example:**

```json
{
  "transaction_id": "txn_0",
  "feature_source": "Feast Online Store",
  "features": {
    "amount": 9839.6396484375,
    "log_amount": 9.19427,
    "step": 1,
    "type_TRANSFER": 0,
    "type_CASH_OUT": 0
  }
}
```

The feature values shown are from a previously tested Feast lookup. Exact response formatting should be confirmed against the running API.

## 4. Error Handling

The API uses HTTP status codes to indicate request outcomes.

| Status code | Meaning |
|---|---|
| 200 | Successful request |
| 422 | Invalid request data |
| 500 | Internal server error |

## 5. Running the API

Activate the project virtual environment:

```powershell
.venv\Scripts\Activate.ps1
```

Start the FastAPI server:

```powershell
uvicorn backend.api.main:app --reload
```

Open Swagger documentation:

`http://127.0.0.1:8000/docs`

## 6. Model Integration

The prediction endpoint uses the production XGBoost model:

`models/realtime_fraud_model.json`

The model receives engineered transaction features and returns a fraud probability and classification.

Feast feature retrieval is provided through a separate endpoint. The current prediction endpoint constructs its model input from the transaction request rather than directly feeding Feast's 19-feature vector into the 12-feature model.

## 7. Testing

The following integrations have been verified:

- FastAPI prediction endpoint.
- Fraudulent transaction classification.
- Legitimate transaction classification through Kafka evaluation.
- Feast Online Store feature retrieval.
- Kafka consumer integration with FastAPI.

A 10-transaction Kafka + Feast integration test achieved 10 successful lookups and 10 correct predictions.

This is a small integration test, not a general performance guarantee.
