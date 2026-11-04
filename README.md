
# NEXVIA – Real-Time Financial Fraud Detection Platform

## 1. Project Overview

AIML-01 is a Real-Time Financial Fraud Detection Platform developed as part of the Nexvia Internship Capstone Project.

The platform uses Machine Learning, streaming technologies, feature management, and network analysis to identify potentially fraudulent financial transactions.

It provides fraud probability scores, risk levels, analyst feedback, and data drift monitoring through an interactive dashboard.

## 2. Objectives

- Detect fraudulent financial transactions using Machine Learning.
- Handle highly imbalanced transaction datasets.
- Process transactions through Kafka streaming.
- Retrieve transaction features using Feast.
- Identify suspicious account relationships through graph analysis.
- Monitor feature drift using Evidently AI.
- Train, validate, and promote candidate models through a controlled workflow.
- Provide an analyst dashboard for fraud investigation.

## 3. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Dataset | PaySim |
| Data Processing | Pandas, NumPy |
| Machine Learning | XGBoost, Scikit-learn |
| Feature Store | Feast |
| Streaming | Apache Kafka |
| API | FastAPI |
| Dashboard | Streamlit |
| Experiment Tracking | MLflow |
| Drift Monitoring | Evidently AI |
| Graph Analysis | NetworkX |
| Visualization | Matplotlib |

## 4. Dataset

The project uses the PaySim synthetic financial transaction dataset.

- Total transactions: 6,362,620
- Fraudulent transactions: 8,213
- Legitimate transactions: 6,354,407
- Fraud percentage: Approximately 0.1291%

The dataset includes transaction types, amounts, sender and receiver balances, account identifiers, and fraud labels.

## 5. Machine Learning Model

An XGBoost classifier is used for fraud detection.

The training workflow includes:

1. Data preprocessing.
2. Feature engineering.
3. Chronological train, validation, and test splitting.
4. Handling class imbalance through fraud-aware sampling.
5. Model training.
6. Threshold selection using validation data.
7. Final evaluation on the test dataset.

### Model Performance

| Metric | Result |
|---|---:|
| PR-AUC | 0.999987 |
| ROC-AUC | 0.9999998 |
| Precision | 99.92% |
| Recall | 99.92% |
| F1-score | 99.92% |

Test dataset:
- Transactions: 89,466
- Fraud cases: 1,252
- True positives: 1,251
- True negatives: 88,213
- False positives: 1
- False negatives: 1

These results are based on the synthetic PaySim dataset and should not be interpreted as guaranteed real-world banking performance.

## 6. Real-Time Fraud Detection

Apache Kafka is used to stream transaction messages.

The Kafka consumer:
- Reads transaction messages.
- Retrieves features from Feast.
- Sends transaction information to the FastAPI prediction service.
- Receives fraud predictions and risk levels.
- Compares predictions with dataset labels during evaluation.

## 7. Feature Store – Feast

Feast provides centralized transaction feature management.

Implemented functionality:
- Feature definitions.
- Historical feature storage.
- Online feature retrieval.
- Transaction feature lookup through the FastAPI service.
- Kafka consumer integration.

A 10-transaction integration test achieved:
- Successful Feast lookups: 10/10
- Failed lookups: 0
- Correct predictions: 10/10

This is an integration test, not a generalization benchmark.

## 8. Fraud Network Analysis

NetworkX is used to analyze relationships between transaction accounts.

The analysis identifies:
- Suspicious accounts.
- Connected account groups.
- Candidate fraud clusters.
- Relationships between accounts involved in recorded fraudulent transactions.

The resulting clusters are investigative candidates, not confirmed criminal networks.

## 9. Model Monitoring and Retraining

Evidently AI is used to monitor changes in feature distributions.

Implemented workflow:

1. Compare reference and current historical dataset samples.
2. Generate a data drift report.
3. Detect drift when the configured threshold is exceeded.
4. Generate a retraining request.
5. Train a candidate XGBoost model.
6. Validate candidate performance and feature compatibility.
7. Promote the candidate model.
8. Back up the previous production model.

The demonstrated drift comparison uses historical PaySim samples. Candidate retraining currently uses the same PaySim dataset rather than newly collected production labels.

## 10. Analyst Dashboard

The Streamlit dashboard provides:

- Transaction analysis.
- Fraud probability.
- Risk classification.
- Fraud alert monitoring.
- Network intelligence.
- Analyst review and feedback.
- Feedback export.

Analyst feedback is stored in:

`reports/analyst_feedback.csv`

## 11. Project Structure

```text
AIML-01/
│
├── backend/
│   ├── api/
│   ├── kafka/
│   └── features/
│
├── dashboard/
├── data/
├── docs/
├── models/
│   └── backups/
├── monitoring/
├── notebooks/
├── reports/
│   ├── evidently/
│   ├── fraud_network/
│   └── screenshots/
├── src/
│   ├── preprocessing/
│   ├── features/
│   ├── models/
│   └── graph/
│
├── tests/
├── docker/
├── k8s/
├── requirements.txt
└── README.md
```

## 12. Installation and Execution

### Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AIML-01
```

### Create a virtual environment

```bash
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start FastAPI

```bash
uvicorn backend.api.main:app --reload
```

API documentation:

`http://127.0.0.1:8000/docs`

### Start Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

### Start Kafka Consumer

Configure and start the Kafka broker and create the required topic first.

```powershell
$env:PYTHONPATH = (Get-Location).Path
python backend\kafka\consumer.py
```

The Kafka producer and Feast repository must also be configured before running the streaming workflow.

## 13. API Endpoint

### POST /predict

Accepts transaction details and returns:

- Fraud probability.
- Prediction label.
- Fraud classification.
- Risk level.
- Response time.

### GET /features/{transaction_id}

Retrieves transaction features from the Feast Online Store.

## 14. Limitations

- PaySim is a synthetic dataset and does not represent all real-world banking patterns.
- Device and geographic metadata are not available in the selected dataset.
- Candidate retraining currently uses the existing dataset.
- Analyst feedback is stored but is not yet automatically incorporated into model training.
- The fraud network clusters require further investigation and validation.
- Kubernetes deployment and live production infrastructure are not yet verified.

## 15. Future Scope

- Integrate additional real-world transaction datasets.
- Incorporate device and geographic intelligence when suitable data becomes available.
- Explore Graph Neural Networks for fraud detection.
- Use newly labelled transactions for continuous model improvement.
- Connect analyst feedback to a reviewed retraining pipeline.
- Deploy the services using Kubernetes.
- Improve monitoring and automated model lifecycle management.

## 16. Project Information

**Project:** AIML-01 – Real-Time Financial Fraud Detection Platform

**Program:** Nexvia Internship Capstone Project

**Developer:** Vedant Mahesh Kumbhar

**Branch:** CSE (Artificial Intelligence and Machine Learning)

---

*Developed as an academic and internship capstone project.*
