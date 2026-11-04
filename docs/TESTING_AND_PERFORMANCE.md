
# AIML-01 – Testing and Performance Report

## 1. Overview

This document records the testing and evaluation results of the Real-Time Financial Fraud Detection Platform.

Testing covers machine learning performance, API functionality, Kafka streaming, Feast feature retrieval, graph analysis, dashboard functionality, and model lifecycle management.

## 2. Machine Learning Model Evaluation

**Model:** XGBoost Classifier

**Dataset:** PaySim

**Test dataset size:** 89,466 transactions

**Fraud cases:** 1,252

### Evaluation Metrics

| Metric | Result |
|---|---:|
| PR-AUC | 0.999987 |
| ROC-AUC | 0.9999998 |
| Precision | 99.92% |
| Recall | 99.92% |
| F1-score | 99.92% |

### Confusion Matrix

| Actual / Predicted | Legitimate | Fraud |
|---|---:|---:|
| Legitimate | 88,213 | 1 |
| Fraud | 1 | 1,251 |

The model achieved strong classification results on the chronological PaySim test split.

These results are specific to the synthetic dataset and do not establish equivalent performance on real banking transactions.

## 3. FastAPI Testing

The following functionality was tested:

| Test | Result |
|---|---|
| API server startup | PASS |
| Swagger documentation | PASS |
| POST /predict | PASS |
| Fraudulent transaction prediction | PASS |
| Risk-level generation | PASS |
| GET /features/{transaction_id} | PASS |

A known high-risk TRANSFER transaction was successfully classified as fraudulent.

Observed response time during one test: 5.643 ms.

This is a single-request observation, not a formal latency benchmark.

## 4. Kafka and Feast Integration Testing

**Topic:** fraud-transactions-feast

**Consumer:** backend/kafka/consumer.py

### Results

| Test | Result |
|---|---:|
| Transactions evaluated | 10 |
| Successful Feast lookups | 10 |
| Failed Feast lookups | 0 |
| True negatives | 8 |
| True positives | 2 |
| False positives | 0 |
| False negatives | 0 |
| Integration test accuracy | 100% |

All 10 transactions were correctly classified during the integration test.

This result is limited to the tested messages and should not be presented as the model's general accuracy.

## 5. Fraud Network Analysis

NetworkX was used to analyze transaction relationships.

Recorded outputs:

- Accounts analyzed: 73,863
- Network edges: 65,694
- Connected components: 8,169
- Candidate fraud clusters: 44

Generated files:

- reports/fraud_network/suspicious_accounts.csv
- reports/fraud_network/fraud_rings.csv
- reports/fraud_network/account_network.graphml

These are candidate suspicious clusters derived from the dataset's recorded fraud labels, not independently confirmed criminal networks.

## 6. Evidently Drift Monitoring

The drift report compared two historical PaySim samples.

| Parameter | Result |
|---|---:|
| Reference rows | 50,000 |
| Current rows | 50,000 |
| Features monitored | 19 |
| Drifted features | 10 |
| Drift share | 52.632% |
| Configured threshold | 50% |

**Result:** Drift detected.

Generated report:

`reports/evidently/data_drift_report.html`

The monitoring experiment demonstrates historical feature-distribution comparison, not live production drift monitoring.

## 7. Candidate Model Retraining

The retraining pipeline successfully:

1. Verified the retraining request.
2. Loaded the PaySim dataset.
3. Recreated model features.
4. Applied chronological splitting.
5. Trained a candidate XGBoost model.
6. Evaluated candidate performance.
7. Saved the candidate model and metrics separately.

Candidate model:

`models/candidate_fraud_model.json`

Candidate metrics:

`reports/candidate_model_metrics.json`

The candidate achieved the same reported evaluation metrics as the existing production model.

The candidate was trained using the existing PaySim dataset, not newly collected production transactions.

## 8. Candidate Validation

The validation script checked:

- Feature compatibility.
- PR-AUC.
- Precision.
- Recall.
- F1-score.

**Result:** VALIDATION PASSED.

The candidate was eligible for promotion.

## 9. Controlled Model Promotion

The promotion process successfully:

- Rechecked candidate validation conditions.
- Backed up the previous production model.
- Promoted the candidate model.
- Updated production metrics.
- Recorded the promotion event.

Backup location:

`models/backups/20261104_172845`

Promotion log:

`reports/model_promotion_log.json`

The promoted model was subsequently verified through FastAPI and the Streamlit dashboard.

## 10. Streamlit Dashboard Testing

The dashboard was tested for:

| Functionality | Result |
|---|---|
| Dashboard startup | PASS |
| API connectivity | PASS |
| Transaction analysis | PASS |
| Fraud probability display | PASS |
| Risk-level display | PASS |
| Analyst feedback submission | PASS |
| Feedback CSV persistence | PASS |
| Network intelligence display | PASS |

Analyst feedback file:

`reports/analyst_feedback.csv`

## 11. Overall Testing Summary

| Component | Status |
|---|---|
| Data preprocessing | PASS |
| Feature engineering | PASS |
| XGBoost model training | PASS |
| FastAPI | PASS |
| Kafka streaming integration | PASS |
| Feast feature retrieval | PASS |
| Fraud network analysis | PASS |
| MLflow experiment tracking | PASS |
| Evidently drift monitoring | PASS |
| Candidate retraining | PASS |
| Candidate validation | PASS |
| Model promotion and backup | PASS |
| Streamlit dashboard | PASS |
| Analyst feedback | PASS |

## 12. Known Limitations

- PaySim is a synthetic dataset.
- Model performance may not transfer to real financial systems.
- The current drift experiment uses historical dataset samples.
- Candidate retraining uses the same dataset rather than newly labelled live data.
- Analyst feedback is stored but is not automatically used for retraining.
- Graph clusters require further investigation.
- Production-scale load testing and deployment have not been verified.
- Kubernetes deployment has not yet been verified.

## 13. Conclusion

The AIML-01 project demonstrates an integrated academic fraud detection workflow combining XGBoost, Kafka, Feast, FastAPI, MLflow, Evidently AI, NetworkX, and Streamlit.

The implemented system supports transaction scoring, streaming integration, feature retrieval, fraud network analysis, drift reporting, candidate retraining, controlled model promotion, and analyst feedback.
