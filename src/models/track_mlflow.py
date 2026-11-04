from pathlib import Path
import json
import joblib
import mlflow
import mlflow.xgboost

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "realtime_fraud_model.json"
METRICS_PATH = PROJECT_ROOT / "reports" / "realtime_model_metrics.json"
FEATURES_PATH = PROJECT_ROOT / "models" / "realtime_feature_columns.pkl"

# MLflow tracking directory
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

# Verify required files
for path in [MODEL_PATH, METRICS_PATH, FEATURES_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

# Configure MLflow
mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.as_posix()}"
)
mlflow.set_experiment("AIML-01-Fraud-Detection")

# Load existing metrics and model information
with open(METRICS_PATH, "r") as file:
    metrics = json.load(file)

feature_columns = joblib.load(FEATURES_PATH)

with mlflow.start_run(run_name="XGBoost-Realtime-Fraud-Model"):

    # Log model parameters
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("dataset", "PaySim")
    mlflow.log_param("feature_count", len(feature_columns))
    mlflow.log_param("decision_threshold", 0.6539098024368286)

    # Log evaluation metrics
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(name, value)

    # Log model artifacts
    mlflow.log_artifact(str(MODEL_PATH))
    mlflow.log_artifact(str(FEATURES_PATH))

    print("MLflow experiment completed successfully.")
    print("Experiment: AIML-01-Fraud-Detection")
    print("Logged model parameters, metrics and artifacts.")

print(f"MLflow database: {MLFLOW_DB}")