from pathlib import Path
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
)

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT / "data" / "archive (5)" / "fraud.csv"
REQUEST_FILE = ROOT / "reports" / "evidently" / "retraining_request.json"

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

CANDIDATE_MODEL = MODEL_DIR / "candidate_fraud_model.json"
CANDIDATE_FEATURES = MODEL_DIR / "candidate_feature_columns.pkl"
CANDIDATE_METRICS = REPORT_DIR / "candidate_model_metrics.json"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


def main():

    # 1. Check retraining request
    if not REQUEST_FILE.exists():
        raise FileNotFoundError(
            "No retraining request found. Run drift monitoring first."
        )

    with open(REQUEST_FILE, "r") as file:
        request = json.load(file)

    if request.get("status") != "RETRAINING_REQUESTED":
        raise RuntimeError("Retraining has not been requested.")

    print("Retraining request verified.")
    print("Reason:", request["reason"])

    # 2. Load dataset
    print("\nLoading PaySim dataset...")
    df = pd.read_csv(DATA_FILE)

    # 3. Create real-time features
    print("Creating features...")

    df["log_amount"] = np.log1p(df["amount"])

    df["amount_to_sender_balance"] = (
        df["amount"] / (df["oldbalanceOrg"] + 1)
    )

    df["amount_to_receiver_balance"] = (
        df["amount"] / (df["oldbalanceDest"] + 1)
    )

    type_features = pd.get_dummies(
        df["type"],
        prefix="type",
        dtype="int8"
    )

    df = pd.concat([df, type_features], axis=1)

    features = [
        "step",
        "amount",
        "log_amount",
        "oldbalanceOrg",
        "oldbalanceDest",
        "amount_to_sender_balance",
        "amount_to_receiver_balance",
    ] + list(type_features.columns)

    df[features] = df[features].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0).astype("float32")

    target = "isFraud"

    # 4. Chronological split
    print("Creating chronological splits...")

    steps = np.sort(df["step"].unique())

    train_end = steps[int(len(steps) * 0.70) - 1]
    validation_end = steps[int(len(steps) * 0.85) - 1]

    train_df = df[df["step"] <= train_end]

    val_df = df[
        (df["step"] > train_end) &
        (df["step"] <= validation_end)
    ]

    test_df = df[df["step"] > validation_end]

    print("Training rows:", len(train_df))
    print("Validation rows:", len(val_df))
    print("Testing rows:", len(test_df))

    # 5. Training sample
    fraud = train_df[train_df[target] == 1]
    normal = train_df[train_df[target] == 0]

    normal_sample = normal.sample(
        n=min(len(normal), len(fraud) * 20),
        random_state=42
    )

    train_sample = pd.concat(
        [fraud, normal_sample]
    ).sample(frac=1, random_state=42)

    X_train = train_sample[features]
    y_train = train_sample[target]

    X_val = val_df[features]
    y_val = val_df[target]

    X_test = test_df[features]
    y_test = test_df[target]

    del df, train_df, val_df, test_df
    del fraud, normal, normal_sample, train_sample

    # 6. Train candidate model
    print("\nTraining candidate XGBoost model...")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=4,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    # 7. Select threshold using validation data
    val_prob = model.predict_proba(X_val)[:, 1]

    precision, recall, thresholds = precision_recall_curve(
        y_val, val_prob
    )

    f1_values = (
        2 * precision[:-1] * recall[:-1] /
        (precision[:-1] + recall[:-1] + 1e-12)
    )

    threshold = float(thresholds[np.argmax(f1_values)])

    # 8. Evaluate candidate on test data
    print("\nEvaluating candidate model...")

    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype("int8")

    metrics = {
        "model": "Candidate XGBoost",
        "threshold": threshold,
        "pr_auc": float(
            average_precision_score(y_test, test_prob)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, test_prob)
        ),
        "precision": float(
            precision_score(y_test, test_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, test_pred, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, test_pred, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, test_pred
        ).tolist(),
        "test_transactions": int(len(y_test)),
        "test_fraud_cases": int(y_test.sum()),
        "features": features,
    }

    # 9. Save candidate separately
    model.save_model(CANDIDATE_MODEL)
    joblib.dump(features, CANDIDATE_FEATURES)

    with open(CANDIDATE_METRICS, "w") as file:
        json.dump(metrics, file, indent=4)

    print("\nCANDIDATE TRAINING COMPLETED")
    print("=" * 50)

    for key, value in metrics.items():
        if key != "features":
            print(f"{key}: {value}")

    print("\nCandidate model:", CANDIDATE_MODEL)
    print("Candidate metrics:", CANDIDATE_METRICS)
    print("\nProduction model has NOT been modified.")


if __name__ == "__main__":
    main()