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
    classification_report,
    precision_recall_curve,
)

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT / "data" / "archive (5)" / "fraud.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("REAL-TIME FRAUD MODEL - LEAKAGE CONTROLLED")
print("=" * 60)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)

# --------------------------------------------------
# 2. CREATE PRE-TRANSACTION FEATURES ONLY
# --------------------------------------------------

print("\nCreating real-time features...")

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

FEATURES = [
    "step",
    "amount",
    "log_amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "amount_to_sender_balance",
    "amount_to_receiver_balance",
]

FEATURES += list(type_features.columns)

TARGET = "isFraud"

# Excludes:
# newbalanceOrig
# newbalanceDest
# isFlaggedFraud
# all post-transaction and outcome-related features

df[FEATURES] = df[FEATURES].replace(
    [np.inf, -np.inf], np.nan
).fillna(0).astype("float32")

# --------------------------------------------------
# 3. CHRONOLOGICAL SPLIT
# --------------------------------------------------

print("\nCreating chronological splits...")

steps = np.sort(df["step"].unique())

train_end = steps[int(len(steps) * 0.70) - 1]
validation_end = steps[int(len(steps) * 0.85) - 1]

train_df = df[df["step"] <= train_end]
val_df = df[
    (df["step"] > train_end) &
    (df["step"] <= validation_end)
]
test_df = df[df["step"] > validation_end]

for name, part in [
    ("Training", train_df),
    ("Validation", val_df),
    ("Testing", test_df),
]:
    print(
        f"{name}: {len(part)} rows, "
        f"{int(part[TARGET].sum())} fraud cases"
    )

# --------------------------------------------------
# 4. TRAINING SAMPLE
# --------------------------------------------------

print("\nSampling training data...")

fraud = train_df[train_df[TARGET] == 1]
normal = train_df[train_df[TARGET] == 0]

normal_sample = normal.sample(
    n=min(len(normal), len(fraud) * 20),
    random_state=42
)

train_sample = pd.concat(
    [fraud, normal_sample]
).sample(frac=1, random_state=42)

X_train = train_sample[FEATURES]
y_train = train_sample[TARGET]

X_val = val_df[FEATURES]
y_val = val_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

del df, train_df, val_df, test_df
del fraud, normal, normal_sample, train_sample

# --------------------------------------------------
# 5. TRAIN MODEL
# --------------------------------------------------

print("\nTraining XGBoost...")

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

# --------------------------------------------------
# 6. VALIDATION THRESHOLD
# --------------------------------------------------

print("\nSelecting threshold...")

val_prob = model.predict_proba(X_val)[:, 1]

precision, recall, thresholds = precision_recall_curve(
    y_val, val_prob
)

f1_values = (
    2 * precision[:-1] * recall[:-1] /
    (precision[:-1] + recall[:-1] + 1e-12)
)

best_index = np.argmax(f1_values)
threshold = float(thresholds[best_index])

# --------------------------------------------------
# 7. TEST EVALUATION
# --------------------------------------------------

print("\nTesting model...")

test_prob = model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= threshold).astype("int8")

metrics = {
    "model": "Real-Time XGBoost",
    "features": FEATURES,
    "threshold": threshold,
    "pr_auc": float(average_precision_score(y_test, test_prob)),
    "roc_auc": float(roc_auc_score(y_test, test_prob)),
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
}

print("\nFINAL RESULTS")
print("=" * 50)

for key, value in metrics.items():
    if key != "features":
        print(f"{key}: {value}")

print("\nClassification report:")
print(classification_report(
    y_test, test_pred, zero_division=0
))

# --------------------------------------------------
# 8. SAVE SEPARATELY
# --------------------------------------------------

model.save_model(
    MODEL_DIR / "realtime_fraud_model.json"
)

joblib.dump(
    FEATURES,
    MODEL_DIR / "realtime_feature_columns.pkl"
)

with open(
    REPORT_DIR / "realtime_model_metrics.json",
    "w"
) as file:
    json.dump(metrics, file, indent=4)

print("\nReal-time model saved successfully.")
print("REAL-TIME MODEL TRAINING COMPLETED")