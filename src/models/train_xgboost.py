from pathlib import Path
import json
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
import joblib

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT / "data" / "processed" / "fraud_features.csv.gz"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
NEGATIVE_RATIO = 20

print("=" * 60)
print("XGBOOST FINANCIAL FRAUD DETECTION TRAINING")
print("=" * 60)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

print("\nLoading feature dataset...")

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)

TARGET = "isFraud"

FEATURES = [col for col in df.columns if col != TARGET]

# Convert features to compact numeric types
df[FEATURES] = df[FEATURES].astype("float32")

# --------------------------------------------------
# 2. TIME-BASED DATA SPLIT
# --------------------------------------------------

print("\nCreating chronological train/validation/test splits...")

steps = np.sort(df["step"].unique())

train_end = steps[int(len(steps) * 0.70) - 1]
validation_end = steps[int(len(steps) * 0.85) - 1]

train_df = df[df["step"] <= train_end].copy()

validation_df = df[
    (df["step"] > train_end) &
    (df["step"] <= validation_end)
].copy()

test_df = df[df["step"] > validation_end].copy()

for name, data in [
    ("Training", train_df),
    ("Validation", validation_df),
    ("Testing", test_df),
]:
    print(f"\n{name}:")
    print("Rows:", len(data))
    print("Fraud:", int(data[TARGET].sum()))
    print("Fraud rate:", round(data[TARGET].mean() * 100, 4), "%")

# --------------------------------------------------
# 3. DOWNSAMPLE TRAINING NEGATIVES ONLY
# --------------------------------------------------

print("\nPreparing balanced training sample...")

fraud_rows = train_df[train_df[TARGET] == 1]

normal_rows = train_df[train_df[TARGET] == 0]

normal_sample = normal_rows.sample(
    n=min(
        len(normal_rows),
        len(fraud_rows) * NEGATIVE_RATIO
    ),
    random_state=RANDOM_STATE
)

training_data = pd.concat(
    [fraud_rows, normal_sample],
    ignore_index=True
).sample(
    frac=1,
    random_state=RANDOM_STATE
)

X_train = training_data[FEATURES]
y_train = training_data[TARGET]

X_val = validation_df[FEATURES]
y_val = validation_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

print("Training rows after sampling:", len(training_data))
print("Training fraud cases:", int(y_train.sum()))

# Release unnecessary dataframes
del df, train_df, validation_df, test_df
del fraud_rows, normal_rows, normal_sample, training_data

# --------------------------------------------------
# 4. TRAIN XGBOOST
# --------------------------------------------------

print("\nTraining XGBoost model...")

model = XGBClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    n_jobs=4,
    random_state=RANDOM_STATE,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=50,
)

# --------------------------------------------------
# 5. SELECT THRESHOLD ON VALIDATION DATA
# --------------------------------------------------

print("\nSelecting fraud detection threshold...")

val_probabilities = model.predict_proba(X_val)[:, 1]

precision, recall, thresholds = precision_recall_curve(
    y_val,
    val_probabilities
)

f1_values = (
    2 * precision[:-1] * recall[:-1] /
    (precision[:-1] + recall[:-1] + 1e-12)
)

best_index = int(np.argmax(f1_values))
best_threshold = float(thresholds[best_index])

print("Selected threshold:", best_threshold)
print("Validation PR-AUC:", average_precision_score(
    y_val, val_probabilities
))

# --------------------------------------------------
# 6. FINAL TEST EVALUATION
# --------------------------------------------------

print("\nEvaluating on untouched test data...")

test_probabilities = model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype("int8")

pr_auc = average_precision_score(
    y_test, test_probabilities
)

roc_auc = roc_auc_score(
    y_test, test_probabilities
)

precision_value = precision_score(
    y_test, test_predictions, zero_division=0
)

recall_value = recall_score(
    y_test, test_predictions, zero_division=0
)

f1_value = f1_score(
    y_test, test_predictions, zero_division=0
)

cm = confusion_matrix(y_test, test_predictions)

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print("PR-AUC:", round(pr_auc, 6))
print("ROC-AUC:", round(roc_auc, 6))
print("Precision:", round(precision_value, 6))
print("Recall:", round(recall_value, 6))
print("F1 Score:", round(f1_value, 6))

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_test,
    test_predictions,
    zero_division=0
))

# --------------------------------------------------
# 7. SAVE MODEL AND REPORT
# --------------------------------------------------

model_path = MODEL_DIR / "xgboost_fraud_model.json"

model.save_model(model_path)

joblib.dump(
    FEATURES,
    MODEL_DIR / "feature_columns.pkl"
)

metrics = {
    "model": "XGBoost",
    "training_negative_ratio": NEGATIVE_RATIO,
    "threshold": best_threshold,
    "pr_auc": float(pr_auc),
    "roc_auc": float(roc_auc),
    "precision": float(precision_value),
    "recall": float(recall_value),
    "f1_score": float(f1_value),
    "confusion_matrix": cm.tolist(),
    "test_transactions": int(len(y_test)),
    "test_fraud_cases": int(y_test.sum()),
}

with open(
    REPORT_DIR / "xgboost_metrics.json",
    "w"
) as file:
    json.dump(metrics, file, indent=4)

print("\nModel saved:", model_path)
print("Metrics saved:", REPORT_DIR / "xgboost_metrics.json")
print("\nTRAINING COMPLETED")