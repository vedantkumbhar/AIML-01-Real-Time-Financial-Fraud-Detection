from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data" / "archive (5)" / "fraud.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "fraud_features.csv.gz"

print("Loading PaySim dataset...")

df = pd.read_csv(INPUT_FILE)

print("Original shape:", df.shape)

# Reduce memory usage
df["step"] = df["step"].astype("int32")
df["amount"] = df["amount"].astype("float32")

for col in [
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]:
    df[col] = df[col].astype("float32")

df["isFraud"] = df["isFraud"].astype("int8")

# Ensure chronological ordering
if not df["step"].is_monotonic_increasing:
    print("Sorting transactions by time...")
    df.sort_values("step", kind="stable", inplace=True)
    df.reset_index(drop=True, inplace=True)

print("Creating transaction features...")

# --------------------------------------------------
# 1. TRANSACTION TYPE FEATURES
# --------------------------------------------------

type_features = pd.get_dummies(
    df["type"],
    prefix="type",
    dtype="int8"
)

df = pd.concat([df, type_features], axis=1)

# --------------------------------------------------
# 2. SENDER HISTORY FEATURES
# --------------------------------------------------

print("Calculating sender history...")

sender_group = df.groupby("nameOrig", sort=False)

df["sender_txn_count_before"] = (
    sender_group.cumcount().astype("int32")
)

sender_cumulative_amount = sender_group["amount"].cumsum()

previous_sender_amount = (
    sender_cumulative_amount - df["amount"]
)

df["sender_avg_amount_before"] = np.where(
    df["sender_txn_count_before"] > 0,
    previous_sender_amount /
    df["sender_txn_count_before"].replace(0, np.nan),
    0
)

previous_sender_step = sender_group["step"].shift(1)

df["sender_hours_since_previous"] = (
    df["step"] - previous_sender_step
).fillna(-1).astype("float32")

# --------------------------------------------------
# 3. RECEIVER HISTORY FEATURES
# --------------------------------------------------

print("Calculating receiver history...")

receiver_group = df.groupby("nameDest", sort=False)

df["receiver_txn_count_before"] = (
    receiver_group.cumcount().astype("int32")
)

receiver_cumulative_amount = receiver_group["amount"].cumsum()

previous_receiver_amount = (
    receiver_cumulative_amount - df["amount"]
)

df["receiver_avg_amount_before"] = np.where(
    df["receiver_txn_count_before"] > 0,
    previous_receiver_amount /
    df["receiver_txn_count_before"].replace(0, np.nan),
    0
)

# --------------------------------------------------
# 4. BALANCE FEATURES
# --------------------------------------------------

df["origin_balance_error"] = (
    df["oldbalanceOrg"]
    - df["amount"]
    - df["newbalanceOrig"]
)

df["destination_balance_error"] = (
    df["newbalanceDest"]
    - df["oldbalanceDest"]
    - df["amount"]
)

df["origin_balance_zero_before"] = (
    df["oldbalanceOrg"] == 0
).astype("int8")

df["origin_balance_zero_after"] = (
    df["newbalanceOrig"] == 0
).astype("int8")

df["destination_balance_zero_before"] = (
    df["oldbalanceDest"] == 0
).astype("int8")

# --------------------------------------------------
# 5. AMOUNT FEATURES
# --------------------------------------------------

df["log_amount"] = np.log1p(df["amount"])

df["amount_vs_sender_average"] = (
    df["amount"] /
    (df["sender_avg_amount_before"] + 1)
)

# --------------------------------------------------
# 6. PREPARE MODEL DATASET
# --------------------------------------------------

feature_columns = [
    "step",
    "amount",
    "log_amount",
    "sender_txn_count_before",
    "sender_avg_amount_before",
    "sender_hours_since_previous",
    "receiver_txn_count_before",
    "receiver_avg_amount_before",
    "origin_balance_error",
    "destination_balance_error",
    "origin_balance_zero_before",
    "origin_balance_zero_after",
    "destination_balance_zero_before",
    "amount_vs_sender_average",
]

feature_columns += list(type_features.columns)

# Keep target separately
output_columns = feature_columns + ["isFraud"]

model_df = df[output_columns].replace(
    [np.inf, -np.inf], np.nan
).fillna(0)

print("\nFeature dataset shape:", model_df.shape)
print("Fraud distribution:")
print(model_df["isFraud"].value_counts())

print("\nSaving compressed feature dataset...")

model_df.to_csv(
    OUTPUT_FILE,
    index=False,
    compression="gzip"
)

print("\nFEATURE ENGINEERING COMPLETED")
print("Saved:", OUTPUT_FILE)
print("Total features:", len(feature_columns))