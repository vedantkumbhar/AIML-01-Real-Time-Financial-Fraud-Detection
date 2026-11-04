from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fraud_features.csv.gz"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "evidently"
REPORT_PATH = REPORT_DIR / "data_drift_report.html"

# Actual engineered features (exclude the target label)
FEATURE_COLUMNS = [
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
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
]

print("Loading engineered PaySim features...")

# Read two different samples from the dataset
reference = pd.read_csv(
    DATASET_PATH,
    usecols=FEATURE_COLUMNS,
    nrows=50000
)

current = pd.read_csv(
    DATASET_PATH,
    usecols=FEATURE_COLUMNS,
    skiprows=range(1, 3000001),
    nrows=50000
)

# Clean invalid values
reference = reference.replace(
    [float("inf"), float("-inf")], float("nan")
).dropna()

current = current.replace(
    [float("inf"), float("-inf")], float("nan")
).dropna()

print("Reference rows:", len(reference))
print("Current rows:", len(current))

print("Generating data drift report...")

report = Report([
    DataDriftPreset()
])

result = report.run(
    reference_data=reference,
    current_data=current
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)
result.save_html(str(REPORT_PATH))

print("\nData drift report generated successfully!")
print("Report location:", REPORT_PATH)