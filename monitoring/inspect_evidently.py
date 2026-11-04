from pathlib import Path
import pandas as pd
import json

from evidently import Report
from evidently.presets import DataDriftPreset

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "processed" / "fraud_features.csv.gz"

columns = [
    "step", "amount", "log_amount",
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
    "type_CASH_IN", "type_CASH_OUT",
    "type_DEBIT", "type_PAYMENT", "type_TRANSFER"
]

reference = pd.read_csv(DATA, usecols=columns, nrows=50000)

current = pd.read_csv(
    DATA,
    usecols=columns,
    skiprows=range(1, 3000001),
    nrows=50000
)

reference = reference.replace([float("inf"), float("-inf")], float("nan")).dropna()
current = current.replace([float("inf"), float("-inf")], float("nan")).dropna()

report = Report([DataDriftPreset()])
result = report.run(
    reference_data=reference,
    current_data=current
)

print("\nRESULT TYPE:", type(result))
print("\nAVAILABLE METHODS:")
print([x for x in dir(result) if not x.startswith("_")])

if hasattr(result, "dict"):
    data = result.dict()
    print("\nRESULT DICTIONARY:")
    print(json.dumps(data, indent=2, default=str)[:12000])