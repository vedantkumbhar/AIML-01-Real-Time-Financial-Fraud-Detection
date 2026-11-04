from pathlib import Path
from datetime import datetime, timezone
import json
from unittest import result

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fraud_features.csv.gz"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "evidently"
HTML_REPORT = REPORT_DIR / "data_drift_report.html"

TRIGGER_FILE = REPORT_DIR / "retraining_request.json"

DRIFT_THRESHOLD = 0.50

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


def find_drift_metrics(data):
    """Find the dataset drift summary in Evidently's result."""

    if isinstance(data, dict):
        if (
            "count" in data
            and "share" in data
            and "drift_by_columns" in data
        ):
            return data

        for value in data.values():
            result = find_drift_metrics(value)
            if result is not None:
                return result

    elif isinstance(data, list):
        for value in data:
            result = find_drift_metrics(value)
            if result is not None:
                return result

    return None


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    print("Loading reference and current data...")

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

    reference = reference.replace(
        [float("inf"), float("-inf")], float("nan")
    ).dropna()

    current = current.replace(
        [float("inf"), float("-inf")], float("nan")
    ).dropna()

    print("Reference rows:", len(reference))
    print("Current rows:", len(current))

    print("Running Evidently drift detection...")

    report = Report([
        DataDriftPreset()
    ])

    result = report.run(
        reference_data=reference,
        current_data=current
    )

    result.save_html(str(HTML_REPORT))

    # Extract drift summary from Evidently Snapshot
    result_data = result.dict()

    drift_metric = next(
    (
        metric
        for metric in result_data["metrics"]
        if metric["metric_name"].startswith("DriftedColumnsCount")
    ),
    None
)

    if drift_metric is None:
        raise RuntimeError(
        "DriftedColumnsCount metric not found. "
        "No retraining decision was made."
    )

    drifted_count = int(drift_metric["value"]["count"])
    drift_share = float(drift_metric["value"]["share"])

    print("\n========== DRIFT RESULTS ==========")
    print("Total features:", len(FEATURE_COLUMNS))
    print("Drifted features:", drifted_count)
    print(f"Drift share: {drift_share:.3%}")
    print(f"Trigger threshold: {DRIFT_THRESHOLD:.0%}")

    if drift_share >= DRIFT_THRESHOLD:

        request = {
            "status": "RETRAINING_REQUESTED",
            "reason": "Data drift threshold exceeded",
            "drifted_features": drifted_count,
            "total_features": len(FEATURE_COLUMNS),
            "drift_share": drift_share,
            "threshold": DRIFT_THRESHOLD,
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "model_replacement": "Requires validation and approval"
        }

        TRIGGER_FILE.write_text(
            json.dumps(request, indent=4),
            encoding="utf-8"
        )

        print("\nDRIFT DETECTED!")
        print("Automatic retraining request created.")
        print("Request:", TRIGGER_FILE)

    else:
        if TRIGGER_FILE.exists():
            TRIGGER_FILE.unlink()

        print("\nNo retraining required.")
        print("Continue normal monitoring.")

    print("\nHTML report:", HTML_REPORT)


if __name__ == "__main__":
    main()