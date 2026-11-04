from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "fraud_features.csv.gz"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "src" / "features" / "feast_repo" / "data"
)

OUTPUT_FILE = OUTPUT_DIR / "feast_features.parquet"
LABEL_FILE = OUTPUT_DIR / "feast_labels.parquet"

CHUNK_SIZE = 100_000

# ============================================================
# FEATURES — EXCLUDING isFraud
# ============================================================

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

INTEGER_COLUMNS = [
    "step",
    "sender_txn_count_before",
    "receiver_txn_count_before",
    "origin_balance_zero_before",
    "origin_balance_zero_after",
    "destination_balance_zero_before",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
]

FLOAT_COLUMNS = [
    column for column in FEATURE_COLUMNS
    if column not in INTEGER_COLUMNS
]


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Remove old incomplete outputs only.
    for path in [OUTPUT_FILE, LABEL_FILE]:
        if path.exists():
            path.unlink()

    feature_writer = None
    label_writer = None

    total_rows = 0
    fraud_rows = 0
    chunk_number = 0

    print("Starting full PaySim Feast preparation...")
    print("Input:", INPUT_FILE)
    print("Chunk size:", CHUNK_SIZE)

    try:
        reader = pd.read_csv(
            INPUT_FILE,
            chunksize=CHUNK_SIZE
        )

        for chunk in reader:

            chunk_number += 1

            missing = [
                col for col in FEATURE_COLUMNS + ["isFraud"]
                if col not in chunk.columns
            ]

            if missing:
                raise ValueError(
                    f"Missing columns: {missing}"
                )

            # Stable transaction IDs based on original row order.
            start_id = total_rows

            transaction_ids = [
                f"txn_{i}"
                for i in range(start_id, start_id + len(chunk))
            ]

            # PaySim step represents simulated hours.
            timestamps = (
                pd.Timestamp("2020-01-01", tz="UTC")
                + pd.to_timedelta(chunk["step"], unit="h")
            )

            # Prepare Feast features.
            features = chunk[FEATURE_COLUMNS].copy()

            for col in INTEGER_COLUMNS:
                features[col] = features[col].astype("int64")

            for col in FLOAT_COLUMNS:
                features[col] = features[col].astype("float32")

            features.insert(0, "transaction_id", transaction_ids)
            features["event_timestamp"] = timestamps

            # Keep ground-truth labels separately.
            labels = pd.DataFrame({
                "transaction_id": transaction_ids,
                "isFraud": chunk["isFraud"].astype("int64")
            })

            fraud_rows += int(labels["isFraud"].sum())

            feature_table = pa.Table.from_pandas(
                features,
                preserve_index=False
            )

            label_table = pa.Table.from_pandas(
                labels,
                preserve_index=False
            )

            if feature_writer is None:
                feature_writer = pq.ParquetWriter(
                    OUTPUT_FILE,
                    feature_table.schema,
                    compression="snappy"
                )

                label_writer = pq.ParquetWriter(
                    LABEL_FILE,
                    label_table.schema,
                    compression="snappy"
                )

            feature_writer.write_table(feature_table)
            label_writer.write_table(label_table)

            total_rows += len(chunk)

            print(
                f"Chunk {chunk_number} completed | "
                f"Rows processed: {total_rows:,} | "
                f"Fraud records: {fraud_rows:,}"
            )

    finally:
        if feature_writer is not None:
            feature_writer.close()

        if label_writer is not None:
            label_writer.close()

    print("\n========================================")
    print("FEAST DATA PREPARATION COMPLETED")
    print("========================================")
    print(f"Total transactions: {total_rows:,}")
    print(f"Fraud transactions: {fraud_rows:,}")
    print(f"Feature columns: {len(FEATURE_COLUMNS)}")
    print("Fraud label included in features: No")
    print("Feature file:", OUTPUT_FILE)
    print("Label file:", LABEL_FILE)


if __name__ == "__main__":
    main()