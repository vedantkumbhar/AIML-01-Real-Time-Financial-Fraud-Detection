
import json
import time
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer
from kafka.errors import KafkaError

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT / "data" / "archive (5)" / "fraud.csv"
)

KAFKA_SERVER = "localhost:9092"

# Separate topic for Feast integration testing
TOPIC = "fraud-transactions-feast"

MAX_MESSAGES = 10
DELAY_SECONDS = 0.1


def create_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_SERVER,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        acks="all",
        retries=3
    )


def main():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    print("Loading PaySim dataset...")

    producer = create_producer()
    count = 0

    try:
        for chunk in pd.read_csv(
            DATASET_PATH,
            usecols=[
                "step",
                "type",
                "amount",
                "nameOrig",
                "oldbalanceOrg",
                "oldbalanceDest",
                "isFraud"
            ],
            chunksize=MAX_MESSAGES
        ):

            for transaction in chunk.to_dict(orient="records"):

                # Preserve the original dataset row index.
                transaction_id = f"txn_{count}"

                transaction["transaction_id"] = transaction_id

                # Retain the label for offline evaluation only.
                # Never send this label to the prediction API.
                transaction["isFraud"] = int(
                    transaction["isFraud"]
                )

                producer.send(
                    TOPIC,
                    value=transaction
                ).get(timeout=30)

                count += 1

                print(
                    f"Published {transaction_id} "
                    f"({count}/{MAX_MESSAGES})"
                )

                if count >= MAX_MESSAGES:
                    break

                time.sleep(DELAY_SECONDS)

            if count >= MAX_MESSAGES:
                break

        producer.flush()

        print("\n================================")
        print("FEAST KAFKA PRODUCER COMPLETED")
        print(f"Total transactions: {count}")
        print(f"Topic: {TOPIC}")
        print("================================")

    except KafkaError as error:
        print(f"Kafka error: {error}")

    finally:
        producer.close()


if __name__ == "__main__":
    main()
