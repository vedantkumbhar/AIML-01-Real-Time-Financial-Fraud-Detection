
import json
import requests

from kafka import KafkaConsumer
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

from backend.features.feast_service import get_transaction_features


KAFKA_SERVER = "localhost:9092"
TOPIC = "fraud-transactions-feast"

API_URL = "http://127.0.0.1:8000/predict"

MAX_MESSAGES = 10


def create_consumer():
    return KafkaConsumer(
        TOPIC,
        bootstrap_servers=KAFKA_SERVER,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        group_id="fraud-detection-feast-evaluation-v1",
        value_deserializer=lambda value: json.loads(
            value.decode("utf-8")
        )
    )


def process_transaction(transaction):

    transaction_id = transaction["transaction_id"]

    # Retrieve matching historical features from Feast.
    feast_features = get_transaction_features(transaction_id)

    feast_lookup_success = (
        feast_features
        and all(value is not None for value in feast_features.values())
    )

    # Preserve the existing XGBoost API input schema.
    payload = {
        "step": int(transaction["step"]),
        "transaction_type": transaction["type"],
        "amount": float(transaction["amount"]),
        "oldbalanceOrg": float(transaction["oldbalanceOrg"]),
        "oldbalanceDest": float(transaction["oldbalanceDest"])
    }

    response = requests.post(
        API_URL,
        json=payload,
        timeout=30
    )

    response.raise_for_status()

    result = response.json()

    # Read actual label only after the prediction.
    actual = int(transaction["isFraud"])
    predicted = int(result["prediction"])

    print("\n----------------------------------")
    print("Transaction ID:", transaction_id)
    print("Original Account:", transaction.get("nameOrig"))
    print("Feast Lookup:", "SUCCESS" if feast_lookup_success else "MISSING")
    print("Feast Amount:", feast_features.get("amount"))
    print("Feast Step:", feast_features.get("step"))
    print("Transaction Type:", payload["transaction_type"])
    print("Actual Label:", actual)
    print("Predicted Label:", predicted)
    print("Fraud Probability:", result["fraud_probability"])
    print("Risk Level:", result["risk_level"])
    print("Prediction:", "CORRECT" if actual == predicted else "INCORRECT")
    print("----------------------------------")

    return actual, predicted, feast_lookup_success


def main():

    consumer = create_consumer()

    actual_labels = []
    predicted_labels = []
    feast_success_count = 0

    print("Connected to Kafka.")
    print("Topic:", TOPIC)
    print("Evaluating:", MAX_MESSAGES, "transactions")

    try:

        while len(actual_labels) < MAX_MESSAGES:

            records = consumer.poll(timeout_ms=2000)

            if not records:
                print("Waiting for messages...")
                continue

            for _, messages in records.items():

                for message in messages:

                    try:

                        actual, predicted, feast_success = (
                            process_transaction(message.value)
                        )

                        actual_labels.append(actual)
                        predicted_labels.append(predicted)

                        if feast_success:
                            feast_success_count += 1

                    except (
                        requests.RequestException,
                        KeyError,
                        ValueError
                    ) as error:

                        print("Processing error:", error)

                    if len(actual_labels) >= MAX_MESSAGES:
                        break

                if len(actual_labels) >= MAX_MESSAGES:
                    break

        tn, fp, fn, tp = confusion_matrix(
            actual_labels,
            predicted_labels,
            labels=[0, 1]
        ).ravel()

        print("\n========== FEAST + FRAUD EVALUATION ==========")

        print("Transactions evaluated:", len(actual_labels))
        print("Successful Feast lookups:", feast_success_count)
        print("Failed Feast lookups:", len(actual_labels) - feast_success_count)

        print("\nConfusion Matrix:")
        print("True Negatives:", tn)
        print("False Positives:", fp)
        print("False Negatives:", fn)
        print("True Positives:", tp)

        print(
            "\nAccuracy:",
            round(
                accuracy_score(actual_labels, predicted_labels),
                4
            )
        )

        print("\nClassification Report:")

        print(
            classification_report(
                actual_labels,
                predicted_labels,
                labels=[0, 1],
                target_names=["Legitimate", "Fraud"],
                zero_division=0
            )
        )

        print("==============================================")

    except KeyboardInterrupt:
        print("\nEvaluation interrupted.")

    finally:
        consumer.close()


if __name__ == "__main__":
    main()
