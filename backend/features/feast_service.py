from pathlib import Path

from feast import FeatureStore


# --------------------------------------------------
# FEAST CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEAST_REPO = (
    PROJECT_ROOT
    / "src"
    / "features"
    / "feast_repo"
)

FEATURE_STORE = FeatureStore(
    repo_path=str(FEAST_REPO)
)


# --------------------------------------------------
# HISTORICAL FEATURE RETRIEVAL
# --------------------------------------------------

def get_transaction_features(transaction_id: str):

    if not transaction_id:
        raise ValueError("transaction_id cannot be empty")

    result = FEATURE_STORE.get_online_features(
        features=[
            "fraud_transaction_features:amount",
            "fraud_transaction_features:log_amount",
            "fraud_transaction_features:step",
            "fraud_transaction_features:type_TRANSFER",
            "fraud_transaction_features:type_CASH_OUT",
        ],
        entity_rows=[
            {"transaction_id": transaction_id}
        ]
    ).to_dict()

    return {
        key: value[0]
        for key, value in result.items()
        if key != "transaction_id"
    }