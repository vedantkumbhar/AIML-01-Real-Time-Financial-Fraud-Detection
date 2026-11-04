from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64
from feast.value_type import ValueType

transaction = Entity(
    name="transaction_id",
    join_keys=["transaction_id"],
    value_type=ValueType.STRING,
    description="Unique identifier for each financial transaction"
)

transaction_source = FileSource(
    name="fraud_transaction_source",
    path="data/feast_features.parquet",
    timestamp_field="event_timestamp"
)

fraud_transaction_features = FeatureView(
    name="fraud_transaction_features",
    entities=[transaction],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="step", dtype=Int64),
        Field(name="amount", dtype=Float32),
        Field(name="log_amount", dtype=Float32),
        Field(name="sender_txn_count_before", dtype=Int64),
        Field(name="sender_avg_amount_before", dtype=Float32),
        Field(name="sender_hours_since_previous", dtype=Float32),
        Field(name="receiver_txn_count_before", dtype=Int64),
        Field(name="receiver_avg_amount_before", dtype=Float32),
        Field(name="origin_balance_error", dtype=Float32),
        Field(name="destination_balance_error", dtype=Float32),
        Field(name="origin_balance_zero_before", dtype=Int64),
        Field(name="origin_balance_zero_after", dtype=Int64),
        Field(name="destination_balance_zero_before", dtype=Int64),
        Field(name="amount_vs_sender_average", dtype=Float32),
        Field(name="type_CASH_IN", dtype=Int64),
        Field(name="type_CASH_OUT", dtype=Int64),
        Field(name="type_DEBIT", dtype=Int64),
        Field(name="type_PAYMENT", dtype=Int64),
        Field(name="type_TRANSFER", dtype=Int64),
    ],
    source=transaction_source,
    online=True,
    description="PaySim transaction features for fraud detection"
)