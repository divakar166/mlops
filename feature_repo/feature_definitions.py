from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64

merchant = Entity(
    name="merchant_category",
    description="Merchant category for the transaction (for example, 'online', 'grocery')",
    value_type=ValueType.STRING,
)

merchant_stats_source = FileSource(
    name="merchant_stats_source",
    path="/app/data/merchant_features.parquet",
    timestamp_field="event_timestamp",
)

merchant_stats_fv = FeatureView(
    name="merchant_stats",
    description="Aggregated statistics per merchant category",
    entities=[merchant],
    ttl=timedelta(days=7),
    schema=[
        Field(name="avg_amount", dtype=Float32, description="Average transaction amount"),
        Field(name="transaction_count", dtype=Int64, description="Number of transactions"),
        Field(name="fraud_rate", dtype=Float32, description="Historical fraud rate"),
    ],
    source=merchant_stats_source,
    online=True,
)