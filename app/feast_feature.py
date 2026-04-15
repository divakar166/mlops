import pandas as pd
from feast import FeatureStore
from datetime import datetime
import logging

logger = logging.getLogger("fraud_detection.feast")

_store: FeatureStore | None = None
FEAST_REPO_PATH = "feature_repo"

FEAST_FALLBACK: dict = {
    "merchant_avg_amount": 100.0,
    "merchant_tx_count":   0,
    "merchant_fraud_rate": 0.05,
}

def get_store() -> FeatureStore:
    """Return the shared FeatureStore, initializing it on first call."""
    global _store
    if _store is None:
        logger.info("Initializing FeatureStore from repo: %s", FEAST_REPO_PATH)
        _store = FeatureStore(repo_path=FEAST_REPO_PATH)
        logger.info("FeatureStore initialized successfully")
    return _store

def get_training_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Retrieving training features from Feast offline store...")
    store = get_store()
    
    entity_df = df[['merchant_category']].copy()
    entity_df['event_timestamp'] = datetime.now()
    entity_df = entity_df.drop_duplicates()
    
    training_data = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "merchant_stats:avg_amount",
            "merchant_stats:transaction_count",
            "merchant_stats:fraud_rate",
        ],
    ).to_df()
    
    result = df.merge(
        training_data[['merchant_category', 'avg_amount', 'transaction_count', 'fraud_rate']],
        on='merchant_category',
        how='left'
    )
    
    print(f"Retrieved features for {len(entity_df)} unique merchants")
    return result

def get_online_features(merchant_category: str) -> tuple[dict, bool]:
    try:
        store = get_store()
        feature_vector = store.get_online_features(
            features=[
                "merchant_stats:avg_amount",
                "merchant_stats:transaction_count",
                "merchant_stats:fraud_rate",
            ],
            entity_rows=[{"merchant_category": merchant_category}],
        ).to_dict()

        avg_amount = feature_vector["avg_amount"][0]
        tx_count   = feature_vector["transaction_count"][0]
        fraud_rate = feature_vector["fraud_rate"][0]

        if any(v is None for v in (avg_amount, tx_count, fraud_rate)):
            logger.warning(
                "Feast returned None for merchant_category=%r — using fallback defaults",
                merchant_category,
            )
            return FEAST_FALLBACK.copy(), False

        return {
            "merchant_avg_amount": avg_amount,
            "merchant_tx_count":   tx_count,
            "merchant_fraud_rate": fraud_rate,
        }, True

    except Exception:
        logger.exception(
            "Feast store error for merchant_category=%r — using fallback defaults",
            merchant_category,
        )
        return FEAST_FALLBACK.copy(), False

def get_online_features_batch(merchant_categories: list) -> pd.DataFrame:
    store = get_store()
    return store.get_online_features(
        features=[
            "merchant_stats:avg_amount",
            "merchant_stats:transaction_count",
            "merchant_stats:fraud_rate",
        ],
        entity_rows=[{"merchant_category": mc} for mc in merchant_categories],
    ).to_df()

if __name__ == "__main__":
    # Test the feature retrieval functions
    print("-"*60)
    print("TESTING FEAST FEATURE RETRIEVAL")
    print("-"*60)
    
    # Test offline retrieval (for training)
    print("\n1. Testing OFFLINE feature retrieval (for training)...")
    train_df = pd.read_csv('data/train.csv').head(10)
    enriched = get_training_features(train_df)
    print("\n   Sample enriched training data:")
    print(enriched[['amount', 'merchant_category', 'avg_amount', 'fraud_rate']].head())
    
    # Test online retrieval (for serving)
    print("\n2. Testing ONLINE feature retrieval (for serving)...")
    for category in ['online', 'grocery', 'travel', 'restaurant', 'retail']:
        features = get_online_features(category)
        print(f"   {category}: avg_amount=${features['merchant_avg_amount']:.2f}, "
              f"fraud_rate={features['merchant_fraud_rate']:.2%}")
    
    # Test batch retrieval
    print("\n3. Testing BATCH online retrieval...")
    batch_features = get_online_features_batch(['online', 'grocery', 'travel'])
    print(batch_features)
    
    print("\n" + "-"*60)
    print("FEAST FEATURE RETRIEVAL TEST COMPLETE!")
    print("-"*60)