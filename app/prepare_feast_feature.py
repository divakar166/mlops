import os
import subprocess
import logging
from datetime import datetime

import numpy as np
import pandas as pd


logger = logging.getLogger("feast_prep")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)


def compute_merchant_features(df: pd.DataFrame, split: str = "train") -> pd.DataFrame:
    if split != "train":
        raise ValueError("Merchant stats must only be computed from the training split.")
    logger.info("   Computing merchant-level features...")

    # Group by merchant category and compute aggregates
    stats = (
        df.groupby("merchant_category")
        .agg(
            avg_amount=("amount", "mean"),
            transaction_count=("amount", "count"),
            fraud_rate=("is_fraud", "mean"),
        )
        .reset_index()
    )

    last_seen = (
        df.groupby("merchant_category")["event_timestamp"]
        .max()
        .reset_index()
    )
    last_seen.rename(columns={"event_timestamp": "event_timestamp"}, inplace=True)
    stats = stats.merge(last_seen, on="merchant_category")

    stats["avg_amount"] = stats["avg_amount"].astype("float32")
    stats["transaction_count"] = stats["transaction_count"].astype("int64")
    stats["fraud_rate"] = stats["fraud_rate"].astype("float32")

    logger.info("   Merchant feature summary:\n%s", stats.to_string(index=False))
    return stats


def main():
    logger.info("-" * 60)
    logger.info("FEAST FEATURE PREPARATION START")
    logger.info("-" * 60)

    # 1. Load training data
    logger.info("1. Loading training data from data/train.csv ...")
    train_df = pd.read_csv("data/train.csv")
    logger.info("   Loaded %d transactions", len(train_df))

    # 2. Compute merchant features
    logger.info("2. Computing merchant features...")
    merchant_features = compute_merchant_features(train_df, split="train")

    # Ensure event_timestamp is tz-aware datetime
    logger.info("   Normalizing event_timestamp to UTC datetimes ...")
    merchant_features["event_timestamp"] = pd.to_datetime(
        merchant_features["event_timestamp"], utc=True, errors="raise"
    )
    logger.info(
        "   event_timestamp dtype after normalization: %s",
        merchant_features["event_timestamp"].dtype,
    )

    # 3. Save as Parquet
    logger.info("3. Saving features to Parquet ...")
    os.makedirs("data", exist_ok=True)
    output_path = "data/merchant_features.parquet"
    merchant_features.to_parquet(output_path, index=False)
    logger.info("   Saved merchant features to %s", output_path)

    # 4. Apply Feast feature definitions
    logger.info("4. Applying Feast feature definitions (feast apply) ...")
    try:
        result = subprocess.run(
            ["feast", "apply"],
            cwd="feature_repo",
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info("   Feast apply completed successfully.")
        if result.stdout:
            logger.info("   feast apply stdout:\n%s", result.stdout)
        if result.stderr:
            logger.warning("   feast apply stderr:\n%s", result.stderr)
    except subprocess.CalledProcessError as e:
        logger.error("   Feast apply failed with return code %s", e.returncode)
        if e.stdout:
            logger.error("   feast apply stdout:\n%s", e.stdout)
        if e.stderr:
            logger.error("   feast apply stderr:\n%s", e.stderr)
        raise

    # 5. Materialize features to online store
    logger.info("5. Materializing features to online store (feast materialize-incremental) ...")
    ts = datetime.now().isoformat()
    logger.info("   Materialize cutoff timestamp: %s", ts)
    try:
        result = subprocess.run(
            ["feast", "materialize-incremental", ts],
            cwd="feature_repo",
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info("   Feast materialize-incremental completed successfully.")
        if result.stdout:
            logger.info("   feast materialize stdout:\n%s", result.stdout)
        if result.stderr:
            logger.warning("   feast materialize stderr:\n%s", result.stderr)
    except subprocess.CalledProcessError as e:
        logger.error(
            "   Feast materialize-incremental failed with return code %s", e.returncode
        )
        if e.stdout:
            logger.error("   feast materialize stdout:\n%s", e.stdout)
        if e.stderr:
            logger.error("   feast materialize stderr:\n%s", e.stderr)
        raise

    logger.info("-" * 60)
    logger.info("FEAST FEATURE PREPARATION COMPLETE")
    logger.info("-" * 60)


if __name__ == "__main__":
    main()