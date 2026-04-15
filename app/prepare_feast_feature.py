# src/prepare_feast_feature.py

import pandas as pd
import numpy as np
from datetime import datetime
import subprocess
import os

def compute_merchant_features(df: pd.DataFrame, split: str = "train") -> pd.DataFrame:
    if split != "train":
        raise ValueError("Merchant stats must only be computed from the training split.")
    print("Computing merchant-level features...")
    
    # Group by merchant category and compute aggregates
    stats = df.groupby("merchant_category").agg(
        avg_amount=("amount", "mean"),
        transaction_count=("amount", "count"),
        fraud_rate=("is_fraud", "mean"),
    ).reset_index()

    last_seen = df.groupby("merchant_category")["event_timestamp"].max().reset_index()
    last_seen.rename(columns={"event_timestamp": "event_timestamp"}, inplace=True)
    stats = stats.merge(last_seen, on="merchant_category")
    
    stats["avg_amount"]        = stats["avg_amount"].astype("float32")
    stats["transaction_count"] = stats["transaction_count"].astype("int64")
    stats["fraud_rate"]        = stats["fraud_rate"].astype("float32")
    return stats

def main():
    print("-"*60)
    print("FEAST FEATURE PREPARATION")
    print("-"*60)
    
    # Load training data
    print("\n1. Loading training data...")
    train_df = pd.read_csv('data/train.csv')
    print(f"   Loaded {len(train_df):,} transactions")
    
    # Compute merchant features
    print("\n2. Computing merchant features...")
    merchant_features = compute_merchant_features(train_df, split="train")
    
    print("\n   Computed features:")
    print(merchant_features.to_string(index=False))
    
    # Save as Parquet (required format for Feast file source)
    print("\n3. Saving features to Parquet...")
    os.makedirs('data', exist_ok=True)
    output_path = 'data/merchant_features.parquet'
    merchant_features.to_parquet(output_path, index=False)
    print(f"   Saved to {output_path}")
    
    # Apply Feast feature definitions
    print("\n4. Applying Feast feature definitions...")
    try:
        result = subprocess.run(
            ['feast', 'apply'],
            cwd='feature_repo',
            capture_output=True,
            text=True,
            check=True
        )
        print("   Feature definitions applied successfully!")
        if result.stdout:
            print(f"   {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"   Error applying Feast: {e.stderr}")
        raise
    
    # Materialize features to online store
    print("\n5. Materializing features to online store...")
    try:
        result = subprocess.run(
            ['feast', 'materialize-incremental', datetime.now().isoformat()],
            cwd='feature_repo',
            capture_output=True,
            text=True,
            check=True
        )
        print("   Features materialized successfully!")
        if result.stdout:
            print(f"   {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"   Error materializing: {e.stderr}")
        raise
    
    print("\n" + "-"*60)
    print("FEAST FEATURE PREPARATION COMPLETE!")
    print("-"*60)

if __name__ == "__main__":
    main()