import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_transactions(n_samples=10000, fraud_ratio=0.02, seed=42):
    np.random.seed(seed)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    end_time   = datetime.utcnow()
    start_time = end_time - timedelta(days=90)
    total_seconds = int((end_time - start_time).total_seconds())

    def random_timestamps(n):
        offsets = np.random.randint(0, total_seconds, size=n)
        return [start_time + timedelta(seconds=int(s)) for s in offsets]

    # Legitimate transactions: normal shopping patterns
    # - Amounts follow a log-normal distribution (most small, some large)
    # - Hours are uniformly distributed throughout the day
    # - Merchant categories weighted toward everyday shopping
    legit = pd.DataFrame({
        "amount": np.random.lognormal(mean=3.5, sigma=1.2, size=n_legit),  # ~$33 average
        "hour": np.random.randint(0, 24, size=n_legit),
        "day_of_week": np.random.randint(0, 7, size=n_legit),
        "merchant_category": np.random.choice(
            ["grocery", "restaurant", "retail", "online", "travel"],
            size=n_legit,
            p=[0.30, 0.25, 0.25, 0.15, 0.05]  # Weighted toward everyday shopping
        ),
        "is_fraud": 0,
        "event_timestamp": random_timestamps(n_legit)
    })
    
    # Fraudulent transactions: suspicious patterns
    # - Higher amounts (fraudsters go big)
    # - Late night hours (less scrutiny)
    # - More online and travel (easier to exploit)
    fraud = pd.DataFrame({
        "amount": np.random.lognormal(mean=5.5, sigma=1.5, size=n_fraud),  # ~$245 average
        "hour": np.random.choice([0, 1, 2, 3, 4, 5, 23], size=n_fraud),  # Late night
        "day_of_week": np.random.randint(0, 7, size=n_fraud),
        "merchant_category": np.random.choice(
            ["grocery", "restaurant", "retail", "online", "travel"],
            size=n_fraud,
            p=[0.05, 0.05, 0.10, 0.60, 0.20]  # Weighted toward online/travel
        ),
        "is_fraud": 1,
        "event_timestamp": random_timestamps(n_fraud)
    })
    
    # Combine and shuffle
    df = pd.concat([legit, fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    # Generate dataset
    print("Generating synthetic fraud detection dataset...")
    df = generate_transactions(n_samples=10000, fraud_ratio=0.02)
    
    # Split into train (80%) and test (20%)
    df_sorted  = df.sort_values("event_timestamp").reset_index(drop=True)
    split_idx  = int(len(df_sorted) * 0.8)
    train_df   = df_sorted.iloc[:split_idx]
    test_df    = df_sorted.iloc[split_idx:]
    
    # Save to CSV files
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)
    
    # Print summary statistics
    print(f"\nDataset generated successfully!")
    print(f"Training set: {len(train_df):,} transactions")
    print(f"Test set: {len(test_df):,} transactions")
    print(f"Overall fraud ratio: {df['is_fraud'].mean():.2%}")
    print(f"\nLegitimate transactions - Average amount: ${df[df['is_fraud']==0]['amount'].mean():.2f}")
    print(f"Fraudulent transactions - Average amount: ${df[df['is_fraud']==1]['amount'].mean():.2f}")
    print(f"\nMerchant category distribution (fraud):")
    print(df[df['is_fraud']==1]['merchant_category'].value_counts(normalize=True))