import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    confusion_matrix,
    classification_report
)

def main():
    print("Loading data...")
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    
    print(f"Training samples: {len(train_df):,}")
    print(f"Test samples: {len(test_df):,}")
    print(f"Training fraud ratio: {train_df['is_fraud'].mean():.2%}")
    
    # Encode the categorical feature
    # We need to save the encoder to use the same mapping at inference time
    print("\nEncoding categorical features...")
    encoder = LabelEncoder()
    train_df["merchant_encoded"] = encoder.fit_transform(train_df["merchant_category"])
    test_df["merchant_encoded"] = encoder.transform(test_df["merchant_category"])
    
    print(f"Merchant category mapping: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")
    
    # Prepare features and labels
    feature_cols = ["amount", "hour", "day_of_week", "merchant_encoded"]
    X_train = train_df[feature_cols]
    y_train = train_df["is_fraud"]
    X_test = test_df[feature_cols]
    y_test = test_df["is_fraud"]
    
    # Train a Random Forest classifier
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,      # Number of trees
        max_depth=10,          # Maximum depth of each tree
        random_state=42,       # For reproducibility
        n_jobs=-1              # Use all CPU cores
    )
    model.fit(X_train, y_train)
    print("Training complete!")
    
    # Evaluate on test data
    print("\n" + "-"*50)
    print("MODEL EVALUATION")
    print("-"*50)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(f"\nAccuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  True Negatives:  {cm[0][0]:,} (correctly identified legitimate)")
    print(f"  False Positives: {cm[0][1]:,} (legitimate flagged as fraud)")
    print(f"  False Negatives: {cm[1][0]:,} (fraud missed - DANGEROUS!)")
    print(f"  True Positives:  {cm[1][1]:,} (correctly caught fraud)")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
    
    # Feature importance
    print("\nFeature Importance:")
    for name, importance in sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"  {name}: {importance:.4f}")
    
    # Save the model and encoder together
    print("\nSaving model to models/model.pkl...")
    with open("models/model.pkl", "wb") as f:
        pickle.dump((model, encoder), f)
    
    print("\nModel trained and saved successfully!")

if __name__ == "__main__":
    main()