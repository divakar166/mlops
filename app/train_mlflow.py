import logging
from datetime import datetime

import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
import pickle
from dotenv import load_dotenv
load_dotenv()

from app.feast_feature import get_training_features

logger = logging.getLogger("train_mlflow")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("fraud-detection")
logger.info(f"MLFLOW TRACKING URL: {os.getenv('MLFLOW_TRACKING_URI')}")

def load_and_preprocess_data():
    """Load and preprocess the training and test data."""
    logger.info("Loading data from data/train.csv and data/test.csv ...")
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    logger.info("Loaded %d train and %d test rows", len(train_df), len(test_df))

    logger.info("Retrieving training features from Feast offline store...")
    train_df = get_training_features(train_df)
    test_df = get_training_features(test_df)
    logger.info(
        "Enriched data with Feast features — train shape=%s, test shape=%s",
        train_df.shape,
        test_df.shape,
    )

    # Encode categorical feature
    logger.info("Encoding merchant_category with LabelEncoder ...")
    encoder = LabelEncoder()
    train_df["merchant_encoded"] = encoder.fit_transform(train_df["merchant_category"])
    test_df["merchant_encoded"] = encoder.transform(test_df["merchant_category"])

    # Prepare features
    feature_cols = [
        "amount",
        "hour",
        "day_of_week",
        "merchant_encoded",
        "avg_amount",
        "transaction_count",
        "fraud_rate",
    ]
    logger.info("Using feature columns: %s", feature_cols)

    X_train = train_df[feature_cols]
    y_train = train_df["is_fraud"]
    X_test = test_df[feature_cols]
    y_test = test_df["is_fraud"]

    logger.info(
        "Final feature matrices — X_train=%s, X_test=%s", X_train.shape, X_test.shape
    )
    logger.info(
        "Fraud ratio in train: %.4f, test: %.4f",
        float(y_train.mean()),
        float(y_test.mean()),
    )

    return X_train, y_train, X_test, y_test, encoder


def train_and_log_model(
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    class_weight: str = "balanced",
):
    logger.info(
        "Starting training run with params: n_estimators=%d, max_depth=%d, "
        "min_samples_split=%d, min_samples_leaf=%d, class_weight=%s",
        n_estimators,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        class_weight,
    )

    X_train, y_train, X_test, y_test, encoder = load_and_preprocess_data()

    with mlflow.start_run():
        run_name = f"rf_est{n_estimators}_depth{max_depth}_{datetime.now().strftime('%H%M%S')}"
        mlflow.set_tag("mlflow.runName", run_name)
        logger.info("MLflow run started: %s", run_name)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("class_weight", class_weight)

        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        ratio = n_neg / max(n_pos, 1)
        mlflow.log_param("class_0_count", n_neg)
        mlflow.log_param("class_1_count", n_pos)
        mlflow.log_param("imbalance_ratio", round(ratio, 2))

        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("fraud_ratio", float(y_train.mean()))
        mlflow.log_param("n_features", X_train.shape[1])

        logger.info(
            "Class balance — class_0=%d, class_1=%d, imbalance_ratio=%.2f",
            n_neg,
            n_pos,
            ratio,
        )

        # Train the model
        logger.info(
            "Training RandomForest model: n_estimators=%d, max_depth=%d",
            n_estimators,
            max_depth,
        )
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        logger.info("Model training complete.")

        # Business cost weights
        FP_COST = 1.0
        FN_COST = 10.0
        logger.info(
            "Using business cost weights FP_COST=%.1f, FN_COST=%.1f", FP_COST, FN_COST
        )

        y_prob_test = None
        y_test_true = None

        for dataset_name, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1]

            if dataset_name == "test":
                y_prob_test = y_prob
                y_test_true = y

            # Calculate all metrics
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, zero_division=0)
            recall = recall_score(y, y_pred, zero_division=0)
            f1 = f1_score(y, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y, y_prob)
            pr_auc = average_precision_score(y, y_prob)

            # Log metrics with dataset prefix
            mlflow.log_metric(f"{dataset_name}_accuracy", accuracy)
            mlflow.log_metric(f"{dataset_name}_precision", precision)
            mlflow.log_metric(f"{dataset_name}_recall", recall)
            mlflow.log_metric(f"{dataset_name}_f1", f1)
            mlflow.log_metric(f"{dataset_name}_roc_auc", roc_auc)
            mlflow.log_metric(f"{dataset_name}_pr_auc", pr_auc)

            logger.info(
                "%s metrics — Accuracy=%.4f, Precision=%.4f, Recall=%.4f, "
                "F1=%.4f, ROC-AUC=%.4f, PR-AUC=%.4f",
                dataset_name.upper(),
                accuracy,
                precision,
                recall,
                f1,
                roc_auc,
                pr_auc,
            )

        # Threshold sweep on TEST set
        if y_prob_test is not None and y_test_true is not None:
            logger.info("Running threshold sweep on TEST set for optimal_threshold ...")
            thresholds = np.linspace(0.01, 0.99, 99)
            best_cost = float("inf")
            best_t = 0.5
            best_f1 = 0.0

            for t in thresholds:
                y_pred_t = (y_prob_test >= t).astype(int)
                tp = int(((y_test_true == 1) & (y_pred_t == 1)).sum())
                fp = int(((y_test_true == 0) & (y_pred_t == 1)).sum())
                fn = int(((y_test_true == 1) & (y_pred_t == 0)).sum())
                tn = int(((y_test_true == 0) & (y_pred_t == 0)).sum())

                prec_t = precision_score(y_test_true, y_pred_t, zero_division=0)
                rec_t = recall_score(y_test_true, y_pred_t, zero_division=0)
                f1_t = f1_score(y_test_true, y_pred_t, zero_division=0)
                cost_t = FP_COST * fp + FN_COST * fn

                # Optional detailed logging per threshold:
                # mlflow.log_metric(f"thr_{t:.2f}_cost", cost_t)
                # mlflow.log_metric(f"thr_{t:.2f}_f1", f1_t)

                if cost_t < best_cost or (cost_t == best_cost and f1_t > best_f1):
                    best_cost = cost_t
                    best_t = float(t)
                    best_f1 = float(f1_t)

            mlflow.log_param("optimal_threshold", best_t)
            mlflow.log_metric("optimal_threshold_cost", best_cost)
            mlflow.log_metric("optimal_threshold_f1", best_f1)

            logger.info(
                "Optimal threshold on TEST (FP=%.1f, FN=%.1f): t=%.2f, cost=%.1f, F1=%.4f",
                FP_COST,
                FN_COST,
                best_t,
                best_cost,
                best_f1,
            )
            print(
                f"\nOptimal threshold on TEST (FP={FP_COST}, FN={FN_COST}): "
                f"t={best_t:.2f}, cost={best_cost:.1f}, F1={best_f1:.4f}"
            )
        else:
            logger.warning("No test probabilities available — skipping threshold sweep.")

        # Log feature importance
        logger.info("Logging feature importances ...")
        for feature, importance in zip(X_train.columns, model.feature_importances_):
            mlflow.log_metric(f"importance_{feature}", importance)
        logger.info("Feature importances logged.")

        for col in X_train.columns:
            X_train[col] = X_train[col].astype("float64")

        logger.info("Registering model in MLflow Model Registry ...")
        print("\nRegistering model in MLflow Model Registry...")
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name="fraud-detection-model",
            input_example=X_train.iloc[:5],
        )

        os.makedirs("models", exist_ok=True)
        encoder_path = "models/encoder.pkl"
        with open(encoder_path, "wb") as f:
            pickle.dump(encoder, f)
        mlflow.log_artifact(encoder_path)
        logger.info("Saved and logged encoder artifact at %s", encoder_path)

        run_id = mlflow.active_run().info.run_id
        logger.info("MLflow Run ID: %s", run_id)
        print(f"\nMLflow Run ID: {run_id}")
        print(f"View this run: http://localhost:5000/#/experiments/1/runs/{run_id}")

        return model, encoder


def run_experiment_sweep():
    logger.info("-" * 60)
    logger.info("RUNNING HYPERPARAMETER EXPERIMENT SWEEP")
    logger.info("-" * 60)
    print("-" * 60)
    print("RUNNING HYPERPARAMETER EXPERIMENT SWEEP")
    print("-" * 60)

    # Define different configurations to try
    experiments = [
        {"n_estimators": 100, "max_depth": 10, "class_weight": None},
        {"n_estimators": 50,  "max_depth": 5,  "class_weight": "balanced"},
        {"n_estimators": 100, "max_depth": 10, "class_weight": "balanced"},
        {"n_estimators": 100, "max_depth": 15, "class_weight": "balanced"},
        {"n_estimators": 200, "max_depth": 10, "class_weight": "balanced"},
        {"n_estimators": 200, "max_depth": 20, "class_weight": "balanced"},
    ]

    for i, params in enumerate(experiments, 1):
        logger.info("Starting experiment %d/%d with params=%s", i, len(experiments), params)
        print(f"\n--- Experiment {i}/{len(experiments)} ---")
        train_and_log_model(**params)

    logger.info("EXPERIMENT SWEEP COMPLETE")
    print("\n" + "-" * 60)
    print("EXPERIMENT SWEEP COMPLETE!")
    print("-" * 60)
    print("\nView all experiments at: http://localhost:5000")
    print("Compare runs to find the best hyperparameters!")


if __name__ == "__main__":
    run_experiment_sweep()