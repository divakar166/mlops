import logging
from datetime import datetime

import os
import shutil
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

EXPERIMENT_NAME = "fraud-detection"
REGISTERED_MODEL_NAME = "fraud-detection-model"
CHAMPION_ALIAS = "champion"

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(EXPERIMENT_NAME)
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
        run_id = mlflow.active_run().info.run_id
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
                mlflow.log_metric(f"thr_{t:.2f}_cost", cost_t)
                mlflow.log_metric(f"thr_{t:.2f}_f1", f1_t)

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
            logger.info(
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

        logger.info("Logging model artifact to MLflow run ...")
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=X_train.iloc[:5],
        )

        os.makedirs("models", exist_ok=True)
        encoder_path = f"models/encoder_{run_id}.pkl"
        with open(encoder_path, "wb") as f:
            pickle.dump(encoder, f)
        mlflow.log_artifact(encoder_path, artifact_path="encoder")
        logger.info("Saved and logged encoder artifact at %s", encoder_path)

        logger.info("MLflow Run ID: %s", run_id)
        logger.info(f"\nMLflow Run ID: {run_id}")
        logger.info(f"View this run: http://localhost:5000/#/experiments/1/runs/{run_id}")

        return model, encoder

REQUIRED_SELECTION_METRICS = [
    "test_recall",
    "test_precision",
    "train_f1",
    "test_f1",
    "test_pr_auc",
    "optimal_threshold_cost",
]


def _artifact_exists(run_id: str, artifact_path: str) -> bool:
    try:
        artifacts = mlflow.artifacts.list_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
        )
        return bool(artifacts)
    except Exception:
        logger.exception(
            "Failed to inspect artifact_path=%s for run_id=%s",
            artifact_path,
            run_id,
        )
        return False


def _encoder_artifact_exists(run_id: str) -> bool:
    expected_path = f"encoder/encoder_{run_id}.pkl"
    try:
        artifacts = mlflow.artifacts.list_artifacts(
            run_id=run_id,
            artifact_path="encoder",
        )
        return any(artifact.path == expected_path for artifact in artifacts)
    except Exception:
        logger.exception("Failed to inspect encoder artifact for run_id=%s", run_id)
        return False


def select_best_model() -> dict:
    """Select the champion run from the current MLflow experiment."""
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment not found: {EXPERIMENT_NAME}")

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        output_format="pandas",
    )
    if runs_df.empty:
        raise RuntimeError(f"No runs found in MLflow experiment: {EXPERIMENT_NAME}")

    logger.info("Total runs evaluated for champion selection: %d", len(runs_df))

    metric_cols = [f"metrics.{metric}" for metric in REQUIRED_SELECTION_METRICS]
    for col in metric_cols:
        if col not in runs_df.columns:
            runs_df[col] = np.nan

    candidate_cols = ["run_id"] + metric_cols
    candidates = runs_df[candidate_cols].copy()
    candidates = candidates.rename(
        columns={f"metrics.{metric}": metric for metric in REQUIRED_SELECTION_METRICS}
    )

    for metric in REQUIRED_SELECTION_METRICS:
        candidates[metric] = pd.to_numeric(candidates[metric], errors="coerce")

    metric_missing = candidates[REQUIRED_SELECTION_METRICS].isna().any(axis=1)
    artifact_missing = []
    for run_id in candidates["run_id"]:
        has_model = _artifact_exists(run_id, "model")
        has_encoder = _encoder_artifact_exists(run_id)
        artifact_missing.append(not (has_model and has_encoder))

    candidates["rejection_reason"] = ""
    candidates.loc[metric_missing, "rejection_reason"] = "missing_metrics"
    candidates.loc[artifact_missing, "rejection_reason"] = "missing_artifacts"

    valid_runs = candidates[candidates["rejection_reason"] == ""].copy()
    if valid_runs.empty:
        raise RuntimeError(
            "No runs have the required metrics plus model and encoder artifacts."
        )

    eligible = valid_runs[
        (valid_runs["test_recall"] >= 0.75)
        & (valid_runs["test_precision"] >= 0.50)
        & ((valid_runs["train_f1"] - valid_runs["test_f1"]).abs() <= 0.15)
    ].copy()

    fallback_used = eligible.empty
    if fallback_used:
        logger.warning(
            "No runs satisfied champion eligibility constraints; falling back to "
            "highest test_recall model."
        )
        selected_pool = valid_runs.copy()
    else:
        selected_pool = eligible

    min_cost = selected_pool["optimal_threshold_cost"].min()
    max_cost = selected_pool["optimal_threshold_cost"].max()
    if max_cost == min_cost:
        selected_pool["normalized_cost"] = 0.0
    else:
        selected_pool["normalized_cost"] = (
            (selected_pool["optimal_threshold_cost"] - min_cost)
            / (max_cost - min_cost)
        )

    selected_pool["score"] = (
        0.35 * selected_pool["test_pr_auc"]
        + 0.25 * selected_pool["test_recall"]
        + 0.20 * selected_pool["test_f1"]
        + 0.10 * selected_pool["test_precision"]
        - 0.10 * selected_pool["normalized_cost"]
    )

    if fallback_used:
        selected_pool = selected_pool.sort_values(
            by=[
                "test_recall",
                "score",
                "optimal_threshold_cost",
                "test_pr_auc",
                "test_f1",
            ],
            ascending=[False, False, True, False, False],
        )
    else:
        selected_pool = selected_pool.sort_values(
            by=["score", "optimal_threshold_cost", "test_pr_auc", "test_f1"],
            ascending=[False, True, False, False],
        )

    champion = selected_pool.iloc[0]
    champion_run_id = champion["run_id"]
    champion_model_uri = f"runs:/{champion_run_id}/model"
    champion_metrics = {
        metric: float(champion[metric]) for metric in REQUIRED_SELECTION_METRICS
    }
    champion_metrics["normalized_cost"] = float(champion["normalized_cost"])
    champion_metrics["score"] = float(champion["score"])

    eligible_count = len(eligible)
    rejected_count = len(candidates) - len(valid_runs)
    logger.info("Eligible runs count: %d", eligible_count)
    logger.info("Rejected runs count: %d", rejected_count)
    logger.info("Selected champion run_id: %s", champion_run_id)
    logger.info("Champion score: %.6f", champion_metrics["score"])
    logger.info("Fallback used: %s", fallback_used)

    return {
        "run_id": champion_run_id,
        "metrics": champion_metrics,
        "model_uri": champion_model_uri,
        "fallback_used": fallback_used,
        "eligible_runs": eligible_count,
        "rejected_runs": rejected_count,
        "total_runs": len(runs_df),
    }


def register_champion(champion: dict) -> dict:
    """Register the selected run model and promote it with the champion alias."""
    run_id = champion["run_id"]
    model_uri = champion["model_uri"]
    encoder_artifact_path = f"encoder/encoder_{run_id}.pkl"
    encoder_artifact_uri = f"runs:/{run_id}/{encoder_artifact_path}"

    if not _artifact_exists(run_id, "model"):
        raise RuntimeError(f"Champion model artifact missing for run_id={run_id}")
    if not _encoder_artifact_exists(run_id):
        raise RuntimeError(f"Champion encoder artifact missing for run_id={run_id}")

    logger.info(
        "Registering champion model from %s as %s",
        model_uri,
        REGISTERED_MODEL_NAME,
    )
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    client = mlflow.MlflowClient()
    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        CHAMPION_ALIAS,
        model_version.version,
    )
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        model_version.version,
        "alias",
        CHAMPION_ALIAS,
    )
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        model_version.version,
        "champion_run_id",
        run_id,
    )
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        model_version.version,
        "encoder_artifact_uri",
        encoder_artifact_uri,
    )

    os.makedirs("models", exist_ok=True)
    champion_encoder_path = f"models/encoder_{run_id}.pkl"
    runtime_encoder_path = "models/encoder.pkl"
    if not os.path.exists(champion_encoder_path):
        downloaded_path = mlflow.artifacts.download_artifacts(
            artifact_uri=encoder_artifact_uri,
            dst_path="models",
        )
        champion_encoder_path = downloaded_path
    shutil.copyfile(champion_encoder_path, runtime_encoder_path)
    logger.info("Champion encoder linked at %s", runtime_encoder_path)

    logger.info(
        "Promoted model version %s to alias @%s",
        model_version.version,
        CHAMPION_ALIAS,
    )
    return {
        "model_name": REGISTERED_MODEL_NAME,
        "model_version": model_version.version,
        "alias": CHAMPION_ALIAS,
        "run_id": run_id,
        "model_uri": model_uri,
        "encoder_artifact_uri": encoder_artifact_uri,
    }

def run_experiment_sweep():
    logger.info("-" * 60)
    logger.info("RUNNING HYPERPARAMETER EXPERIMENT SWEEP")
    logger.info("-" * 60)

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
        logger.info(f"\n--- Experiment {i}/{len(experiments)} ---")
        train_and_log_model(**params)
    champion = select_best_model()
    registered_champion = register_champion(champion)

    logger.info("Champion selected: %s", champion)
    logger.info("Champion registered: %s", registered_champion)

    logger.info("EXPERIMENT SWEEP COMPLETE!")
    logger.info("-" * 60)
    logger.info("View all experiments at: http://localhost:5000")
    logger.info("Champion model is available as @%s", CHAMPION_ALIAS)


if __name__ == "__main__":
    run_experiment_sweep()
