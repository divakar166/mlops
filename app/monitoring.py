import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

from app.db import persist_drift_result, get_drift_history

logger = logging.getLogger("fraud_detection.monitoring")


class DriftMonitor:
    def __init__(
        self,
        reference_data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        default_threshold: float = 0.1,
    ):
        self.reference = reference_data
        self.feature_columns = feature_columns or reference_data.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        self.default_threshold = default_threshold
        logger.info("Drift monitor initialized with %d reference samples", len(self.reference))
        logger.info("Monitoring columns: %s", self.feature_columns)

    def check_drift(
        self,
        current_data: pd.DataFrame,
        threshold: Optional[float] = None,
        window_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run KS drift for each feature, persist result, and return the payload."""
        if threshold is None:
            threshold = self.default_threshold

        ref_subset = self.reference[self.feature_columns]
        cur_subset = current_data[self.feature_columns]

        drifted_columns: List[str] = []
        column_stats: Dict[str, Dict[str, float]] = {}

        for col in self.feature_columns:
            statistic, p_value = stats.ks_2samp(
                ref_subset[col].dropna(),
                cur_subset[col].dropna(),
            )
            column_stats[col] = {
                "ks_statistic": round(float(statistic), 4),
                "p_value":      round(float(p_value), 4),
            }
            if p_value < 0.05:
                drifted_columns.append(col)

        n_features = len(self.feature_columns)
        n_drifted = len(drifted_columns)
        drift_share = n_drifted / n_features if n_features > 0 else 0.0

        result: Dict[str, Any] = {
            "timestamp":        datetime.now().isoformat(),
            "drift_detected":   n_drifted > 0,
            "drift_share":      drift_share,
            "drifted_columns":  drifted_columns,
            "column_stats":     column_stats,
            "n_features":       n_features,
            "n_drifted":        n_drifted,
            "current_samples":  len(current_data),
            "threshold":        float(threshold),
            "alert":            drift_share > threshold,
        }

        # Persist to DB so history survives restarts
        try:
            persist_drift_result(result=result, window_size=window_size or len(current_data))
        except Exception:
            logger.exception("Failed to persist drift result — continuing")

        return result

    def get_alerts(self, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Return alert rows derived from persisted drift history."""
        if threshold is None:
            threshold = self.default_threshold

        history = get_drift_history(limit=200)
        alerts: List[Dict[str, Any]] = []
        for r in history:
            if r["drift_share"] <= threshold:
                continue
            alerts.append(
                {
                    "timestamp": r["timestamp"],
                    "severity":  "HIGH" if r["drift_share"] > 0.3 else "MEDIUM",
                    "drift_share": r["drift_share"],
                    "message":   f"Drift detected: {r['drift_share']:.1%} of features drifted",
                    "drifted_columns": r.get("drifted_columns", []),
                }
            )
        return alerts

    def summary(self) -> Dict[str, Any]:
        """Aggregate stats from persisted history."""
        history = get_drift_history(limit=500)
        if not history:
            return {"message": "No drift checks performed yet"}

        drift_shares = [h["drift_share"] for h in history]
        alerts = [h for h in history if h.get("alert")]

        return {
            "total_checks":    len(history),
            "total_alerts":    len(alerts),
            "avg_drift_share": float(np.mean(drift_shares)),
            "max_drift_share": float(np.max(drift_shares)),
            "first_check":     history[-1]["timestamp"],
            "last_check":      history[0]["timestamp"],
        }