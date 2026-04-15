import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger("fraud_detection.monitoring")


class DriftMonitor:
    def __init__(self, reference_data: pd.DataFrame, feature_columns: Optional[List[str]] = None):
        self.reference = reference_data
        self.feature_columns = feature_columns or reference_data.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        self.history: List[Dict[str, Any]] = []
        logger.info("Drift monitor initialized with %d reference samples", len(self.reference))
        logger.info("Monitoring columns: %s", self.feature_columns)

    def check_drift(self, current_data: pd.DataFrame, threshold: float = 0.1) -> Dict[str, Any]:
        ref_subset = self.reference[self.feature_columns]
        cur_subset = current_data[self.feature_columns]

        drifted_columns = []
        column_stats = {}
        for col in self.feature_columns:
            statistic, p_value = stats.ks_2samp(
                ref_subset[col].dropna(),
                cur_subset[col].dropna(),
            )
            column_stats[col] = {"ks_statistic": round(float(statistic), 4), "p_value": round(float(p_value), 4)}
            if p_value < 0.05:
                drifted_columns.append(col)

        n_features = len(self.feature_columns)
        n_drifted = len(drifted_columns)
        drift_share = n_drifted / n_features if n_features > 0 else 0

        result = {
            "timestamp": datetime.now().isoformat(),
            "drift_detected": n_drifted > 0,
            "drift_share": drift_share,
            "drifted_columns": drifted_columns,
            "column_stats": column_stats,
            "n_features": n_features,
            "n_drifted": n_drifted,
            "current_samples": len(current_data),
            "threshold": threshold,
            "alert": drift_share > threshold,
        }
        self.history.append(result)
        return result

    def get_alerts(self, threshold: float = 0.1) -> List[Dict[str, Any]]:
        return [
            {
                "timestamp": r["timestamp"],
                "severity": "HIGH" if r["drift_share"] > 0.3 else "MEDIUM",
                "drift_share": r["drift_share"],
                "message": f"Drift detected: {r['drift_share']:.1%} of features drifted",
                "drifted_columns": r["drifted_columns"],
            }
            for r in self.history
            if r["drift_share"] > threshold
        ]

    def summary(self) -> Dict[str, Any]:
        if not self.history:
            return {"message": "No drift checks performed yet"}
        drift_shares = [r["drift_share"] for r in self.history]
        alerts = [r for r in self.history if r["alert"]]
        return {
            "total_checks": len(self.history),
            "total_alerts": len(alerts),
            "avg_drift_share": float(np.mean(drift_shares)),
            "max_drift_share": float(np.max(drift_shares)),
            "first_check": self.history[0]["timestamp"],
            "last_check": self.history[-1]["timestamp"],
        }