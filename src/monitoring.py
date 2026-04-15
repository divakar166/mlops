import pandas as pd
import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset
from datetime import datetime
from typing import List, Dict, Any, Optional

class DriftMonitor:  
    def __init__(self, reference_data: pd.DataFrame, feature_columns: Optional[List[str]] = None):
        self.reference = reference_data
        self.feature_columns = feature_columns or reference_data.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        self.history: List[Dict[str, Any]] = []
        
        print(f"Drift monitor initialized with {len(self.reference):,} reference samples")
        print(f"Monitoring columns: {self.feature_columns}")
    
    def check_drift(self, current_data: pd.DataFrame, threshold: float = 0.1) -> Dict[str, Any]:
        from scipy import stats
        
        ref_subset = self.reference[self.feature_columns]
        cur_subset = current_data[self.feature_columns]
        
        # Simple statistical drift detection using KS test
        drifted_columns = []
        for col in self.feature_columns:
            statistic, p_value = stats.ks_2samp(
                ref_subset[col].dropna(),
                cur_subset[col].dropna()
            )
            if p_value < 0.05:  # 5% significance level
                drifted_columns.append(col)
        
        n_features = len(self.feature_columns)
        n_drifted = len(drifted_columns)
        drift_share = n_drifted / n_features if n_features > 0 else 0
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'drift_detected': n_drifted > 0,
            'drift_share': drift_share,
            'drifted_columns': drifted_columns,
            'n_features': n_features,
            'n_drifted': n_drifted,
            'current_samples': len(current_data),
            'threshold': threshold,
            'alert': drift_share > threshold
        }
        
        self.history.append(result)
        
        return result
    
    def generate_report(self, current_data: pd.DataFrame, output_path: str = "drift_report.html"):
        ref_subset = self.reference[self.feature_columns]
        cur_subset = current_data[self.feature_columns]
        
        try:
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=ref_subset, current_data=cur_subset)
            
            # Save HTML report
            with open(output_path, 'w') as f:
                f.write(report.show(mode='inline').data)
            
            print(f"Drift report saved to {output_path}")
            print(f"Open this file in a browser to view detailed visualizations.")
        except Exception as e:
            print(f"Could not generate Evidently report: {e}")
            print(f"Using simplified drift detection instead.")
    
    def get_alerts(self, threshold: float = 0.1) -> List[Dict[str, Any]]:
        return [
            {
                'timestamp': r['timestamp'],
                'severity': 'HIGH' if r['drift_share'] > 0.3 else 'MEDIUM',
                'drift_share': r['drift_share'],
                'message': f"Drift detected: {r['drift_share']:.1%} of features drifted",
                'drifted_columns': r['drifted_columns']
            }
            for r in self.history
            if r['drift_share'] > threshold
        ]
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics from monitoring history."""
        if not self.history:
            return {"message": "No drift checks performed yet"}
        
        drift_shares = [r['drift_share'] for r in self.history]
        alerts = [r for r in self.history if r['alert']]
        
        return {
            'total_checks': len(self.history),
            'total_alerts': len(alerts),
            'avg_drift_share': np.mean(drift_shares),
            'max_drift_share': np.max(drift_shares),
            'first_check': self.history[0]['timestamp'],
            'last_check': self.history[-1]['timestamp']
        }


def simulate_drift_scenarios():
    from generate_data import generate_transactions
    
    print("-"*70)
    print("DRIFT DETECTION SIMULATION")
    print("-"*70)
    
    # Load reference (training) data
    print("\n1. Loading reference data (training set)...")
    reference = pd.read_csv('data/train.csv')
    feature_cols = ['amount', 'hour', 'day_of_week']
    
    # Initialize drift monitor
    monitor = DriftMonitor(reference, feature_cols)
    
    # Scenario 1: Similar data (should show minimal drift)
    print("\n" + "-"*70)
    print("SCENARIO 1: Test data (similar distribution)")
    print("-"*70)
    test_data = pd.read_csv('data/test.csv')
    result = monitor.check_drift(test_data)
    print(f"  Drift detected: {result['drift_detected']}")
    print(f"  Drift share: {result['drift_share']:.1%}")
    print(f"  Drifted columns: {result['drifted_columns']}")
    print(f"  Alert triggered: {result['alert']}")
    
    # Scenario 2: Fraud spike (10% fraud instead of 2%)
    print("\n" + "-"*70)
    print("SCENARIO 2: Fraud spike (10% fraud rate instead of 2%)")
    print("-"*70)
    fraud_spike = generate_transactions(n_samples=2000, fraud_ratio=0.10, seed=101)
    result = monitor.check_drift(fraud_spike)
    print(f"  Drift detected: {result['drift_detected']}")
    print(f"  Drift share: {result['drift_share']:.1%}")
    print(f"  Drifted columns: {result['drifted_columns']}")
    print(f"  Alert triggered: {result['alert']}")
    
    # Scenario 3: Amount inflation (everything costs more)
    print("\n" + "-"*70)
    print("SCENARIO 3: Amount inflation (2x multiplier)")
    print("-"*70)
    inflated = test_data.copy()
    inflated['amount'] = inflated['amount'] * 2
    result = monitor.check_drift(inflated)
    print(f"  Drift detected: {result['drift_detected']}")
    print(f"  Drift share: {result['drift_share']:.1%}")
    print(f"  Drifted columns: {result['drifted_columns']}")
    print(f"  Alert triggered: {result['alert']}")
    
    # Scenario 4: Time shift (more late-night transactions)
    print("\n" + "-"*70)
    print("SCENARIO 4: Time shift (mostly late-night transactions)")
    print("-"*70)
    night_shift = test_data.copy()
    night_shift['hour'] = np.random.choice([0, 1, 2, 3, 22, 23], size=len(night_shift))
    result = monitor.check_drift(night_shift)
    print(f"  Drift detected: {result['drift_detected']}")
    print(f"  Drift share: {result['drift_share']:.1%}")
    print(f"  Drifted columns: {result['drifted_columns']}")
    print(f"  Alert triggered: {result['alert']}")
    
    # Generate detailed report for the most drifted scenario
    print("\n" + "-"*70)
    print("GENERATING DETAILED REPORT")
    print("-"*70)
    monitor.generate_report(night_shift, "drift_report.html")
    
    # Print summary
    print("\n" + "-"*70)
    print("MONITORING SUMMARY")
    print("-"*70)
    summary = monitor.summary()
    print(f"  Total checks: {summary['total_checks']}")
    print(f"  Total alerts: {summary['total_alerts']}")
    print(f"  Average drift share: {summary['avg_drift_share']:.1%}")
    print(f"  Maximum drift share: {summary['max_drift_share']:.1%}")
    
    # Print alerts
    alerts = monitor.get_alerts()
    if alerts:
        print(f"\n  Alerts ({len(alerts)}):")
        for alert in alerts:
            print(f"    [{alert['severity']}] {alert['message']}")
    
    print("\n" + "-"*70)
    print("DRIFT DETECTION SIMULATION COMPLETE")
    print("-"*70)


if __name__ == "__main__":
    simulate_drift_scenarios()