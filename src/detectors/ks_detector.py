import numpy as np
import pandas as pd
from scipy import stats


class KSDriftDetector:
    """
    Kolmogorov-Smirnov drift detector.
    Compares feature distributions between reference 
    and production data using the KS statistical test.
    
    Standard approach documented in:
    Dal Pozzolo et al. (2015), MDPI Survey (2024)
    """

    def __init__(self, threshold=0.05, drift_fraction=0.5):
        """
        threshold: p-value below which a feature is considered drifted
        drift_fraction: fraction of features that must drift to trigger alert
        """
        self.threshold = threshold
        self.drift_fraction = drift_fraction
        self.reference_data = None
        
        # For ensemble weighting — tracks false positive history
        self.fp_count = 0
        self.total_calls = 0

    def fit(self, reference_data: pd.DataFrame):
        """Store reference data distribution."""
        self.reference_data = reference_data.copy()
        print(f"[KS] Reference data set: {len(reference_data)} samples, "
              f"{len(reference_data.columns)} features")

    def detect(self, new_data: pd.DataFrame):
        """
        Compare new_data against reference distribution.
        
        Returns:
            drift_detected (bool): True if drift detected
            drift_score (float): fraction of features that drifted
            feature_scores (dict): per-feature p-values
        """
        if self.reference_data is None:
            raise ValueError("Must call fit() before detect()")

        self.total_calls += 1
        feature_scores = {}
        drifted_features = []

        for col in self.reference_data.columns:
            if col not in new_data.columns:
                continue
            
            # KS test compares two distributions
            ks_stat, p_value = stats.ks_2samp(
                self.reference_data[col].values,
                new_data[col].values
            )
            
            feature_scores[col] = {
                'ks_statistic': round(ks_stat, 4),
                'p_value': round(p_value, 4),
                'drifted': p_value < self.threshold
            }
            
            if p_value < self.threshold:
                drifted_features.append(col)

        # Drift score = fraction of features that drifted
        drift_score = len(drifted_features) / len(self.reference_data.columns)
        drift_detected = drift_score >= self.drift_fraction

        return {
            'detector': 'KS-test',
            'drift_detected': drift_detected,
            'drift_score': round(drift_score, 4),
            'drifted_features': drifted_features,
            'num_drifted': len(drifted_features),
            'total_features': len(self.reference_data.columns),
            'feature_scores': feature_scores
        }

    def update_fp_history(self, was_false_positive: bool):
        """
        Called by ensemble after retraining to record
        whether this detector's alarm was a false positive.
        Used to compute weights in ensemble voting.
        """
        if was_false_positive:
            self.fp_count += 1

    def get_fp_rate(self):
        """
        Returns historical false positive rate.
        Used by ensemble to compute detector weight.
        Lower FP rate = higher weight in ensemble.
        """
        if self.total_calls == 0:
            return 0.0
        return self.fp_count / self.total_calls