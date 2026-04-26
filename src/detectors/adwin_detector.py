import numpy as np
import pandas as pd
from river import drift


class ADWINDriftDetector:
    """
    ADWIN (Adaptive Windowing) drift detector.
    Unlike KS and PSI which compare two static snapshots,
    ADWIN monitors a continuous data stream and detects
    when statistics within the window change significantly.
    
    Particularly effective for gradual drift detection.
    
    From River library — documented in:
    ADWIN-U ResearchGate (2025), MDPI Survey (2024)
    """

    def __init__(self, delta=0.002, drift_fraction=0.5):
        """
        delta: confidence parameter (lower = more sensitive)
        drift_fraction: fraction of features that must drift to trigger alert
        """
        self.delta = delta
        self.drift_fraction = drift_fraction
        self.reference_data = None

        # For ensemble weighting
        self.fp_count = 0
        self.total_calls = 0

    def fit(self, reference_data: pd.DataFrame):
        """Store reference data for comparison."""
        self.reference_data = reference_data.copy()
        print(f"[ADWIN] Reference data set: {len(reference_data)} samples, "
              f"{len(reference_data.columns)} features")

    def _detect_feature_drift(self, reference_col, new_col):
        """
        Run ADWIN on a single feature by feeding reference
        then new data through the detector sequentially.
        ADWIN signals drift when it detects a change point.
        """
        adwin = drift.ADWIN(delta=self.delta)
        drift_detected = False
        change_point = None

        # Feed reference data first to establish baseline window
        for i, val in enumerate(reference_col):
            adwin.update(float(val))

        # Feed new data — ADWIN signals if distribution changed
        for i, val in enumerate(new_col):
            adwin.update(float(val))
            if adwin.drift_detected:
                drift_detected = True
                change_point = i
                break

        return drift_detected, change_point

    def detect(self, new_data: pd.DataFrame):
        """
        Run ADWIN on each feature and detect drift.

        Returns:
            drift_detected (bool): True if drift detected
            drift_score (float): fraction of features where drift detected
            feature_scores (dict): per-feature drift results
        """
        if self.reference_data is None:
            raise ValueError("Must call fit() before detect()")

        self.total_calls += 1
        feature_scores = {}
        drifted_features = []

        for col in self.reference_data.columns:
            if col not in new_data.columns:
                continue

            drifted, change_point = self._detect_feature_drift(
                self.reference_data[col].values,
                new_data[col].values
            )

            feature_scores[col] = {
                'drifted': drifted,
                'change_point': change_point
            }

            if drifted:
                drifted_features.append(col)

        # Drift score = fraction of features where ADWIN detected drift
        drift_score = len(drifted_features) / len(self.reference_data.columns)
        drift_detected = drift_score >= self.drift_fraction

        return {
            'detector': 'ADWIN',
            'drift_detected': drift_detected,
            'drift_score': round(drift_score, 4),
            'drifted_features': drifted_features,
            'num_drifted': len(drifted_features),
            'total_features': len(self.reference_data.columns),
            'feature_scores': feature_scores
        }

    def update_fp_history(self, was_false_positive: bool):
        """Record whether this detector's alarm was a false positive."""
        if was_false_positive:
            self.fp_count += 1

    def get_fp_rate(self):
        """Returns historical false positive rate for ensemble weighting."""
        if self.total_calls == 0:
            return 0.0
        return self.fp_count / self.total_calls