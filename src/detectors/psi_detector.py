import numpy as np
import pandas as pd


class PSIDriftDetector:
    """
    Population Stability Index (PSI) drift detector.
    Standard drift detection method in financial ML systems.
    
    PSI < 0.1:  No significant drift
    PSI 0.1-0.2: Moderate drift
    PSI > 0.2:  Significant drift
    
    Documented in financial ML literature:
    IJSRA (2024), IJCSMC (2023)
    """

    def __init__(self, threshold=0.2, drift_fraction=0.5, bins=10):
        """
        threshold: PSI value above which a feature is considered drifted
        drift_fraction: fraction of features that must drift to trigger alert
        bins: number of bins for distribution comparison
        """
        self.threshold = threshold
        self.drift_fraction = drift_fraction
        self.bins = bins
        self.reference_data = None
        self.bin_edges = {}

        # For ensemble weighting
        self.fp_count = 0
        self.total_calls = 0

    def _compute_psi(self, reference_col, new_col, bin_edges):
        """
        Compute PSI for a single feature.
        PSI = sum((actual% - expected%) * ln(actual% / expected%))
        """
        # Bin both distributions using reference bin edges
        ref_counts, _ = np.histogram(reference_col, bins=bin_edges)
        new_counts, _ = np.histogram(new_col, bins=bin_edges)

        # Convert to proportions
        ref_proportions = ref_counts / len(reference_col)
        new_proportions = new_counts / len(new_col)

        # Avoid division by zero and log(0)
        ref_proportions = np.where(ref_proportions == 0, 0.0001, ref_proportions)
        new_proportions = np.where(new_proportions == 0, 0.0001, new_proportions)

        # PSI formula
        psi = np.sum(
            (new_proportions - ref_proportions) *
            np.log(new_proportions / ref_proportions)
        )
        return round(float(psi), 4)

    def fit(self, reference_data: pd.DataFrame):
        """
        Store reference data and compute bin edges for each feature.
        Bin edges are fixed from reference data so production
        data is compared against the same bins.
        """
        self.reference_data = reference_data.copy()

        # Pre-compute bin edges from reference data for each feature
        for col in reference_data.columns:
            _, bin_edges = np.histogram(reference_data[col].values, bins=self.bins)
            # Extend edges to capture out-of-range production values
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
            self.bin_edges[col] = bin_edges

        print(f"[PSI] Reference data set: {len(reference_data)} samples, "
              f"{len(reference_data.columns)} features")

    def detect(self, new_data: pd.DataFrame):
        """
        Compute PSI for each feature and detect drift.

        Returns:
            drift_detected (bool): True if drift detected
            drift_score (float): average PSI across all features
            feature_scores (dict): per-feature PSI values
        """
        if self.reference_data is None:
            raise ValueError("Must call fit() before detect()")

        self.total_calls += 1
        feature_scores = {}
        drifted_features = []
        psi_values = []

        for col in self.reference_data.columns:
            if col not in new_data.columns:
                continue

            psi = self._compute_psi(
                self.reference_data[col].values,
                new_data[col].values,
                self.bin_edges[col]
            )

            drifted = psi > self.threshold
            feature_scores[col] = {
                'psi': psi,
                'drifted': drifted,
                'severity': 'none' if psi < 0.1 else 'moderate' if psi < 0.2 else 'significant'
            }

            psi_values.append(psi)
            if drifted:
                drifted_features.append(col)

        # Drift score = average PSI across features
        drift_score = np.mean(psi_values) if psi_values else 0.0
        drift_detected = (
            len(drifted_features) / len(self.reference_data.columns)
            >= self.drift_fraction
        )

        return {
            'detector': 'PSI',
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