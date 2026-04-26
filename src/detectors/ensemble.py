import numpy as np
import pandas as pd
import mlflow
import json
from datetime import datetime
from detectors.ks_detector import KSDriftDetector
from detectors.psi_detector import PSIDriftDetector
from detectors.adwin_detector import ADWINDriftDetector


class WeightedEnsembleDriftDetector:
    """
    Weighted Ensemble Drift Detector — Core Research Contribution.
    
    Combines KS-test, PSI, and ADWIN drift detectors using a
    weighted voting mechanism where each detector's vote is
    weighted by its historical false positive rate.
    
    Detectors with fewer false alarms carry more weight in the
    retraining decision, making the trigger more reliable than
    any single detector alone.
    
    This is the proposed improvement over single-detector
    baselines documented in existing MLOps literature.
    """

    def __init__(
        self,
        ks_threshold=0.05,
        psi_threshold=0.2,
        adwin_delta=0.002,
        drift_fraction=0.5,
        vote_threshold=0.5
    ):
        """
        vote_threshold: weighted vote must exceed this to trigger retraining
                       0.5 = majority weighted vote required
        """
        self.vote_threshold = vote_threshold
        self.drift_fraction = drift_fraction

        # Initialize all three detectors
        self.ks = KSDriftDetector(
            threshold=ks_threshold,
            drift_fraction=drift_fraction
        )
        self.psi = PSIDriftDetector(
            threshold=psi_threshold,
            drift_fraction=drift_fraction
        )
        self.adwin = ADWINDriftDetector(
            delta=adwin_delta,
            drift_fraction=drift_fraction
        )

        self.detectors = {
            'ks': self.ks,
            'psi': self.psi,
            'adwin': self.adwin
        }

        # History tracking for experiment analysis
        self.detection_history = []
        self.retraining_count = 0
        self.false_trigger_count = 0
        self.batch_count = 0

    def fit(self, reference_data: pd.DataFrame):
        """Fit all three detectors on reference data."""
        print("[Ensemble] Fitting all detectors on reference data...")
        for name, detector in self.detectors.items():
            detector.fit(reference_data)
        print("[Ensemble] All detectors fitted successfully")

    def _compute_weights(self):
        """
        Compute each detector's weight based on historical FP rate.
        
        Weight = 1 - FP_rate
        A detector with 0% FP rate gets weight 1.0 (full trust)
        A detector with 50% FP rate gets weight 0.5 (half trust)
        
        Weights are normalized so they sum to 1.
        """
        weights = {}
        for name, detector in self.detectors.items():
            fp_rate = detector.get_fp_rate()
            weights[name] = max(1 - fp_rate, 0.01)  # minimum weight 0.01

        # Normalize weights to sum to 1
        total = sum(weights.values())
        weights = {k: round(v / total, 4) for k, v in weights.items()}
        return weights

    def detect(self, new_data: pd.DataFrame, batch_id=None):
        """
        Run all detectors and compute weighted vote.
        
        Returns:
            result (dict): full detection result including
                          individual detector results,
                          weights, weighted vote, and
                          final retraining decision
        """
        self.batch_count += 1
        batch_id = batch_id or self.batch_count
        timestamp = datetime.now().isoformat()

        # Run all three detectors
        ks_result = self.ks.detect(new_data)
        psi_result = self.psi.detect(new_data)
        adwin_result = self.adwin.detect(new_data)

        detector_results = {
            'ks': ks_result,
            'psi': psi_result,
            'adwin': adwin_result
        }

        # Get current weights based on FP history
        weights = self._compute_weights()

        # Compute weighted vote
        # Each detector votes 1 (drift) or 0 (no drift)
        votes = {
            'ks': 1 if ks_result['drift_detected'] else 0,
            'psi': 1 if psi_result['drift_detected'] else 0,
            'adwin': 1 if adwin_result['drift_detected'] else 0
        }

        weighted_vote = sum(
            weights[name] * votes[name]
            for name in self.detectors.keys()
        )
        weighted_vote = round(weighted_vote, 4)

        # Final decision
        retrain_triggered = weighted_vote >= self.vote_threshold

        if retrain_triggered:
            self.retraining_count += 1

        # Build full result
        result = {
            'batch_id': batch_id,
            'timestamp': timestamp,
            'detector_results': detector_results,
            'weights': weights,
            'votes': votes,
            'weighted_vote': weighted_vote,
            'vote_threshold': self.vote_threshold,
            'retrain_triggered': retrain_triggered,
            'individual_scores': {
                'ks_drift_score': ks_result['drift_score'],
                'psi_drift_score': psi_result['drift_score'],
                'adwin_drift_score': adwin_result['drift_score']
            }
        }

        self.detection_history.append(result)
        return result

    def update_false_positive(self, detector_name: str):
        """
        Called after retraining to record if a detector
        gave a false alarm. Updates weights for future batches.
        """
        if detector_name in self.detectors:
            self.detectors[detector_name].update_fp_history(True)
            self.false_trigger_count += 1
            print(f"[Ensemble] FP recorded for {detector_name}, "
                  f"weight will decrease")

    def log_to_mlflow(self, result: dict, experiment_name: str):
        """
        Log ensemble detection result to MLflow.
        This is how results appear in your MLflow dashboard.
        """
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(
            run_name=f"ensemble_batch_{result['batch_id']}"
        ):
            # Log individual detector scores
            mlflow.log_metric("ks_drift_score",
                            result['individual_scores']['ks_drift_score'])
            mlflow.log_metric("psi_drift_score",
                            result['individual_scores']['psi_drift_score'])
            mlflow.log_metric("adwin_drift_score",
                            result['individual_scores']['adwin_drift_score'])

            # Log weighted vote and decision
            mlflow.log_metric("weighted_vote", result['weighted_vote'])
            mlflow.log_metric("retrain_triggered",
                            1 if result['retrain_triggered'] else 0)

            # Log weights used
            mlflow.log_metric("ks_weight", result['weights']['ks'])
            mlflow.log_metric("psi_weight", result['weights']['psi'])
            mlflow.log_metric("adwin_weight", result['weights']['adwin'])

            # Log individual votes
            mlflow.log_metric("ks_vote", result['votes']['ks'])
            mlflow.log_metric("psi_vote", result['votes']['psi'])
            mlflow.log_metric("adwin_vote", result['votes']['adwin'])

            # Log parameters
            mlflow.log_param("vote_threshold", self.vote_threshold)
            mlflow.log_param("batch_id", result['batch_id'])
            mlflow.log_param("timestamp", result['timestamp'])

            print(f"[Ensemble] Batch {result['batch_id']} logged to MLflow")

    def get_summary(self):
        """Print summary of all detection activity."""
        print("\n" + "="*50)
        print("ENSEMBLE DETECTION SUMMARY")
        print("="*50)
        print(f"Total batches processed: {self.batch_count}")
        print(f"Retraining triggered:    {self.retraining_count}")
        print(f"False triggers recorded: {self.false_trigger_count}")
        print(f"Current weights:")
        weights = self._compute_weights()
        for name, weight in weights.items():
            fp_rate = self.detectors[name].get_fp_rate()
            print(f"  {name:>6}: weight={weight:.4f} | FP rate={fp_rate:.4f}")
        print("="*50)