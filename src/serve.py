import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
import time
import requests
import json
from datetime import datetime
from detectors.ensemble import WeightedEnsembleDriftDetector

app = Flask(__name__)

# ── Prometheus Metrics ─────────────────────────────────────
# These are what Prometheus scrapes and Grafana visualizes

prediction_counter = Counter(
    'fraud_predictions_total',
    'Total number of predictions made',
    ['model_version', 'prediction']
)

latency_histogram = Histogram(
    'fraud_prediction_latency_seconds',
    'Time taken to make a prediction',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0]
)

drift_score_gauge = Gauge(
    'drift_score_ks',
    'Current KS drift score'
)

psi_score_gauge = Gauge(
    'drift_score_psi',
    'Current PSI drift score'
)

adwin_score_gauge = Gauge(
    'drift_score_adwin',
    'Current ADWIN drift score'
)

weighted_vote_gauge = Gauge(
    'ensemble_weighted_vote',
    'Current ensemble weighted vote'
)

retraining_counter = Counter(
    'retraining_triggered_total',
    'Total number of times retraining was triggered'
)

batch_counter = Counter(
    'batches_processed_total',
    'Total number of batches processed by drift detector'
)

# ── Global State ───────────────────────────────────────────
model = None
model_version = "unknown"
ensemble = None
reference_data = None
MODEL_NAME = "fraud_detector"

# ── Load Model From MLflow ─────────────────────────────────
def load_model():
    global model, model_version
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    try:
        # Load latest version from registry
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME)
        latest = max(versions, key=lambda v: int(v.version))
        model_version = latest.version
        model_uri = f"models:/{MODEL_NAME}/{model_version}"
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"[Serve] Loaded model '{MODEL_NAME}' version {model_version}")
    except Exception as e:
        print(f"[Serve] Error loading model: {e}")
        raise

# ── Load Reference Data For Drift Detection ────────────────
def load_reference_data():
    global reference_data, ensemble
    
    # Load and preprocess reference data same way as training
    from sklearn.preprocessing import StandardScaler
    df = pd.read_csv('data/creditcard.csv')
    
    scaler = StandardScaler()
    df['scaled_amount'] = scaler.fit_transform(df[['Amount']])
    df['scaled_time'] = scaler.fit_transform(df[['Time']])
    df.drop(['Amount', 'Time'], axis=1, inplace=True)
    
    # Use first 70% as reference (training distribution)
    split = int(len(df) * 0.7)
    reference_data = df.iloc[:split].drop('Class', axis=1)
    
    # Initialize and fit ensemble
    ensemble = WeightedEnsembleDriftDetector(
        ks_threshold=0.05,
        psi_threshold=0.2,
        adwin_delta=0.002,
        drift_fraction=0.3,
        vote_threshold=0.5
    )
    ensemble.fit(reference_data)
    print(f"[Serve] Reference data loaded: {len(reference_data)} samples")
    print(f"[Serve] Ensemble detector initialized")

# ── Routes ─────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model': MODEL_NAME,
        'model_version': model_version,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Single prediction endpoint.
    Accepts JSON with feature values, returns fraud prediction.
    """
    start_time = time.time()
    
    try:
        data = request.get_json()
        features = pd.DataFrame([data['features']])
        
        prediction = model.predict(features)
        pred_label = 'fraud' if prediction[0] == 1 else 'legitimate'
        
        latency = time.time() - start_time
        
        # Update Prometheus metrics
        with latency_histogram.time():
            pass
        prediction_counter.labels(
            model_version=model_version,
            prediction=pred_label
        ).inc()
        
        return jsonify({
            'prediction': int(prediction[0]),
            'label': pred_label,
            'model_version': model_version,
            'latency_ms': round(latency * 1000, 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/detect_drift', methods=['POST'])
def detect_drift():
    """
    Drift detection endpoint.
    Accepts a batch of new data, runs ensemble detection,
    updates Prometheus metrics, triggers retraining if needed.
    """
    try:
        data = request.get_json()
        batch_df = pd.DataFrame(data['batch'])
        batch_id = data.get('batch_id', ensemble.batch_count + 1)
        
        # Run ensemble detection
        result = ensemble.detect(batch_df, batch_id=batch_id)
        
        # Update Prometheus gauges
        drift_score_gauge.set(result['individual_scores']['ks_drift_score'])
        psi_score_gauge.set(result['individual_scores']['psi_drift_score'])
        adwin_score_gauge.set(result['individual_scores']['adwin_drift_score'])
        weighted_vote_gauge.set(result['weighted_vote'])
        batch_counter.inc()
        
        # Log to MLflow
        ensemble.log_to_mlflow(result, "fraud-drift-detection")
        
        # Trigger retraining if ensemble votes yes
        if result['retrain_triggered']:
            retraining_counter.inc()
            print(f"[Serve] Retraining triggered! Weighted vote: "
                  f"{result['weighted_vote']}")
        
        return jsonify({
            'batch_id': batch_id,
            'drift_detected': result['retrain_triggered'],
            'weighted_vote': result['weighted_vote'],
            'votes': result['votes'],
            'weights': result['weights'],
            'individual_scores': result['individual_scores'],
            'retraining_triggered': result['retrain_triggered']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/metrics', methods=['GET'])
def metrics():
    """
    Prometheus metrics endpoint.
    Prometheus scrapes this every 15 seconds.
    """
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route('/summary', methods=['GET'])
def summary():
    """Returns ensemble detection summary."""
    weights = ensemble._compute_weights()
    return jsonify({
        'batches_processed': ensemble.batch_count,
        'retraining_triggered': ensemble.retraining_count,
        'false_triggers': ensemble.false_trigger_count,
        'current_weights': weights,
        'model_version': model_version
    })

# ── Startup ────────────────────────────────────────────────
if __name__ == '__main__':
    print("[Serve] Starting fraud detection API...")
    load_model()
    load_reference_data()
    print("[Serve] API ready at http://127.0.0.1:5001")
    app.run(host='0.0.0.0', port=5001, debug=False)