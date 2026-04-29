import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, roc_auc_score,
    precision_score, recall_score,
    classification_report
)
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import warnings
import os
warnings.filterwarnings('ignore')
print("Script started")
# ── 1. Load Data ──────────────────────────────────────────
df = pd.read_csv('data/creditcard.csv')
print(f"Loaded {len(df):,} records")

# ── 2. Preprocessing ──────────────────────────────────────
# Scale Amount and Time (standard in fraud detection literature)
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df[['Amount']])
df['scaled_time']   = scaler.fit_transform(df[['Time']])
df.drop(['Amount', 'Time'], axis=1, inplace=True)

X = df.drop('Class', axis=1)
y = df['Class']

# Stratified split preserves fraud ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}")
print(f"Fraud in train: {y_train.sum()} | Fraud in test: {y_test.sum()}")

# ── 3. Fraud ratio for XGBoost imbalance handling ─────────
fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight for XGBoost: {fraud_ratio:.1f}")

#--4 mlflow setup for ci/cd
mlflow.set_tracking_uri(
    os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
)

mlflow.set_experiment("fraud-detection-baseline")

# ── 5. Train Logistic Regression (v1 Baseline) ────────────
print("\nTraining Logistic Regression (v1)...")
with mlflow.start_run(run_name="logistic_regression_v1"):
    
    lr = LogisticRegression(
        class_weight='balanced',  # handles class imbalance
        max_iter=1000,
        random_state=42
    )
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    
    # Metrics
    f1_lr      = f1_score(y_test, y_pred_lr)
    auc_lr     = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
    prec_lr    = precision_score(y_test, y_pred_lr)
    recall_lr  = recall_score(y_test, y_pred_lr)
    
    # Log to MLflow
    mlflow.log_param("model_type", "logistic_regression")
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_metric("f1_score", f1_lr)
    mlflow.log_metric("auc_roc", auc_lr)
    mlflow.log_metric("precision", prec_lr)
    mlflow.log_metric("recall", recall_lr)
    mlflow.sklearn.log_model(
        lr, "model",
        registered_model_name="fraud_detector"
    )
    
    print(f"  F1:        {f1_lr:.4f}")
    print(f"  AUC-ROC:   {auc_lr:.4f}")
    print(f"  Precision: {prec_lr:.4f}")
    print(f"  Recall:    {recall_lr:.4f}")
    print(classification_report(y_test, y_pred_lr))

# ── 6. Train XGBoost (v2 Improved Model) ──────────────────
print("\nTraining XGBoost (v2)...")
with mlflow.start_run(run_name="xgboost_v2"):
    
    xgb = XGBClassifier(
        scale_pos_weight=fraud_ratio,  # handles class imbalance
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    
    # Metrics
    f1_xgb     = f1_score(y_test, y_pred_xgb)
    auc_xgb    = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
    prec_xgb   = precision_score(y_test, y_pred_xgb)
    recall_xgb = recall_score(y_test, y_pred_xgb)
    
    # Log to MLflow
    mlflow.log_param("model_type", "xgboost")
    mlflow.log_param("scale_pos_weight", fraud_ratio)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_metric("f1_score", f1_xgb)
    mlflow.log_metric("auc_roc", auc_xgb)
    mlflow.log_metric("precision", prec_xgb)
    mlflow.log_metric("recall", recall_xgb)
    mlflow.xgboost.log_model(
        xgb, "model",
        registered_model_name="fraud_detector"
    )
    
    print(f"  F1:        {f1_xgb:.4f}")
    print(f"  AUC-ROC:   {auc_xgb:.4f}")
    print(f"  Precision: {prec_xgb:.4f}")
    print(f"  Recall:    {recall_xgb:.4f}")
    print(classification_report(y_test, y_pred_xgb))

# ── 7. Summary ────────────────────────────────────────────
print("\n" + "="*50)
print("TRAINING SUMMARY")
print("="*50)
print(f"Logistic Regression - F1: {f1_lr:.4f} | AUC: {auc_lr:.4f}")
print(f"XGBoost             - F1: {f1_xgb:.4f} | AUC: {auc_xgb:.4f}")
winner = "XGBoost" if f1_xgb > f1_lr else "Logistic Regression"
print(f"Better model: {winner}")
print("Both models logged to MLflow")
print("="*50)