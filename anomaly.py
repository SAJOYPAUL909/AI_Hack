from sklearn.ensemble import IsolationForest
import numpy as np
from .features import build_features
from ..utils.logs import console
import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[2] / 'backend' / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
IF_MODEL_PATH = MODELS_DIR / 'isolation_forest.joblib'

def run_anomaly_detection(contamination=0.05):
    df = build_features()
    if df.empty:
        console("No features available for anomaly detection")
        return df
    console("Running anomaly detection")
    numeric_cols = [c for c in df.columns if any(s in c for s in ['mean','std','min','max'])]
    if not numeric_cols:
        console("No numeric feature columns found, skipping anomaly detection")
        df['anomaly'] = 0
        return df
    X = df[numeric_cols].fillna(0).values
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X)
    preds = model.predict(X)
    df['anomaly'] = np.where(preds == -1, 1, 0)
    joblib.dump(model, IF_MODEL_PATH)
    console(f"Saved IsolationForest to {IF_MODEL_PATH}")
    console(f"Detected {int(df['anomaly'].sum())} anomalous devices")
    return df
