from sklearn.ensemble import RandomForestRegressor
import joblib
from .features import build_features
from ..utils.logs import console
from pathlib import Path
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parents[2] / 'backend' / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MAINT_MODEL_PATH = MODELS_DIR / 'maintenance_regressor.joblib'

def train_maintenance_model():
    df = build_features()
    if df.empty:
        console("No features to train maintenance model")
        return None
    if 'next_due' not in df.columns or 'performed_at' not in df.columns:
        console("Insufficient maintenance columns to train model; skipping")
        return None
    df = df.dropna(subset=['performed_at','next_due'])
    if df.empty:
        console("No completed maintenance records present for training")
        return None
    df['next_due'] = pd.to_datetime(df['next_due'], errors='coerce')
    df['performed_at'] = pd.to_datetime(df['performed_at'], errors='coerce')
    df['interval_days'] = (df['next_due'] - df['performed_at']).dt.days
    df = df.dropna(subset=['interval_days'])
    numeric_cols = [c for c in df.columns if any(s in c for s in ['mean','std','min','max'])]
    if not numeric_cols:
        console("No numeric features for training")
        return None
    X = df[numeric_cols].fillna(0).values
    y = df['interval_days'].values
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MAINT_MODEL_PATH)
    console(f"Saved maintenance regressor to {MAINT_MODEL_PATH}")
    return model

def predict_intervals(df_features):
    if df_features.empty:
        return df_features
    try:
        model = joblib.load(MAINT_MODEL_PATH)
    except Exception:
        console("Maintenance model not found; please run train_maintenance_model() first")
        df_features['predicted_interval_days'] = pd.NA
        return df_features
    numeric_cols = [c for c in df_features.columns if any(s in c for s in ['mean','std','min','max'])]
    if not numeric_cols:
        df_features['predicted_interval_days'] = pd.NA
        return df_features
    X = df_features[numeric_cols].fillna(0).values
    preds = model.predict(X)
    df_features['predicted_interval_days'] = preds
    return df_features
