import pandas as pd
from .cleaner import run_all_cleaners
from ..utils.logs import console

def build_features():
    df_logs, df_sensors, df_maint = run_all_cleaners()
    console("Building device-level features")

    if df_logs.empty:
        console("No logs available; returning empty features")
        return pd.DataFrame()

    telemetry_cols = [c for c in ['temperature_c','battery_pct','humidity_pct','signal_dbm'] if c in df_logs.columns]
    # compute aggregates
    agg = {}
    for col in telemetry_cols:
        agg[col] = ['mean','std','min','max']
    df_tele = df_logs.groupby('device_id').agg(agg)
    # flatten
    df_tele.columns = ['_'.join(col).strip() for col in df_tele.columns.values]
    df_tele = df_tele.reset_index().rename(columns={'device_id':'device_id'})

    # last maintenance if present
    if not df_maint.empty and 'performed_at' in df_maint.columns:
        df_maint_latest = df_maint.sort_values('performed_at').groupby('device_id').tail(1)
        df_maint_latest = df_maint_latest[['device_id','performed_at','next_due']]
    else:
        df_maint_latest = pd.DataFrame(columns=['device_id','performed_at','next_due'])

    # join
    df = df_tele.merge(df_maint_latest, on='device_id', how='left')
    # days since last maintenance
    if 'performed_at' in df.columns:
        df['performed_at'] = pd.to_datetime(df['performed_at'], errors='coerce')
        df['days_since_last_maint'] = (pd.Timestamp.now() - df['performed_at']).dt.days
    else:
        df['days_since_last_maint'] = pd.NA

    console(f"Built features for {len(df)} devices")
    return df
