from pathlib import Path
import pandas as pd
from ..utils.logs import console
from ..utils.io_utils import read_csv_safe, write_csv_safe

BASE_DIR = Path(__file__).resolve().parents[2]  # project root/backend
RAW_DIR = BASE_DIR / 'backend' / 'data' / 'raw'
CLEAN_DIR = BASE_DIR / 'backend' / 'data' / 'cleaned'

def _safe_read(pathname):
    p = RAW_DIR / pathname
    console(f"Loading {p}")
    try:
        df = read_csv_safe(p)
        console(f"Loaded {len(df)} rows from {p.name}")
        # Log what columns pandas detected (very useful for debugging)
        console(f"Detected columns: {list(df.columns[:20])}")
        if len(df) > 0:
            # show the first parsed row (as a dict) so we can see delimiter mis-parsing
            console(f"First parsed row sample: {df.iloc[0].to_dict()}")
        return df
    except Exception as e:
        console(f"Failed to load {p}: {e}")
        return pd.DataFrame()


def clean_maintenance():
    df = _safe_read('maintenance_records.csv')
    if df.empty:
        return df
    # normalize col names
    df.columns = [c.strip() for c in df.columns]
    # ensure expected columns
    expected = ['device_id', 'performed_at', 'details', 'next_due']
    for col in expected:
        if col not in df.columns:
            df[col] = pd.NA
    # parse datetimes
    df['performed_at'] = pd.to_datetime(df['performed_at'], errors='coerce')
    df['next_due'] = pd.to_datetime(df['next_due'], errors='coerce')
    # drop rows with missing device_id
    missing_dev = df['device_id'].isna().sum()
    console(f"maintenance_records: missing device_id: {missing_dev}")
    df = df.dropna(subset=['device_id'])
    console(f"maintenance_records after drop: {len(df)}")
    out = CLEAN_DIR / 'maintenance_records.cleaned.csv'
    write_csv_safe(df, out)
    console(f"Saved cleaned maintenance to {out}")
    return df

def clean_sensors():
    df = _safe_read('sensor_data_sample.csv')
    if df.empty:
        return df
    df.columns = [c.strip() for c in df.columns]
    if 'device_id' not in df.columns:
        df['device_id'] = pd.NA
    missing_dev = df['device_id'].isna().sum()
    console(f"sensor_data_sample: missing device_id: {missing_dev}")
    df = df.dropna(subset=['device_id'])
    # keep raw reading and a numeric column where possible
    if 'reading_value' in df.columns:
        df['reading_value_raw'] = df['reading_value']
        df['reading_value'] = pd.to_numeric(df['reading_value'], errors='coerce')
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    out = CLEAN_DIR / 'sensor_data_sample.cleaned.csv'
    write_csv_safe(df, out)
    console(f"Saved cleaned sensors to {out}")
    return df

def clean_logs():
    df = _safe_read('iot_device_logs_raw.csv')
    if df.empty:
        return df
    df.columns = [c.strip() for c in df.columns]
    if 'device_id' not in df.columns:
        df['device_id'] = pd.NA
    missing_dev = df['device_id'].isna().sum()
    console(f"iot_device_logs_raw: missing device_id: {missing_dev}")
    df = df.dropna(subset=['device_id'])
    # numeric coercion
    for col in ['temperature_c', 'battery_pct', 'humidity_pct', 'signal_dbm']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    if 'raw' not in df.columns:
        df['raw'] = ''
    out = CLEAN_DIR / 'iot_device_logs_raw.cleaned.csv'
    write_csv_safe(df, out)
    console(f"Saved cleaned logs to {out}")
    return df

def run_all_cleaners():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    console("Starting cleaning pipeline")
    df_m = clean_maintenance()
    df_s = clean_sensors()
    df_l = clean_logs()
    console("Cleaning pipeline complete")
    return df_l, df_s, df_m
