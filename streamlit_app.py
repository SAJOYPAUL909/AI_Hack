# streamlit_app.py
import streamlit as st
from pathlib import Path
import json
import pandas as pd

# make backend importable
import sys
sys.path.append(str(Path.cwd()))

from backend.pipeline.cleaner import run_all_cleaners
from backend.pipeline.anomaly import run_anomaly_detection
from backend.pipeline.maintenance_model import train_maintenance_model, predict_intervals
from backend.pipeline.llm_report import generate_reports
from backend.utils.logs import console

st.set_page_config(page_title="IoT Maintenance Assistant", layout="wide")
st.title("IoT Device Status & Maintenance Assistant")

st.markdown("""
**Instructions**
1. Put your CSVs in `backend/data/raw/` OR upload them here.
2. Provide your LangChain/OpenAI API key below (optional). If not provided, template text will be used.
3. Click **Run pipeline** to clean data, run ML, and produce reports.
""")

with st.expander("Upload CSVs (optional)"):
    up_logs = st.file_uploader("iot_device_logs_raw.csv", type=["csv","txt"], key="logs")
    up_sensors = st.file_uploader("sensor_data_sample.csv", type=["csv","txt"], key="sensors")
    up_maint = st.file_uploader("maintenance_records.csv", type=["csv","txt"], key="maint")

st.write("")
api_key = st.text_input("LLM API key (paste here, not stored)", type="password")
base_url = st.text_input("LLM base URL (optional, default genailab.tcs.in)", value="https://genailab.tcs.in")
model_name = st.text_input("LLM model name (optional)", value="azure_ai/genailab-maas-DeepSeek-V3-0324")

def save_uploaded_files():
    RAW = Path('backend/data/raw')
    RAW.mkdir(parents=True, exist_ok=True)
    saved = []
    mapping = {
        'iot_device_logs_raw.csv': up_logs,
        'sensor_data_sample.csv': up_sensors,
        'maintenance_records.csv': up_maint
    }
    for fname, fileobj in mapping.items():
        if fileobj is not None:
            dest = RAW / fname
            with open(dest, 'wb') as f:
                f.write(fileobj.getbuffer())
            saved.append(str(dest))
            st.info(f"Saved {fname} to {dest}")
    return saved

if st.button("Run pipeline"):
    st.info("Starting pipeline...")
    saved = save_uploaded_files()
    # Step 1: Cleaning
    st.info("1) Cleaning data (check logs area for details)")
    df_logs, df_sensors, df_maint = run_all_cleaners()
    st.success("Data cleaning finished")
    st.write("Cleaned samples:")
    if not df_logs.empty:
        st.write("Logs sample")
        st.dataframe(df_logs.head())
    if not df_sensors.empty:
        st.write("Sensors sample")
        st.dataframe(df_sensors.head())
    if not df_maint.empty:
        st.write("Maintenance sample")
        st.dataframe(df_maint.head())

    # Step 2: Anomaly detection (device-level)
    st.info("2) Building features & running anomaly detection")
    df_features = run_anomaly_detection()
    st.success("Anomaly detection done")
    st.write("Device-level features (head):")
    st.dataframe(df_features.head())

    # Step 3: Train maintenance model (if possible) and predict
    st.info("3) Training maintenance model (if possible)")
    train_maintenance_model()
    st.info("Predicting maintenance intervals")
    df_features = predict_intervals(df_features)
    st.success("Maintenance predictions done")

    # Step 4: LLM reports
    st.info("4) Generating natural-language reports")
    reports = generate_reports(df_features, api_key=api_key if api_key.strip() else None, base_url=base_url, model=model_name)
    st.success("Reports generated")

    # Display condensed dashboard
    st.write("---")
    st.subheader("Device Summary")
    for r in reports:
        color = "🔴" if r['anomaly'] else "🟢"
        st.markdown(f"**{color} {r['device_id']}** — Pred. interval: {r.get('predicted_interval_days')}")
        st.write(r['report'])
        st.write("----")

    st.subheader("Anomalies table")
    if 'anomaly' in df_features.columns:
        st.dataframe(df_features[['device_id','anomaly']].sort_values('anomaly', ascending=False))

    st.subheader("Download Reports")
    st.download_button("Download JSON reports", data=json.dumps(reports, default=str, indent=2).encode('utf-8'), file_name="device_reports.json", help="Download generated LLM/template reports")
    st.success("Pipeline complete. Check logs in your terminal for detailed console prints.")
