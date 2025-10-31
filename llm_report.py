from ..utils.logs import console
import os

def _call_langchain_llm(prompt: str, api_key: str = None, base_url: str = "https://genailab.tcs.in", model: str = "azure_ai/genailab-maas-DeepSeek-V3-0324"):
    """
    Call langchain_openai.ChatOpenAI.invoke with the provided api_key.
    Returns text result or None on failure.
    """
    try:
        # Import at runtime - if not available, we fallback gracefully
        from langchain_openai import ChatOpenAI
        import httpx
        client = httpx.Client(verify=False)
        llm = ChatOpenAI(
            base_url=base_url,
            model=model,
            api_key=api_key,
            http_client=client
        )
        # Using invoke as per your snippet
        resp = llm.invoke(prompt)
        return str(resp)
    except Exception as e:
        console(f"LangChain LLM call failed: {e}")
        return None

def simple_template(device, anomaly, pred):
    level = 'HIGH' if anomaly else 'NORMAL'
    pred_text = f"predicted interval days: {pred:.1f}" if (pred is not None and not pd_isna(pred)) else 'no prediction available'
    return f"Device {device} status: {level}. {pred_text}. Recommended: inspect device if level is HIGH."

def pd_isna(x):
    try:
        import pandas as pd
        return pd.isna(x)
    except Exception:
        return x is None

def generate_reports(df_features, api_key: str = None, base_url: str = None, model: str = None):
    """
    df_features: DataFrame with 'device_id', 'anomaly', 'predicted_interval_days' (optional)
    api_key: runtime key provided by user (string)
    base_url, model: optional endpoint overrides
    """
    reports = []
    console("Generating reports using LLM (if available) or templates")
    base_url = base_url or "https://genailab.tcs.in"
    model = model or "azure_ai/genailab-maas-DeepSeek-R1"
    for _, row in df_features.iterrows():
        device = row.get('device_id')
        anomaly = int(row.get('anomaly', 0)) if row.get('anomaly') is not None else 0
        pred = row.get('predicted_interval_days', None)
        # Build a richer prompt
        prompt = (
            f"You are an IoT maintenance expert.\n\n"
            f"Device: {device}\n"
            f"Anomaly flag: {anomaly}\n"
            f"Predicted interval days: {pred}\n\n"
            "Using this information plus best practices, provide:\n"
            "1) A one-line current condition summary.\n"
            "2) Top 2 likely causes (if any).\n"
            "3) Concrete next 3 recommended actions with urgency labels (Immediate / 24h / Next week).\n"
            "4) If applicable, a short optimization suggestion for enterprise context (cost/downtime tradeoff).\n"
        )
        text = None
        if api_key is not None:
            text = _call_langchain_llm(prompt, api_key=api_key, base_url=base_url, model=model)
        if text is None:
            text = simple_template(device, anomaly, pred)
        reports.append({
            'device_id': device,
            'report': text,
            'anomaly': int(anomaly),
            'predicted_interval_days': float(pred) if (pred is not None and not pd_isna(pred)) else None
        })
    console("LLM/template reports generated")
    return reports
