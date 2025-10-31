import pandas as pd
from pathlib import Path
import io

def read_csv_safe(path, **kwargs):
    """
    Read CSV while auto-detecting delimiter using pandas (sep=None, engine='python').
    Falls back to simple read_csv if that fails.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found")
    try:
        # Let pandas sniff delimiter
        return pd.read_csv(p, sep=None, engine='python', **kwargs)
    except Exception:
        # Fallback: try common delimiters
        for sep in [',', '\t', ';', '|']:
            try:
                return pd.read_csv(p, sep=sep, engine='python', **kwargs)
            except Exception:
                continue
        # As last resort, read whole file as a single column
        text = p.read_text(encoding='utf-8', errors='ignore')
        return pd.read_csv(io.StringIO(text), header=None, names=['raw_line'])

def write_csv_safe(df, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
