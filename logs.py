import sys
import datetime

def console(msg: str):
    t = datetime.datetime.now().isoformat()
    print(f"[{t}] {msg}")
    sys.stdout.flush()