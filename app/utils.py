# app/utils.py
import time
import csv
from pathlib import Path
import json
import datetime
from typing import Dict, Optional, Any

# Metrics storage
METRICS_DIR = Path("metrics")
METRICS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_FILE = METRICS_DIR / "metrics.csv"
_METRICS_HEADER = ["timestamp", "endpoint", "latency_ms", "model_version", "metadata"]

def timeit() -> float:
    """
    Simple wrapper returning current epoch seconds (float).
    Use like:
        start = timeit()
        ...
        latency_ms = (timeit() - start) * 1000
    """
    return time.time()

def _ensure_header():
    """Ensure metrics CSV has a header row (best-effort)."""
    if not METRICS_FILE.exists():
        try:
            with METRICS_FILE.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(_METRICS_HEADER)
        except Exception:
            pass

def record_metric(
    endpoint: str,
    latency_ms: float,
    metadata: Optional[Dict[str, Any]] = None,
    model_version: Optional[str] = None,
):
    """
    Append a metric row to metrics/metrics.csv.

    - endpoint: API route name (e.g. "/summarize-text")
    - latency_ms: float latency in milliseconds
    - metadata: optional simple dict (e.g. {"success": True, "summary_len": 123})
    - model_version: optional string for model identifier

    This function is best-effort and will not raise exceptions.
    """
    _ensure_header()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md = metadata or {}
    if model_version:
        md["model_version"] = model_version
    try:
        with METRICS_FILE.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ts, endpoint, float(latency_ms), model_version or "", json.dumps(md, ensure_ascii=False)])
    except Exception:
        # never fail an endpoint because metric logging failed
        pass
