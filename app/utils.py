import time
import csv
from pathlib import Path

metrics_file = Path("metrics/metrics.csv")
metrics_file.parent.mkdir(parents=True, exist_ok=True)

def timeit():
    return time.time()

def record_metric(endpoint, latency, metadata):
    """Save simple performance metrics to CSV."""
    row = {
        "endpoint": endpoint,
        "latency_ms": round(latency, 2),
        "metadata": str(metadata),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    header = ["endpoint", "latency_ms", "metadata", "timestamp"]

    new_file = not metrics_file.exists()
    with open(metrics_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
