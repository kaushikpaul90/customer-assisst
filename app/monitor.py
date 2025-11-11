"""LLMOps Monitor: Logging LLM interactions for observability and quality analysis.

This module captures the inputs, outputs, and metadata for key LLM calls
to enable downstream monitoring and analysis (e.g., detecting prompt drift,
analyzing failure modes, or logging cost/latency).
"""

import csv
import json
import time
from pathlib import Path

# Setup logging file
monitor_file = Path("metrics/llm_monitor.csv")
monitor_file.parent.mkdir(parents=True, exist_ok=True)
HEADER = ["timestamp", "model_task", "latency_ms", "prompt", "output", "metadata"]


def _get_active_model_version(task_name: str) -> str:
    """A placeholder to simulate looking up the active model version (LLMOps)."""
    # In a real system, this would query a proper Model Registry service.
    # Here, we use a simple mapping based on the task name.
    if task_name == "/question-answering":
        return "qa_model/v1"
    elif task_name == "/summarize-text":
        return "summarizer/v1"
    elif task_name == "/translate-text":
        return "translator/v1"
    # Add other tasks here as needed
    return task_name


def log_llm_interaction(
    endpoint: str,
    latency: float,
    prompt: str,
    output: str,
    metadata: dict = None,
):
    """
    Logs a single LLM interaction row to a CSV file for monitoring.

    Args:
        endpoint: The API endpoint that triggered the LLM (e.g., '/question-answering').
        latency: Latency in milliseconds.
        prompt: The full text prompt sent to the LLM.
        output: The full text response from the LLM.
        metadata: Optional dict for extra info (e.g., user ID, token count).
    """
    metadata = metadata or {}
    
    # Add model version to metadata for traceability
    metadata["model_version"] = _get_active_model_version(endpoint)

    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_task": endpoint,
        "latency_ms": round(latency, 2),
        "prompt": prompt.replace('\n', ' ').strip(), # Single-line cleanup for CSV
        "output": output.replace('\n', ' ').strip(),
        "metadata": json.dumps(metadata)
    }

    new_file = not monitor_file.exists()
    with open(monitor_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        if new_file:
            writer.writeheader()
        writer.writerow(row)