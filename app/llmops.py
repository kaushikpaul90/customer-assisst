"""
app/llmops.py

Lightweight LLMOps utilities for an example project.

- Aggregates operational metrics from metrics/metrics.csv
- Basic QA (Exact Match, token F1) & summarization (ROUGE-1 recall)
- Simple data-drift detector (token-frequency + KL divergence)
- FastAPI router exposing /llmops/summary and /llmops/metrics_csv
- instrument_endpoint decorator to automatically record latency + success
- dump_summary_json helper to write a JSON summary file
"""
from fastapi import APIRouter
from pathlib import Path
import csv
import time
import math
import statistics
import json
from collections import Counter, defaultdict
from typing import List, Dict, Optional
import datetime

# metrics CSV file used by the demo (relative to project root)
METRICS_FILE = Path("metrics/metrics.csv")

# -----------------------
# CSV parsing / helpers
# -----------------------
def read_metrics_csv(path: Path = METRICS_FILE) -> List[Dict]:
    """Read the metrics CSV into a list of dicts. Returns empty list if file not found."""
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Normalize types
            try:
                r["latency_ms"] = float(r.get("latency_ms", 0.0))
            except Exception:
                r["latency_ms"] = 0.0
            rows.append(r)
    return rows

def _to_float_safe(x):
    try:
        return float(x)
    except:
        return 0.0

# -----------------------
# Aggregation metrics
# -----------------------
def aggregate_metrics(rows: List[Dict]) -> Dict:
    """Compute a set of operational metrics from raw rows.

    Returns:
        dict with keys: total_requests, avg_latency_ms, p50_latency_ms, p95_latency_ms,
        throughput_rps (approx), error_rate, avg_response_len, per_endpoint stats.
    """
    if not rows:
        return {}
    latencies = [r["latency_ms"] for r in rows]
    total = len(latencies)
    avg_latency = statistics.mean(latencies) if latencies else 0.0
    p50 = statistics.median(latencies) if latencies else 0.0
    p95 = _percentile(latencies, 95)
    # approximate throughput: requests / wall_time_seconds
    times = [ _parse_ts(r.get("timestamp")) for r in rows ]
    min_t = min(times)
    max_t = max(times)
    wall_sec = max(1.0, max_t - min_t)
    throughput_rps = total / wall_sec

    # error rate: look into metadata strings for '"success": False' or 'success=False'
    errors = 0
    succ_counts = 0
    resp_lens = []
    endpoints = defaultdict(list)
    for r in rows:
        endpoints[r.get("endpoint","<unknown>")].append(r)
        m = r.get("metadata","")
        if '"success"' in m or "success" in m:
            # crude parse
            if "False" in m or "false" in m or "0" in m:
                errors += 1
            succ_counts += 1
        # try to parse out response length hints
        try:
            if "summary_len" in m or "answer_len" in m or "out_len" in m:
                # attempt JSON parse
                md = json.loads(m.replace("'",'"'))
                for k in ("summary_len","answer_len","out_len"):
                    if k in md:
                        resp_lens.append(int(md[k]))
        except Exception:
            pass

    error_rate = (errors / succ_counts) if succ_counts>0 else None
    avg_resp_len = (statistics.mean(resp_lens) if resp_lens else None)

    per_endpoint = {}
    for ep, items in endpoints.items():
        lats = [i.get("latency_ms",0.0) for i in items]
        per_endpoint[ep] = {
            "count": len(items),
            "avg_latency_ms": statistics.mean(lats) if lats else 0.0,
            "p95_latency_ms": _percentile(lats,95)
        }

    return {
        "total_requests": total,
        "avg_latency_ms": avg_latency,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "throughput_rps": throughput_rps,
        "error_rate": error_rate,
        "avg_response_len": avg_resp_len,
        "per_endpoint": per_endpoint,
        "wall_time_seconds": wall_sec
    }

def _percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    data = sorted(data)
    k = (len(data)-1) * (p/100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f==c:
        return data[int(k)]
    d0 = data[int(f)] * (c-k)
    d1 = data[int(c)] * (k-f)
    return d0 + d1

def _parse_ts(ts_str: Optional[str]) -> float:
    """Parse timestamp string in format 'YYYY-%m-%d %H:%M:%S' into epoch seconds.
    On parse failure return current time."""
    if not ts_str:
        return time.time()
    try:
        dt = datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        return dt.timestamp()
    except Exception:
        try:
            # try ISO format
            dt = datetime.datetime.fromisoformat(ts_str)
            return dt.timestamp()
        except Exception:
            return time.time()

# -----------------------
# Simple evaluation helpers
# -----------------------
def normalize_answer(s: str) -> str:
    """Lower, strip, remove punctuation for simple token-level metrics."""
    import re, string
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return s

def f1_score_from_tokens(pred: str, gold: str) -> float:
    """Compute token-level F1 between predicted and gold string."""
    p_tokens = normalize_answer(pred).split()
    g_tokens = normalize_answer(gold).split()
    if not p_tokens and not g_tokens:
        return 1.0
    if not p_tokens or not g_tokens:
        return 0.0
    common = Counter(p_tokens) & Counter(g_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_tokens)
    recall = num_same / len(g_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0

def evaluate_qa(predictions: List[Dict], references: List[Dict]) -> Dict:
    """Compute EM and F1 over lists of QA dicts.
    Each element is {'prediction': '...'} and {'gold': '...'} respectively.
    """
    ems = []
    f1s = []
    for p, g in zip(predictions, references):
        pred = p.get("prediction", "")
        gold = g.get("gold", "")
        ems.append(exact_match(pred, gold))
        f1s.append(f1_score_from_tokens(pred, gold))
    return {
        "exact_match": sum(ems)/len(ems) if ems else None,
        "f1": sum(f1s)/len(f1s) if f1s else None,
        "n": len(ems)
    }

def rouge1_recall(pred: str, gold: str) -> float:
    """Very small ROUGE-1 recall: overlap of unigrams / gold unigrams."""
    p_tokens = normalize_answer(pred).split()
    g_tokens = normalize_answer(gold).split()
    if not g_tokens:
        return 1.0 if not p_tokens else 0.0
    common = Counter(p_tokens) & Counter(g_tokens)
    return sum(common.values()) / len(g_tokens)

def evaluate_summaries(predictions: List[str], references: List[str]) -> Dict:
    recalls = [rouge1_recall(p,r) for p,r in zip(predictions,references)]
    avg_recall = sum(recalls)/len(recalls) if recalls else None
    return {"rouge1_recall_avg": avg_recall, "n": len(recalls)}

# -----------------------
# Data drift (simple)
# -----------------------
def token_freq(texts: List[str]) -> Counter:
    cnt = Counter()
    for t in texts:
        toks = normalize_answer(t).split()
        cnt.update(toks)
    return cnt

def kl_divergence(p: Counter, q: Counter, eps=1e-8) -> float:
    """KL divergence D(P||Q) where P and Q are Counters (raw counts)."""
    all_keys = set(p.keys()) | set(q.keys())
    p_total = sum(p.values()) or 1
    q_total = sum(q.values()) or 1
    kl = 0.0
    for k in all_keys:
        p_prob = p.get(k,0)/p_total + eps
        q_prob = q.get(k,0)/q_total + eps
        kl += p_prob * math.log(p_prob / q_prob)
    return kl

def detect_drift(current_texts: List[str], baseline_texts: List[str]) -> Dict:
    """Return simple drift score (KL divergence) and top-delta tokens."""
    p = token_freq(current_texts)
    q = token_freq(baseline_texts)
    kl = kl_divergence(p,q)
    # compute top tokens that increased the most (by normalized frequency)
    p_total = sum(p.values()) or 1
    q_total = sum(q.values()) or 1
    deltas = []
    for tok in set(p.keys()) | set(q.keys()):
        p_norm = p.get(tok,0)/p_total
        q_norm = q.get(tok,0)/q_total
        deltas.append((tok, p_norm - q_norm))
    deltas.sort(key=lambda x: x[1], reverse=True)
    top_increases = deltas[:10]
    return {"kl_divergence": kl, "top_increases": top_increases}

# -----------------------
# FastAPI router (optional)
# -----------------------
def create_router():
    router = APIRouter()

    @router.get("/llmops/summary")
    async def llmops_summary():
        rows = read_metrics_csv()
        agg = aggregate_metrics(rows)
        return {"ok": True, "summary": agg}

    @router.get("/llmops/metrics_csv")
    async def llmops_metrics_csv():
        # return raw CSV as text for debugging / ingestion
        if not METRICS_FILE.exists():
            return {"ok": True, "csv": ""}
        return {"ok": True, "csv": METRICS_FILE.read_text(encoding='utf-8')}

    return router

# -----------------------
# Endpoint instrumentation decorator (usage: @instrument_endpoint("name"))
# -----------------------
def instrument_endpoint(endpoint_name: str):
    """Return a decorator that wraps an async FastAPI endpoint to measure latency and success.

    Example:
        @app.post("/my-endpoint")
        @instrument_endpoint("/my-endpoint")
        async def my_endpoint(req: ReqType):
            ...
    This decorator requires that `app.utils.record_metric` is importable and
    available to call. The decorator will attempt to import it lazily so
    it doesn't force imports at module load time.
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            import time
            start = time.time()
            success = True
            try:
                out = await func(*args, **kwargs)
                return out
            except Exception:
                success = False
                raise
            finally:
                latency = (time.time() - start) * 1000
                # try to call app.utils.record_metric
                try:
                    # import lazily in case of circular imports
                    from app.utils import record_metric
                    meta = {"success": success}
                    # record minimal metric: timestamp, endpoint, latency_ms, metadata
                    record_metric(endpoint_name, latency, meta)
                except Exception:
                    # best-effort; don't fail if utils isn't available
                    pass
        return wrapper
    return decorator

# -----------------------
# CLI-style helper to produce a JSON summary file
# -----------------------
def dump_summary_json(out_path: str = "metrics/llmops_summary.json"):
    rows = read_metrics_csv()
    agg = aggregate_metrics(rows)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(agg, indent=2))
    return p.resolve()
