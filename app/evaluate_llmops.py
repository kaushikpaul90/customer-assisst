"""
app/evaluate_llmops.py

Small helper script to run evaluation on a JSONL file for QA or summarization.

Usage (from project root):
    python -m app.evaluate_llmops path/to/eval.jsonl --task qa
"""
import argparse
import json
from pathlib import Path
from typing import List
from app.llmops import evaluate_qa, evaluate_summaries

def load_jsonl(path: Path):
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            out.append(json.loads(line))
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="Path to JSONL evaluation file")
    parser.add_argument("--task", type=str, choices=["qa","summarization"], default="qa")
    args = parser.parse_args()

    data = load_jsonl(args.path)
    if args.task == "qa":
        preds = [{"prediction": d.get("prediction","")} for d in data]
        refs = [{"gold": d.get("gold","")} for d in data]
        res = evaluate_qa(preds, refs)
        print("QA Eval:", res)
    else:
        preds = [d.get("prediction","") for d in data]
        refs = [d.get("reference","") for d in data]
        res = evaluate_summaries(preds, refs)
        print("Summarization Eval:", res)

if __name__ == "__main__":
    main()
