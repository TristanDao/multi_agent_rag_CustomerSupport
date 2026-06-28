"""Run evaluation: classify intent, check guardrail, compute metrics."""
import argparse
import asyncio
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agents.intent_classifier import classify_intent
from app.agents.orchestrator import run_orchestrator
from app.core.security import new_request_id
from app.guardrails.input_guardrail import run_input_guardrail

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval")


def load_queries(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def evaluate_intent(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    correct = 0
    total = 0
    per_intent: Dict[str, Dict[str, int]] = {}
    failures: List[Dict[str, Any]] = []
    for r in rows:
        expected = r["expected_intent"]
        # prompt_injection and PII_redaction are guardrail-level concerns, not intents
        if expected in ("prompt_injection", "PII_redaction"):
            continue
        total += 1
        try:
            decision = asyncio.run(classify_intent(r["query"]))
            predicted = decision.get("intent", "unknown")
        except Exception as e:
            predicted = "error"
            logger.warning("intent_eval_error query=%s err=%s", r["query"][:50], str(e))
        ok = predicted == expected
        correct += int(ok)
        per_intent.setdefault(expected, {"total": 0, "correct": 0})
        per_intent[expected]["total"] += 1
        if ok:
            per_intent[expected]["correct"] += 1
        else:
            failures.append({"query": r["query"], "expected": expected, "predicted": predicted})
    accuracy = correct / total if total else 0.0
    per_intent_acc = {
        k: (v["correct"] / v["total"] if v["total"] else 0.0) for k, v in per_intent.items()
    }
    return {
        "intent_accuracy": accuracy,
        "total": total,
        "correct": correct,
        "per_intent_accuracy": per_intent_acc,
        "failures": failures[:20],
    }


def evaluate_guardrail(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    tp = fp = tn = fn = 0
    for r in rows:
        gr, _ = run_input_guardrail(r["query"])
        blocked = bool(gr.get("blocked"))
        expected_blocked = r["expected_intent"] in ("prompt_injection",)
        if blocked and expected_blocked:
            tp += 1
        elif blocked and not expected_blocked:
            fp += 1
        elif not blocked and not expected_blocked:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fpr,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def evaluate_end_to_end(rows: List[Dict[str, Any]], limit: int = 25) -> Dict[str, Any]:
    latencies: List[int] = []
    pii_redacted = 0
    blocked = 0
    for r in rows[:limit]:
        req_id = new_request_id()
        start = time.time()
        result = asyncio.run(
            run_orchestrator(message=r["query"], request_id=req_id, customer_id=None)
        )
        latencies.append(int((time.time() - start) * 1000))
        if result.get("pii_redacted"):
            pii_redacted += 1
        if result.get("guardrail", {}).get("input") == "blocked":
            blocked += 1
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    return {
        "samples": len(latencies),
        "avg_latency_ms": round(avg_latency, 1),
        "max_latency_ms": max(latencies) if latencies else 0,
        "pii_redactions": pii_redacted,
        "blocked_responses": blocked,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="data/eval/test_queries.jsonl")
    parser.add_argument("--e2e-limit", type=int, default=15)
    parser.add_argument("--out", default="data/eval/results.json")
    args = parser.parse_args()

    rows = load_queries(args.queries)
    logger.info("loaded_queries count=%s", len(rows))

    intent_metrics = evaluate_intent(rows)
    guardrail_metrics = evaluate_guardrail(rows)
    e2e_metrics = evaluate_end_to_end(rows, limit=args.e2e_limit)

    summary = {
        "intent_routing": intent_metrics,
        "guardrail": guardrail_metrics,
        "end_to_end": e2e_metrics,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    # save failures
    if intent_metrics.get("failures"):
        with open(args.out.replace(".json", "_failures.jsonl"), "w", encoding="utf-8") as f:
            for f_rec in intent_metrics["failures"]:
                f.write(json.dumps(f_rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
