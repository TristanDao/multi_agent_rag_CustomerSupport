"""Metric helpers used by the evaluator."""
from typing import Dict, List


def accuracy(correct: int, total: int) -> float:
    return (correct / total) if total else 0.0


def precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def latency_stats(latencies_ms: List[int]) -> Dict[str, float]:
    if not latencies_ms:
        return {"avg": 0, "p50": 0, "p95": 0, "max": 0}
    s = sorted(latencies_ms)
    p50 = s[len(s) // 2]
    p95 = s[max(0, int(len(s) * 0.95) - 1)]
    return {
        "avg": round(sum(s) / len(s), 1),
        "p50": p50,
        "p95": p95,
        "max": max(s),
    }
