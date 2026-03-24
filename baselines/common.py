"""
Shared utilities for CVSS v3.1 prediction baselines.

Provides:
  - METRICS, METRIC_CLASSES, LABEL2IDX, IDX2LABEL, NUM_CLASSES
  - load_jsonl, encode_labels
  - cvss31_score, score_to_severity
  - evaluate, print_results
"""

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

# ── Metric definitions ────────────────────────────────────────────────────────

METRICS = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]

METRIC_CLASSES = {
    "AV": ["N", "A", "L", "P"],   # Network, Adjacent, Local, Physical
    "AC": ["L", "H"],              # Low, High
    "PR": ["N", "L", "H"],        # None, Low, High
    "UI": ["N", "R"],              # None, Required
    "S":  ["U", "C"],             # Unchanged, Changed
    "C":  ["N", "L", "H"],        # None, Low, High
    "I":  ["N", "L", "H"],        # None, Low, High
    "A":  ["N", "L", "H"],        # None, Low, High
}

METRIC_FULL_NAMES = {
    "AV": "Attack Vector",
    "AC": "Attack Complexity",
    "PR": "Privileges Required",
    "UI": "User Interaction",
    "S":  "Scope",
    "C":  "Confidentiality",
    "I":  "Integrity",
    "A":  "Availability",
}

LABEL2IDX = {m: {v: i for i, v in enumerate(vals)} for m, vals in METRIC_CLASSES.items()}
IDX2LABEL = {m: {i: v for i, v in enumerate(vals)} for m, vals in METRIC_CLASSES.items()}
NUM_CLASSES = {m: len(v) for m, v in METRIC_CLASSES.items()}

# Task weights for multi-task loss (reflects prediction difficulty)
TASK_WEIGHTS = {
    "AV": 1.0,
    "AC": 1.0,
    "PR": 1.5,   # Hardest metric
    "UI": 1.0,
    "S":  1.0,
    "C":  1.2,   # Impact metrics are medium difficulty
    "I":  1.2,
    "A":  1.2,
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def encode_labels(records: list[dict]) -> dict[str, list[int]]:
    """Return {metric: [label_idx, ...]} for every record."""
    encoded = {m: [] for m in METRICS}
    for rec in records:
        mvals = rec.get("metrics", {})
        for m in METRICS:
            val = mvals.get(m, METRIC_CLASSES[m][0])
            encoded[m].append(LABEL2IDX[m].get(val, 0))
    return encoded


# ── CVSS v3.1 Base Score ──────────────────────────────────────────────────────

_AV_W   = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.20}
_AC_W   = {"L": 0.77, "H": 0.44}
_UI_W   = {"N": 0.85, "R": 0.62}
_CIA_W  = {"N": 0.00, "L": 0.22, "H": 0.56}
_PR_W_U = {"N": 0.85, "L": 0.62, "H": 0.27}   # Scope=Unchanged
_PR_W_C = {"N": 0.85, "L": 0.68, "H": 0.50}   # Scope=Changed


def _roundup(x: float) -> float:
    """CVSS Roundup: round up to 1 decimal place."""
    return math.ceil(x * 10) / 10


def cvss31_score(metrics: dict) -> float:
    """Compute CVSS v3.1 Base Score from a metric dict (letter codes)."""
    av    = _AV_W.get(metrics.get("AV", "N"), 0.85)
    ac    = _AC_W.get(metrics.get("AC", "L"), 0.77)
    scope = metrics.get("S", "U")
    pr    = (_PR_W_C if scope == "C" else _PR_W_U).get(metrics.get("PR", "N"), 0.85)
    ui    = _UI_W.get(metrics.get("UI", "N"), 0.85)
    c     = _CIA_W.get(metrics.get("C", "N"), 0.00)
    i     = _CIA_W.get(metrics.get("I", "N"), 0.00)
    a     = _CIA_W.get(metrics.get("A", "N"), 0.00)

    isc_base = 1.0 - (1.0 - c) * (1.0 - i) * (1.0 - a)

    if scope == "U":
        isc = 6.42 * isc_base
    else:
        isc = 7.52 * (isc_base - 0.029) - 3.25 * ((isc_base - 0.02) ** 15)

    exploit = 8.22 * av * ac * pr * ui

    if isc <= 0:
        return 0.0
    elif scope == "U":
        return _roundup(min(isc + exploit, 10.0))
    else:
        return _roundup(min(1.08 * (isc + exploit), 10.0))


def score_to_severity(score: float) -> str:
    if score == 0.0:
        return "NONE"
    elif score < 4.0:
        return "LOW"
    elif score < 7.0:
        return "MEDIUM"
    elif score < 9.0:
        return "HIGH"
    else:
        return "CRITICAL"


def labels_to_vector_string(label_dict: dict[str, str]) -> str:
    """Convert {metric: letter} dict to CVSS:3.1/AV:N/... string."""
    parts = "/".join(f"{m}:{label_dict[m]}" for m in METRICS)
    return f"CVSS:3.1/{parts}"


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    true_metrics: dict[str, list[int]],
    pred_metrics: dict[str, list[int]],
    records: list[dict],
) -> dict:
    """
    Compute all evaluation metrics.

    Args:
        true_metrics: {metric: [true_label_idx, ...]}
        pred_metrics: {metric: [pred_label_idx, ...]}
        records:      original dataset records (used for ground-truth base_score)

    Returns:
        Dict with per_metric_accuracy, mean_accuracy, exact_match,
        hamming_distance, score_mae, severity_accuracy, critical_fnr.
    """
    n = len(records)
    results: dict = {}

    # ── Per-metric accuracy ──────────────────────────────────────────────────
    per_acc = {}
    for m in METRICS:
        t = np.array(true_metrics[m])
        p = np.array(pred_metrics[m])
        per_acc[m] = float(np.mean(t == p))
    results["per_metric_accuracy"] = per_acc
    results["mean_accuracy"] = float(np.mean(list(per_acc.values())))

    # ── Exact match ──────────────────────────────────────────────────────────
    exact = np.ones(n, dtype=bool)
    for m in METRICS:
        exact &= np.array(true_metrics[m]) == np.array(pred_metrics[m])
    results["exact_match"] = float(np.mean(exact))

    # ── Hamming distance ─────────────────────────────────────────────────────
    wrong = np.zeros(n, dtype=float)
    for m in METRICS:
        wrong += (np.array(true_metrics[m]) != np.array(pred_metrics[m])).astype(float)
    results["hamming_distance"] = float(np.mean(wrong))

    # ── CVSS Score MAE ───────────────────────────────────────────────────────
    true_scores = np.array([
        float(records[i].get("base_score") or cvss31_score(
            {m: IDX2LABEL[m][true_metrics[m][i]] for m in METRICS}
        ))
        for i in range(n)
    ])
    pred_scores = np.array([
        cvss31_score({m: IDX2LABEL[m][pred_metrics[m][i]] for m in METRICS})
        for i in range(n)
    ])
    results["score_mae"] = float(np.mean(np.abs(true_scores - pred_scores)))

    # ── Severity accuracy ────────────────────────────────────────────────────
    true_sev = [score_to_severity(s) for s in true_scores]
    pred_sev = [score_to_severity(s) for s in pred_scores]
    results["severity_accuracy"] = float(np.mean([t == p for t, p in zip(true_sev, pred_sev)]))

    # ── Critical False Negative Rate (score ≥ 7.0 predicted as < 7.0) ───────
    crit_mask = true_scores >= 7.0
    if crit_mask.sum() > 0:
        fn = ((true_scores >= 7.0) & (pred_scores < 7.0)).sum()
        results["critical_fnr"] = float(fn / crit_mask.sum())
    else:
        results["critical_fnr"] = 0.0

    return results


def print_results(name: str, results: dict):
    print(f"\n{'='*62}")
    print(f"  {name}")
    print(f"{'='*62}")
    print(f"  Mean Accuracy    : {results['mean_accuracy']*100:6.2f}%")
    print(f"  Exact Match      : {results['exact_match']*100:6.2f}%")
    print(f"  Hamming Distance : {results['hamming_distance']:.4f}")
    print(f"  Score MAE        : {results['score_mae']:.4f}")
    print(f"  Severity Accuracy: {results['severity_accuracy']*100:6.2f}%")
    print(f"  Critical FNR     : {results['critical_fnr']*100:6.2f}%")
    print(f"\n  Per-Metric Accuracy:")
    for m, acc in results["per_metric_accuracy"].items():
        bar = "#" * int(acc * 30)
        name_str = METRIC_FULL_NAMES.get(m, m)
        print(f"    {m} ({name_str:<22}): {acc*100:5.1f}%  {bar}")
