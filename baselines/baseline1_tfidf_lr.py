"""
Baseline 1: TF-IDF + Logistic Regression (one model per metric).

Based on:
  Elbaz et al. (2020) — BoW + Linear Regression
  Khazaei et al. (2016) — TF-IDF + SVM/RF

Architecture:
  CVE Description
       ↓
  TF-IDF Vectorization (5 000 features, bigrams)
       ↓
  8 separate Logistic Regression classifiers
       ↓
  Prediction for each CVSS metric

Expected results:
  Per-metric accuracy : 60–70%
  Exact match         : 30–35%

Usage:
  python -m baselines.baseline1_tfidf_lr \\
      --data-dir dataset \\
      --results-out results/baseline1.json
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from baselines.common import (
    METRICS,
    LABEL2IDX,
    load_jsonl,
    encode_labels,
    evaluate,
    print_results,
)


# ── Model definition ──────────────────────────────────────────────────────────

def build_pipeline(metric: str) -> Pipeline:
    """TF-IDF + Logistic Regression pipeline for one CVSS metric."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5_000,
            ngram_range=(1, 2),
            sublinear_tf=True,          # log(1+tf) instead of raw tf
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
        )),
        ("clf", LogisticRegression(
            class_weight="balanced",    # combat label imbalance
            max_iter=1_000,
            solver="lbfgs",
            random_state=42,
            C=1.0,
        )),
    ])


# ── Training ──────────────────────────────────────────────────────────────────

def train(train_records: list[dict]) -> dict[str, Pipeline]:
    """
    Train 8 independent TF-IDF + LogReg pipelines, one per CVSS metric.

    Returns:
        Dict mapping metric name → fitted Pipeline.
    """
    texts  = [r.get("description", "") or "" for r in train_records]
    labels = encode_labels(train_records)

    models: dict[str, Pipeline] = {}
    for m in METRICS:
        print(f"  [{m}] training ...", end=" ", flush=True)
        pipe = build_pipeline(m)
        pipe.fit(texts, labels[m])
        models[m] = pipe
        # Show dominant class baseline for reference
        from collections import Counter
        cnt = Counter(labels[m])
        majority_acc = max(cnt.values()) / len(labels[m])
        print(f"done  (majority baseline: {majority_acc:.1%})")

    return models


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(models: dict[str, Pipeline], records: list[dict]) -> dict[str, list[int]]:
    texts = [r.get("description", "") or "" for r in records]
    return {m: list(models[m].predict(texts)) for m in METRICS}


# ── Feature importance ────────────────────────────────────────────────────────

def top_features(models: dict[str, Pipeline], top_k: int = 10):
    """Print the most informative n-grams per metric."""
    print("\n[Top features per metric]")
    for m, pipe in models.items():
        tfidf: TfidfVectorizer = pipe.named_steps["tfidf"]
        clf: LogisticRegression  = pipe.named_steps["clf"]
        feature_names = tfidf.get_feature_names_out()

        if clf.coef_.shape[0] == 1:
            # Binary classification: single coefficient array
            coefs = clf.coef_[0]
            top_idx = np.argsort(np.abs(coefs))[-top_k:][::-1]
            tokens = [feature_names[i] for i in top_idx]
        else:
            # Multiclass: average absolute coefficient across classes
            coefs = np.abs(clf.coef_).mean(axis=0)
            top_idx = np.argsort(coefs)[-top_k:][::-1]
            tokens = [feature_names[i] for i in top_idx]

        print(f"  {m}: {', '.join(tokens)}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Baseline 1: TF-IDF + Logistic Regression"
    )
    parser.add_argument("--data-dir",    default="dataset",
                        help="Directory containing train/val/test.jsonl")
    parser.add_argument("--save-model",  default=None,
                        help="Path to save trained pipelines (.pkl)")
    parser.add_argument("--load-model",  default=None,
                        help="Path to load pre-trained pipelines (.pkl)")
    parser.add_argument("--results-out", default=None,
                        help="JSON file to write evaluation results")
    parser.add_argument("--show-features", action="store_true",
                        help="Print top TF-IDF features per metric")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # ── Load data ────────────────────────────────────────────────────────────
    print("Loading data ...")
    train_records = load_jsonl(data_dir / "train.jsonl")
    val_records   = load_jsonl(data_dir / "val.jsonl")
    test_records  = load_jsonl(data_dir / "test.jsonl")
    print(f"  train={len(train_records):,}  val={len(val_records):,}  test={len(test_records):,}")

    # ── Train or load model ──────────────────────────────────────────────────
    if args.load_model:
        print(f"Loading model from {args.load_model} ...")
        with open(args.load_model, "rb") as f:
            models = pickle.load(f)
    else:
        print("\nTraining TF-IDF + LogReg (8 models) ...")
        models = train(train_records)

    if args.save_model:
        Path(args.save_model).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_model, "wb") as f:
            pickle.dump(models, f)
        print(f"Model saved → {args.save_model}")

    if args.show_features:
        top_features(models)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print("\nEvaluating on validation set ...")
    val_true = encode_labels(val_records)
    val_pred = predict(models, val_records)
    val_results = evaluate(val_true, val_pred, val_records)
    print_results("Baseline 1 — TF-IDF + LogReg  [Validation]", val_results)

    print("\nEvaluating on test set ...")
    test_true = encode_labels(test_records)
    test_pred = predict(models, test_records)
    test_results = evaluate(test_true, test_pred, test_records)
    print_results("Baseline 1 — TF-IDF + LogReg  [Test]", test_results)

    # ── Save results ─────────────────────────────────────────────────────────
    if args.results_out:
        Path(args.results_out).parent.mkdir(parents=True, exist_ok=True)
        out = {
            "baseline": "tfidf_lr",
            "description": "TF-IDF (5k features, bigrams) + Logistic Regression (balanced)",
            "val":  val_results,
            "test": test_results,
        }
        with open(args.results_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved → {args.results_out}")


if __name__ == "__main__":
    main()
