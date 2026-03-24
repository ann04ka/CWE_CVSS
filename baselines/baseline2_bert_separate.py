"""
Baseline 2: BERT Separate Models (DistilBERT, one model per metric).

Based on:
  Shahid & Debar (2021)  — BERT-small, 8 separate models, 84–96% accuracy
  Costa et al. (2022)    — DistilBERT-E, state-of-the-art results
  Aghaei & Al-Shaer (2023) — SecureBERT, 80–96% accuracy

Architecture:
  CVE Description
       ↓
  DistilBERT tokenization (max_length=128)
       ↓
  DistilBERT encoder (66M params, shared weights per metric)
       ↓
  Pooled [CLS] token
       ↓
  Dropout (0.3)
       ↓
  Linear classifier (hidden_size → num_classes)
       ↓
  Prediction for ONE metric
  (Repeat 8 times)

Expected results:
  Per-metric accuracy : 70–80%  (easy: AV/AC/UI/S ≈ 90–95%)
  Exact match         : 40–42%

Usage:
  python -m baselines.baseline2_bert_separate \\
      --data-dir dataset \\
      --model-dir models/baseline2 \\
      --results-out results/baseline2.json
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

from baselines.common import (
    METRICS, NUM_CLASSES, LABEL2IDX,
    load_jsonl, encode_labels, evaluate, print_results,
)
from baselines.bert_common import (
    CVSSDataset,
    compute_class_weights,
    get_device,
    train_epoch_single,
    predict_single,
    save_checkpoint,
    load_checkpoint,
)


# ── Model ─────────────────────────────────────────────────────────────────────

class BertSingleMetricModel(nn.Module):
    """
    DistilBERT encoder + linear classification head for one CVSS metric.
    """

    def __init__(
        self,
        metric: str,
        model_name: str = "distilbert-base-uncased",
        dropout: float = 0.3,
    ):
        super().__init__()
        self.metric = metric
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size  = self.encoder.config.hidden_size
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, NUM_CLASSES[metric])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # DistilBERT: last_hidden_state; BERT: pooler_output or last_hidden_state[:,0]
        pooled = out.last_hidden_state[:, 0, :]   # [CLS] token
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


# ── Training one metric ───────────────────────────────────────────────────────

def train_one_metric(
    metric: str,
    train_records: list[dict],
    val_records: list[dict],
    tokenizer,
    args,
    device: torch.device,
) -> tuple[BertSingleMetricModel, list[dict]]:
    """
    Fine-tune a DistilBERT model for a single CVSS metric.

    Returns:
        (model, history)  where history is a list of per-epoch dicts:
        [{"epoch": 1, "train_loss": ..., "val_loss": ..., "val_acc": ...}, ...]
    """
    # ── Datasets & loaders ─────────────────────────────────────────────────
    train_ds = CVSSDataset(train_records, tokenizer, args.max_length)
    val_ds   = CVSSDataset(val_records,   tokenizer, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2,
                              shuffle=False, num_workers=0, pin_memory=False)

    # ── Loss with class weights ─────────────────────────────────────────────
    train_labels = encode_labels(train_records)[metric]
    weights = compute_class_weights(train_labels, NUM_CLASSES[metric], device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # ── Model, optimizer, scheduler ────────────────────────────────────────
    model = BertSingleMetricModel(metric, args.model_name, args.dropout).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01
    )
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.06)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and args.fp16) else None

    # ── Training loop ───────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_state    = None
    patience_cnt  = 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch_single(
            model, train_loader, optimizer, scheduler,
            criterion, metric, device, scaler,
        )

        # Validation loss + per-metric accuracy
        model.eval()
        val_loss   = 0.0
        n_correct  = 0
        n_total    = 0
        with torch.no_grad():
            for batch in val_loader:
                ids  = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                lbls = batch["labels"][metric].to(device)
                logits = model(ids, mask)
                val_loss  += criterion(logits, lbls).item()
                n_correct += (logits.argmax(-1) == lbls).sum().item()
                n_total   += lbls.size(0)
        val_loss /= len(val_loader)
        val_acc   = n_correct / max(n_total, 1)

        history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss,   6),
            "val_acc":    round(val_acc,     6),
        })

        print(f"    Epoch {epoch}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"    Early stopping at epoch {epoch}")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Baseline 2: BERT Separate Models (DistilBERT)"
    )
    parser.add_argument("--data-dir",    default="dataset")
    parser.add_argument("--model-dir",   default="models/baseline2",
                        help="Directory to save/load per-metric model checkpoints")
    parser.add_argument("--results-out", default=None,
                        help="JSON file to write evaluation results")
    parser.add_argument("--model-name",  default="distilbert-base-uncased",
                        help="HuggingFace model name/path")
    # Training hyperparameters
    parser.add_argument("--epochs",     type=int,   default=4)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--max-length", type=int,   default=128)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--patience",   type=int,   default=3,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--fp16",       action="store_true",
                        help="Use mixed precision (requires CUDA)")
    parser.add_argument("--device",     default=None,
                        help="Force device: cuda / cpu")
    # Load pre-trained models (skip training)
    parser.add_argument("--load-models", action="store_true",
                        help="Load checkpoints from --model-dir instead of training")
    # Evaluate on specific metrics only (for quick testing)
    parser.add_argument("--metrics",    nargs="*", default=None,
                        help="Subset of metrics to train/evaluate (default: all 8)")
    args = parser.parse_args()

    device    = get_device(args.device)
    data_dir  = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    active_metrics = args.metrics or METRICS

    print(f"Device: {device}")
    print(f"Model:  {args.model_name}")
    print(f"Metrics to process: {active_metrics}")

    # ── Load data ─────────────────────────────────────────────────────────
    print("\nLoading data ...")
    train_records = load_jsonl(data_dir / "train.jsonl")
    val_records   = load_jsonl(data_dir / "val.jsonl")
    test_records  = load_jsonl(data_dir / "test.jsonl")
    print(f"  train={len(train_records):,}  val={len(val_records):,}  test={len(test_records):,}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ── Train or load models ──────────────────────────────────────────────
    models: dict[str, BertSingleMetricModel] = {}
    all_histories: dict[str, list[dict]] = {}

    for m in active_metrics:
        ckpt_path  = model_dir / f"metric_{m}.pt"
        hist_path  = model_dir / f"history_{m}.json"

        if args.load_models and ckpt_path.exists():
            print(f"\n[{m}] Loading checkpoint from {ckpt_path}")
            model = BertSingleMetricModel(m, args.model_name, args.dropout).to(device)
            load_checkpoint(model, ckpt_path, device)
            # Load saved history if present
            if hist_path.exists():
                with open(hist_path) as f:
                    all_histories[m] = json.load(f)
        else:
            print(f"\n[{m}] Training ({args.epochs} epochs, bs={args.batch_size}, lr={args.lr})")
            model, history = train_one_metric(
                m, train_records, val_records, tokenizer, args, device
            )
            save_checkpoint(model, ckpt_path)
            model_dir.mkdir(parents=True, exist_ok=True)
            with open(hist_path, "w") as f:
                json.dump(history, f, indent=2)
            all_histories[m] = history
            print(f"  Checkpoint saved → {ckpt_path}")
            print(f"  History   saved  → {hist_path}")

        models[m] = model

    # ── Build full prediction dicts ───────────────────────────────────────
    # For metrics that were not trained/loaded, fall back to majority class
    all_true_labels = encode_labels(val_records)

    def predict_all(records: list[dict]) -> dict[str, list[int]]:
        ds     = CVSSDataset(records, tokenizer, args.max_length)
        loader = DataLoader(ds, batch_size=args.batch_size * 2,
                            shuffle=False, num_workers=0)
        preds: dict[str, list[int]] = {}
        for m in METRICS:
            if m in models:
                preds[m] = predict_single(models[m], loader, m, device)
            else:
                # Fallback: predict majority class from training set
                from collections import Counter
                train_labels = encode_labels(train_records)[m]
                majority = Counter(train_labels).most_common(1)[0][0]
                preds[m] = [majority] * len(records)
        return preds

    # ── Evaluate ──────────────────────────────────────────────────────────
    print("\nEvaluating on validation set ...")
    val_true = encode_labels(val_records)
    val_pred = predict_all(val_records)
    val_results = evaluate(val_true, val_pred, val_records)
    print_results("Baseline 2 — BERT Separate  [Validation]", val_results)

    print("\nEvaluating on test set ...")
    test_true = encode_labels(test_records)
    test_pred = predict_all(test_records)
    test_results = evaluate(test_true, test_pred, test_records)
    print_results("Baseline 2 — BERT Separate  [Test]", test_results)

    # ── Save results ──────────────────────────────────────────────────────
    if args.results_out:
        Path(args.results_out).parent.mkdir(parents=True, exist_ok=True)
        out = {
            "baseline": "bert_separate",
            "description": (
                f"DistilBERT ({args.model_name}), 8 independent classifiers; "
                f"epochs={args.epochs}, bs={args.batch_size}, lr={args.lr}"
            ),
            "val":  val_results,
            "test": test_results,
        }
        with open(args.results_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved → {args.results_out}")


if __name__ == "__main__":
    main()
