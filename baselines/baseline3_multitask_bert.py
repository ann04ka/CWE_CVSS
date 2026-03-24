"""
Baseline 3: Multi-task BERT — Scientific Contribution #1.

One shared DistilBERT encoder fine-tuned simultaneously for all 8 CVSS
metrics via multi-task learning. Unlike all prior work (Shahid 2021,
Costa 2022, Aghaei 2023, Sanvito 2025) that trains 8 separate models,
this approach:
  • Uses a single shared representation
  • Optimises exact match jointly (all metrics at once)
  • Is faster at inference (one forward pass)
  • Captures cross-metric dependencies through the shared encoder

Architecture:
  CVE Description
       ↓
  DistilBERT tokenization (max_length=128)
       ↓
  Shared DistilBERT encoder (66M params)
       ↓
  [CLS] pooled output
       ↓
  Dropout (0.3)
       ↓
  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
  │ AV   │ AC   │ PR   │ UI   │  S   │  C   │  I   │  A   │
  │(4 cl)│(2 cl)│(3 cl)│(2 cl)│(2 cl)│(3 cl)│(3 cl)│(3 cl)│
  └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
       ↓
  Complete CVSS vector

Total loss = Σ task_weight_i × CrossEntropy_i
  task weights: PR=1.5 (hardest), C/I/A=1.2, rest=1.0

Expected results:
  Per-metric accuracy : 68–75%  (slightly below separate)
  Exact match         : 55–60%  (significantly above separate!)

Usage:
  python -m baselines.baseline3_multitask_bert \\
      --data-dir dataset \\
      --model-dir models/baseline3 \\
      --results-out results/baseline3.json
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

from baselines.common import (
    METRICS, NUM_CLASSES, TASK_WEIGHTS,
    load_jsonl, encode_labels, evaluate, print_results,
)
from baselines.bert_common import (
    CVSSDataset,
    compute_class_weights,
    get_device,
    train_epoch_multitask,
    predict_multitask,
    save_checkpoint,
    load_checkpoint,
)


# ── Model ─────────────────────────────────────────────────────────────────────

class BertMultiTaskModel(nn.Module):
    """
    Shared DistilBERT encoder with 8 task-specific linear classification heads.

    All heads are trained simultaneously through a weighted sum of
    per-metric cross-entropy losses.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        dropout: float  = 0.3,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size  = self.encoder.config.hidden_size   # 768

        self.dropout = nn.Dropout(dropout)

        # One linear head per CVSS metric
        self.heads = nn.ModuleDict({
            m: nn.Linear(hidden_size, NUM_CLASSES[m])
            for m in METRICS
        })

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            Dict {metric: logits_tensor [batch, num_classes]}
        """
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]    # [CLS]
        pooled = self.dropout(pooled)
        return {m: self.heads[m](pooled) for m in METRICS}


# ── Training ──────────────────────────────────────────────────────────────────

def build_criteria(
    train_records: list[dict],
    device: torch.device,
) -> dict[str, nn.CrossEntropyLoss]:
    """One weighted CrossEntropyLoss per metric (handles class imbalance)."""
    all_labels = encode_labels(train_records)
    criteria = {}
    for m in METRICS:
        w = compute_class_weights(all_labels[m], NUM_CLASSES[m], device)
        criteria[m] = nn.CrossEntropyLoss(weight=w)
    return criteria


def train(
    train_records: list[dict],
    val_records: list[dict],
    tokenizer,
    args,
    device: torch.device,
) -> BertMultiTaskModel:
    """
    Fine-tune the multi-task model. Returns best checkpoint (min val loss).
    """
    # ── Datasets & loaders ────────────────────────────────────────────────
    train_ds = CVSSDataset(train_records, tokenizer, args.max_length)
    val_ds   = CVSSDataset(val_records,   tokenizer, args.max_length)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2,
        shuffle=False, num_workers=0, pin_memory=False,
    )

    # ── Criteria & model ──────────────────────────────────────────────────
    criteria = build_criteria(train_records, device)
    model    = BertMultiTaskModel(args.model_name, args.dropout).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01
    )
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.06)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )
    scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and args.fp16) else None

    task_weights = TASK_WEIGHTS   # PR=1.5, C/I/A=1.2, rest=1.0

    best_val_loss = float("inf")
    best_state    = None
    patience_cnt  = 0
    history: list[dict] = []

    import numpy as np

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch_multitask(
            model, train_loader, optimizer, scheduler,
            criteria, device, task_weights,
            is_multimodal=False, scaler=scaler,
        )

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                ids  = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                logits_dict = model(ids, mask)
                for m in METRICS:
                    lbls = batch["labels"][m].to(device)
                    w    = task_weights.get(m, 1.0)
                    val_loss += w * criteria[m](logits_dict[m], lbls).item()
        val_loss /= len(val_loader)

        # Per-metric accuracy + exact match on val
        val_preds = predict_multitask(model, val_loader, device, is_multimodal=False)
        val_true  = encode_labels(val_records)
        per_acc = {
            m: float(np.mean(np.array(val_true[m]) == np.array(val_preds[m])))
            for m in METRICS
        }
        mean_acc = float(np.mean(list(per_acc.values())))
        exact = np.ones(len(val_records), dtype=bool)
        for m in METRICS:
            exact &= np.array(val_true[m]) == np.array(val_preds[m])
        exact_match = float(np.mean(exact))

        history.append({
            "epoch":       epoch,
            "train_loss":  round(train_loss,  6),
            "val_loss":    round(val_loss,     6),
            "mean_acc":    round(mean_acc,     6),
            "exact_match": round(exact_match,  6),
            "per_metric_acc": {m: round(v, 6) for m, v in per_acc.items()},
        })

        print(
            f"  Epoch {epoch}/{args.epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"mean_acc={mean_acc:.3f}  exact={exact_match:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Baseline 3: Multi-task BERT (Scientific Contribution #1)"
    )
    parser.add_argument("--data-dir",    default="dataset")
    parser.add_argument("--model-dir",   default="models/baseline3")
    parser.add_argument("--results-out", default=None)
    parser.add_argument("--model-name",  default="distilbert-base-uncased")
    parser.add_argument("--epochs",      type=int,   default=4)
    parser.add_argument("--batch-size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--max-length",  type=int,   default=128)
    parser.add_argument("--dropout",     type=float, default=0.3)
    parser.add_argument("--patience",    type=int,   default=3)
    parser.add_argument("--fp16",        action="store_true")
    parser.add_argument("--device",      default=None)
    parser.add_argument("--load-model",  default=None,
                        help="Path to saved checkpoint (.pt); skips training")
    args = parser.parse_args()

    device    = get_device(args.device)
    data_dir  = Path(args.data_dir)
    model_dir = Path(args.model_dir)

    print(f"Device: {device}")
    print(f"Model:  {args.model_name}")
    print(f"Task weights: {TASK_WEIGHTS}")

    # ── Load data ─────────────────────────────────────────────────────────
    print("\nLoading data ...")
    train_records = load_jsonl(data_dir / "train.jsonl")
    val_records   = load_jsonl(data_dir / "val.jsonl")
    test_records  = load_jsonl(data_dir / "test.jsonl")
    print(f"  train={len(train_records):,}  val={len(val_records):,}  test={len(test_records):,}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ── Train or load model ───────────────────────────────────────────────
    ckpt_path = Path(args.load_model) if args.load_model else model_dir / "multitask.pt"

    hist_path = model_dir / "history.json"

    if args.load_model and ckpt_path.exists():
        print(f"\nLoading checkpoint from {ckpt_path} ...")
        model = BertMultiTaskModel(args.model_name, args.dropout).to(device)
        load_checkpoint(model, ckpt_path, device)
    else:
        print(f"\nTraining multi-task BERT ({args.epochs} epochs, "
              f"bs={args.batch_size}, lr={args.lr}) ...")
        model, history = train(train_records, val_records, tokenizer, args, device)
        save_checkpoint(model, ckpt_path)
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Checkpoint saved → {ckpt_path}")
        print(f"History   saved  → {hist_path}")

    # ── Evaluate ──────────────────────────────────────────────────────────
    def run_eval(records, split_name):
        ds     = CVSSDataset(records, tokenizer, args.max_length)
        loader = DataLoader(ds, batch_size=args.batch_size * 2,
                            shuffle=False, num_workers=0)
        true_labels = encode_labels(records)
        pred_labels = predict_multitask(model, loader, device, is_multimodal=False)
        results     = evaluate(true_labels, pred_labels, records)
        print_results(f"Baseline 3 — Multi-task BERT  [{split_name}]", results)
        return results

    print("\nEvaluating on validation set ...")
    val_results  = run_eval(val_records,  "Validation")
    print("\nEvaluating on test set ...")
    test_results = run_eval(test_records, "Test")

    # ── Save results ──────────────────────────────────────────────────────
    if args.results_out:
        Path(args.results_out).parent.mkdir(parents=True, exist_ok=True)
        out = {
            "baseline": "multitask_bert",
            "description": (
                f"Single DistilBERT ({args.model_name}) with 8 task heads; "
                f"weighted multi-task loss; "
                f"epochs={args.epochs}, bs={args.batch_size}, lr={args.lr}"
            ),
            "task_weights": TASK_WEIGHTS,
            "val":  val_results,
            "test": test_results,
        }
        with open(args.results_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved → {args.results_out}")


if __name__ == "__main__":
    main()
