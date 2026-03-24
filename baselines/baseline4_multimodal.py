"""
Baseline 4: Multi-modal (Text + Code Diff) — Scientific Contribution #2.

The only prior work using code for CVSS scoring is Du et al. (2024), which
showed +3–5% F1. This baseline combines:
  • DistilBERT for CVE description (text modality)
  • CodeBERT for security patch diff (code modality)
  • Fusion layer (concatenation → projection) + 8 task heads

Key design choices:
  • When no diff is available → code embedding is zeroed out (graceful fallback)
  • Training strategy: full fine-tune from epoch 1 (both encoders)
  • Ablation: compare with/without diff on the patch-covered subset

Architecture:
  CVE Description          Security Patch (diff)
       ↓                          ↓
  DistilBERT               CodeBERT encoder
  encoder (768-d)          (microsoft/codebert-base, 768-d)
       ↓                          ↓
  text_emb                 code_emb  (× has_code mask)
       └──────────┬─────────────┘
                  ↓
          concat (1 536-d)
                  ↓
          Fusion  Linear (1 536 → 768) + GELU + Dropout
                  ↓
          8 Classification Heads
                  ↓
          Complete CVSS Vector

Expected results:
  Per-metric accuracy : 73–78%  (+3–5% over baseline 3 on diff subset)
  Exact match         : 60–65%  (+5–10% over baseline 3 on diff subset)
  C/I/A improvement   : +5–7%   (impact metrics benefit most from code)

Usage:
  python -m baselines.baseline4_multimodal \\
      --data-dir dataset \\
      --model-dir models/baseline4 \\
      --results-out results/baseline4.json
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

from baselines.common import (
    METRICS, NUM_CLASSES, TASK_WEIGHTS,
    load_jsonl, encode_labels, evaluate, print_results,
)
from baselines.bert_common import (
    MultiModalDataset,
    compute_class_weights,
    get_device,
    train_epoch_multitask,
    predict_multitask,
    save_checkpoint,
    load_checkpoint,
)


# ── Default model names ───────────────────────────────────────────────────────

DEFAULT_TEXT_MODEL = "distilbert-base-uncased"
DEFAULT_CODE_MODEL = "microsoft/codebert-base"


# ── Model ─────────────────────────────────────────────────────────────────────

class MultiModalCVSSModel(nn.Module):
    """
    Dual-encoder model: DistilBERT (text) + CodeBERT (code diff).

    Fusion via concatenation → linear projection → 8 classification heads.

    When has_code=False for a sample, the code embedding is masked to zeros
    so the model learns to fall back on the text-only representation.
    """

    def __init__(
        self,
        text_model_name: str = DEFAULT_TEXT_MODEL,
        code_model_name: str = DEFAULT_CODE_MODEL,
        dropout: float = 0.3,
        fusion_hidden: int = 768,
    ):
        super().__init__()

        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.code_encoder = AutoModel.from_pretrained(code_model_name)

        text_dim = self.text_encoder.config.hidden_size   # 768
        code_dim = self.code_encoder.config.hidden_size   # 768

        # Fusion: concatenated embeddings → shared representation
        self.fusion = nn.Sequential(
            nn.Linear(text_dim + code_dim, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

        # 8 task-specific classification heads
        self.heads = nn.ModuleDict({
            m: nn.Linear(fusion_hidden, NUM_CLASSES[m])
            for m in METRICS
        })

    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        code_input_ids: torch.Tensor,
        code_attention_mask: torch.Tensor,
        has_code: torch.Tensor,           # BoolTensor [batch]
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            Dict {metric: logits_tensor [batch, num_classes]}
        """
        # Text branch
        text_out = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
        )
        text_emb = text_out.last_hidden_state[:, 0, :]   # [CLS]

        # Code branch (pass through encoder regardless; mask afterward)
        code_out = self.code_encoder(
            input_ids=code_input_ids,
            attention_mask=code_attention_mask,
        )
        code_emb = code_out.last_hidden_state[:, 0, :]   # [CLS]

        # Zero-out code embeddings where no diff is available
        # has_code: [batch] bool → [batch, 1] float mask
        mask = has_code.unsqueeze(1).float()
        code_emb = code_emb * mask

        # Fusion
        fused  = torch.cat([text_emb, code_emb], dim=-1)   # [batch, 1536]
        fused  = self.fusion(fused)                          # [batch, 768]

        return {m: self.heads[m](fused) for m in METRICS}


# ── Training ──────────────────────────────────────────────────────────────────

def build_criteria(
    train_records: list[dict],
    device: torch.device,
) -> dict[str, nn.CrossEntropyLoss]:
    all_labels = encode_labels(train_records)
    return {
        m: nn.CrossEntropyLoss(
            weight=compute_class_weights(all_labels[m], NUM_CLASSES[m], device)
        )
        for m in METRICS
    }


def train(
    train_records: list[dict],
    val_records: list[dict],
    text_tokenizer,
    code_tokenizer,
    args,
    device: torch.device,
) -> MultiModalCVSSModel:
    """Fine-tune the multi-modal model."""

    # ── Datasets & loaders ────────────────────────────────────────────────
    train_ds = MultiModalDataset(
        train_records, text_tokenizer, code_tokenizer,
        args.text_max_length, args.code_max_length,
    )
    val_ds = MultiModalDataset(
        val_records, text_tokenizer, code_tokenizer,
        args.text_max_length, args.code_max_length,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2,
        shuffle=False, num_workers=0, pin_memory=False,
    )

    # ── Model & optimisation ──────────────────────────────────────────────
    criteria = build_criteria(train_records, device)
    model    = MultiModalCVSSModel(
        args.text_model, args.code_model, args.dropout
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01
    )
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.06)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )
    scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and args.fp16) else None

    best_val_loss = float("inf")
    best_state    = None
    patience_cnt  = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch_multitask(
            model, train_loader, optimizer, scheduler,
            criteria, device, TASK_WEIGHTS,
            is_multimodal=True, scaler=scaler,
        )

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                t_ids  = batch["text_input_ids"].to(device)
                t_mask = batch["text_attention_mask"].to(device)
                c_ids  = batch["code_input_ids"].to(device)
                c_mask = batch["code_attention_mask"].to(device)
                hc     = batch["has_code"].to(device)
                logits_dict = model(t_ids, t_mask, c_ids, c_mask, hc)
                for m in METRICS:
                    lbls = batch["labels"][m].to(device)
                    val_loss += TASK_WEIGHTS.get(m, 1.0) * criteria[m](logits_dict[m], lbls).item()
        val_loss /= len(val_loader)

        # Quick exact-match on val
        val_preds = predict_multitask(model, val_loader, device, is_multimodal=True)
        val_true  = encode_labels(val_records)
        import numpy as np
        exact = np.ones(len(val_records), dtype=bool)
        for m in METRICS:
            exact &= np.array(val_true[m]) == np.array(val_preds[m])
        exact_match = float(np.mean(exact))
        mean_acc    = float(np.mean([
            np.mean(np.array(val_true[m]) == np.array(val_preds[m]))
            for m in METRICS
        ]))

        # How many val samples actually had a diff
        n_with_diff = sum(1 for r in val_records if r.get("patch_diff"))
        print(
            f"  Epoch {epoch}/{args.epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"mean_acc={mean_acc:.3f}  exact={exact_match:.3f}  "
            f"(val with diff: {n_with_diff}/{len(val_records)})"
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

    return model


# ── Ablation: diff subset evaluation ─────────────────────────────────────────

def evaluate_diff_subset(
    model: MultiModalCVSSModel,
    records: list[dict],
    text_tokenizer,
    code_tokenizer,
    args,
    device: torch.device,
    split_name: str = "Test",
) -> dict:
    """
    Ablation Study 2: evaluate separately on records that have a diff
    vs records without a diff. Shows the direct contribution of code information.
    """
    with_diff    = [r for r in records if r.get("patch_diff")]
    without_diff = [r for r in records if not r.get("patch_diff")]

    print(f"\n  [{split_name}] with diff: {len(with_diff):,}  "
          f"without diff: {len(without_diff):,}")

    results_out = {}

    for subset_name, subset in [("with_diff", with_diff), ("without_diff", without_diff)]:
        if not subset:
            continue
        ds = MultiModalDataset(
            subset, text_tokenizer, code_tokenizer,
            args.text_max_length, args.code_max_length,
        )
        loader = DataLoader(ds, batch_size=args.batch_size * 2,
                            shuffle=False, num_workers=0)
        true_labels = encode_labels(subset)
        pred_labels = predict_multitask(model, loader, device, is_multimodal=True)
        res = evaluate(true_labels, pred_labels, subset)
        print_results(
            f"Baseline 4 — Multi-modal  [{split_name} / {subset_name}]", res
        )
        results_out[subset_name] = res

    return results_out


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Baseline 4: Multi-modal BERT+CodeBERT (Scientific Contribution #2)"
    )
    parser.add_argument("--data-dir",         default="dataset")
    parser.add_argument("--model-dir",        default="models/baseline4")
    parser.add_argument("--results-out",      default=None)
    parser.add_argument("--text-model",       default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--code-model",       default=DEFAULT_CODE_MODEL)
    parser.add_argument("--epochs",           type=int,   default=4)
    parser.add_argument("--batch-size",       type=int,   default=16,
                        help="Smaller batch due to two encoders in memory")
    parser.add_argument("--lr",               type=float, default=2e-5)
    parser.add_argument("--text-max-length",  type=int,   default=128)
    parser.add_argument("--code-max-length",  type=int,   default=256)
    parser.add_argument("--dropout",          type=float, default=0.3)
    parser.add_argument("--patience",         type=int,   default=3)
    parser.add_argument("--fp16",             action="store_true")
    parser.add_argument("--device",           default=None)
    parser.add_argument("--load-model",       default=None,
                        help="Path to saved checkpoint; skips training")
    parser.add_argument("--ablation-diff",    action="store_true",
                        help="Also evaluate separately on with/without diff subsets")
    args = parser.parse_args()

    device    = get_device(args.device)
    data_dir  = Path(args.data_dir)
    model_dir = Path(args.model_dir)

    print(f"Device    : {device}")
    print(f"Text model: {args.text_model}")
    print(f"Code model: {args.code_model}")

    # ── Load data ─────────────────────────────────────────────────────────
    print("\nLoading data ...")
    train_records = load_jsonl(data_dir / "train.jsonl")
    val_records   = load_jsonl(data_dir / "val.jsonl")
    test_records  = load_jsonl(data_dir / "test.jsonl")
    n_train_diff = sum(1 for r in train_records if r.get("patch_diff"))
    n_test_diff  = sum(1 for r in test_records  if r.get("patch_diff"))
    print(f"  train={len(train_records):,} ({n_train_diff} with diff)  "
          f"val={len(val_records):,}  "
          f"test={len(test_records):,} ({n_test_diff} with diff)")

    text_tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    code_tokenizer = AutoTokenizer.from_pretrained(args.code_model)

    # ── Train or load model ───────────────────────────────────────────────
    ckpt_path = Path(args.load_model) if args.load_model else model_dir / "multimodal.pt"

    if args.load_model and ckpt_path.exists():
        print(f"\nLoading checkpoint from {ckpt_path} ...")
        model = MultiModalCVSSModel(args.text_model, args.code_model, args.dropout).to(device)
        load_checkpoint(model, ckpt_path, device)
    else:
        print(f"\nTraining multi-modal model ({args.epochs} epochs, "
              f"bs={args.batch_size}, lr={args.lr}) ...")
        model = train(
            train_records, val_records,
            text_tokenizer, code_tokenizer,
            args, device,
        )
        save_checkpoint(model, ckpt_path)
        print(f"Checkpoint saved → {ckpt_path}")

    # ── Standard evaluation ───────────────────────────────────────────────
    def run_eval(records, split_name):
        ds = MultiModalDataset(
            records, text_tokenizer, code_tokenizer,
            args.text_max_length, args.code_max_length,
        )
        loader = DataLoader(ds, batch_size=args.batch_size * 2,
                            shuffle=False, num_workers=0)
        true_labels = encode_labels(records)
        pred_labels = predict_multitask(model, loader, device, is_multimodal=True)
        results     = evaluate(true_labels, pred_labels, records)
        print_results(f"Baseline 4 — Multi-modal  [{split_name}]", results)
        return results

    print("\nEvaluating on validation set ...")
    val_results  = run_eval(val_records,  "Validation")
    print("\nEvaluating on test set ...")
    test_results = run_eval(test_records, "Test")

    # ── Ablation: diff vs no-diff ─────────────────────────────────────────
    ablation_results = {}
    if args.ablation_diff:
        print("\n[Ablation Study 2: with diff vs without diff]")
        ablation_results = evaluate_diff_subset(
            model, test_records,
            text_tokenizer, code_tokenizer,
            args, device, split_name="Test",
        )

    # ── Save results ──────────────────────────────────────────────────────
    if args.results_out:
        Path(args.results_out).parent.mkdir(parents=True, exist_ok=True)
        out = {
            "baseline": "multimodal",
            "description": (
                f"DistilBERT ({args.text_model}) + CodeBERT ({args.code_model}); "
                f"concatenation fusion; "
                f"epochs={args.epochs}, bs={args.batch_size}, lr={args.lr}"
            ),
            "task_weights": TASK_WEIGHTS,
            "val":      val_results,
            "test":     test_results,
            "ablation": ablation_results,
        }
        with open(args.results_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved → {args.results_out}")


if __name__ == "__main__":
    main()
