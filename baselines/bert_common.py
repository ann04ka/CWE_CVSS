"""
Shared PyTorch / Transformers utilities for BERT-based baselines.

Provides:
  - CVSSDataset          — standard text-only dataset (baselines 2 & 3)
  - MultiModalDataset    — text + code-diff dataset (baseline 4)
  - compute_class_weights
  - preprocess_diff
  - get_device
  - train_epoch / predict_epoch
  - save_checkpoint / load_checkpoint
"""

import re
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

from baselines.common import METRICS, METRIC_CLASSES, LABEL2IDX, NUM_CLASSES


# ── Diff preprocessing ───────────────────────────────────────────────────────

def preprocess_diff(diff: str, max_lines: int = 120) -> str:
    """
    Extract changed lines (+/-) from a git unified diff.
    Skips file headers (+++/---) and keeps added/removed code lines only.
    Returns a single whitespace-joined string suitable for tokenization.
    """
    lines = []
    for raw in diff.split("\n"):
        if raw.startswith("+++") or raw.startswith("---"):
            continue
        if raw.startswith("+") or raw.startswith("-"):
            # Strip the leading +/- marker
            stripped = raw[1:].strip()
            if stripped:
                lines.append(stripped)
        if len(lines) >= max_lines:
            break
    return " ".join(lines)


# ── Datasets ─────────────────────────────────────────────────────────────────

class CVSSDataset(Dataset):
    """
    Text-only dataset for baselines 2 (BERT separate) and 3 (multi-task BERT).

    Each item returns:
      input_ids        : LongTensor [max_length]
      attention_mask   : LongTensor [max_length]
      labels           : dict {metric: LongTensor scalar}
    """

    def __init__(
        self,
        records: list[dict],
        tokenizer,
        max_length: int = 128,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        text = rec.get("description", "") or ""

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        labels = {}
        mvals = rec.get("metrics", {})
        for m in METRICS:
            val = mvals.get(m, METRIC_CLASSES[m][0])
            labels[m] = torch.tensor(LABEL2IDX[m].get(val, 0), dtype=torch.long)

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         labels,
        }


class MultiModalDataset(Dataset):
    """
    Text + code-diff dataset for baseline 4 (multi-modal).

    Each item returns:
      text_input_ids        : LongTensor [text_max_length]
      text_attention_mask   : LongTensor [text_max_length]
      code_input_ids        : LongTensor [code_max_length]  (zeros if no diff)
      code_attention_mask   : LongTensor [code_max_length]  (zeros if no diff)
      has_code              : BoolTensor scalar
      labels                : dict {metric: LongTensor scalar}
    """

    def __init__(
        self,
        records: list[dict],
        text_tokenizer,
        code_tokenizer,
        text_max_length: int = 128,
        code_max_length: int = 256,
    ):
        self.records = records
        self.text_tokenizer = text_tokenizer
        self.code_tokenizer = code_tokenizer
        self.text_max_length = text_max_length
        self.code_max_length = code_max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]

        # ── Text encoding ─────────────────────────────────────────────────
        text = rec.get("description", "") or ""
        text_enc = self.text_tokenizer(
            text,
            max_length=self.text_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # ── Code encoding ─────────────────────────────────────────────────
        diff = rec.get("patch_diff") or ""
        has_code = bool(diff.strip())

        if has_code:
            code_text = preprocess_diff(diff)
            code_enc = self.code_tokenizer(
                code_text,
                max_length=self.code_max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            code_input_ids      = code_enc["input_ids"].squeeze(0)
            code_attention_mask = code_enc["attention_mask"].squeeze(0)
        else:
            code_input_ids      = torch.zeros(self.code_max_length, dtype=torch.long)
            code_attention_mask = torch.zeros(self.code_max_length, dtype=torch.long)

        # ── Labels ────────────────────────────────────────────────────────
        labels = {}
        mvals = rec.get("metrics", {})
        for m in METRICS:
            val = mvals.get(m, METRIC_CLASSES[m][0])
            labels[m] = torch.tensor(LABEL2IDX[m].get(val, 0), dtype=torch.long)

        return {
            "text_input_ids":      text_enc["input_ids"].squeeze(0),
            "text_attention_mask": text_enc["attention_mask"].squeeze(0),
            "code_input_ids":      code_input_ids,
            "code_attention_mask": code_attention_mask,
            "has_code":            torch.tensor(has_code, dtype=torch.bool),
            "labels":              labels,
        }


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(
    labels: list[int],
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights to handle label imbalance.
    weight_c = total / (num_classes * count_c)
    """
    counts = Counter(labels)
    total = len(labels)
    weights = [
        total / (num_classes * max(counts.get(c, 1), 1))
        for c in range(num_classes)
    ]
    return torch.tensor(weights, dtype=torch.float, device=device)


# ── Device helper ─────────────────────────────────────────────────────────────

def get_device(device_str: Optional[str] = None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Training / Prediction loops ───────────────────────────────────────────────

def train_epoch_multitask(
    model: nn.Module,
    dataloader,
    optimizer,
    scheduler,
    criteria: dict,           # {metric: CrossEntropyLoss}
    device: torch.device,
    task_weights: dict,       # {metric: float}
    is_multimodal: bool = False,
    scaler=None,              # torch.cuda.amp.GradScaler or None
) -> float:
    """One training epoch for multi-task models (baselines 3 & 4)."""
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="  train", leave=False, ncols=80):
        optimizer.zero_grad()

        if is_multimodal:
            text_ids   = batch["text_input_ids"].to(device)
            text_mask  = batch["text_attention_mask"].to(device)
            code_ids   = batch["code_input_ids"].to(device)
            code_mask  = batch["code_attention_mask"].to(device)
            has_code   = batch["has_code"].to(device)
            forward_fn = lambda: model(text_ids, text_mask, code_ids, code_mask, has_code)
        else:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            forward_fn = lambda: model(ids, mask)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits_dict = forward_fn()
                loss = _compute_multitask_loss(logits_dict, batch, criteria, task_weights, device)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits_dict = forward_fn()
            loss = _compute_multitask_loss(logits_dict, batch, criteria, task_weights, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


def train_epoch_single(
    model: nn.Module,
    dataloader,
    optimizer,
    scheduler,
    criterion: nn.Module,
    metric: str,
    device: torch.device,
    scaler=None,
) -> float:
    """One training epoch for a single-metric model (baseline 2)."""
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc=f"  train[{metric}]", leave=False, ncols=80):
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        lbls  = batch["labels"][metric].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(ids, mask)
                loss = criterion(logits, lbls)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(ids, mask)
            loss = criterion(logits, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


def _compute_multitask_loss(
    logits_dict: dict,
    batch: dict,
    criteria: dict,
    task_weights: dict,
    device: torch.device,
) -> torch.Tensor:
    loss = torch.tensor(0.0, device=device)
    for m in METRICS:
        lbls = batch["labels"][m].to(device)
        w = task_weights.get(m, 1.0)
        loss = loss + w * criteria[m](logits_dict[m], lbls)
    return loss


@torch.no_grad()
def predict_multitask(
    model: nn.Module,
    dataloader,
    device: torch.device,
    is_multimodal: bool = False,
) -> dict[str, list[int]]:
    """Return predicted label indices for all metrics."""
    model.eval()
    preds = {m: [] for m in METRICS}

    for batch in tqdm(dataloader, desc="  predict", leave=False, ncols=80):
        if is_multimodal:
            text_ids  = batch["text_input_ids"].to(device)
            text_mask = batch["text_attention_mask"].to(device)
            code_ids  = batch["code_input_ids"].to(device)
            code_mask = batch["code_attention_mask"].to(device)
            has_code  = batch["has_code"].to(device)
            logits_dict = model(text_ids, text_mask, code_ids, code_mask, has_code)
        else:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits_dict = model(ids, mask)

        for m in METRICS:
            preds[m].extend(logits_dict[m].argmax(dim=-1).cpu().tolist())

    return preds


@torch.no_grad()
def predict_single(
    model: nn.Module,
    dataloader,
    metric: str,
    device: torch.device,
) -> list[int]:
    """Return predicted label indices for one metric."""
    model.eval()
    preds = []

    for batch in tqdm(dataloader, desc=f"  predict[{metric}]", leave=False, ncols=80):
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        logits = model(ids, mask)
        preds.extend(logits.argmax(dim=-1).cpu().tolist())

    return preds


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(model: nn.Module, path: Path, extra: dict = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(model: nn.Module, path: Path, device: torch.device):
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model_state"])
    return payload
