"""
Run all baselines and produce a comparison table + ablation studies.

This script either:
  (a) Calls each baseline module programmatically, OR
  (b) Loads pre-computed JSON results from --results-dir

Usage examples:

  # Run everything from scratch (CPU, quick smoke-test with tiny data):
  python -m baselines.run_all --data-dir dataset --results-dir results

  # Load pre-computed results and just print the comparison table:
  python -m baselines.run_all --results-dir results --compare-only

  # Run only classical ML baseline (fast):
  python -m baselines.run_all --data-dir dataset --baselines tfidf_lr

Ablation studies included:
  Study 1: Separate vs Multi-task (baseline 2 vs 3) — exact match gain
  Study 2: With diff vs without diff (baseline 3 vs 4) — code value
  Study 3: Per-metric breakdown for impact metrics C/I/A
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


# ── Colour helpers (ANSI, disabled on Windows without colour support) ─────────
try:
    import colorama; colorama.init()
    BOLD  = "\033[1m"
    GREEN = "\033[92m"
    CYAN  = "\033[96m"
    RESET = "\033[0m"
except ImportError:
    BOLD = GREEN = CYAN = RESET = ""


# ── Metric labels ─────────────────────────────────────────────────────────────
METRICS = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]

BASELINE_CONFIGS = {
    "tfidf_lr": {
        "module":      "baselines.baseline1_tfidf_lr",
        "result_file": "baseline1_tfidf_lr.json",
        "label":       "1. TF-IDF + LogReg",
    },
    "bert_separate": {
        "module":      "baselines.baseline2_bert_separate",
        "result_file": "baseline2_bert_separate.json",
        "label":       "2. BERT Separate",
    },
    "multitask_bert": {
        "module":      "baselines.baseline3_multitask_bert",
        "result_file": "baseline3_multitask_bert.json",
        "label":       "3. Multi-task BERT ✨",
    },
    "multimodal": {
        "module":      "baselines.baseline4_multimodal",
        "result_file": "baseline4_multimodal.json",
        "label":       "4. Multi-modal     ✨",
    },
}


# ── Running baselines ─────────────────────────────────────────────────────────

def run_baseline(key: str, data_dir: Path, results_dir: Path, extra_args: list[str]):
    cfg  = BASELINE_CONFIGS[key]
    out  = results_dir / cfg["result_file"]
    cmd  = [
        sys.executable, "-m", cfg["module"],
        "--data-dir",    str(data_dir),
        "--results-out", str(out),
    ] + extra_args
    print(f"\n{BOLD}{'─'*62}{RESET}")
    print(f"{BOLD}Running: {cfg['label']}{RESET}")
    print(f"{'─'*62}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[WARNING] {key} exited with code {result.returncode}")
    return out


# ── Loading results ───────────────────────────────────────────────────────────

def load_results(results_dir: Path, keys: list[str]) -> dict[str, dict]:
    loaded = {}
    for key in keys:
        cfg  = BASELINE_CONFIGS[key]
        path = results_dir / cfg["result_file"]
        if path.exists():
            with open(path) as f:
                loaded[key] = json.load(f)
        else:
            print(f"[WARNING] Result not found: {path}")
    return loaded


# ── Comparison table ──────────────────────────────────────────────────────────

def print_comparison_table(all_results: dict[str, dict], split: str = "test"):
    print(f"\n{BOLD}{'='*72}{RESET}")
    print(f"{BOLD}  COMPARISON TABLE  [{split.upper()}]{RESET}")
    print(f"{BOLD}{'='*72}{RESET}")

    header = (
        f"{'Model':<28} {'MeanAcc':>8} {'ExactM':>8} "
        f"{'HammDist':>9} {'ScoreMAE':>9} {'SevAcc':>8} {'CritFNR':>8}"
    )
    print(header)
    print("─" * 72)

    best = {
        "mean_accuracy":   ("key", -1.0),
        "exact_match":     ("key", -1.0),
        "hamming_distance":("key", 999.0),
        "score_mae":       ("key", 999.0),
        "severity_accuracy":("key", -1.0),
        "critical_fnr":    ("key", 999.0),
    }

    rows = []
    for key, data in all_results.items():
        r = data.get(split, {})
        if not r:
            continue
        rows.append((key, r))
        # Track bests
        if r.get("mean_accuracy",    -1) > best["mean_accuracy"][1]:
            best["mean_accuracy"] = (key, r["mean_accuracy"])
        if r.get("exact_match",      -1) > best["exact_match"][1]:
            best["exact_match"] = (key, r["exact_match"])
        if r.get("hamming_distance", 999) < best["hamming_distance"][1]:
            best["hamming_distance"] = (key, r["hamming_distance"])
        if r.get("score_mae",        999) < best["score_mae"][1]:
            best["score_mae"] = (key, r["score_mae"])
        if r.get("severity_accuracy",-1) > best["severity_accuracy"][1]:
            best["severity_accuracy"] = (key, r["severity_accuracy"])
        if r.get("critical_fnr",     999) < best["critical_fnr"][1]:
            best["critical_fnr"] = (key, r["critical_fnr"])

    for key, r in rows:
        label = BASELINE_CONFIGS[key]["label"]
        acc   = r.get("mean_accuracy",    float("nan"))
        em    = r.get("exact_match",      float("nan"))
        hd    = r.get("hamming_distance", float("nan"))
        mae   = r.get("score_mae",        float("nan"))
        sa    = r.get("severity_accuracy",float("nan"))
        fnr   = r.get("critical_fnr",     float("nan"))

        is_best_acc = best["mean_accuracy"][0]    == key
        is_best_em  = best["exact_match"][0]      == key
        is_best_hd  = best["hamming_distance"][0] == key
        is_best_mae = best["score_mae"][0]        == key
        is_best_sa  = best["severity_accuracy"][0] == key
        is_best_fnr = best["critical_fnr"][0]     == key

        def fmt(v, is_best, higher_better=True):
            s = f"{v*100:7.2f}%" if higher_better else f"{v:.4f} "
            return (GREEN + BOLD + s + RESET) if is_best else s

        def fmt_raw(v, is_best):
            s = f"{v:.4f}  "
            return (GREEN + BOLD + s + RESET) if is_best else s

        line = (
            f"{label:<28} "
            f"{fmt(acc, is_best_acc):>8} "
            f"{fmt(em,  is_best_em):>8} "
            f"{fmt_raw(hd,  is_best_hd):>9} "
            f"{fmt_raw(mae, is_best_mae):>9} "
            f"{fmt(sa,  is_best_sa):>8} "
            f"{fmt(fnr, is_best_fnr, higher_better=False):>8}"
        )
        print(line)

    print("─" * 72)
    print("  (Green = best in column)")


def print_per_metric_table(all_results: dict[str, dict], split: str = "test"):
    print(f"\n{BOLD}{'='*72}{RESET}")
    print(f"{BOLD}  PER-METRIC ACCURACY  [{split.upper()}]{RESET}")
    print(f"{BOLD}{'='*72}{RESET}")

    # Header
    metric_header = "".join(f"{m:>7}" for m in METRICS)
    print(f"{'Model':<28}{metric_header}")
    print("─" * 72)

    for key, data in all_results.items():
        r = data.get(split, {})
        if not r:
            continue
        label = BASELINE_CONFIGS[key]["label"]
        pma = r.get("per_metric_accuracy", {})
        accs = "".join(f"{pma.get(m, float('nan'))*100:6.1f}%" for m in METRICS)
        print(f"{label:<28}{accs}")

    print("─" * 72)


# ── Ablation study tables ─────────────────────────────────────────────────────

def print_ablation_separate_vs_multitask(all_results: dict, split: str = "test"):
    """
    Study 1: BERT Separate (B2) vs Multi-task BERT (B3).
    Expected: B3 loses ~3–5% mean accuracy but gains +15% exact match.
    """
    b2 = all_results.get("bert_separate",  {}).get(split, {})
    b3 = all_results.get("multitask_bert", {}).get(split, {})

    if not (b2 and b3):
        return

    print(f"\n{BOLD}{'='*62}{RESET}")
    print(f"{BOLD}  ABLATION 1: Separate vs Multi-task  [{split.upper()}]{RESET}")
    print(f"{BOLD}{'='*62}{RESET}")
    print(f"{'Metric':<30} {'BERT Sep':>10} {'Multi-task':>11} {'Δ':>8}")
    print("─" * 62)

    metrics_to_show = [
        ("Mean Accuracy",      "mean_accuracy",     True),
        ("Exact Match",        "exact_match",       True),
        ("Hamming Distance",   "hamming_distance",  False),
        ("Score MAE",          "score_mae",         False),
    ]
    for label, key, higher_better in metrics_to_show:
        v2  = b2.get(key, float("nan"))
        v3  = b3.get(key, float("nan"))
        diff = v3 - v2
        is_improvement = (diff > 0) if higher_better else (diff < 0)
        sign = "↑" if diff > 0 else "↓"
        color = GREEN if is_improvement else ""

        if key in ("mean_accuracy", "exact_match"):
            line = f"  {label:<28} {v2*100:8.2f}%   {v3*100:8.2f}%  {color}{sign}{abs(diff)*100:5.2f}%{RESET}"
        else:
            line = f"  {label:<28} {v2:9.4f}   {v3:9.4f}  {color}{sign}{abs(diff):.4f}{RESET}"
        print(line)

    print("─" * 62)
    em2 = b2.get("exact_match", 0)
    em3 = b3.get("exact_match", 0)
    gain = (em3 - em2) * 100
    print(f"  Exact Match gain: {GREEN}{BOLD}+{gain:.1f}pp{RESET}  "
          f"(hypothesis: joint optimisation boosts full-vector accuracy)")


def print_ablation_diff_contribution(all_results: dict):
    """
    Study 2: Multi-task text-only (B3) vs Multi-modal (B4).
    Shows the value of code diff information.
    """
    b3 = all_results.get("multitask_bert", {}).get("test", {})
    b4_all  = all_results.get("multimodal", {}).get("test", {})
    b4_diff = all_results.get("multimodal", {}).get("ablation", {}).get("with_diff", {})

    if not (b3 or b4_all):
        return

    print(f"\n{BOLD}{'='*62}{RESET}")
    print(f"{BOLD}  ABLATION 2: Text-only vs Text+Code  [TEST]{RESET}")
    print(f"{BOLD}{'='*62}{RESET}")
    print(f"  Note: B4 'diff subset' restricted to CVEs with a patch.")
    print()
    print(f"{'Metric':<30} {'B3 text-only':>13} {'B4 all':>9} {'B4 diff-only':>13}")
    print("─" * 62)

    for label, key, pct in [
        ("Mean Accuracy",    "mean_accuracy", True),
        ("Exact Match",      "exact_match",   True),
        ("Score MAE",        "score_mae",     False),
    ]:
        v3   = b3.get(key, float("nan"))
        v4a  = b4_all.get(key, float("nan"))
        v4d  = b4_diff.get(key, float("nan"))

        if pct:
            print(f"  {label:<28} {v3*100:10.2f}%  {v4a*100:7.2f}%  {v4d*100:10.2f}%")
        else:
            print(f"  {label:<28} {v3:11.4f}   {v4a:7.4f}   {v4d:10.4f}")
    print("─" * 62)


def print_ablation_impact_metrics(all_results: dict):
    """
    Study 3: Where does code diff help most? (C, I, A impact metrics)
    """
    b3 = all_results.get("multitask_bert", {}).get("test", {})
    b4 = all_results.get("multimodal", {}).get("ablation", {}).get("with_diff", {})

    if not (b3 and b4):
        return

    print(f"\n{BOLD}{'='*62}{RESET}")
    print(f"{BOLD}  ABLATION 3: Impact Metric Analysis (diff subset)  [TEST]{RESET}")
    print(f"{BOLD}{'='*62}{RESET}")
    print(f"  Hypothesis: diff helps most on C/I/A (impact metrics).")
    print()
    print(f"{'Metric':<10} {'B3 text-only':>13} {'B4 text+code':>14} {'Δ':>8}")
    print("─" * 40)

    pma3 = b3.get("per_metric_accuracy", {})
    pma4 = b4.get("per_metric_accuracy", {})

    for m in METRICS:
        v3 = pma3.get(m, float("nan"))
        v4 = pma4.get(m, float("nan"))
        diff = v4 - v3
        color = GREEN if diff > 0.005 else ""
        tag   = " ← impact" if m in ("C", "I", "A") else ""
        print(f"  {m:<8} {v3*100:12.1f}%  {v4*100:12.1f}%  "
              f"{color}+{diff*100:.1f}%{RESET}{tag}")

    print("─" * 40)


# ── Save consolidated results ─────────────────────────────────────────────────

def save_consolidated(all_results: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nConsolidated results saved → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run all CVSS baselines and compare results"
    )
    parser.add_argument("--data-dir",     default="dataset",
                        help="Dataset directory")
    parser.add_argument("--results-dir",  default="results",
                        help="Directory for JSON result files")
    parser.add_argument("--model-dir",    default="models",
                        help="Parent directory for model checkpoints")
    parser.add_argument("--compare-only", action="store_true",
                        help="Skip training; only load results and compare")
    parser.add_argument("--baselines",    nargs="*",
                        choices=list(BASELINE_CONFIGS.keys()),
                        default=list(BASELINE_CONFIGS.keys()),
                        help="Which baselines to run (default: all 4)")
    parser.add_argument("--split",        default="test",
                        choices=["val", "test"],
                        help="Which split to show in the comparison table")
    # Pass-through args to BERT baselines
    parser.add_argument("--epochs",       type=int, default=None)
    parser.add_argument("--batch-size",   type=int, default=None)
    parser.add_argument("--fp16",         action="store_true")
    parser.add_argument("--device",       default=None)
    # Optional: ablation flag for baseline 4
    parser.add_argument("--ablation-diff", action="store_true",
                        help="Run ablation diff analysis for baseline 4")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Build extra args to pass through ─────────────────────────────────
    extra: list[str] = []
    if args.epochs:
        extra += ["--epochs", str(args.epochs)]
    if args.batch_size:
        extra += ["--batch-size", str(args.batch_size)]
    if args.fp16:
        extra += ["--fp16"]
    if args.device:
        extra += ["--device", args.device]

    # ── Run baselines ─────────────────────────────────────────────────────
    if not args.compare_only:
        for key in args.baselines:
            cfg = BASELINE_CONFIGS[key]
            kw_extra = list(extra)

            # Model directory for BERT baselines
            if key in ("bert_separate", "multitask_bert", "multimodal"):
                kw_extra += ["--model-dir", str(Path(args.model_dir) / key)]

            # Baseline 4: ablation diff flag
            if key == "multimodal" and args.ablation_diff:
                kw_extra += ["--ablation-diff"]

            run_baseline(key, Path(args.data_dir), results_dir, kw_extra)

    # ── Load all results ──────────────────────────────────────────────────
    all_results = load_results(results_dir, args.baselines)

    if not all_results:
        print("No results found. Run without --compare-only first.")
        return

    # ── Print tables ──────────────────────────────────────────────────────
    print_comparison_table(all_results, split=args.split)
    print_per_metric_table(all_results, split=args.split)

    # ── Ablation studies ──────────────────────────────────────────────────
    print_ablation_separate_vs_multitask(all_results, split=args.split)
    if args.ablation_diff:
        print_ablation_diff_contribution(all_results)
        print_ablation_impact_metrics(all_results)

    # ── Save consolidated ─────────────────────────────────────────────────
    save_consolidated(all_results, results_dir / "all_results.json")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{BOLD}Summary:{RESET}")
    best_em_key = max(
        ((k, v.get(args.split, {}).get("exact_match", 0)) for k, v in all_results.items()),
        key=lambda x: x[1],
        default=("?", 0),
    )
    best_label = BASELINE_CONFIGS.get(best_em_key[0], {}).get("label", best_em_key[0])
    print(f"  Best Exact Match ({args.split}): {GREEN}{BOLD}{best_label}{RESET} "
          f"— {best_em_key[1]*100:.2f}%")


if __name__ == "__main__":
    main()
