"""
Visualization of CVSS v3.1 baseline results.

Reads evaluation JSON files produced by baseline1/2/3 and optionally
the per-epoch history files saved by baseline2/3. Generates 7 figures:

  01_per_metric_accuracy.png  — grouped bar chart (B1 vs B2 vs B3)
  02_summary_metrics.png      — exact match / mean acc / MAE / Hamming
  03_training_curves.png      — train & val loss over epochs
  04_b3_epoch_metrics.png     — B3 exact match & accuracy per epoch
  05_radar_chart.png          — spider chart per-metric accuracy
  06_error_metrics.png        — Score MAE, Hamming distance, Critical FNR
  07_dashboard.png            — all-in-one overview figure

Usage:
  # Results already present in results/ from the three baseline runs:
  python -m baselines.visualize --results-dir results --out-dir plots

  # Also draw training curves (requires history files from B2/B3):
  python -m baselines.visualize \\
      --results-dir results \\
      --history-b2  models/baseline2 \\
      --history-b3  models/baseline3/history.json \\
      --out-dir plots

  # Show figures interactively instead of saving:
  python -m baselines.visualize --results-dir results --show
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        130,
    "savefig.dpi":       180,
    "savefig.bbox":      "tight",
})

# ── Palette & labels ──────────────────────────────────────────────────────────

BASELINES = {
    "b1": {"label": "B1: TF-IDF + LogReg", "color": "#4878CF", "marker": "o"},
    "b2": {"label": "B2: BERT Separate",   "color": "#E07B39", "marker": "s"},
    "b3": {"label": "B3: Multi-task BERT", "color": "#3DAA6D", "marker": "^"},
}

METRICS = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]

METRIC_LABELS = {
    "AV": "Attack\nVector",
    "AC": "Attack\nComplexity",
    "PR": "Privileges\nRequired",
    "UI": "User\nInteraction",
    "S":  "Scope",
    "C":  "Confidentiality",
    "I":  "Integrity",
    "A":  "Availability",
}


# ── Data loading helpers ──────────────────────────────────────────────────────

def load_result(path: Path) -> dict | None:
    if path and Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_b2_history(model_dir: Path) -> dict[str, list[dict]] | None:
    """Load per-metric history files saved by baseline2."""
    if not model_dir or not Path(model_dir).exists():
        return None
    hist = {}
    for m in METRICS:
        p = Path(model_dir) / f"history_{m}.json"
        if p.exists():
            with open(p) as f:
                hist[m] = json.load(f)
    return hist if hist else None


def load_b3_history(path: Path) -> list[dict] | None:
    if not path or not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_pma(result: dict, split: str) -> dict[str, float]:
    """Per-metric accuracy dict from result JSON for the given split."""
    return result.get(split, {}).get("per_metric_accuracy", {})


def get_metric(result: dict, split: str, key: str) -> float:
    return result.get(split, {}).get(key, float("nan"))


# ── Figure 1: Per-metric accuracy grouped bar chart ──────────────────────────

def plot_per_metric_accuracy(
    results: dict,
    split: str,
    out: Path | None,
    show: bool,
):
    keys   = [k for k in ["b1", "b2", "b3"] if k in results]
    n_bl   = len(keys)
    n_met  = len(METRICS)
    width  = 0.22
    x      = np.arange(n_met)

    fig, ax = plt.subplots(figsize=(13, 5.5))

    offsets = np.linspace(-(n_bl - 1) / 2, (n_bl - 1) / 2, n_bl) * width

    for i, key in enumerate(keys):
        pma = get_pma(results[key], split)
        vals = [pma.get(m, float("nan")) * 100 for m in METRICS]
        bars = ax.bar(
            x + offsets[i], vals, width,
            label=BASELINES[key]["label"],
            color=BASELINES[key]["color"],
            edgecolor="white", linewidth=0.6, zorder=3,
        )
        # Value labels on top of bars
        for bar, v in zip(bars, vals):
            if not math.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4,
                    f"{v:.1f}",
                    ha="center", va="bottom",
                    fontsize=7.5, color="#333333",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=9.5)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(60, 102)
    ax.set_title(f"Per-Metric Classification Accuracy  [{split.upper()}]")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", framealpha=0.9)

    _save_or_show(fig, out, show, "01_per_metric_accuracy.png")


# ── Figure 2: Summary metrics bar chart ──────────────────────────────────────

def plot_summary_metrics(
    results: dict,
    split: str,
    out: Path | None,
    show: bool,
):
    keys = [k for k in ["b1", "b2", "b3"] if k in results]

    summary_keys = [
        ("mean_accuracy",    "Mean\nAccuracy (%)",    True,  100),
        ("exact_match",      "Exact\nMatch (%)",      True,  100),
        ("severity_accuracy","Severity\nAccuracy (%)",True,  100),
    ]

    n_groups = len(summary_keys)
    n_bl     = len(keys)
    width    = 0.22
    x        = np.arange(n_groups)
    offsets  = np.linspace(-(n_bl - 1) / 2, (n_bl - 1) / 2, n_bl) * width

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, key in enumerate(keys):
        vals = [
            get_metric(results[key], split, sk) * scale
            for sk, _, _, scale in summary_keys
        ]
        bars = ax.bar(
            x + offsets[i], vals, width,
            label=BASELINES[key]["label"],
            color=BASELINES[key]["color"],
            edgecolor="white", linewidth=0.6, zorder=3,
        )
        for bar, v in zip(bars, vals):
            if not math.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{v:.1f}",
                    ha="center", va="bottom",
                    fontsize=9, color="#333333",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label, _, _ in summary_keys], fontsize=10)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(40, 102)
    ax.set_title(f"Summary Performance Metrics  [{split.upper()}]")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", framealpha=0.9)

    _save_or_show(fig, out, show, "02_summary_metrics.png")


# ── Figure 3: Training loss curves ───────────────────────────────────────────

def plot_training_curves(
    b2_hist: dict[str, list[dict]] | None,
    b3_hist: list[dict] | None,
    out: Path | None,
    show: bool,
):
    has_b2 = b2_hist is not None
    has_b3 = b3_hist is not None

    if not has_b2 and not has_b3:
        print("  [skip] No training history found for B2 or B3.")
        return

    n_cols = sum([has_b2, has_b3])
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5), squeeze=False)
    ax_list = list(axes[0])

    col = 0

    # ── B2: average train/val loss over all 8 metrics ─────────────────────
    if has_b2:
        ax = ax_list[col]; col += 1
        max_epoch = max(len(v) for v in b2_hist.values())

        # Build epoch-averaged arrays (metrics may have different lengths)
        train_avg = np.full(max_epoch, np.nan)
        val_avg   = np.full(max_epoch, np.nan)
        for ep_idx in range(max_epoch):
            t_vals = [h[ep_idx]["train_loss"] for h in b2_hist.values() if ep_idx < len(h)]
            v_vals = [h[ep_idx]["val_loss"]   for h in b2_hist.values() if ep_idx < len(h)]
            if t_vals: train_avg[ep_idx] = np.mean(t_vals)
            if v_vals: val_avg[ep_idx]   = np.mean(v_vals)

        epochs = np.arange(1, max_epoch + 1)
        ax.plot(epochs, train_avg, "o-",
                color=BASELINES["b2"]["color"], label="Train loss (avg 8 metrics)", lw=2)
        ax.plot(epochs, val_avg, "s--",
                color=BASELINES["b2"]["color"], alpha=0.6, label="Val loss (avg)", lw=2)

        # Individual metric val loss as thin lines
        for m, h in b2_hist.items():
            ep = np.arange(1, len(h) + 1)
            vl = [e["val_loss"] for e in h]
            ax.plot(ep, vl, "-", alpha=0.18, color=BASELINES["b2"]["color"], lw=1)

        ax.set_title("B2: BERT Separate — Loss Curves\n(thick = avg over 8 metrics, thin = per metric)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.legend(fontsize=9)
        ax.xaxis.grid(True, linestyle="--", alpha=0.4)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_xticks(epochs)

    # ── B3: single model train/val loss ───────────────────────────────────
    if has_b3:
        ax = ax_list[col]; col += 1
        epochs = [e["epoch"]     for e in b3_hist]
        train  = [e["train_loss"] for e in b3_hist]
        val    = [e["val_loss"]   for e in b3_hist]

        ax.plot(epochs, train, "o-",
                color=BASELINES["b3"]["color"], label="Train loss", lw=2)
        ax.plot(epochs, val, "^--",
                color=BASELINES["b3"]["color"], alpha=0.6, label="Val loss", lw=2)

        # Annotate best epoch
        best_ep = epochs[int(np.argmin(val))]
        best_vl = min(val)
        ax.annotate(
            f"best epoch {best_ep}\n({best_vl:.4f})",
            xy=(best_ep, best_vl),
            xytext=(best_ep + 0.2, best_vl + (max(val) - min(val)) * 0.15),
            fontsize=8.5,
            arrowprops=dict(arrowstyle="->", color="#555"),
        )

        ax.set_title("B3: Multi-task BERT — Loss Curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Weighted Multi-task Loss")
        ax.legend(fontsize=9)
        ax.xaxis.grid(True, linestyle="--", alpha=0.4)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_xticks(epochs)

    fig.suptitle("Training Dynamics", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_or_show(fig, out, show, "03_training_curves.png")


# ── Figure 4: B3 per-epoch exact match + mean accuracy ───────────────────────

def plot_b3_epoch_metrics(
    b3_hist: list[dict] | None,
    out: Path | None,
    show: bool,
):
    if not b3_hist:
        print("  [skip] No B3 history found for epoch-metrics plot.")
        return

    epochs     = [e["epoch"]      for e in b3_hist]
    exact      = [e["exact_match"] * 100 for e in b3_hist]
    mean_acc   = [e["mean_acc"]    * 100 for e in b3_hist]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    color_g = BASELINES["b3"]["color"]
    color_b = "#5B6ABF"

    ax.plot(epochs, exact,    "^-",  color=color_g, lw=2.2, label="Exact Match (%)")
    ax.plot(epochs, mean_acc, "o--", color=color_b, lw=2.2, label="Mean Accuracy (%)", alpha=0.8)

    # Fill area under exact match
    ax.fill_between(epochs, exact, alpha=0.12, color=color_g)

    # Per-metric accuracy thin lines
    if "per_metric_acc" in b3_hist[0]:
        for m in METRICS:
            vals = [e["per_metric_acc"].get(m, np.nan) * 100 for e in b3_hist]
            ax.plot(epochs, vals, "-", lw=0.9, alpha=0.25, color="#888888")

    # Annotate final values
    ax.annotate(
        f"{exact[-1]:.1f}%",
        xy=(epochs[-1], exact[-1]),
        xytext=(epochs[-1] - 0.05, exact[-1] - 3.5),
        fontsize=9, color=color_g, fontweight="bold",
        ha="right",
    )
    ax.annotate(
        f"{mean_acc[-1]:.1f}%",
        xy=(epochs[-1], mean_acc[-1]),
        xytext=(epochs[-1] - 0.05, mean_acc[-1] + 1.5),
        fontsize=9, color=color_b, fontweight="bold",
        ha="right",
    )

    ax.set_xticks(epochs)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score (%)")
    ax.set_title("B3: Multi-task BERT — Val Metrics per Epoch\n"
                 "(grey thin lines = individual CVSS metrics)")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(max(0, min(min(exact), min(mean_acc)) - 5), 100)

    _save_or_show(fig, out, show, "04_b3_epoch_metrics.png")


# ── Figure 5: Radar / spider chart ───────────────────────────────────────────

def plot_radar_chart(
    results: dict,
    split: str,
    out: Path | None,
    show: bool,
):
    keys = [k for k in ["b1", "b2", "b3"] if k in results]

    angles = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

    for key in keys:
        pma  = get_pma(results[key], split)
        vals = [pma.get(m, 0) * 100 for m in METRICS]
        vals += vals[:1]   # close

        ax.plot(
            angles, vals,
            color=BASELINES[key]["color"],
            linewidth=2, linestyle="solid",
            label=BASELINES[key]["label"],
            marker=BASELINES[key]["marker"], markersize=6,
        )
        ax.fill(angles, vals, color=BASELINES[key]["color"], alpha=0.08)

    # Axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([METRIC_LABELS[m].replace("\n", " ") for m in METRICS],
                       fontsize=9.5)
    ax.set_ylim(60, 100)
    ax.set_yticks([65, 75, 85, 95])
    ax.set_yticklabels(["65%", "75%", "85%", "95%"], fontsize=8, color="grey")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)

    ax.set_title(f"Per-Metric Accuracy — Radar Chart  [{split.upper()}]",
                 pad=18, fontsize=13)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.13),
              ncol=3, framealpha=0.9, fontsize=9.5)

    _save_or_show(fig, out, show, "05_radar_chart.png")


# ── Figure 6: Error metrics (MAE, Hamming, Critical FNR) ─────────────────────

def plot_error_metrics(
    results: dict,
    split: str,
    out: Path | None,
    show: bool,
):
    keys = [k for k in ["b1", "b2", "b3"] if k in results]

    panels = [
        ("score_mae",        "CVSS Score MAE\n(lower is better)",     False),
        ("hamming_distance", "Hamming Distance\n(lower is better)",   False),
        ("critical_fnr",     "Critical FNR (%)\n(lower is better)",   True),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

    for ax, (metric_key, ylabel, pct) in zip(axes, panels):
        vals  = [get_metric(results[k], split, metric_key) for k in keys]
        if pct:
            vals = [v * 100 for v in vals]
        colors = [BASELINES[k]["color"] for k in keys]
        labels = [BASELINES[k]["label"] for k in keys]

        bars = ax.bar(labels, vals, color=colors, edgecolor="white",
                      linewidth=0.7, zorder=3, width=0.55)

        # Value labels
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.015,
                f"{v:.2f}" + ("%" if pct else ""),
                ha="center", va="bottom", fontsize=9.5, color="#222",
            )

        # Best marker (min = best for all three)
        best_idx = int(np.nanargmin(vals))
        ax.patches[best_idx].set_edgecolor("#222222")
        ax.patches[best_idx].set_linewidth(2.0)
        ax.text(
            axes[0].get_xlim()[0] if False else 0,  # skip
            0, "",
        )

        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split("\n")[0])
        ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=8.5)
        margin = max(vals) * 0.18
        ax.set_ylim(0, max(vals) + margin)

    fig.suptitle(f"Error & Safety Metrics  [{split.upper()}]",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_or_show(fig, out, show, "06_error_metrics.png")


# ── Figure 7: Dashboard (all in one) ─────────────────────────────────────────

def plot_dashboard(
    results: dict,
    split: str,
    out: Path | None,
    show: bool,
):
    keys = [k for k in ["b1", "b2", "b3"] if k in results]

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        "CVSS v3.1 Prediction — Baseline Comparison Dashboard",
        fontsize=16, fontweight="bold", y=0.98,
    )

    # ── Layout: 2 rows × 3 cols ───────────────────────────────────────────
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.35)
    ax_bar   = fig.add_subplot(gs[0, :2])   # wide: per-metric grouped bars
    ax_summ  = fig.add_subplot(gs[0, 2])    # summary bars
    ax_radar = fig.add_subplot(gs[1, 0], polar=True)  # radar
    ax_mae   = fig.add_subplot(gs[1, 1])    # MAE + Hamming
    ax_fnr   = fig.add_subplot(gs[1, 2])    # Critical FNR

    # ── Per-metric accuracy (top left) ────────────────────────────────────
    n_bl    = len(keys)
    width   = 0.22
    x       = np.arange(len(METRICS))
    offsets = np.linspace(-(n_bl - 1) / 2, (n_bl - 1) / 2, n_bl) * width

    for i, key in enumerate(keys):
        pma  = get_pma(results[key], split)
        vals = [pma.get(m, np.nan) * 100 for m in METRICS]
        ax_bar.bar(x + offsets[i], vals, width,
                   label=BASELINES[key]["label"],
                   color=BASELINES[key]["color"],
                   edgecolor="white", linewidth=0.5, zorder=3)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([METRIC_LABELS[m].replace("\n", " ") for m in METRICS],
                            fontsize=8.5)
    ax_bar.set_ylabel("Accuracy (%)")
    ax_bar.set_ylim(60, 102)
    ax_bar.set_title("Per-Metric Accuracy")
    ax_bar.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax_bar.set_axisbelow(True)
    ax_bar.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # ── Summary metrics (top right) ───────────────────────────────────────
    summ_keys = ["mean_accuracy", "exact_match", "severity_accuracy"]
    summ_lbls = ["Mean\nAcc", "Exact\nMatch", "Severity\nAcc"]
    xs = np.arange(len(summ_keys))
    offsets2 = np.linspace(-(n_bl - 1) / 2, (n_bl - 1) / 2, n_bl) * 0.25

    for i, key in enumerate(keys):
        vals = [get_metric(results[key], split, k) * 100 for k in summ_keys]
        bars = ax_summ.bar(xs + offsets2[i], vals, 0.25,
                           color=BASELINES[key]["color"],
                           edgecolor="white", linewidth=0.5, zorder=3)
        for bar, v in zip(bars, vals):
            if not math.isnan(v):
                ax_summ.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7.5,
                )

    ax_summ.set_xticks(xs)
    ax_summ.set_xticklabels(summ_lbls, fontsize=9)
    ax_summ.set_ylim(40, 105)
    ax_summ.set_ylabel("Score (%)")
    ax_summ.set_title("Summary Metrics")
    ax_summ.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax_summ.set_axisbelow(True)
    ax_summ.spines["top"].set_visible(False)
    ax_summ.spines["right"].set_visible(False)

    # ── Radar (bottom left) ────────────────────────────────────────────────
    angles = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False).tolist()
    angles += angles[:1]

    for key in keys:
        pma  = get_pma(results[key], split)
        vals = [pma.get(m, 0) * 100 for m in METRICS] + [pma.get(METRICS[0], 0) * 100]
        ax_radar.plot(angles, vals,
                      color=BASELINES[key]["color"], lw=2,
                      label=BASELINES[key]["label"])
        ax_radar.fill(angles, vals, color=BASELINES[key]["color"], alpha=0.07)

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels([m for m in METRICS], fontsize=9)
    ax_radar.set_ylim(60, 100)
    ax_radar.set_yticks([70, 80, 90])
    ax_radar.set_yticklabels(["70", "80", "90"], fontsize=7, color="grey")
    ax_radar.set_title("Radar Chart", pad=12, fontsize=10)

    # ── MAE + Hamming side by side (bottom middle) ─────────────────────────
    ax_mae2 = ax_mae.twinx()
    bar_w   = 0.3
    xpos    = np.arange(n_bl)
    mae_v   = [get_metric(results[k], split, "score_mae")        for k in keys]
    ham_v   = [get_metric(results[k], split, "hamming_distance") for k in keys]
    colors  = [BASELINES[k]["color"] for k in keys]

    b1 = ax_mae.bar(xpos - bar_w / 2, mae_v, bar_w, color=colors,
                    edgecolor="white", label="Score MAE", alpha=0.9, zorder=3)
    b2 = ax_mae2.bar(xpos + bar_w / 2, ham_v, bar_w, color=colors,
                     edgecolor="white", label="Hamming Dist", alpha=0.55,
                     linewidth=1.2, hatch="///", zorder=3)

    ax_mae.set_xticks(xpos)
    ax_mae.set_xticklabels([BASELINES[k]["label"].split(":")[0] for k in keys], fontsize=9)
    ax_mae.set_ylabel("Score MAE", fontsize=9)
    ax_mae2.set_ylabel("Hamming Distance", fontsize=9)
    ax_mae.set_title("Error Metrics\n(solid=MAE, hatched=Hamming)", fontsize=9.5)
    ax_mae.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax_mae.set_axisbelow(True)
    ax_mae.spines["top"].set_visible(False)

    # ── Critical FNR (bottom right) ────────────────────────────────────────
    fnr_v = [get_metric(results[k], split, "critical_fnr") * 100 for k in keys]
    ax_fnr.bar(
        [BASELINES[k]["label"].split(":")[0] for k in keys],
        fnr_v,
        color=[BASELINES[k]["color"] for k in keys],
        edgecolor="white", linewidth=0.6, zorder=3, width=0.55,
    )
    for i, (lbl, v) in enumerate(zip(keys, fnr_v)):
        ax_fnr.text(i, v + 0.3, f"{v:.1f}%",
                    ha="center", va="bottom", fontsize=9)

    ax_fnr.axhline(5, color="#cc3333", linestyle="--", lw=1.4, label="Target ≤5%")
    ax_fnr.set_ylabel("Rate (%)")
    ax_fnr.set_title("Critical FNR\n(score≥7.0 predicted as <7.0)", fontsize=9.5)
    ax_fnr.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax_fnr.set_axisbelow(True)
    ax_fnr.legend(fontsize=8)
    ax_fnr.spines["top"].set_visible(False)
    ax_fnr.spines["right"].set_visible(False)

    _save_or_show(fig, out, show, "07_dashboard.png")


# ── Save / show helper ────────────────────────────────────────────────────────

def _save_or_show(fig: plt.Figure, out_dir: Path | None, show: bool, filename: str):
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        path = Path(out_dir) / filename
        fig.savefig(path)
        print(f"  Saved -> {path}")
    if show:
        plt.show()
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize CVSS baseline results"
    )
    parser.add_argument("--results-dir",  default="results",
                        help="Directory with b1.json / b2.json / b3.json")
    parser.add_argument("--result-b1",   default=None,
                        help="Override path for baseline-1 result JSON")
    parser.add_argument("--result-b2",   default=None,
                        help="Override path for baseline-2 result JSON")
    parser.add_argument("--result-b3",   default=None,
                        help="Override path for baseline-3 result JSON")
    parser.add_argument("--history-b2",  default=None,
                        help="Directory with history_AV.json … (models/baseline2)")
    parser.add_argument("--history-b3",  default=None,
                        help="Path to models/baseline3/history.json")
    parser.add_argument("--out-dir",     default="plots",
                        help="Output directory for PNG files")
    parser.add_argument("--split",       default="test",
                        choices=["val", "test"],
                        help="Which evaluation split to visualize")
    parser.add_argument("--show",        action="store_true",
                        help="Display figures interactively (in addition to saving)")
    parser.add_argument("--no-save",     action="store_true",
                        help="Do not save — only show (useful with --show)")
    args = parser.parse_args()

    rdir = Path(args.results_dir)
    out  = None if args.no_save else Path(args.out_dir)

    # ── Load result JSONs ──────────────────────────────────────────────────
    r_b1 = load_result(Path(args.result_b1) if args.result_b1 else rdir / "b1.json")
    r_b2 = load_result(Path(args.result_b2) if args.result_b2 else rdir / "b2.json")
    r_b3 = load_result(Path(args.result_b3) if args.result_b3 else rdir / "b3.json")

    results: dict = {}
    if r_b1: results["b1"] = r_b1
    if r_b2: results["b2"] = r_b2
    if r_b3: results["b3"] = r_b3

    if not results:
        print("ERROR: no result files found. Check --results-dir or --result-b* paths.")
        return

    found = ", ".join(results.keys())
    print(f"Loaded results: {found}  (split={args.split})")

    # ── Load history ───────────────────────────────────────────────────────
    b2_hist = load_b2_history(
        Path(args.history_b2) if args.history_b2 else None
    )
    b3_hist = load_b3_history(
        Path(args.history_b3) if args.history_b3
        else Path("models/baseline3/history.json")
    )

    if b2_hist: print(f"  B2 history: {len(b2_hist)} metrics loaded")
    if b3_hist: print(f"  B3 history: {len(b3_hist)} epochs loaded")

    # ── Generate plots ─────────────────────────────────────────────────────
    print(f"\nGenerating plots -> {out or '(display only)'}")

    print("  [1/7] Per-metric accuracy...")
    plot_per_metric_accuracy(results, args.split, out, args.show)

    print("  [2/7] Summary metrics...")
    plot_summary_metrics(results, args.split, out, args.show)

    print("  [3/7] Training curves...")
    plot_training_curves(b2_hist, b3_hist, out, args.show)

    print("  [4/7] B3 epoch metrics...")
    plot_b3_epoch_metrics(b3_hist, out, args.show)

    print("  [5/7] Radar chart...")
    plot_radar_chart(results, args.split, out, args.show)

    print("  [6/7] Error metrics...")
    plot_error_metrics(results, args.split, out, args.show)

    print("  [7/7] Dashboard...")
    plot_dashboard(results, args.split, out, args.show)

    print("\nDone.")


if __name__ == "__main__":
    main()
