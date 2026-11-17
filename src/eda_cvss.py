"""Audit the historical CVSS splits and render aggregate EDA figures."""

import argparse
import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from baselines.common import METRIC_CLASSES, METRICS, cvss31_score, score_to_severity


QUALITY_FIELDS = (
    "malformed_id",
    "invalid_date",
    "blank_description",
    "cvss_in_description",
    "invalid_labels",
    "vector_mismatch",
    "score_mismatch",
    "severity_mismatch",
    "without_cwe",
    "with_diff",
    "patch_flag_mismatch",
)
SEVERITIES = ("NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL")


def read_split(path):
    digest = hashlib.sha256()
    with path.open(encoding="utf-8") as source:
        records = []
        for line in source:
            digest.update(line.encode("utf-8"))
            if line.strip():
                records.append(json.loads(line))
    return records, digest.hexdigest()


def audit(paths):
    splits = {}
    by_year = defaultdict(Counter)
    labels_by_year = defaultdict(lambda: {metric: Counter() for metric in METRICS})
    ids_by_split = defaultdict(set)
    texts_by_split = defaultdict(Counter)
    vectors_by_text = defaultdict(set)
    text_occurrences = Counter()
    for split, path in paths.items():
        records, digest = read_split(path)
        issues = Counter(dict.fromkeys(QUALITY_FIELDS, 0))
        years, severity, points, ids = Counter(), Counter(), Counter(), Counter()
        patch_severity = Counter()
        labels = {metric: Counter() for metric in METRICS}
        dates, lengths = [], []
        for row in records:
            cve = row.get("cve_id") or ""
            if re.fullmatch(r"CVE-\d{4}-\d{4,}", cve):
                ids[cve] += 1
            else:
                issues["malformed_id"] += 1
            year = None
            try:
                date = datetime.fromisoformat(row.get("published") or "").date()
                dates.append(date)
                year = str(date.year)
                years[year] += 1
                by_year[year][split] += 1
            except ValueError:
                issues["invalid_date"] += 1

            text = " ".join((row.get("description") or "").casefold().split())
            issues["blank_description"] += not text
            issues["cvss_in_description"] += bool(re.search(r"\bcvss\b", text))
            if text:
                lengths.append(len(text.split()))
                texts_by_split[split][text] += 1
                text_occurrences[text] += 1

            recorded_severity = row.get("base_severity") or "MISSING"
            severity[recorded_severity] += 1
            diff = bool((row.get("patch_diff") or "").strip())
            issues["with_diff"] += diff
            issues["patch_flag_mismatch"] += bool(row.get("has_patch")) != diff
            issues["without_cwe"] += not row.get("cwes")
            if diff:
                patch_severity[recorded_severity] += 1

            metrics = row.get("metrics") or {}
            valid = all(metrics.get(metric) in METRIC_CLASSES[metric] for metric in METRICS)
            issues["invalid_labels"] += not valid
            for metric in METRICS:
                value = metrics.get(metric, "MISSING")
                labels[metric][value] += 1
                if year:
                    labels_by_year[year][metric][value] += 1
            if not valid:
                continue
            version, *parts = (row.get("vector_string") or "").split("/")
            vector_fields = [part.partition(":") for part in parts]
            vector = {key: value for key, colon, value in vector_fields if colon}
            issues["vector_mismatch"] += version != "CVSS:3.1" or any(
                vector.get(metric) != metrics[metric]
                or sum(field[0] == metric for field in vector_fields) != 1
                for metric in METRICS
            )
            expected_score = cvss31_score(metrics)
            issues["score_mismatch"] += row.get("base_score") != expected_score
            issues["severity_mismatch"] += recorded_severity != score_to_severity(expected_score)
            if isinstance(row.get("base_score"), (int, float)):
                points[row["base_score"]] += 1
            if text:
                vectors_by_text[text].add(tuple(metrics[metric] for metric in METRICS))

        ids_by_split[split] = set(ids)
        text_lengths = {}
        if lengths:
            text_lengths = {
                "p05": float(np.percentile(lengths, 5)),
                "median": float(np.median(lengths)),
                "p95": float(np.percentile(lengths, 95)),
                "max": max(lengths),
            }
        splits[split] = {
            "rows": len(records),
            "sha256_utf8_lf": digest,
            "date_min": min(dates).isoformat() if dates else None,
            "date_max": max(dates).isoformat() if dates else None,
            "by_year": years,
            "quality": issues,
            "duplicate_id_excess_rows": sum(count - 1 for count in ids.values()),
            "duplicate_text_excess_rows": sum(
                count - 1 for count in texts_by_split[split].values()
            ),
            "text_word_counts": text_lengths,
            "label_counts": labels,
            "severity_counts": severity,
            "severity_counts_with_diff": patch_severity,
            "base_score_counts": dict(sorted(points.items())),
        }

    overlap = {}
    for left, right in itertools.combinations(splits, 2):
        shared = texts_by_split[left].keys() & texts_by_split[right].keys()
        overlap[f"{left}/{right}"] = {
            "shared_ids": len(ids_by_split[left] & ids_by_split[right]),
            "shared_text_groups": len(shared),
            "left_rows_with_shared_text": sum(texts_by_split[left][text] for text in shared),
            "right_rows_with_shared_text": sum(texts_by_split[right][text] for text in shared),
        }
    conflicts = [text for text, vectors in vectors_by_text.items() if len(vectors) > 1]
    return {
        "rows": sum(split["rows"] for split in splits.values()),
        "splits": splits,
        "by_year": dict(sorted(by_year.items())),
        "labels_by_year": dict(sorted(labels_by_year.items())),
        "overlap": overlap,
        "duplicate_text_groups": sum(count > 1 for count in text_occurrences.values()),
        "duplicate_text_groups_with_conflicting_vectors": len(conflicts),
        "rows_in_conflicting_text_groups": sum(text_occurrences[text] for text in conflicts),
    }


def render(stats, out):
    plt.switch_backend("Agg")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    splits = stats["splits"]
    colors = dict(zip(splits, ("#276582", "#44a589", "#c6802a")))
    years = list(stats["by_year"])
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    bottom = np.zeros(len(years), dtype=int)
    for split in splits:
        counts = np.array([stats["by_year"][year].get(split, 0) for year in years])
        ax.bar(years, counts, bottom=bottom, label=split, color=colors[split])
        bottom += counts
    ax.set(title="Публикации CVE по годам и исходным сплитам", ylabel="Число записей")
    ax.bar_label(
        ax.containers[-1],
        labels=[f"{total:,}".replace(",", " ") for total in bottom],
        padding=4,
    )
    ax.set_ylim(0, max(bottom) * 1.16)
    ax.legend(frameon=False)
    fig.savefig(out / "publication_years.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(10, 5), sharey=True, layout="constrained")
    fig.suptitle("Распределение классов CVSS v3.1 по сплитам")
    fig.supylabel("Доля записей в сплите, %")
    for ax, metric in zip(axes.flat, METRICS):
        classes = METRIC_CLASSES[metric]
        x = np.arange(len(classes))
        width = 0.75 / len(splits)
        for idx, (split, data) in enumerate(splits.items()):
            values = [100 * data["label_counts"][metric].get(c, 0) / data["rows"] for c in classes]
            offset = (idx - (len(splits) - 1) / 2) * width
            ax.bar(x + offset, values, width, color=colors[split], label=split)
        ax.set(title=metric, xticks=x, xticklabels=classes, ylim=(0, 100))
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0, -1].legend(frameon=False)
    fig.savefig(out / "class_balance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for metric, value in (("AV", "P"), ("AC", "H"), ("S", "C"), ("A", "L")):
        rates = [
            100
            * stats["labels_by_year"][year][metric].get(value, 0)
            / sum(stats["by_year"][year].values())
            for year in years
        ]
        ax.plot(years, rates, marker="o", label=f"{metric}:{value}")
    ax.set(title="Доли отдельных классов по году публикации", ylabel="Доля записей в году, %")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    fig.savefig(out / "label_drift.png", dpi=150)
    plt.close(fig)

    all_counts, with_diff = Counter(), Counter()
    for data in splits.values():
        all_counts.update(data["severity_counts"])
        with_diff.update(data["severity_counts_with_diff"])
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    present = [severity for severity in SEVERITIES if all_counts[severity]]
    coverage = [100 * with_diff[level] / all_counts[level] for level in present]
    bars = ax.bar(present, coverage, color="#276582")
    ax.bar_label(
        bars,
        labels=[
            f"{rate:.2f}%\n{with_diff[level]}/{all_counts[level]}"
            for level, rate in zip(present, coverage)
        ],
        padding=4,
    )
    ax.set(title="Доступность diff по категории severity", ylabel="Доля записей с diff, %")
    ax.set_ylim(0, max(max(coverage, default=0) * 1.35, 0.1))
    fig.savefig(out / "patch_coverage.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--train", type=Path, help="Path to a local historical train.jsonl")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/eda"))
    args = parser.parse_args()
    paths = {split: args.data_dir / f"{split}.jsonl" for split in ("train", "val", "test")}
    if args.train:
        paths["train"] = args.train
    elif not paths["train"].is_file():
        del paths["train"]
    stats = audit(paths)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    render(stats, args.out_dir)


if __name__ == "__main__":
    main()
