import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]

METRIC_LABELS = {
    "AV": {"N": "Network", "A": "Adjacent", "L": "Local", "P": "Physical"},
    "AC": {"L": "Low", "H": "High"},
    "PR": {"N": "None", "L": "Low", "H": "High"},
    "UI": {"N": "None", "R": "Required"},
    "S":  {"U": "Unchanged", "C": "Changed"},
    "C":  {"N": "None", "L": "Low", "H": "High"},
    "I":  {"N": "None", "L": "Low", "H": "High"},
    "A":  {"N": "None", "L": "Low", "H": "High"},
}


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def analyze_nvd(records: list[dict]) -> dict:
    stats = {
        "total": len(records),
        "metrics_distribution": {},
        "severity_distribution": Counter(),
        "cwe_distribution": Counter(),
        "year_distribution": Counter(),
        "desc_len": [],
        "base_scores": [],
    }

    metric_counters = {m: Counter() for m in METRICS}

    for rec in records:
        metrics = rec.get("metrics", {})
        for m in METRICS:
            val = metrics.get(m)
            if val:
                metric_counters[m][val] += 1

        stats["severity_distribution"][rec.get("base_severity", "UNKNOWN")] += 1

        for cwe in rec.get("cwes", []):
            stats["cwe_distribution"][cwe] += 1

        pub = rec.get("published", "")
        if pub:
            year = pub[:4]
            stats["year_distribution"][year] += 1

        desc = rec.get("description", "")
        stats["desc_len"].append(len(desc.split()))

        score = rec.get("base_score")
        if score is not None:
            stats["base_scores"].append(float(score))

    stats["metrics_distribution"] = {m: dict(c) for m, c in metric_counters.items()}

    # Дисбаланс: отношение мажоритарного класса к миноритарному
    imbalance = {}
    for m, c in metric_counters.items():
        if c:
            counts = sorted(c.values(), reverse=True)
            ratio = counts[0] / counts[-1] if counts[-1] > 0 else float("inf")
            imbalance[m] = round(ratio, 2)
    stats["class_imbalance_ratio"] = imbalance

    # Desc stats
    lens = np.array(stats["desc_len"])
    stats["desc_stats"] = {
        "mean": float(np.mean(lens)),
        "median": float(np.median(lens)),
        "p95": float(np.percentile(lens, 95)),
        "max": int(np.max(lens)),
    }

    scores = np.array(stats["base_scores"])
    stats["score_stats"] = {
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "critical_pct": float(np.mean(scores >= 9.0) * 100),
        "high_pct": float(np.mean((scores >= 7.0) & (scores < 9.0)) * 100),
    }

    return stats


def plot_metric_distributions(stats: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    dist = stats["metrics_distribution"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("CVSS v3.1 Metric Distributions", fontsize=14)

    for ax, metric in zip(axes.flat, METRICS):
        counts = dist.get(metric, {})
        labels = list(counts.keys())
        values = [counts[k] for k in labels]

        # Полные метки
        full_labels = [METRIC_LABELS[metric].get(k, k) for k in labels]
        bars = ax.bar(full_labels, values, color="steelblue", edgecolor="white")
        ax.set_title(f"{metric}")
        ax.set_ylabel("Count")

        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{v:,}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.tick_params(axis="x", labelsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "metric_distributions.png", dpi=150)
    plt.close()
    print(f"Сохранено: {out_dir / 'metric_distributions.png'}")


def plot_yearly(stats: dict, out_dir: Path):
    year_dist = stats["year_distribution"]
    years = sorted(year_dist.keys())
    counts = [year_dist[y] for y in years]

    plt.figure(figsize=(10, 4))
    plt.bar(years, counts, color="teal")
    plt.title("CVE с CVSS v3.1 по годам")
    plt.xlabel("Год")
    plt.ylabel("Количество")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / "yearly_distribution.png", dpi=150)
    plt.close()


def print_summary(stats: dict):
    print("\n" + "=" * 60)
    print(f"  Датасет: {stats['total']:,} записей")
    print("=" * 60)

    print("\n[Severity]")
    for sev, cnt in sorted(stats["severity_distribution"].items(), key=lambda x: -x[1]):
        pct = cnt / stats["total"] * 100
        print(f"  {sev:<12} {cnt:>7,}  ({pct:.1f}%)")

    print("\n[Распределение по метрикам]")
    for m in METRICS:
        counts = stats["metrics_distribution"].get(m, {})
        total = sum(counts.values())
        print(f"  {m}:")
        for val, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            label = METRIC_LABELS[m].get(val, val)
            pct = cnt / total * 100 if total else 0
            print(f"    {val} ({label:<12}) {cnt:>7,}  ({pct:.1f}%)")

    print("\n[Дисбаланс классов (max/min)]")
    for m, ratio in stats["class_imbalance_ratio"].items():
        flag = " ← высокий!" if ratio > 10 else ""
        print(f"  {m}: {ratio:.1f}x{flag}")

    print("\n[Описания (слов)]")
    ds = stats["desc_stats"]
    print(f"  mean={ds['mean']:.0f}, median={ds['median']:.0f}, p95={ds['p95']:.0f}, max={ds['max']}")

    print("\n[Top-20 CWE]")
    for cwe, cnt in stats["cwe_distribution"].most_common(20):
        print(f"  {cwe:<15} {cnt:>6,}")

    print("\n[CVSS Base Score]")
    ss = stats["score_stats"]
    print(f"  mean={ss['mean']:.2f}, CRITICAL(≥9)={ss['critical_pct']:.1f}%, HIGH(7-9)={ss['high_pct']:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Анализ датасета CVE/CVSS")
    parser.add_argument("--nvd", required=True, help="nvd_cves.jsonl")
    parser.add_argument("--patches", default=None, help="patches.jsonl (опционально)")
    parser.add_argument("--out-dir", default="plots", help="Папка для графиков")
    parser.add_argument("--stats-out", default="stats.json")
    args = parser.parse_args()

    print(f"Загружаю {args.nvd}...")
    records = load_jsonl(Path(args.nvd))
    stats = analyze_nvd(records)

    # Покрытие патчами
    if args.patches:
        patch_records = load_jsonl(Path(args.patches))
        patched_cves = {r.get("cve_id") for r in patch_records if r.get("cve_id")}
        nvd_cves = {r.get("cve_id") for r in records}
        overlap = patched_cves & nvd_cves
        stats["patch_coverage"] = {
            "patch_records": len(patch_records),
            "unique_cves_with_patch": len(patched_cves),
            "overlap_with_nvd": len(overlap),
            "coverage_pct": round(len(overlap) / len(nvd_cves) * 100, 2),
        }
        print(f"\nПокрытие патчами: {len(overlap):,}/{len(nvd_cves):,} CVE ({stats['patch_coverage']['coverage_pct']}%)")

    print_summary(stats)

    out_dir = Path(args.out_dir)
    plot_metric_distributions(stats, out_dir)
    plot_yearly(stats, out_dir)

    # Сохраняем stats (Counter → dict)
    stats_serializable = json.loads(
        json.dumps(stats, default=lambda x: dict(x) if isinstance(x, Counter) else str(x))
    )
    with open(args.stats_out, "w", encoding="utf-8") as f:
        json.dump(stats_serializable, f, ensure_ascii=False, indent=2)
    print(f"\nСтатистика сохранена: {args.stats_out}")


if __name__ == "__main__":
    main()
