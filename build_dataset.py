import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


METRICS = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_patch_index(patch_files: list[Path]) -> dict[str, list[dict]]:
    """
    cve_id → список патч-объектов: {"diff": str, "repo": str, "sha": str, ...}
    Поддерживает форматы:
      - старый linux: patch_diff + patch_repo + patch_commit_sha
      - refs/refs-ext/osv: patches = [{diff, repo, sha, platform, url}, ...]
    """
    index: dict[str, list[dict]] = {}
    for path in patch_files:
        if not path.exists():
            log.warning(f"Файл патчей не найден: {path}")
            continue
        for rec in load_jsonl(path):
            cve_id = rec.get("cve_id")
            if not cve_id:
                continue

            # Формат 1: linux collect_patches (patch_diff на верхнем уровне)
            if rec.get("patch_diff"):
                index.setdefault(cve_id, []).append({
                    "diff":     rec["patch_diff"],
                    "repo":     rec.get("patch_repo", ""),
                    "sha":      rec.get("patch_commit_sha", ""),
                    "platform": "github",
                    "url":      rec.get("patch_commit_url", ""),
                })
                continue

            # Формат 2: patches = [{diff, repo, sha, platform?, url?}, ...]
            for p in rec.get("patches", []):
                d = p.get("diff", "")
                if d:
                    index.setdefault(cve_id, []).append({
                        "diff":     d,
                        "repo":     p.get("repo", ""),
                        "sha":      p.get("sha", ""),
                        "platform": p.get("platform", ""),
                        "url":      p.get("url", ""),
                    })
    return index


def merge_records(nvd_records: list[dict], patch_index: dict[str, list[dict]]) -> list[dict]:
    merged = []
    for rec in nvd_records:
        cve_id = rec.get("cve_id", "")
        patches = patch_index.get(cve_id, [])

        # Основной diff — первый найденный; дополнительные — для ablation / multi-patch экспериментов
        best = patches[0] if patches else None
        out = {
            "cve_id": cve_id,
            "description": rec.get("description", ""),
            "vector_string": rec.get("vector_string", ""),
            "base_score": rec.get("base_score"),
            "base_severity": rec.get("base_severity", ""),
            "metrics": rec.get("metrics", {}),
            "cwes": rec.get("cwes", []),
            "published": rec.get("published", ""),
            "has_patch": bool(patches),
            # Основной diff (строка) — для baseline-моделей
            "patch_diff": best["diff"] if best else None,
            # Метаданные первого патча
            "patch_repo":     best["repo"] if best else None,
            "patch_sha":      best["sha"]  if best else None,
            "patch_platform": best["platform"] if best else None,
            # Все патчи — для мультипатч-экспериментов
            "all_patches": [
                {"diff": p["diff"], "repo": p["repo"], "sha": p["sha"],
                 "platform": p["platform"], "url": p["url"]}
                for p in patches
            ],
        }
        merged.append(out)
    return merged


def stratified_split(
    records: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list, list, list]:
    """
    Стратифицированный сплит по вектору CVSS (или просто random если векторов мало).
    Стратифицируем по AV*Severity — самые важные для баланса.
    """
    random.seed(seed)

    # Группировка по страте: severity + AV
    strata: dict[str, list] = {}
    for rec in records:
        metrics = rec.get("metrics", {})
        av = metrics.get("AV", "?")
        sev = rec.get("base_severity", "?")
        key = f"{sev}_{av}"
        strata.setdefault(key, []).append(rec)

    train, val, test = [], [], []
    for key, items in strata.items():
        random.shuffle(items)
        n = len(items)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


def save_jsonl(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def compute_label_counts(records: list[dict]) -> dict:
    counts = {m: Counter() for m in METRICS}
    for rec in records:
        metrics = rec.get("metrics", {})
        for m in METRICS:
            v = metrics.get(m)
            if v:
                counts[m][v] += 1
    return {m: dict(c) for m, c in counts.items()}


def main():
    parser = argparse.ArgumentParser(description="Сборка финального датасета")
    parser.add_argument("--nvd", required=True, help="nvd_cves.jsonl")
    parser.add_argument("--patches", nargs="*", default=[], help="patch jsonl files")
    parser.add_argument("--out-dir", default="dataset", help="Выходная папка")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Загружаю NVD данные...")
    nvd = load_jsonl(Path(args.nvd))
    print(f"  {len(nvd):,} записей с CVSS v3.1")

    patch_index = {}
    if args.patches:
        print("Загружаю патчи...")
        patch_index = build_patch_index([Path(p) for p in args.patches])
        print(f"  Диффы для {len(patch_index):,} CVE")

    print("Объединяю...")
    merged = merge_records(nvd, patch_index)
    with_patch = sum(1 for r in merged if r["has_patch"])
    print(f"  Итого: {len(merged):,} записей, из них с диффом: {with_patch:,} ({with_patch/len(merged)*100:.1f}%)")

    print("Разбиваю на train/val/test...")
    train, val, test = stratified_split(merged, args.train_ratio, args.val_ratio, args.seed)
    print(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

    out_dir = Path(args.out_dir)
    save_jsonl(train, out_dir / "train.jsonl")
    save_jsonl(val, out_dir / "val.jsonl")
    save_jsonl(test, out_dir / "test.jsonl")

    meta = {
        "total": len(merged),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "with_patch": with_patch,
        "patch_coverage_pct": round(with_patch / len(merged) * 100, 2),
        "train_label_counts": compute_label_counts(train),
        "val_label_counts": compute_label_counts(val),
        "test_label_counts": compute_label_counts(test),
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nДатасет сохранён в {out_dir}/")
    print(f"  train.jsonl, val.jsonl, test.jsonl, meta.json")


if __name__ == "__main__":
    main()
