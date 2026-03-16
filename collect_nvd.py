import argparse
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

NVD_BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"

# CVSS v3.1 метрики — все компоненты вектора
CVSS31_METRICS = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]

CVSS31_VALUES = {
    "AV": ["N", "A", "L", "P"],       # Network, Adjacent, Local, Physical
    "AC": ["L", "H"],                  # Low, High
    "PR": ["N", "L", "H"],            # None, Low, High
    "UI": ["N", "R"],                  # None, Required
    "S":  ["U", "C"],                  # Unchanged, Changed
    "C":  ["N", "L", "H"],            # None, Low, High
    "I":  ["N", "L", "H"],
    "A":  ["N", "L", "H"],
}


def parse_cvss31_vector(vector_string: str) -> dict | None:
    """
    'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H'
    → {'AV': 'N', 'AC': 'L', ...}
    """
    if not vector_string or not vector_string.startswith("CVSS:3.1/"):
        return None
    parts = vector_string.split("/")[1:]  # убрать 'CVSS:3.1'
    metrics = {}
    for part in parts:
        k, v = part.split(":")
        metrics[k] = v
    # валидация полноты вектора
    if not all(m in metrics for m in CVSS31_METRICS):
        return None
    return metrics


def extract_cve_record(item: dict) -> dict | None:
    """Из NVD CVE item извлечь нужные поля."""
    cve = item.get("cve", {})
    cve_id = cve.get("id", "")

    # Описание (en)
    descriptions = cve.get("descriptions", [])
    description = ""
    for d in descriptions:
        if d.get("lang") == "en":
            description = d.get("value", "")
            break

    # CVSS v3.1
    metrics = cve.get("metrics", {})
    cvss31_data = metrics.get("cvssMetricV31", [])
    if not cvss31_data:
        return None  # нет v3.1 — пропускаем

    primary = next((m for m in cvss31_data if m.get("type") == "Primary"), cvss31_data[0])
    cvss_data = primary.get("cvssData", {})
    vector_string = cvss_data.get("vectorString", "")
    base_score = cvss_data.get("baseScore")
    base_severity = cvss_data.get("baseSeverity", "")

    parsed_vector = parse_cvss31_vector(vector_string)
    if parsed_vector is None:
        return None

    # References
    refs = [r.get("url", "") for r in cve.get("references", [])]

    # Weaknesses (CWE)
    cwes = []
    for w in cve.get("weaknesses", []):
        for wd in w.get("description", []):
            val = wd.get("value", "")
            if val.startswith("CWE-"):
                cwes.append(val)

    # Даты
    published = cve.get("published", "")
    modified = cve.get("lastModified", "")

    return {
        "cve_id": cve_id,
        "description": description,
        "vector_string": vector_string,
        "base_score": base_score,
        "base_severity": base_severity,
        "metrics": parsed_vector,
        "cwes": cwes,
        "references": refs,
        "published": published,
        "last_modified": modified,
    }


def fetch_page(session: requests.Session, params: dict, api_key: str | None) -> dict:
    headers = {}
    if api_key:
        headers["apiKey"] = api_key

    for attempt in range(5):
        try:
            r = session.get(NVD_BASE, params=params, headers=headers, timeout=30)
            if r.status_code == 429:
                wait = 35
                log.warning(f"Rate limited, ждём {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            log.warning(f"Ошибка запроса (попытка {attempt+1}): {e}")
            time.sleep(10 * (attempt + 1))

    raise RuntimeError("Не удалось получить страницу после 5 попыток")


def date_chunks(start_year: int, end_year: int, chunk_days: int = 90):
    """
    NVD API 2.0 ограничение: максимум 120 дней на запрос.
    Разбиваем диапазон годов на чанки по chunk_days дней.
    """
    start = datetime(start_year, 1, 1, tzinfo=timezone.utc)
    end   = datetime(end_year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    cur = start
    while cur < end:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), end)
        yield (
            cur.strftime("%Y-%m-%dT%H:%M:%S.000"),
            chunk_end.strftime("%Y-%m-%dT%H:%M:%S.999"),
        )
        cur = chunk_end + timedelta(seconds=1)


def collect_nvd(
    api_key: str | None,
    start_year: int,
    end_year: int,
    out_path: Path,
    results_per_page: int = 2000,
    chunk_days: int = 90,
):
    session = requests.Session()
    # NVD рекомендует не быстрее 1 req/6 sec без ключа, 1 req/0.6 sec с ключом
    sleep_between = 0.7 if api_key else 6.5

    total_collected = 0
    skipped_no_cvss31 = 0
    chunks = list(date_chunks(start_year, end_year, chunk_days))
    log.info(f"Итого чанков (по {chunk_days} дней): {len(chunks)}")

    with out_path.open("w", encoding="utf-8") as fout:
        for pub_start, pub_end in chunks:
            params = {
                "pubStartDate": pub_start,
                "pubEndDate": pub_end,
                "resultsPerPage": results_per_page,
                "startIndex": 0,
            }
            data = fetch_page(session, params, api_key)
            total = data.get("totalResults", 0)
            log.info(f"{pub_start[:10]} → {pub_end[:10]}: {total} CVE")

            with tqdm(total=total, desc=pub_start[:10], unit="CVE", leave=False) as pbar:
                start_idx = 0
                while True:
                    for item in data.get("vulnerabilities", []):
                        record = extract_cve_record(item)
                        if record is None:
                            skipped_no_cvss31 += 1
                        else:
                            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                            total_collected += 1
                        pbar.update(1)

                    fetched = start_idx + len(data.get("vulnerabilities", []))
                    if fetched >= total:
                        break

                    start_idx = fetched
                    params["startIndex"] = start_idx
                    time.sleep(sleep_between)
                    data = fetch_page(session, params, api_key)

            time.sleep(sleep_between)

    log.info(f"Готово. Собрано: {total_collected}, пропущено (нет CVSS 3.1): {skipped_no_cvss31}")
    log.info(f"Файл: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Сбор CVE с CVSS v3.1 из NVD")
    parser.add_argument("--api-key", default=None, help="NVD API key (опционально)")
    parser.add_argument("--start", type=int, default=2018, help="Начальный год")
    parser.add_argument("--end", type=int, default=2024, help="Конечный год")
    parser.add_argument("--out", default="nvd_cves.jsonl", help="Выходной файл .jsonl")
    parser.add_argument("--results-per-page", type=int, default=2000)
    parser.add_argument("--chunk-days", type=int, default=90,
                        help="Размер чанка в днях (max 120, NVD ограничение)")
    args = parser.parse_args()

    # Пустой API key → None
    api_key = args.api_key if args.api_key else None

    out = Path(args.out)
    collect_nvd(
        api_key=api_key,
        start_year=args.start,
        end_year=args.end,
        out_path=out,
        results_per_page=args.results_per_page,
        chunk_days=args.chunk_days,
    )


if __name__ == "__main__":
    main()
