from __future__ import annotations

import argparse
import base64
import json
import logging
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MAX_DIFF_CHARS = 60_000   

def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def save_jsonl(records: list[dict], path: Path):
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"Сохранено {len(records)} записей → {path}")


def make_session(gh_token: str | None = None) -> requests.Session:
    s = requests.Session()
    if gh_token:
        s.headers.update({
            "Authorization": f"token {gh_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
    s.headers.update({"User-Agent": "cvss-research-bot/1.0"})
    return s


def safe_get(session: requests.Session, url: str,
             headers: dict | None = None, timeout: int = 20,
             retries: int = 3) -> requests.Response | None:
    for attempt in range(retries):
        try:
            r = session.get(url, headers=headers, timeout=timeout)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 30))
                log.warning(f"Rate limit, ждём {wait}s…")
                time.sleep(wait)
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                log.debug(f"Не удалось получить {url}: {e}")
    return None

SHA_RE   = r"[0-9a-f]{7,40}"
SHA40_RE = r"[0-9a-f]{40}"

URL_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    # GitHub commit
    ("github",
     re.compile(rf"github\.com/([^/]+/[^/]+)/commit/({SHA_RE})", re.I),
     r"\1|\2"),

    # GitHub pull request commit  github.com/A/B/pull/123/commits/SHA
    ("github",
     re.compile(rf"github\.com/([^/]+/[^/]+)/pull/\d+/commits/({SHA_RE})", re.I),
     r"\1|\2"),

    # kernel.org cgit: /commit/?id=SHA  or  /commit/?h=...&id=SHA
    ("kernel_cgit",
     re.compile(rf"git\.kernel\.org/[^?#]+\?(?:[^&]*&)*id=({SHA_RE})", re.I),
     r"torvalds/linux|\1"),

    # kernel.org /linus/SHA
    ("kernel_cgit",
     re.compile(rf"git\.kernel\.org/linus/({SHA_RE})", re.I),
     r"torvalds/linux|\1"),

    # kernel.org pub URL  .../linux.git/commit/?id=SHA
    ("kernel_cgit",
     re.compile(rf"kernel\.org/pub/scm/linux/kernel/git/[^/]+/[^/]+\.git/commit\?(?:[^&]*&)*id=({SHA_RE})", re.I),
     r"torvalds/linux|\1"),

    # GitLab: gitlab.com/a/b/-/commit/SHA  or  gitlab.com/a/b/commit/SHA
    ("gitlab",
     re.compile(rf"gitlab\.com/([^/]+/[^/]+)(?:/-)?/commit/({SHA_RE})", re.I),
     r"\1|\2"),

    # Android googlesource: android.googlesource.com/REPO/+/SHA
    ("googlesource",
     re.compile(rf"android\.googlesource\.com/([^/]+(?:/[^/+]+)*)/\+/({SHA_RE})", re.I),
     r"android/\1|\2"),

    # Chromium googlesource
    ("googlesource",
     re.compile(rf"chromium\.googlesource\.com/([^/]+(?:/[^/+]+)*)/\+/({SHA_RE})", re.I),
     r"chromium/\1|\2"),

    # Generic googlesource  *.googlesource.com/REPO/+/SHA
    ("googlesource",
     re.compile(rf"([\w\-]+)\.googlesource\.com/([^/]+(?:/[^/+]+)*)/\+/({SHA_RE})", re.I),
     r"\1/\2|\3"),
]


def parse_url(url: str) -> list[tuple[str, str, str]]:
    """Возвращает список (platform, repo_key, sha) для URL."""
    results = []
    for platform, pat, tmpl in URL_PATTERNS:
        m = pat.search(url)
        if not m:
            continue
        groups = m.groups()
        # Применяем шаблон подстановки
        repo_sha = tmpl
        for i, g in enumerate(groups, 1):
            repo_sha = repo_sha.replace(f"\\{i}", g or "")
        parts = repo_sha.split("|", 1)
        if len(parts) == 2:
            results.append((platform, parts[0], parts[1]))
    return results

def fetch_github_diff(session: requests.Session, repo: str, sha: str) -> str | None:
    """Diff коммита через GitHub REST API."""
    url = f"https://api.github.com/repos/{repo}/commits/{sha}"
    r = safe_get(session, url, headers={"Accept": "application/vnd.github.v3.diff"})
    if r and r.text:
        return r.text[:MAX_DIFF_CHARS]
    return None


def fetch_kernel_diff_via_github(session: requests.Session, sha: str) -> str | None:
    """
    Ядро Linux доступно через GitHub mirror torvalds/linux.
    Если коммит не найден там — пробуем gregkh/linux (stable).
    """
    for repo in ("torvalds/linux", "gregkh/linux"):
        diff = fetch_github_diff(session, repo, sha)
        if diff:
            return diff
        time.sleep(0.3)
    return None


def fetch_kernel_diff_via_lore(sha: str) -> str | None:
    """
    Запасной вариант: kernel.org cgit напрямую (plain patch endpoint).
    https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/patch/?id=SHA
    """
    url = (f"https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git"
           f"/patch/?id={sha}")
    try:
        r = requests.get(url, timeout=25,
                         headers={"User-Agent": "cvss-research-bot/1.0"})
        if r.status_code == 200 and r.text.startswith("From "):
            return r.text[:MAX_DIFF_CHARS]
    except requests.RequestException:
        pass
    return None


def fetch_gitlab_diff(session: requests.Session, repo: str, sha: str) -> str | None:
    """GitLab REST API v4 — публичные проекты без токена."""
    encoded = repo.replace("/", "%2F")
    url = f"https://gitlab.com/api/v4/projects/{encoded}/repository/commits/{sha}/diff"
    r = safe_get(session, url, timeout=25)
    if not r:
        return None
    try:
        diffs = r.json()
        if not isinstance(diffs, list):
            return None
        # Собираем unified diff из JSON-ответа GitLab
        parts = []
        for d in diffs:
            header = f"diff --git a/{d.get('old_path','')} b/{d.get('new_path','')}\n"
            parts.append(header + d.get("diff", ""))
        combined = "".join(parts)
        return combined[:MAX_DIFF_CHARS] if combined else None
    except (ValueError, KeyError):
        return None


def fetch_googlesource_diff(url_original: str, repo_key: str, sha: str) -> str | None:
    """
    googlesource.com: GET {url}^!/?format=TEXT возвращает base64-encoded patch.
    """
    # Восстанавливаем base URL из repo_key (platform/repo — это ключ,
    # но нам нужен оригинальный хост)
    m = re.search(r"(https?://[^/]*googlesource\.com/[^+]+)/\+/" + re.escape(sha),
                  url_original, re.I)
    if not m:
        return None
    base = m.group(1).rstrip("/")
    patch_url = f"{base}/+/{sha}^!/?format=TEXT"
    try:
        r = requests.get(patch_url, timeout=25,
                         headers={"User-Agent": "cvss-research-bot/1.0"})
        if r.status_code != 200 or not r.text:
            return None
        # Ответ — base64
        try:
            decoded = base64.b64decode(r.text).decode("utf-8", errors="replace")
        except Exception:
            decoded = r.text
        return decoded[:MAX_DIFF_CHARS] if decoded else None
    except requests.RequestException:
        return None


def fetch_diff(session: requests.Session,
               platform: str, repo: str, sha: str,
               original_url: str = "") -> str | None:
    """Универсальный диспетчер по платформе."""
    if platform == "github":
        return fetch_github_diff(session, repo, sha)
    elif platform == "kernel_cgit":
        diff = fetch_kernel_diff_via_github(session, sha)
        if diff:
            return diff
        time.sleep(0.5)
        return fetch_kernel_diff_via_lore(sha)
    elif platform == "gitlab":
        return fetch_gitlab_diff(session, repo, sha)
    elif platform == "googlesource":
        return fetch_googlesource_diff(original_url, repo, sha)
    return None

def collect_refs_extended(
    nvd_path: Path,
    out_path: Path,
    gh_token: str | None,
    sleep_s: float = 0.5,
    max_patches_per_cve: int = 3,
):
    session = make_session(gh_token)
    records = load_jsonl(nvd_path)
    log.info(f"Загружено {len(records)} CVE")

    found_total = 0
    out_records: list[dict] = []

    for rec in tqdm(records, desc="refs-ext"):
        refs = rec.get("references", [])
        patches: list[dict] = []
        seen_sha: set[str] = set()

        for url in refs:
            if len(patches) >= max_patches_per_cve:
                break
            parsed = parse_url(url)
            for platform, repo, sha in parsed:
                if sha in seen_sha:
                    continue
                seen_sha.add(sha)
                diff = fetch_diff(session, platform, repo, sha, original_url=url)
                time.sleep(sleep_s)
                if diff:
                    patches.append({
                        "platform": platform,
                        "repo": repo,
                        "sha": sha,
                        "url": url,
                        "diff": diff,
                    })
                    found_total += 1
                    break  # один diff на ссылку

        if patches:
            out_records.append({**rec, "patches": patches})

    save_jsonl(out_records, out_path)
    log.info(f"refs-ext итого: {found_total} диффов для {len(out_records)} CVE")

OSV_API = "https://api.osv.dev/v1"


def query_osv(session: requests.Session, cve_id: str) -> dict | None:
    """Запросить OSV.dev по CVE-ID."""
    url = f"{OSV_API}/vulns/{cve_id}"
    r = safe_get(session, url, timeout=15)
    if r:
        return r.json()
    return None


def extract_fix_commits_from_osv(osv_entry: dict) -> list[dict]:
    """
    Из OSV-записи извлечь все FIX-коммиты.
    Два источника:
      a) references[type=FIX] → URL → парсим через parse_url()
      b) affected[].ranges[type=GIT] → {repo, events[].fixed}
    """
    fixes: list[dict] = []
    seen: set[str] = set()

    # a) References с type=FIX
    for ref in osv_entry.get("references", []):
        if ref.get("type") in ("FIX", "PATCH", "WEB"):
            url = ref.get("url", "")
            for platform, repo, sha in parse_url(url):
                key = f"{repo}@{sha}"
                if key not in seen:
                    seen.add(key)
                    fixes.append({"platform": platform, "repo": repo,
                                  "sha": sha, "url": url})

    # b) Structured git ranges
    for affected in osv_entry.get("affected", []):
        for rng in affected.get("ranges", []):
            if rng.get("type") != "GIT":
                continue
            repo_url = rng.get("repo", "")
            for event in rng.get("events", []):
                sha = event.get("fixed")
                if not sha or len(sha) < 7:
                    continue
                # Определяем платформу по repo_url
                platform = "github"
                repo_key = repo_url
                if "github.com" in repo_url:
                    m = re.search(r"github\.com/([^/]+/[^/]+?)(?:\.git)?/?$", repo_url)
                    repo_key = m.group(1) if m else repo_url
                elif "kernel.org" in repo_url:
                    platform = "kernel_cgit"
                    repo_key = "torvalds/linux"
                elif "gitlab.com" in repo_url:
                    platform = "gitlab"
                    m = re.search(r"gitlab\.com/([^/]+/[^/]+?)(?:\.git)?/?$", repo_url)
                    repo_key = m.group(1) if m else repo_url
                elif "googlesource.com" in repo_url:
                    platform = "googlesource"
                    repo_key = repo_url

                key = f"{repo_key}@{sha}"
                if key not in seen:
                    seen.add(key)
                    fixes.append({"platform": platform, "repo": repo_key,
                                  "sha": sha, "url": repo_url})
    return fixes


def collect_osv(
    nvd_path: Path,
    out_path: Path,
    gh_token: str | None,
    batch_size: int = 50,
    sleep_between: float = 0.15,
):
    """
    Для каждого CVE в nvd_cves.jsonl запрашиваем OSV.dev,
    извлекаем FIX-коммиты, скачиваем диффы.
    """
    session_osv  = make_session()          # OSV — без авторизации
    session_diff = make_session(gh_token)  # для GitHub/GitLab

    records = load_jsonl(nvd_path)
    log.info(f"Загружено {len(records)} CVE, запрашиваем OSV…")

    found_cve = 0
    found_diff = 0
    out_records: list[dict] = []

    for rec in tqdm(records, desc="OSV"):
        cve_id = rec.get("cve_id", "")
        if not cve_id:
            continue

        osv = query_osv(session_osv, cve_id)
        time.sleep(sleep_between)
        if not osv:
            continue

        fix_commits = extract_fix_commits_from_osv(osv)
        if not fix_commits:
            continue

        patches: list[dict] = []
        for fix in fix_commits[:4]:
            diff = fetch_diff(
                session_diff,
                fix["platform"], fix["repo"], fix["sha"],
                original_url=fix["url"],
            )
            time.sleep(0.4)
            if diff:
                patches.append({**fix, "diff": diff})
                found_diff += 1
            if len(patches) >= 3:
                break

        if patches:
            found_cve += 1
            out_records.append({**rec, "patches": patches})

        # Периодически сбрасываем на диск (на случай прерывания)
        if len(out_records) % 200 == 0 and out_records:
            save_jsonl(out_records, out_path)

    save_jsonl(out_records, out_path)
    log.info(f"OSV итого: {found_diff} диффов для {found_cve} CVE")

KNOWN_HOSTS = [
    "github.com",
    "git.kernel.org",
    "gitlab.com",
    "googlesource.com",
    "android.googlesource.com",
    "chromium.googlesource.com",
    "bugzilla.redhat.com",
    "bugzilla.suse.com",
    "bugzilla.kernel.org",
    "launchpad.net",
    "lists.debian.org",
    "security.gentoo.org",
    "packetstormsecurity.com",
    "exploit-db.com",
    "nvd.nist.gov",
]


def analyze_stats(nvd_path: Path, sample: int = 0):
    records = load_jsonl(nvd_path)
    if sample:
        import random
        random.shuffle(records)
        records = records[:sample]

    host_counter: Counter = Counter()
    pattern_counter: Counter = Counter()
    parseable = 0
    total_refs = 0

    for rec in records:
        for url in rec.get("references", []):
            total_refs += 1
            try:
                host = urlparse(url).netloc.lower()
                host_counter[host] += 1
            except Exception:
                pass
            parsed = parse_url(url)
            if parsed:
                parseable += 1
                for platform, repo, sha in parsed:
                    pattern_counter[platform] += 1

    print(f"\n{'─'*60}")
    print(f"CVE проанализировано : {len(records):,}")
    print(f"Всего references     : {total_refs:,}")
    print(f"Парсируемых (есть SHA): {parseable:,}  ({parseable/max(total_refs,1)*100:.1f}%)")
    print(f"\nТоп-20 хостов в references:")
    for host, cnt in host_counter.most_common(20):
        print(f"  {cnt:6,}  {host}")
    print(f"\nПарсируемые по платформам:")
    for plat, cnt in pattern_counter.most_common():
        print(f"  {cnt:6,}  {plat}")

    # Сколько CVE имеют хотя бы одну парсируемую ссылку
    cve_with_parseable = sum(
        1 for rec in records
        if any(parse_url(u) for u in rec.get("references", []))
    )
    print(f"\nCVE с ≥1 парсируемой ref: {cve_with_parseable:,} "
          f"({cve_with_parseable/len(records)*100:.1f}%)")
    print(f"{'─'*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Расширенный сбор диффов: refs-ext | osv | stats"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # stats
    p_stats = sub.add_parser("stats", help="Анализ паттернов references в NVD данных")
    p_stats.add_argument("--nvd", required=True, help="nvd_cves.jsonl")
    p_stats.add_argument("--sample", type=int, default=0,
                         help="Кол-во CVE для выборки (0 = все)")

    # refs-ext
    p_refs = sub.add_parser("refs-ext",
                             help="Расширенный парсинг NVD references (kernel/gitlab/googlesource)")
    p_refs.add_argument("--nvd",    required=True, help="nvd_cves.jsonl")
    p_refs.add_argument("--out",    default="ref_ext_patches.jsonl")
    p_refs.add_argument("--token",  default=None, help="GitHub PAT (рекомендуется)")
    p_refs.add_argument("--sleep",  type=float, default=0.5,
                        help="Пауза между запросами (сек)")
    p_refs.add_argument("--max-patches", type=int, default=3,
                        help="Макс. диффов на CVE")

    # osv
    p_osv = sub.add_parser("osv", help="OSV.dev API — structured fix commits (~40K CVE)")
    p_osv.add_argument("--nvd",   required=True, help="nvd_cves.jsonl")
    p_osv.add_argument("--out",   default="osv_patches.jsonl")
    p_osv.add_argument("--token", default=None, help="GitHub PAT для скачивания диффов")
    p_osv.add_argument("--sleep", type=float, default=0.15,
                       help="Пауза между OSV запросами (сек)")

    args = parser.parse_args()

    if args.cmd == "stats":
        analyze_stats(Path(args.nvd), args.sample)

    elif args.cmd == "refs-ext":
        collect_refs_extended(
            nvd_path=Path(args.nvd),
            out_path=Path(args.out),
            gh_token=args.token or None,
            sleep_s=args.sleep,
            max_patches_per_cve=args.max_patches,
        )

    elif args.cmd == "osv":
        collect_osv(
            nvd_path=Path(args.nvd),
            out_path=Path(args.out),
            gh_token=args.token or None,
            sleep_between=args.sleep,
        )


if __name__ == "__main__":
    main()
