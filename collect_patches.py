import argparse
import json
import logging
import re
import time
from pathlib import Path

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

GITHUB_GRAPHQL = "https://api.github.com/graphql"
GITHUB_REST    = "https://api.github.com"

GHSA_QUERY = """
query($cursor: String) {
  securityAdvisories(first: 100, after: $cursor, orderBy: {field: UPDATED_AT, direction: DESC}) {
    pageInfo { hasNextPage endCursor }
    nodes {
      ghsaId
      summary
      description
      severity
      cvss { vectorString score }
      cwes(first: 10) { nodes { cweId name } }
      identifiers { type value }
      references { url }
      vulnerabilities(first: 5) {
        nodes {
          package { name ecosystem }
          firstPatchedVersion { identifier }
          vulnerableVersionRange
        }
      }
    }
  }
}
"""


def fetch_ghsa_page(session: requests.Session, cursor: str | None) -> dict:
    variables = {"cursor": cursor}
    r = session.post(
        GITHUB_GRAPHQL,
        json={"query": GHSA_QUERY, "variables": variables},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL errors: {data['errors']}")
    return data["data"]["securityAdvisories"]


def extract_cve_id(identifiers: list) -> str | None:
    for ident in identifiers:
        if ident.get("type") == "CVE":
            return ident.get("value")
    return None


def collect_ghsa(token: str, out_path: Path, limit: int = 0):
    """Собрать GHSA с CVSS v3 вектором и ссылками на патчи."""
    headers = {
        "Authorization": f"bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    session = requests.Session()
    session.headers.update(headers)

    cursor = None
    total = 0

    with out_path.open("w", encoding="utf-8") as fout:
        with tqdm(desc="GHSA", unit=" advisories") as pbar:
            while True:
                page = fetch_ghsa_page(session, cursor)
                for node in page["nodes"]:
                    cvss = node.get("cvss") or {}
                    vector = cvss.get("vectorString", "")
                    # только CVSS v3.x
                    if not vector.startswith("CVSS:3"):
                        continue

                    cve_id = extract_cve_id(node.get("identifiers", []))
                    cwes = [c["cweId"] for c in node.get("cwes", {}).get("nodes", [])]
                    refs = [r["url"] for r in node.get("references", [])]

                    record = {
                        "ghsa_id": node.get("ghsaId"),
                        "cve_id": cve_id,
                        "summary": node.get("summary", ""),
                        "description": node.get("description", ""),
                        "severity": node.get("severity"),
                        "cvss_vector": vector,
                        "cvss_score": cvss.get("score"),
                        "cwes": cwes,
                        "references": refs,
                        "packages": [
                            {
                                "name": v["package"]["name"],
                                "ecosystem": v["package"]["ecosystem"],
                                "first_patch": (v.get("firstPatchedVersion") or {}).get("identifier"),
                            }
                            for v in node.get("vulnerabilities", {}).get("nodes", [])
                        ],
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total += 1
                    pbar.update(1)

                if limit and total >= limit:
                    break
                if not page["pageInfo"]["hasNextPage"]:
                    break
                cursor = page["pageInfo"]["endCursor"]
                time.sleep(0.5)

    log.info(f"GHSA: собрано {total} advisory → {out_path}")

LINUX_REPO = "torvalds/linux"
LINUX_STABLE_REPOS = [
    "torvalds/linux",
    "gregkh/linux",   # stable backports
]


def search_commits_for_cve(
    session: requests.Session,
    cve_id: str,
    repo: str = LINUX_REPO,
) -> list[dict]:
    """Поиск коммитов в репозитории по CVE-ID."""
    url = f"{GITHUB_REST}/search/commits"
    params = {"q": f"{cve_id} repo:{repo}", "per_page": 10}
    r = session.get(url, params=params, timeout=20)
    if r.status_code == 422:
        return []
    if r.status_code == 403:
        log.warning("GitHub rate limit, ждём 60s...")
        time.sleep(60)
        return []
    r.raise_for_status()
    return r.json().get("items", [])


def fetch_commit_diff(session: requests.Session, repo: str, sha: str) -> str | None:
    """Получить diff коммита через REST API."""
    url = f"{GITHUB_REST}/repos/{repo}/commits/{sha}"
    headers_patch = {"Accept": "application/vnd.github.v3.diff"}
    r = session.get(url, headers=headers_patch, timeout=30)
    if r.status_code != 200:
        return None
    return r.text


def collect_linux_patches(
    token: str,
    cve_list_path: Path,
    out_path: Path,
    max_diff_chars: int = 50_000,
):
    """
    Для каждой CVE из nvd_cves.jsonl ищем патч в linux/linux.
    Сохраняем diff (обрезанный) рядом с метаданными CVE.
    """
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    session = requests.Session()
    session.headers.update(headers)

    LINUX_KEYWORDS = re.compile(
        r"\b(linux kernel|linux/kernel|net/|drivers/|fs/|kernel/|mm/|arch/x86|arch/arm)\b",
        re.IGNORECASE,
    )

    cves = []
    with cve_list_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            desc = rec.get("description", "")
            if LINUX_KEYWORDS.search(desc):
                cves.append(rec)

    log.info(f"Linux-релевантных CVE: {len(cves)}")

    found = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for rec in tqdm(cves, desc="Linux patches"):
            cve_id = rec["cve_id"]
            commits = search_commits_for_cve(session, cve_id)
            time.sleep(1.2)  # GitHub Search: 30 req/min для auth

            for commit in commits[:3]:  # берём до 3 коммитов
                sha = commit.get("sha", "")
                repo = commit.get("repository", {}).get("full_name", LINUX_REPO)
                message = commit.get("commit", {}).get("message", "")
                commit_url = commit.get("html_url", "")

                diff = fetch_commit_diff(session, repo, sha)
                if diff:
                    diff = diff[:max_diff_chars]  # обрезаем слишком большие диффы

                out_rec = {
                    **rec,
                    "patch_commit_sha": sha,
                    "patch_repo": repo,
                    "patch_commit_message": message,
                    "patch_commit_url": commit_url,
                    "patch_diff": diff,
                }
                fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                found += 1
                time.sleep(0.5)

    log.info(f"Linux patches: найдено {found} коммитов → {out_path}")

GITHUB_COMMIT_RE = re.compile(
    r"github\.com/([^/]+/[^/]+)/commit/([0-9a-f]{40})", re.IGNORECASE
)


def collect_patches_from_refs(
    token: str,
    cve_list_path: Path,
    out_path: Path,
    max_diff_chars: int = 50_000,
    filter_repo_pattern: str | None = None,
):
    """
    Для каждого CVE смотрим references, ищем прямые ссылки на GitHub commits,
    скачиваем diff. Опционально фильтруем по паттерну репо (напр. 'linux').
    """
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }
    session = requests.Session()
    session.headers.update(headers)

    repo_re = re.compile(filter_repo_pattern, re.IGNORECASE) if filter_repo_pattern else None

    found = 0
    with cve_list_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:

        lines = fin.readlines()
        for line in tqdm(lines, desc="Refs→patches"):
            rec = json.loads(line)
            refs = rec.get("references", [])
            patch_records = []

            for ref in refs:
                m = GITHUB_COMMIT_RE.search(ref)
                if not m:
                    continue
                repo = m.group(1)
                sha = m.group(2)

                if repo_re and not repo_re.search(repo):
                    continue

                diff = fetch_commit_diff(session, repo, sha)
                if diff:
                    patch_records.append({
                        "repo": repo,
                        "sha": sha,
                        "url": ref,
                        "diff": diff[:max_diff_chars],
                    })
                    found += 1
                time.sleep(0.3)

            if patch_records:
                out_rec = {**rec, "patches": patch_records}
                fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    log.info(f"Ref-patches: найдено {found} диффов → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Сбор патчей для CVE")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ghsa
    p_ghsa = sub.add_parser("ghsa", help="GitHub Security Advisories")
    p_ghsa.add_argument("--token", required=True)
    p_ghsa.add_argument("--out", default="ghsa_patches.jsonl")
    p_ghsa.add_argument("--limit", type=int, default=0, help="0 = без лимита")

    # linux
    p_linux = sub.add_parser("linux", help="Linux kernel patches via commit search")
    p_linux.add_argument("--token", required=True)
    p_linux.add_argument("--cve-list", required=True, help="nvd_cves.jsonl")
    p_linux.add_argument("--out", default="linux_patches.jsonl")

    # refs
    p_refs = sub.add_parser("refs", help="Патчи из ссылок NVD references")
    p_refs.add_argument("--token", required=True)
    p_refs.add_argument("--cve-list", required=True)
    p_refs.add_argument("--out", default="ref_patches.jsonl")
    p_refs.add_argument("--filter-repo", default=None, help="Regex фильтр репозитория, напр. 'linux|kernel'")

    args = parser.parse_args()

    if args.cmd == "ghsa":
        collect_ghsa(args.token, Path(args.out), args.limit)
    elif args.cmd == "linux":
        collect_linux_patches(args.token, Path(args.cve_list), Path(args.out))
    elif args.cmd == "refs":
        collect_patches_from_refs(
            args.token, Path(args.cve_list), Path(args.out), filter_repo_pattern=args.filter_repo
        )


if __name__ == "__main__":
    main()
