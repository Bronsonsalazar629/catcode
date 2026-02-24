#!/usr/bin/env python3
"""Collect competitive programming problems, solutions, and editorials from Codeforces."""

import argparse
import json
import time
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

CF_API = "https://codeforces.com/api"
CF_BASE = "https://codeforces.com"
RATE_LIMIT = 2.0  # seconds between requests


class RateLimiter:
    def __init__(self, delay: float):
        self.delay = delay
        self.last_request = 0.0

    def wait(self):
        elapsed = time.time() - self.last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_request = time.time()


rate_limiter = RateLimiter(RATE_LIMIT)


def api_get(method: str, params: dict = None) -> dict | None:
    rate_limiter.wait()
    try:
        resp = requests.get(f"{CF_API}/{method}", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "OK":
            return data.get("result")
    except (requests.RequestException, ValueError) as exc:
        print(f"  [warn] API error for {method}: {exc}", file=sys.stderr)
    return None


def fetch_problem_list() -> list[dict]:
    result = api_get("problemset.problems")
    if result is None:
        return []
    return result.get("problems", [])


def fetch_contest_submissions(contest_id: int, count: int = 50) -> list[dict]:
    result = api_get("contest.status", {
        "contestId": contest_id,
        "from": 1,
        "count": count,
    })
    return result or []


def fetch_page_html(url: str) -> str | None:
    rate_limiter.wait()
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException:
        return None


def scrape_problem_statement(contest_id: int, index: str) -> str:
    html = fetch_page_html(f"{CF_BASE}/problemset/problem/{contest_id}/{index}")
    if html is None:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    statement_div = soup.find("div", class_="problem-statement")
    if statement_div is None:
        return ""
    return statement_div.get_text(separator="\n", strip=True)


def scrape_editorial(contest_id: int) -> str:
    html = fetch_page_html(f"{CF_BASE}/blog/entry/{contest_id}")
    if html is None:
        # Try the contest editorial page pattern
        html = fetch_page_html(f"{CF_BASE}/contest/{contest_id}/editorial")
        if html is None:
            return ""
    soup = BeautifulSoup(html, "html.parser")
    content = soup.find("div", class_="ttypography")
    if content is None:
        return ""
    return content.get_text(separator="\n", strip=True)


def scrape_submission_code(contest_id: int, submission_id: int) -> str:
    html = fetch_page_html(f"{CF_BASE}/contest/{contest_id}/submission/{submission_id}")
    if html is None:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    source = soup.find("pre", id="program-source-text")
    if source is None:
        return ""
    return source.get_text()


def is_cpp(programming_language: str) -> bool:
    lang = programming_language.lower()
    return "c++" in lang or "gnu c" in lang


def collect_codeforces(max_problems: int, min_rating: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Fetching problem list from Codeforces API...")
    problems = fetch_problem_list()
    if not problems:
        print("Failed to fetch problems.", file=sys.stderr)
        return

    # Filter by rating
    problems = [p for p in problems if p.get("rating", 0) >= min_rating]
    problems = problems[:max_problems]
    print(f"Processing {len(problems)} problems (rating >= {min_rating})...")

    # Group problems by contest
    contests: dict[int, list[dict]] = {}
    for p in problems:
        cid = p.get("contestId")
        if cid:
            contests.setdefault(cid, []).append(p)

    records = []
    editorial_cache: dict[int, str] = {}

    for contest_id, contest_problems in tqdm(contests.items(), desc="Contests", unit="contest"):
        # Fetch submissions for this contest
        submissions = fetch_contest_submissions(contest_id, count=100)

        # Filter: accepted C++ solutions from high-rated users
        good_subs: dict[str, dict] = {}  # problem index -> best submission
        for sub in submissions:
            if sub.get("verdict") != "OK":
                continue
            if not is_cpp(sub.get("programmingLanguage", "")):
                continue
            author = sub.get("author", {})
            members = author.get("members", [])
            if not members:
                continue
            # Check if any author member has rating >= 2400
            rating = members[0].get("rating", 0) if members else 0
            if rating < 2400:
                continue

            prob_index = sub.get("problem", {}).get("index", "")
            if prob_index not in good_subs:
                good_subs[prob_index] = sub

        # Fetch editorial for this contest (cached)
        if contest_id not in editorial_cache:
            editorial_cache[contest_id] = scrape_editorial(contest_id)
        editorial = editorial_cache[contest_id]

        for prob in contest_problems:
            index = prob.get("index", "")

            # Get problem statement
            statement = scrape_problem_statement(contest_id, index)
            if not statement:
                continue

            # Get solution code
            sub = good_subs.get(index)
            if sub is None:
                continue

            solution_code = scrape_submission_code(contest_id, sub["id"])
            if not solution_code:
                continue

            records.append({
                "problem": statement,
                "editorial": editorial,
                "solution": solution_code,
                "rating": prob.get("rating", 0),
                "tags": prob.get("tags", []),
            })

    # Write output
    print(f"\nWriting {len(records)} records to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n--- Statistics ---")
    print(f"Contests processed: {len(contests)}")
    print(f"Records written:    {len(records)}")
    print(f"Output file:        {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect CP data from Codeforces.")
    parser.add_argument("--output", type=Path, default=Path("data/raw/codeforces.jsonl"))
    parser.add_argument("--max-problems", type=int, default=500, help="Max problems to process.")
    parser.add_argument("--min-rating", type=int, default=1200, help="Minimum problem difficulty rating.")
    args = parser.parse_args()
    collect_codeforces(args.max_problems, args.min_rating, args.output)


if __name__ == "__main__":
    main()
