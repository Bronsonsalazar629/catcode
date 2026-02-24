#!/usr/bin/env python3
"""Collect C++ competitive programming solutions from IOI winners' GitHub repos."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("tqdm is required. Install it with: pip install tqdm")

IOI_REPOS: list[str] = [
    # Gennady Korotkevich (tourist) -- 6x IOI gold, 2x ICPC champion
    "https://github.com/the-tourist/algo",
    # Benjamin Qi (Benq) -- IOI 2020 gold
    "https://github.com/bqi343/cp-notebook",
    # Andrew He (ecnerwala) -- IOI 2014/2015 gold, ICPC champion
    "https://github.com/ecnerwala/cp-book",
    # Petr Mitrichev -- IOI 2000/2002 gold
    "https://github.com/PetarV-/Algorithms",
    # Errichto -- competitive programmer & educator
    "https://github.com/Errichto/youtube",
    # AtCoder Library (high-quality reference implementations)
    "https://github.com/atcoder/ac-library",
    # jiangly -- IOI 2021 homework solutions
    "https://github.com/jiangly-programmer/ioi2021-homework",
    # William Lin -- competitive programming solutions
    "https://github.com/tmwilliamlin168/CompetitiveProgramming",
]

_SINGLE_LINE_COMMENT = re.compile(r"//(.*)$", re.MULTILINE)
_MULTI_LINE_COMMENT = re.compile(r"/\*(.+?)\*/", re.DOTALL)


def extract_comments(code: str) -> list[str]:
    comments: list[str] = []
    for m in _SINGLE_LINE_COMMENT.finditer(code):
        text = m.group(1).strip()
        if text:
            comments.append(text)
    for m in _MULTI_LINE_COMMENT.finditer(code):
        text = m.group(1).strip()
        if text:
            comments.append(text)
    return comments


def repo_name_from_url(url: str) -> str:
    parts = url.rstrip("/").split("/")
    return f"{parts[-2]}__{parts[-1]}"


def clone_repo(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  [skip] {dest.name} already cloned")
        return
    print(f"  [clone] {url} -> {dest}")
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(dest)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def process_repos(repo_urls: list[str], repos_dir: Path, output_path: Path) -> None:
    repos_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== Cloning repos ===")
    repo_dirs: list[tuple[str, Path]] = []
    for url in repo_urls:
        name = repo_name_from_url(url)
        dest = repos_dir / name
        try:
            clone_repo(url, dest)
            repo_dirs.append((name, dest))
        except subprocess.CalledProcessError as exc:
            print(f"  [error] Failed to clone {url}: {exc}", file=sys.stderr)

    print("\n=== Extracting .cpp files ===")
    total_files = 0
    total_written = 0

    with open(output_path, "w", encoding="utf-8") as fout:
        for source_name, rdir in repo_dirs:
            cpp_files = sorted(rdir.rglob("*.cpp"))
            total_files += len(cpp_files)
            print(f"  {source_name}: {len(cpp_files)} .cpp file(s)")

            for cpp_path in tqdm(cpp_files, desc=f"  {source_name}", unit="file", leave=False):
                try:
                    code = cpp_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue

                record = {
                    "source": source_name,
                    "filename": str(cpp_path.relative_to(rdir)),
                    "code": code,
                    "comments": extract_comments(code),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_written += 1

    print(f"\nDone. {total_written}/{total_files} files written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect C++ solutions from CP GitHub repos.")
    parser.add_argument("--output", type=Path, default=Path("data/raw/ioi_solutions.jsonl"))
    parser.add_argument("--repos-dir", type=Path, default=Path("data/raw/repos"))
    args = parser.parse_args()
    process_repos(repo_urls=IOI_REPOS, repos_dir=args.repos_dir, output_path=args.output)


if __name__ == "__main__":
    main()
