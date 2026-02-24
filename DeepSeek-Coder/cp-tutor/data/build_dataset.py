#!/usr/bin/env python3
"""Merge, deduplicate, balance, and split teaching data into train/eval JSONL."""

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

TARGET_DISTRIBUTION = {
    "[HINT]": 0.30,
    "[EXPLAIN]": 0.30,
    "[TEACH]": 0.25,
    "[SOLVE]": 0.15,
}

KNOWN_PREFIXES = list(TARGET_DISTRIBUTION.keys())


def hash_instruction(instruction: str) -> str:
    return hashlib.sha256(instruction.encode("utf-8")).hexdigest()


def detect_prefix(instruction: str) -> str:
    stripped = instruction.lstrip()
    for prefix in KNOWN_PREFIXES:
        if stripped.startswith(prefix):
            return prefix
    return "UNKNOWN"


def read_jsonl(path: str) -> list[dict]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Reading {os.path.basename(path)}"):
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def deduplicate(entries: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for entry in tqdm(entries, desc="Deduplicating"):
        h = hash_instruction(entry["instruction"])
        if h not in seen:
            seen.add(h)
            unique.append(entry)
    return unique


def categorize(entries: list[dict]) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        prefix = detect_prefix(entry["instruction"])
        buckets[prefix].append(entry)
    return dict(buckets)


def balance(buckets: dict[str, list[dict]], rng: random.Random) -> list[dict]:
    known_counts = {
        prefix: len(buckets.get(prefix, []))
        for prefix in KNOWN_PREFIXES
    }

    if all(c == 0 for c in known_counts.values()):
        print("WARNING: No entries with known prefix tags found.")
        return []

    # Find total N so no category needs > 2x oversample
    candidate_totals = []
    for prefix in KNOWN_PREFIXES:
        ratio = TARGET_DISTRIBUTION[prefix]
        count = known_counts[prefix]
        if count > 0:
            candidate_totals.append(int(2 * count / ratio))

    total_n = min(candidate_totals) if candidate_totals else 0
    total_available = sum(known_counts.values())
    total_n = min(total_n, total_available)

    balanced: list[dict] = []
    for prefix in KNOWN_PREFIXES:
        ratio = TARGET_DISTRIBUTION[prefix]
        target_count = int(total_n * ratio)
        pool = buckets.get(prefix, [])
        if not pool:
            continue

        if target_count <= len(pool):
            sampled = rng.sample(pool, target_count)
        else:
            sampled = list(pool)
            extra = target_count - len(pool)
            sampled.extend(rng.choices(pool, k=extra))

        balanced.extend(sampled)

    return balanced


def write_jsonl(data: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in tqdm(data, desc=f"Writing {os.path.basename(path)}"):
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def print_statistics(raw_buckets, balanced_buckets, train_data, eval_data, balanced_data):
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    print("\n--- Before Balancing ---")
    total_before = sum(len(v) for v in raw_buckets.values())
    for prefix in KNOWN_PREFIXES:
        count = len(raw_buckets.get(prefix, []))
        pct = (count / total_before * 100) if total_before else 0
        print(f"  {prefix:10s}: {count:>7d}  ({pct:5.1f}%)")
    print(f"  {'TOTAL':10s}: {total_before:>7d}")

    print("\n--- After Balancing ---")
    total_after = len(balanced_data)
    for prefix in KNOWN_PREFIXES:
        count = len(balanced_buckets.get(prefix, []))
        pct = (count / total_after * 100) if total_after else 0
        print(f"  {prefix:10s}: {count:>7d}  ({pct:5.1f}%)")
    print(f"  {'TOTAL':10s}: {total_after:>7d}")

    print("\n--- Train / Eval Split ---")
    print(f"  Train : {len(train_data):>7d}")
    print(f"  Eval  : {len(eval_data):>7d}")

    print("\n--- Sample Entry per Format ---")
    for prefix in KNOWN_PREFIXES:
        entries = balanced_buckets.get(prefix, [])
        if entries:
            sample = entries[0]
            instr_preview = sample["instruction"][:120].replace("\n", " ")
            output_preview = sample["output"][:120].replace("\n", " ")
            print(f"\n  [{prefix}]")
            print(f"    instruction: {instr_preview}...")
            print(f"    output:      {output_preview}...")

    print("\n" + "=" * 60)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    print(f"\n[1/6] Reading input from: {args.input}")
    entries = read_jsonl(args.input)
    print(f"  Loaded {len(entries)} entries.")

    print("\n[2/6] Deduplicating by instruction hash...")
    entries = deduplicate(entries)
    print(f"  {len(entries)} unique entries remain.")

    print("\n[3/6] Categorizing and balancing...")
    raw_buckets = categorize(entries)
    balanced_data = balance(raw_buckets, rng)
    print(f"  Balanced dataset size: {len(balanced_data)}")

    print("\n[4/6] Shuffling...")
    rng.shuffle(balanced_data)

    print("\n[5/6] Splitting into train (95%) / eval (5%)...")
    split_idx = int(len(balanced_data) * 0.95)
    train_data = balanced_data[:split_idx]
    eval_data = balanced_data[split_idx:]

    print(f"\n[6/6] Writing outputs...")
    write_jsonl(train_data, args.train_output)
    write_jsonl(eval_data, args.eval_output)

    balanced_buckets = categorize(balanced_data)
    print_statistics(raw_buckets, balanced_buckets, train_data, eval_data, balanced_data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build balanced train/eval splits from teaching data.")
    parser.add_argument("--input", type=str, default="data/raw/teaching_synthetic.jsonl")
    parser.add_argument("--train-output", type=str, default="data/train.jsonl")
    parser.add_argument("--eval-output", type=str, default="data/eval.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
