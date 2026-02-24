#!/usr/bin/env python3
"""Download and filter C++ solutions from DeepMind's CodeContests dataset."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

CPP_LANGUAGE = 2


def extract_cpp_solutions(dataset, max_samples=None):
    records = []
    total_problems = 0
    problems_with_cpp = 0
    total_cpp_solutions = 0

    for split_name in dataset:
        split = dataset[split_name]
        print(f"Processing split '{split_name}' ({len(split)} problems)...")

        for example in tqdm(split, desc=f"  {split_name}"):
            total_problems += 1
            description = example["description"]
            difficulty = example["difficulty"]
            solutions = example.get("solutions", {})
            languages = solutions.get("language", [])
            codes = solutions.get("solution", [])

            found_cpp = False
            for lang, code in zip(languages, codes):
                if lang == CPP_LANGUAGE:
                    if not found_cpp:
                        problems_with_cpp += 1
                        found_cpp = True
                    total_cpp_solutions += 1

                    records.append({
                        "problem": description,
                        "solution": code,
                        "difficulty": difficulty,
                        "source": "codecontests",
                    })

                    if max_samples is not None and len(records) >= max_samples:
                        return records, total_problems, problems_with_cpp, total_cpp_solutions

    return records, total_problems, problems_with_cpp, total_cpp_solutions


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CodeContests and extract C++ solutions.")
    parser.add_argument("--output", type=str, default="data/raw/codecontests.jsonl")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit output samples (for testing).")
    args = parser.parse_args()

    print("Loading deepmind/code_contests from HuggingFace...")
    dataset = load_dataset("deepmind/code_contests")

    records, total_problems, problems_with_cpp, total_cpp_solutions = extract_cpp_solutions(
        dataset, max_samples=args.max_samples
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting {len(records)} records to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n--- Statistics ---")
    print(f"Total problems scanned:      {total_problems}")
    print(f"Problems with C++ solutions: {problems_with_cpp}")
    print(f"Total C++ solutions found:   {total_cpp_solutions}")
    print(f"Records written:             {len(records)}")
    print(f"Output file:                 {output_path}")


if __name__ == "__main__":
    main()
