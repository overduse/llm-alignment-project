#!/usr/bin/env python3
"""
Randomly sample a subset of seed prompts from the cleaned JSONL.
Reads data/processed/clean_seed_prompts.jsonl, writes data/processed/seed_prompts_random.jsonl.
"""
import os
import random
import argparse

OUTPUT_FILENAME = "seed_prompts_random.jsonl"


def _project_root():
    """Project root (parent of src/). Script lives in src/data_prep/."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_golden_prompts(
    target_size: int = 20000,
    random_seed: int = 42,
    output_dir: str | None = None,
) -> str:
    """
    Sample a fixed number of lines from the cleaned seed prompts file using a fixed random seed.
    """
    project_root = _project_root()
    processed_dir = output_dir or os.path.join(project_root, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    input_file = os.path.join(processed_dir, "clean_seed_prompts.jsonl")
    output_file = os.path.join(processed_dir, OUTPUT_FILENAME)

    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Cleaned data not found: {input_file}. Run data_process.py first."
        )

    print(f"Sampling up to {target_size} prompts (seed={random_seed})...")

    all_prompts = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_prompts.append(line)

    total_clean = len(all_prompts)
    print(f"Loaded {total_clean} lines from {input_file}")

    if target_size > total_clean:
        print(f"Warning: target_size {target_size} > available {total_clean}, using all.")
        target_size = total_clean

    random.seed(random_seed)
    sampled = random.sample(all_prompts, target_size)

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(sampled)

    print("-" * 40)
    print(f"Done. Sampled {target_size} / {total_clean} -> {output_file}")
    print("-" * 40)
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample seed prompts from cleaned JSONL (same layout as data_process / download_data)."
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=20000,
        help="Number of lines to sample (default 20000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default 42)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <project>/data/processed)",
    )
    args = parser.parse_args()
    extract_golden_prompts(
        target_size=args.target_size,
        random_seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
