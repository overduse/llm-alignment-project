#!/usr/bin/env python3
"""
Clean raw Magicoder JSONL: stream read, filter by length/blacklist/dedup, keep only instruction field.
Reads data/raw/magicoder_110k.jsonl, writes data/processed/clean_seed_prompts.jsonl.
"""
import os
import json
from tqdm import tqdm

# uncomment the line below to use the hf-mirror
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

RAW_FILENAME = "magicoder_110k.jsonl"
OUTPUT_FILENAME = "clean_seed_prompts.jsonl"

MIN_PROMPT_LEN = 20
MAX_PROMPT_LEN = 2500

# Phrases that often indicate refusals or low-quality meta text; filter these out.
BLACKLIST_KEYWORDS = [
    "as an ai", "i cannot fulfill", "i'm sorry", "i am sorry",
    "i can't fulfill", "language model", "openai", "chatgpt",
    "抱歉", "无法回答", "作为人工智能", "evolve this prompt",
    "sorry, but i cannot",
]


def _project_root():
    """Project root (parent of src/). Script lives in src/data_prep/."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def process_and_clean_magicoder() -> str:
    """
    Stream the raw JSONL, apply length/blacklist/dedup rules, output instruction-only JSONL.
    """
    project_root = _project_root()
    raw_file = os.path.join(project_root, "data", "raw", RAW_FILENAME)
    processed_dir = os.path.join(project_root, "data", "processed")
    output_file = os.path.join(processed_dir, OUTPUT_FILENAME)

    os.makedirs(processed_dir, exist_ok=True)

    if not os.path.exists(raw_file):
        raise FileNotFoundError(
            f"Raw data not found: {raw_file}. Run download_data.py first."
        )

    print(f"Processing: {raw_file}")

    seen_prompts = set()
    cleaned_count = 0
    total_count = 0
    filtered_by_length = 0
    filtered_by_blacklist = 0
    filtered_by_duplicate = 0

    with open(raw_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:

        for line in tqdm(f_in, desc="Cleaning"):
            total_count += 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            prompt = data.get("instruction", "").strip()

            if len(prompt) < MIN_PROMPT_LEN or len(prompt) > MAX_PROMPT_LEN:
                filtered_by_length += 1
                continue

            prompt_lower = prompt.lower()
            if any(kw in prompt_lower for kw in BLACKLIST_KEYWORDS):
                filtered_by_blacklist += 1
                continue

            if prompt in seen_prompts:
                filtered_by_duplicate += 1
                continue
            seen_prompts.add(prompt)

            f_out.write(json.dumps({"instruction": prompt}, ensure_ascii=False) + "\n")
            cleaned_count += 1

    print("-" * 40)
    print(f"Total read: {total_count}")
    print(f"Filtered (length): {filtered_by_length}")
    print(f"Filtered (blacklist): {filtered_by_blacklist}")
    print(f"Filtered (duplicate): {filtered_by_duplicate}")
    print(f"Written: {cleaned_count} -> {output_file}")
    print("-" * 40)
    return output_file


if __name__ == "__main__":
    process_and_clean_magicoder()
