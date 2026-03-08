#!/usr/bin/env python3
"""
Download Magicoder Evol-Instruct 110K from Hugging Face to data/raw.
"""
import os
from datasets import load_dataset


def _project_root():
    """Project root (parent of src/). Script lives in src/download/."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download_magicoder_data() -> str:
    """Download Magicoder Evol-Instruct 110K from Hugging Face to data/raw."""
    dataset_name = "ise-uiuc/Magicoder-Evol-Instruct-110K"
    target_dir = os.path.join(_project_root(), "data", "raw")
    os.makedirs(target_dir, exist_ok=True)

    print(f"Downloading dataset: {dataset_name}...")
    try:
        dataset = load_dataset(dataset_name, split="train")
        output_file = os.path.join(target_dir, "magicoder_110k.jsonl")
        dataset.to_json(output_file, force_ascii=False)
        print("-" * 30)
        print(f"Done. Path: {output_file}")
        print(f"Rows: {len(dataset)}")
        print("-" * 30)
        return output_file
    except Exception as e:
        print(f"Download failed: {e}")
        raise


if __name__ == "__main__":
    download_magicoder_data()
