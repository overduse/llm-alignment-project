import os
from datasets import load_dataset

def download_magicoder_data():
    """Download Magicoder Evol-Instruct 110K from Hugging Face to data/raw."""
    dataset_name = "ise-uiuc/Magicoder-Evol-Instruct-110K"
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(base_dir, "data", "raw")

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
    except Exception as e:
        print(f"Download failed: {e}")

if __name__ == "__main__":
    download_magicoder_data()