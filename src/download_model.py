#!/usr/bin/env python3

import os
from modelscope.hub.snapshot_download import snapshot_download

def download_qwen_model(model_id, base_save_dir="./models"):
    """
    Download a model from ModelScope to the given local directory.
    """
    # Derive model name from id (e.g. "qwen/Qwen3-8B" -> "Qwen3-8B")
    model_name = model_id.split("/")[-1]
    local_dir = os.path.join(base_save_dir, model_name)
    os.makedirs(local_dir, exist_ok=True)

    print(f"Downloading model: {model_id}")
    print(f"Save path: {local_dir}")

    snapshot_download(model_id=model_id, local_dir=local_dir)
    print(f"Done: {model_name}\n")
    print("-" * 50)

if __name__ == "__main__":
    models_to_download = [
        "qwen/Qwen3-Next-80B-A3B-Instruct",
        "qwen/Qwen3-8B"
    ]
    
    for model in models_to_download:
        download_qwen_model(model)