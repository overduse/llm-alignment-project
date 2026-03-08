#!/usr/bin/env python3
"""
Download models from ModelScope to project root models/ (or a given directory).
"""
import os
from modelscope.hub.snapshot_download import snapshot_download


def _project_root():
    """Project root (parent of src/). Script lives in src/download/."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download_qwen_model(model_id: str, base_save_dir: str | None = None) -> str:
    """
    Download a model from ModelScope to the given local directory.
    If base_save_dir is None, uses <project_root>/models.
    """
    if base_save_dir is None:
        base_save_dir = os.path.join(_project_root(), "models")
    model_name = model_id.split("/")[-1]
    local_dir = os.path.join(base_save_dir, model_name)
    os.makedirs(local_dir, exist_ok=True)

    print(f"Downloading model: {model_id}")
    print(f"Save path: {local_dir}")

    snapshot_download(model_id=model_id, local_dir=local_dir)
    print(f"Done: {model_name}\n")
    print("-" * 50)
    return local_dir


if __name__ == "__main__":
    models_to_download = [
        "qwen/Qwen3-Next-80B-A3B-Instruct",
        "qwen/Qwen3-8B",
    ]
    for model in models_to_download:
        download_qwen_model(model)
