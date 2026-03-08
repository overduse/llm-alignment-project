#!/usr/bin/env python3
import os
import sys
import subprocess
import time
import json

def main():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    root = os.path.dirname(os.path.abspath(__file__))
    done_flag = os.path.join(root, ".rank0_done.json")

    if rank != 0:
        # Wait until rank0 writes completion flag, then exit with same code.
        while True:
            if os.path.exists(done_flag):
                try:
                    with open(done_flag, "r", encoding="utf-8") as f:
                        status = json.load(f)
                    code = int(status.get("exit_code", 0))
                    sys.exit(code)
                except Exception:
                    # If flag is corrupted, still exit non-zero to surface failure.
                    sys.exit(1)
            time.sleep(60)

    os.chdir(root)
    try:
        if os.path.exists(done_flag):
            os.remove(done_flag)
    except Exception:
        pass

    # Prefer WORLD_SIZE from torchrun; fallback to visible GPU count.
    import torch
    nproc = world_size if world_size > 1 else torch.cuda.device_count()
    model_path = os.environ.get("VLLM_MODEL", "models/Qwen3-Next-80B-A3B-Instruct")
    served_model_name = os.environ.get("VLLM_SERVED_MODEL_NAME", "Qwen3-Next-80B-A3B-Instruct")
    input_file = os.environ.get("INPUT_FILE", "data/processed/seed_prompts_random.jsonl")
    output_file = os.environ.get("OUTPUT_FILE", "data/generated/sft_pairs.jsonl")
    port = int(os.environ.get("VLLM_PORT", "8000"))
    max_samples = os.environ.get("MAX_SAMPLES", "20000")
    max_new_tokens = os.environ.get("MAX_NEW_TOKENS", "4096")
    max_concurrent = os.environ.get("MAX_CONCURRENT", "64")
    temperature = os.environ.get("TEMPERATURE", "0.7")
    gpu_mem_util = os.environ.get("GPU_MEMORY_UTILIZATION", "0.9")
    max_model_len = os.environ.get("MAX_MODEL_LEN", "32768")

    print(
        f"[rank0] start vLLM model={model_path} served={served_model_name} "
        f"tp={nproc} port={port}"
    )

    cmd_vllm = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--served-model-name", served_model_name,
        "--tensor-parallel-size", str(nproc),
        "--trust-remote-code",
        "--dtype", "auto",
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--max-model-len", str(max_model_len),
        "--host", "0.0.0.0",
        "--port", str(port),
    ]
    proc = subprocess.Popen(cmd_vllm, cwd=root)
    url = f"http://localhost:{port}/v1/models"
    while True:
        try:
            import urllib.request
            urllib.request.urlopen(url, timeout=2)
            break
        except Exception:
            if proc.poll() is not None:
                print("vLLM exited unexpectedly.", file=sys.stderr)
                sys.exit(1)
            time.sleep(5)
    print(f"[rank0] vLLM ready at {url}")

    cmd_gen = [
        sys.executable, "src/generate/generate_sft_data_async.py",
        "--api-base", f"http://localhost:{port}/v1",
        "--model", served_model_name,
        "--input", input_file,
        "--output", output_file,
        "--max-samples", str(max_samples),
        "--max-new-tokens", str(max_new_tokens),
        "--max-concurrent", str(max_concurrent),
        "--temperature", str(temperature),
    ]
    print("[rank0] running async client...")
    exit_code = 0
    try:
        ret = subprocess.run(cmd_gen, cwd=root)
        if ret.returncode != 0:
            exit_code = ret.returncode
        print(f"[rank0] done. output={output_file}")
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=30)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            with open(done_flag, "w", encoding="utf-8") as f:
                json.dump({"exit_code": exit_code}, f)
        except Exception:
            pass
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
