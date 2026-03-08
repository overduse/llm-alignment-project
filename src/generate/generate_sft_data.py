#!/usr/bin/env python3
"""
Generate SFT (instruction, output) pairs using a local small model on a single GPU.
Reads seed prompts from data/processed/seed_prompts_random.jsonl (or --input),
runs the model to produce responses, writes instruction-output JSONL to data/generated/.
"""
import os
import json
import argparse
from pathlib import Path

# Lazy import to avoid loading torch/transformers when only parsing args
def _project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful coding assistant. For each request, provide a clear solution with "
    "step-by-step explanation, comments in code, and handle edge cases. Output the final code in a code block."
)


def run(
    model_name_or_path: str,
    input_file: str,
    output_file: str,
    max_samples: int | None = None,
    system_prompt: str | None = None,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    device: str = "cuda",
) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Resolve local path relative to project root (e.g. models/Qwen3-8B)
    if not model_name_or_path.startswith(("/", "~")) and not os.path.isdir(model_name_or_path):
        candidate = os.path.join(_project_root(), model_name_or_path)
        if os.path.isdir(candidate):
            model_name_or_path = candidate

    system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Loading model: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to(device)

    # Load seed prompts
    prompts = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            instruction = obj.get("instruction", "").strip()
            if instruction:
                prompts.append(instruction)
    if max_samples is not None:
        prompts = prompts[:max_samples]
    print(f"Generating for {len(prompts)} prompts -> {output_file}")

    results = []
    for i, instruction in enumerate(prompts):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        gen_kw: dict = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kw["do_sample"] = True
            gen_kw["temperature"] = temperature
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kw)
        if i == 0 and device == "cuda" and torch.cuda.is_available():
            mem_gb = torch.cuda.memory_allocated() / (1024**3)
            print(f"  [Inference on GPU, allocated: {mem_gb:.2f} GB]")
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        results.append({"instruction": instruction, "output": response.strip()})
        if (i + 1) % 10 == 0 or (i + 1) == len(prompts):
            print(f"  {i + 1}/{len(prompts)}")

    with open(output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("-" * 40)
    print(f"Done. Wrote {len(results)} pairs to {output_file}")
    print("-" * 40)
    return output_file


def main():
    root = _project_root()
    default_input = os.path.join(root, "data", "processed", "seed_prompts_random.jsonl")
    default_output = os.path.join(root, "data", "generated", "sft_pairs.jsonl")

    parser = argparse.ArgumentParser(
        description="Generate SFT (instruction, output) pairs with a small model on one GPU."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
        help="Model name or path (e.g. Qwen/Qwen3-8B or models/Qwen3-8B)",
    )
    parser.add_argument(
        "--input",
        default=default_input,
        help=f"Seed prompts JSONL (default: {default_input})",
    )
    parser.add_argument(
        "--output",
        default=default_output,
        help=f"Output JSONL (default: {default_output})",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of prompts (for quick test)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Max new tokens per response (default 2048)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default 0.7)",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Device (default cuda)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}. Run data_prep/sample_data.py first.")

    run(
        model_name_or_path=args.model,
        input_file=args.input,
        output_file=args.output,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device,
    )


if __name__ == "__main__":
    main()
