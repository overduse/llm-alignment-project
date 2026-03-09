"""
Batch inference script using vLLM for instruction-following or chat models.

Reads prompts from a JSONL file (field: "instruction"), runs batched generation
with vLLM (optionally with tensor parallelism), applies chat template, and
writes instruction + generated output to a JSONL file.
"""

import os
import json
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Run batch inference with vLLM on instruction JSONL input."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model (e.g. /mnt/models/... on ACP).",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file; each line must have an 'instruction' field.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file (instruction + output per line).",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=4,
        help="Tensor parallel size (number of GPUs).",
    )
    args = parser.parse_args()

    # Load all prompts from input JSONL (expect "instruction" field per line)
    prompts = []
    with open(args.input, "r") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line)["instruction"])

    print(f"Loaded {len(prompts)} prompts.")

    # Initialize vLLM engine (tensor_parallel_size splits model across GPUs)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        max_model_len=4096,  # Reduce to 2048 if OOM
        gpu_memory_utilization=0.90,
        enforce_eager=False,
    )

    # Sampling settings for generation
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        max_tokens=4096,
        repetition_penalty=1.05,
        stop=["<|endoftext|>", "<|im_end|>"],  # Qwen stop tokens
    )

    # Apply chat template so base/instruct models get correct system/user format
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    formatted_prompts = []
    for p in prompts:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": p},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        formatted_prompts.append(text)

    # Run batched inference (vLLM handles scheduling)
    print("Starting batch inference...")
    outputs = llm.generate(formatted_prompts, sampling_params)

    # Write results: one JSON object per line (instruction + output)
    data_to_save = []
    for prompt, output in zip(prompts, outputs):
        generated_text = output.outputs[0].text
        data_to_save.append({
            "instruction": prompt,
            "output": generated_text,
        })

    with open(args.output, "w") as f:
        for item in data_to_save:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Done! Saved to {args.output}")


if __name__ == "__main__":
    main()
