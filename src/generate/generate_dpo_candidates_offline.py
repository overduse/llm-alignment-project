"""
Batch generation script for DPO candidates using vLLM for instruction-following models.

This script runs completely offline. It reads prompts from a JSONL file, uses vLLM 
to generate multiple candidates per prompt (using N=num_candidates), and writes 
the results to an output JSONL file ready for the scoring phase.

Usage:
  python src/generate/generate_dpo_candidates_offline.py \
    --model checkpoints/qwen3-8b-sft \
    --input data/processed/dpo_seed_prompts.jsonl \
    --output data/generated/candidates_unscored.jsonl \
    --tp 4 \
    --num-candidates 4
"""

import os
import json
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(
        description="Run batch offline inference with vLLM to generate DPO candidates."
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to the trained SFT model (e.g. checkpoints/qwen3-8b-sft).",
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input JSONL file containing the instructions.",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output JSONL file (instruction + responses list per line).",
    )
    parser.add_argument(
        "--tp", type=int, default=4,
        help="Tensor parallel size (number of GPUs, default: 4 for H100s).",
    )
    parser.add_argument(
        "--num-candidates", type=int, default=4,
        help="Number of responses to generate per instruction.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (higher = more diverse candidates).",
    )
    args = parser.parse_args()

    # Load all prompts from input JSONL
    prompts = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                inst = obj.get("instruction", obj.get("prompt", "")).strip()
                if inst:
                    prompts.append(inst)

    print(f"Loaded {len(prompts)} prompts. Will generate {args.num_candidates} candidates each.")

    # Initialize vLLM engine
    print(f"Initializing vLLM engine with TP={args.tp}...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
    )

    # Apply chat template
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # We use a custom system prompt to match our SFT phase
    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful coding assistant. For each request, provide a clear solution with "
        "step-by-step explanation, comments in code, and handle edge cases. Output the final code in a code block."
    )

    # Temporary fallback if chat template fails
    try:
        formatted_prompts = []
        for p in prompts:
            messages = [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": p},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            formatted_prompts.append(text)
    except Exception as e:
        print(f"Warning: Failed to apply chat template ({e}). Using raw prompts instead.")
        formatted_prompts = [f"<|im_start|>system\n{DEFAULT_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n" for p in prompts]

    # Sampling settings for generation (generate multiple sequences at once)
    sampling_params = SamplingParams(
        n=args.num_candidates,            # Crucial: generate N responses per prompt
        temperature=args.temperature,     # Crucial: > 0 for diversity
        top_p=0.9,
        max_tokens=2048,
        stop=["<|endoftext|>", "<|im_end|>"], 
    )

    # Run batched inference (vLLM handles everything efficiently)
    print("Starting highly parallel offline batch inference...")
    outputs = llm.generate(formatted_prompts, sampling_params)

    # Write results: one JSON object per line (instruction + list of responses)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    data_to_save = []
    for prompt, output in zip(prompts, outputs):
        # Extract the N generated texts for this prompt
        responses = [out.text.strip() for out in output.outputs]
        data_to_save.append({
            "instruction": prompt,
            "responses": responses,
        })

    with open(args.output, "w", encoding="utf-8") as f:
        for item in data_to_save:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Done! Generated candidates saved to {args.output}")
    print(f"Next step: Run src/generate/score_sft_data.py to score these candidates!")

if __name__ == "__main__":
    main()
