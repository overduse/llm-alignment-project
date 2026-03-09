#!/usr/bin/env python3
"""
Score DPO candidates using 80B Judge with vLLM offline (no API server).
Loads the Judge model once with tensor parallelism, runs batch inference for all
(instruction, response) pairs, parses scores, and writes candidates_scored.jsonl.

Usage (4-card node):
  python src/generate/score_candidates_vllm_offline.py \
    --input data/generated/candidates_unscored.jsonl \
    --output data/generated/candidates_scored.jsonl \
    --judge-model models/Qwen3-Next-80B-A3B-Instruct \
    --tp 4
"""
import os
import sys
import json
import argparse

def _project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse Judge prompt and score parsing from score_sft_data
sys.path.insert(0, _project_root())
from src.generate.score_sft_data import JUDGE_SYSTEM, JUDGE_USER_TEMPLATE, _parse_score


def main():
    parser = argparse.ArgumentParser(
        description="Score candidates with 80B Judge via vLLM offline (no server)."
    )
    parser.add_argument("--input", type=str, required=True,
        help="Input JSONL: candidates_unscored.jsonl")
    parser.add_argument("--output", type=str, required=True,
        help="Output JSONL: candidates_scored.jsonl")
    parser.add_argument("--judge-model", type=str, required=True,
        help="Path to 80B Judge model (e.g. models/Qwen3-Next-80B-A3B-Instruct)")
    parser.add_argument("--tp", type=int, default=4,
        help="Tensor parallel size (number of GPUs)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="Batch size for vLLM generate (reduce if OOM)")
    args = parser.parse_args()

    root = _project_root()
    input_path = args.input if os.path.isabs(args.input) else os.path.join(root, args.input)
    output_path = args.output if os.path.isabs(args.output) else os.path.join(root, args.output)
    judge_path = args.judge_model if os.path.isabs(args.judge_model) else os.path.join(root, args.judge_model)

    # 1. Load candidates
    candidates = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            candidates.append(json.loads(line))
    print(f"Loaded {len(candidates)} prompts (each with multiple responses).")

    # 2. Build flat list of (index_in_candidates, index_in_responses) and Judge prompt strings
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(judge_path, trust_remote_code=True)

    prompt_tuples = []  # (i, j) for candidate i, response j
    prompt_strings = []
    for i, item in enumerate(candidates):
        inst = item["instruction"]
        for j, resp in enumerate(item["responses"]):
            if not resp.strip():
                prompt_tuples.append((i, j))
                prompt_strings.append("")  # placeholder, will yield None score
                continue
            user_text = JUDGE_USER_TEMPLATE.format(
                instruction=inst[:8000],
                output=resp[:8000],
            )
            messages = [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_text},
            ]
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                text = f"{JUDGE_SYSTEM}\n\n{user_text}\n\nScore:"
            prompt_tuples.append((i, j))
            prompt_strings.append(text)

    # Filter to only non-empty prompts for inference (we'll fill None for empty later)
    to_run = [(idx, t) for idx, t in enumerate(prompt_tuples) if prompt_strings[idx]]
    prompts_to_run = [prompt_strings[idx] for idx, _ in to_run]
    print(f"Total score requests: {len(prompt_tuples)} (non-empty: {len(prompts_to_run)})")

    # 3. Load vLLM and run batch inference (no server)
    from vllm import LLM, SamplingParams
    print(f"Loading Judge model with tp={args.tp} (this may take a few minutes)...")
    llm = LLM(
        model=judge_path,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.90,
    )
    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)

    all_scores_flat = [None] * len(prompt_tuples)  # default None for empty responses
    for start in range(0, len(prompts_to_run), args.batch_size):
        end = min(start + args.batch_size, len(prompts_to_run))
        batch = prompts_to_run[start:end]
        outputs = llm.generate(batch, sampling_params)
        for k, out in enumerate(outputs):
            text = out.outputs[0].text.strip()
            score = _parse_score(text)
            orig_idx = to_run[start + k][0]
            all_scores_flat[orig_idx] = score
        print(f"  Scored {end}/{len(prompts_to_run)} requests...")

    # 4. Map flat scores back to (instruction, responses, scores) per row
    idx = 0
    scored = []
    for item in candidates:
        n = len(item["responses"])
        scores = all_scores_flat[idx : idx + n]
        idx += n
        scored.append({
            "instruction": item["instruction"],
            "responses": item["responses"],
            "scores": scores,
        })

    # 5. Write output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in scored:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"✅ Saved {len(scored)} rows to {output_path}")
    print("Next: python src/train/build_dpo_data.py --input ... --output data/generated/dpo_pairs.jsonl --format scores")


if __name__ == "__main__":
    main()
