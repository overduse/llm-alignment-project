import os
import json
import argparse
from typing import List

def _project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser(
        description="Offline scoring of generated candidates using a local Judge model."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input JSONL file (output of generate_dpo_candidates_offline.py).",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output JSONL file containing candidates with scores.",
    )
    parser.add_argument(
        "--judge-model", type=str, default="models/Qwen3-8B-Base",
        help="Path to the local Judge model.",
    )
    args = parser.parse_args()

    root = _project_root()
    # Import the local scoring function from your existing script
    import sys
    sys.path.insert(0, root)
    from src.generate.score_sft_data import _score_via_local

    input_path = args.input if os.path.isabs(args.input) else os.path.join(root, args.input)
    output_path = args.output if os.path.isabs(args.output) else os.path.join(root, args.output)
    judge_path = args.judge_model if os.path.isabs(args.judge_model) else os.path.join(root, args.judge_model)

    # 1. Load candidates
    candidates = []
    print(f"Loading generated candidates from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            candidates.append(json.loads(line))

    # 2. Score them sequentially using the local model
    print(f"Starting scoring using local judge: {judge_path}")
    print("This will load the model onto your GPUs (make sure vLLM is closed to free VRAM).")
    
    scored = []
    for i, item in enumerate(candidates):
        inst = item["instruction"]
        responses = item["responses"]
        scores = []
        for j, resp in enumerate(responses):
            if not resp.strip():
                scores.append(None)
                continue
            # Use your exact same judge logic, offline
            s = _score_via_local(inst, resp, judge_path, "cuda")
            scores.append(s)
            
        scored.append({"instruction": inst, "responses": responses, "scores": scores})
        if (i + 1) % 10 == 0 or (i + 1) == len(candidates):
            print(f"  Judged {i + 1}/{len(candidates)} prompts...")

    # 3. Save scored candidates
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in scored:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"✅ Scoring complete! Saved to {output_path}")
    print("Next step: Run 'python src/train/build_dpo_data.py --input data/generated/candidates_scored.jsonl --output data/generated/dpo_pairs.jsonl --format scores'")

if __name__ == "__main__":
    main()
