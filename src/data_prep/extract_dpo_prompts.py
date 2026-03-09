import os
import json
import random
import argparse

def _project_root():
    """Return the absolute path to the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser(description="Extract unused prompts for DPO dataset generation.")
    parser.add_argument("--num", type=int, default=10000, help="Number of prompts to extract.")
    args = parser.parse_args()

    root = _project_root()
    sft_file = os.path.join(root, "data", "generated", "sft_pairs_20k.jsonl")
    all_prompts_file = os.path.join(root, "data", "processed", "clean_seed_prompts.jsonl")
    output_file = os.path.join(root, "data", "processed", "dpo_seed_prompts.jsonl")

    # 1. Load instructions used during the SFT phase to prevent data leakage
    sft_seen_instructions = set()
    print("Loading instructions used in the SFT phase...")
    if os.path.exists(sft_file):
        with open(sft_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): 
                    continue
                obj = json.loads(line)
                # Handle both 'instruction' and 'prompt' keys gracefully
                inst = obj.get("instruction", obj.get("prompt", "")).strip()
                sft_seen_instructions.add(inst)
        print(f"Loaded {len(sft_seen_instructions)} seen instructions to act as a blacklist.")
    else:
        print(f"Warning: SFT data {sft_file} not found. Skipping filtering.")

    # 2. Extract unseen prompts from the full seed dataset
    unused_prompts = []
    print("Extracting unused prompts from the full dataset...")
    with open(all_prompts_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): 
                continue
            obj = json.loads(line)
            inst = obj.get("instruction", obj.get("prompt", "")).strip()
            
            # Keep the prompt only if it was not used in SFT
            if inst and inst not in sft_seen_instructions:
                unused_prompts.append(obj)

    print(f"Found {len(unused_prompts)} unused prompts available for DPO.")

    # 3. Sample the requested number of prompts randomly
    if len(unused_prompts) < args.num:
        print(f"Warning: Only {len(unused_prompts)} prompts available. Extracting all.")
        sampled_prompts = unused_prompts
    else:
        sampled_prompts = random.sample(unused_prompts, args.num)

    # 4. Save the sampled prompts to the output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for p in sampled_prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"✅ Successfully extracted {len(sampled_prompts)} unused prompts to {output_file}")

if __name__ == "__main__":
    main()
