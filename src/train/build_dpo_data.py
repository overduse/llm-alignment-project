#!/usr/bin/env python3
"""
Build DPO preference data (instruction, chosen, rejected) for LLaMA Factory.

Input options:
  A) JSONL with (instruction, responses[], scores[]): pick best as chosen, worst as rejected.
  B) JSONL with (instruction, chosen, rejected): pass through to dpo_pairs.jsonl.

Usage:
  # From scored candidates (e.g. after SFT model + Judge)
  python src/train/build_dpo_data.py \
    --input data/generated/candidates_scored.jsonl \
    --output data/generated/dpo_pairs.jsonl

  # From existing chosen/rejected pairs
  python src/train/build_dpo_data.py \
    --input data/generated/dpo_raw.jsonl \
    --output data/generated/dpo_pairs.jsonl \
    --format pairs

Input format A (candidates_scored.jsonl), one JSON per line:
  {"instruction": "...", "responses": ["resp1", "resp2", ...], "scores": [8, 3, ...]}

Input format B (dpo_raw.jsonl):
  {"instruction": "...", "chosen": "...", "rejected": "..."}
"""
import os
import json
import argparse


def _project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_from_scores(rows: list[dict]) -> list[dict]:
    """From rows with instruction, responses, scores -> pick best/worst -> chosen, rejected.
    Ignores None scores (e.g. when Judge fails to parse); needs at least 2 valid scores per row."""
    out = []
    for row in rows:
        inst = row.get("instruction") or row.get("prompt")
        responses = row.get("responses", [])
        scores = row.get("scores", [])
        if not inst or len(responses) < 2 or len(scores) != len(responses):
            continue
        valid = [(i, s) for i, s in enumerate(scores) if s is not None]
        if len(valid) < 2:
            continue
        best_idx = max(valid, key=lambda x: x[1])[0]
        worst_idx = min(valid, key=lambda x: x[1])[0]
        if best_idx == worst_idx:
            continue
        chosen = responses[best_idx]
        rejected = responses[worst_idx]
        out.append({"instruction": inst, "chosen": chosen, "rejected": rejected})
    return out


def build_from_pairs(rows: list[dict]) -> list[dict]:
    """Pass through rows that already have instruction, chosen, rejected."""
    out = []
    for row in rows:
        inst = row.get("instruction") or row.get("prompt")
        chosen = row.get("chosen")
        rejected = row.get("rejected")
        if not inst or not chosen or not rejected:
            continue
        out.append({"instruction": inst, "chosen": chosen, "rejected": rejected})
    return out


def main():
    parser = argparse.ArgumentParser(description="Build DPO pairs for LLaMA Factory")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL (scored candidates or pairs)")
    parser.add_argument("--output", type=str, default="data/generated/dpo_pairs.jsonl", help="Output dpo_pairs.jsonl path")
    parser.add_argument(
        "--format",
        choices=("scores", "pairs"),
        default="scores",
        help="Input format: scores (instruction,responses,scores) or pairs (instruction,chosen,rejected)",
    )
    args = parser.parse_args()

    root = _project_root()
    inp = args.input if os.path.isabs(args.input) else os.path.join(root, args.input)
    out = args.output if os.path.isabs(args.output) else os.path.join(root, args.output)

    if not os.path.exists(inp):
        raise FileNotFoundError(f"Input not found: {inp}")

    rows = []
    with open(inp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if args.format == "scores":
        out_rows = build_from_scores(rows)
    else:
        out_rows = build_from_pairs(rows)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(out_rows)} DPO pairs to {out}")


if __name__ == "__main__":
    main()
