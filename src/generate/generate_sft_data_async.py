#!/usr/bin/env python3
"""
Generate SFT (instruction, output) pairs by async requests to a vLLM OpenAI-compatible server.
PROJECT_PLAN_V2: 编写异步请求脚本，向本地部署的 80B vLLM 服务发送那 2 万条 Prompt。
Reads data/processed/seed_prompts_random.jsonl, POSTs to server, writes data/generated/sft_pairs.jsonl.
"""
import os
import json
import argparse
import asyncio
from typing import List, Tuple

def _project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful coding assistant. For each request, provide a clear solution with "
    "step-by-step explanation, comments in code, and handle edge cases. Output the final code in a code block."
)


async def request_one(
    client: "OpenAI.AsyncOpenAI",
    model: str,
    instruction: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
) -> Tuple[str, str]:
    async with semaphore:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
        ]
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 0.0,
        )
        text = (resp.choices[0].message.content or "").strip()
        return instruction, text


async def run_async(
    api_base: str,
    model_name: str,
    input_file: str,
    output_file: str,
    max_samples: int | None,
    max_concurrent: int,
    max_new_tokens: int,
    temperature: float,
) -> str:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError("pip install openai")

    client = AsyncOpenAI(base_url=api_base, api_key="EMPTY")
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    prompts = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            inst = obj.get("instruction", "").strip()
            if inst:
                prompts.append(inst)
    if max_samples is not None:
        prompts = prompts[:max_samples]

    print(f"Async generating {len(prompts)} prompts (concurrency={max_concurrent}) -> {output_file}")
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        request_one(client, model_name, inst, DEFAULT_SYSTEM_PROMPT, max_new_tokens, temperature, semaphore)
        for inst in prompts
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    ok = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"  Error at {i}: {r}")
            continue
        inst, text = r
        ok.append({"instruction": inst, "output": text})
        if (len(ok)) % 500 == 0 or len(ok) == len(prompts):
            print(f"  {len(ok)}/{len(prompts)}")

    with open(output_file, "w", encoding="utf-8") as f:
        for item in ok:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("-" * 40)
    print(f"Done. Wrote {len(ok)} pairs to {output_file}")
    print("-" * 40)
    return output_file


def main():
    root = _project_root()
    default_input = os.path.join(root, "data", "processed", "seed_prompts_random.jsonl")
    default_output = os.path.join(root, "data", "generated", "sft_pairs.jsonl")

    parser = argparse.ArgumentParser(
        description="Generate SFT pairs via async requests to vLLM server (V2 style)."
    )
    parser.add_argument("--api-base", default="http://localhost:8000/v1", help="vLLM OpenAI API base URL")
    parser.add_argument("--model", default="Qwen3-Next-80B-A3B-Instruct", help="API model name")
    parser.add_argument("--input", default=default_input)
    parser.add_argument("--output", default=default_output)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-concurrent", type=int, default=64, help="Max concurrent requests")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")

    asyncio.run(
        run_async(
            api_base=args.api_base,
            model_name=args.model,
            input_file=args.input,
            output_file=args.output,
            max_samples=args.max_samples,
            max_concurrent=args.max_concurrent,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    )


if __name__ == "__main__":
    main()
