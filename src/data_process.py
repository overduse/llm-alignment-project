#!/usr/bin/env python3
"""
Day 0 弹药库准备：运行本脚本，读取 data/raw 中的原始代码数据，
进行清洗与格式化，产出约 2 万条高质量的代码 SFT 数据到 data/processed。
"""
from __future__ import annotations

import argparse
import os
import re

import pandas as pd


# 目标条数（与计划表一致）
TARGET_SIZE = 20_000

# 常见代码指令数据集的列名映射：(instruction_col, response_col) 或 None 表示自动推断
COLUMN_ALIASES = [
    ("instruction", "response"),
    ("prompt", "response"),
    ("input", "output"),
    ("question", "answer"),
    ("conversations", None),  # 需解析 JSON 对话
]


def infer_columns(df: pd.DataFrame) -> tuple[str, str] | None:
    """推断 instruction / response 列名。"""
    for inc, out in COLUMN_ALIASES:
        if out is None:
            continue
        if inc in df.columns and out in df.columns:
            return inc, out
    if "messages" in df.columns:
        return "messages", "messages"
    return None


def clean_text(s: str) -> str:
    """简单清洗：去首尾空白、合并多余空行。"""
    if not s or not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s


def basic_quality_filter(instruction: str, response: str) -> bool:
    """过滤过短、空或明显无效的样本。"""
    instruction = clean_text(instruction)
    response = clean_text(response)
    if len(instruction) < 10 or len(response) < 20:
        return False
    # 可在此增加更多规则：如必须包含代码块、语言检测等
    return True


def build_messages(instruction: str, response: str) -> list[dict]:
    """构建 chat 格式的 messages，便于后续 SFT。"""
    return [
        {"role": "user", "content": clean_text(instruction)},
        {"role": "assistant", "content": clean_text(response)},
    ]


def load_raw(raw_dir: str | None = None) -> pd.DataFrame:
    """从 data/raw 加载 parquet 或 json。"""
    raw_dir = raw_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "raw",
    )
    parquet_path = os.path.join(raw_dir, "train.parquet")
    json_path = os.path.join(raw_dir, "train.json")
    if os.path.isfile(parquet_path):
        return pd.read_parquet(parquet_path)
    if os.path.isfile(json_path):
        return pd.read_json(json_path, orient="records")
    raise FileNotFoundError(
        f"未找到原始数据，请先运行: python -m src.download_data\n期望路径: {parquet_path} 或 {json_path}"
    )


def process(df: pd.DataFrame, target_size: int = TARGET_SIZE) -> pd.DataFrame:
    """清洗并采样至 target_size 条，输出带 messages 的 DataFrame。"""
    cols = infer_columns(df)
    if cols is None:
        raise ValueError(
            f"无法推断 instruction/response 列，当前列: {list(df.columns)}。"
            "请确保数据含 instruction/response 或 prompt/response 等。"
        )
    inc_col, out_col = cols

    rows = []
    for _, r in df.iterrows():
        inc = r.get(inc_col, "")
        out = r.get(out_col, "")
        if pd.isna(inc):
            inc = ""
        if pd.isna(out):
            out = ""
        if isinstance(inc, list) or isinstance(out, list):
            continue  # 暂不处理 messages 列
        if not basic_quality_filter(str(inc), str(out)):
            continue
        rows.append({"messages": build_messages(str(inc), str(out))})
        if len(rows) >= target_size:
            break

    return pd.DataFrame(rows)


def save_processed(processed_df: pd.DataFrame, out_dir: str | None = None) -> str:
    """保存到 data/processed，格式为 train.parquet 和 train.jsonl（便于 LLaMA-Factory 等）。"""
    out_dir = out_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "processed",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_parquet = os.path.join(out_dir, "train.parquet")
    out_jsonl = os.path.join(out_dir, "train.jsonl")
    processed_df.to_parquet(out_parquet, index=False)
    processed_df.to_json(out_jsonl, orient="records", lines=True, force_ascii=False)
    print(f"已写入 {len(processed_df)} 条至 {out_parquet} 与 {out_jsonl}")
    return out_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description="清洗代码数据并产出约 2 万条 SFT 数据（Day 0）")
    parser.add_argument("--raw-dir", default=None, help="原始数据目录，默认: <project>/data/raw")
    parser.add_argument("--output-dir", default=None, help="输出目录，默认: <project>/data/processed")
    parser.add_argument("--target-size", type=int, default=TARGET_SIZE, help=f"目标条数，默认 {TARGET_SIZE}")
    args = parser.parse_args()

    df = load_raw(raw_dir=args.raw_dir)
    processed = process(df, target_size=args.target_size)
    save_processed(processed, out_dir=args.output_dir)


if __name__ == "__main__":
    main()
