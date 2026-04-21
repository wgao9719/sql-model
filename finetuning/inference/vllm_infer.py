#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
import jsonlines
from typing import List, Dict
import torch
from vllm import LLM, SamplingParams


def load_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for obj in jsonlines.Reader(f):
            items.append(obj)
    return items


def _strip_evidence(prompt: str) -> str:
    """Drop BIRD evidence lines from the Question block, keep only the final question."""
    def repl(m):
        block = m.group(1)
        lines = [l for l in block.strip("\n").splitlines() if l.strip()]
        question_only = lines[-1] if lines else ""
        return f"Question:\n{question_only}\n\n"
    return re.sub(r"Question:\n(.*?)\n\nInstructions:", lambda m: repl(m) + "Instructions:", prompt, flags=re.DOTALL)


def _swap_profile_descriptions(prompt: str, profiles_dir: str, db_id: str) -> str:
    """Stub: needs profile_db.py + describe_columns.py outputs in profiles_dir."""
    raise NotImplementedError(
        "profile mode requires profile_db.py + describe_columns.py to be run first "
        "and descriptions/ to exist at --profiles_dir"
    )


def apply_metadata_mode(items: List[Dict], mode: str, profiles_dir: str = "") -> List[Dict]:
    """Transform prompts per experiment mode: baseline / no_evidence / profile."""
    if mode == "baseline":
        return items
    out = []
    for it in items:
        row = dict(it)
        p = row["prompt"]
        if mode == "no_evidence":
            p = _strip_evidence(p)
        elif mode == "profile":
            assert profiles_dir, "profile mode requires --profiles_dir"
            p = _swap_profile_descriptions(p, profiles_dir, row.get("db_id", ""))
        else:
            raise ValueError(f"unknown metadata_mode: {mode}")
        row["prompt"] = p
        out.append(row)
    return out

def write_jsonl(path: str, items: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

def write_index_json_map(path: str, pred_sql_list: List[str]):
    data = {str(i): s for i, s in enumerate(pred_sql_list)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
def batches(seq, bs):
    for i in range(0, len(seq), bs):
        yield seq[i:i+bs]


def sql_response_extract(response_string):
    """Extract SQL from a ```sqlite/sql/mysql/postgresql code block; fallback strips fences."""
    sqlite_pattern = re.compile(r"```[ \t]*sqlite\s*([\s\S]*?)```", re.IGNORECASE)
    mysql_pattern = re.compile(r"```[ \t]*mysql\s*([\s\S]*?)```", re.IGNORECASE)
    postgresql_pattern = re.compile(r"```[ \t]*postgresql\s*([\s\S]*?)```", re.IGNORECASE)
    sql_pattern = re.compile(r"```[ \t]*sql\s*([\s\S]*?)```", re.IGNORECASE)

    m = sqlite_pattern.search(response_string)
    if m:
        return m.group(1).strip()
    m = sql_pattern.search(response_string)
    if m:
        return m.group(1).strip()
    m = mysql_pattern.search(response_string)
    if m:
        return m.group(1).strip()
    m = postgresql_pattern.search(response_string)
    if m:
        return m.group(1).strip()
    return response_string.replace("```sql", "").replace("```sqlite", "").replace("```", "")


def run_infer(
    model_path: str,
    prompt_items: List[Dict],
    batch_size: int = 32,
    max_model_len: int = 15000,
    temperature: float = 0.0,
) -> List[str]:
    """
    Returns raw generated texts aligned with prompt_items.
    Each prompt item must contain a 'prompt' string.
    """
    # Heuristic tp_size / gpu_util by model name.
    mp = model_path.lower()
    if "72b" in mp or "70b" in mp:
        tp_size, gpu_util = 4, 0.95
    elif "qwen3-coder-30b" in mp or "a3b" in mp:
        tp_size, gpu_util = 1, 0.92
    elif "gemma3" in mp or "phi-4" in mp:
        tp_size, gpu_util = 2, 0.95
    else:
        tp_size, gpu_util = 1, 0.90

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_util,
        max_model_len=max_model_len,
        disable_custom_all_reduce=True,
    )
    tok = llm.get_tokenizer()

    sampling = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        max_tokens=3000,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        stop=["</FINAL_ANSWER>"],
        stop_token_ids=[tok.eos_token_id],
    )

    use_chat_template = ("sqlcoder-7b-2" not in mp)
    prompts = [it["prompt"] for it in prompt_items]
    generations = []

    for chunk in batches(prompts, batch_size):
        if use_chat_template:
            conversations = [
                tok.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in chunk
            ]
        else:
            conversations = chunk

        with torch.no_grad():
            outs = llm.generate(conversations, sampling)

        for o in outs:
            generations.append(o.outputs[0].text)

    return generations


def main():
    ap = argparse.ArgumentParser("vLLM inference + SQL postprocess")
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--prompt_path", type=str, required=True, help="JSONL with 'prompt' field")
    ap.add_argument("--output_path", type=str, required=True,
                    help="If endswith .json -> write {'0': 'sql', ...}; if .jsonl -> write per-line JSONL")
    ap.add_argument("--raw_output_path", type=str, default="", help="(optional) save raw responses JSONL")
    ap.add_argument("--gpu", type=str, default="0")
    ap.add_argument("--batch_size", type=int, default=50)
    ap.add_argument("--max_token_length", type=int, default=15000)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--metadata_mode", choices=["baseline", "no_evidence", "profile"],
                    default="baseline",
                    help="baseline=prompts as-is; no_evidence=strip BIRD evidence; "
                         "profile=swap descriptions from --profiles_dir (not yet wired up)")
    ap.add_argument("--profiles_dir", type=str, default="",
                    help="Dir with {db_id}.json profile/description files (for --metadata_mode=profile)")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    items = load_jsonl(args.prompt_path)
    assert len(items) > 0 and "prompt" in items[0], "Input must be JSONL with a 'prompt' field."

    # Apply metadata-mode transform before tokenization.
    items = apply_metadata_mode(items, args.metadata_mode, args.profiles_dir)
    print(f"[metadata_mode={args.metadata_mode}] {len(items)} prompts ready")

    # 1) inference
    raw_texts = run_infer(
        model_path=args.model_path,
        prompt_items=items,
        batch_size=args.batch_size,
        max_model_len=args.max_token_length,
        temperature=args.temperature,
    )

    # 2) (optional) dump raw
    if args.raw_output_path:
        raw_dump = []
        for it, txt in zip(items, raw_texts):
            row = dict(it)
            row["response"] = txt
            raw_dump.append(row)
        write_jsonl(args.raw_output_path, raw_dump)

    # 3) postprocess -> pred_sql list
    pred_sql_list = []
    final_rows = []
    for it, txt in zip(items, raw_texts):
        sql = sql_response_extract(txt) or ""
        pred_sql_list.append(sql)
        row = dict(it)
        row["response"] = txt
        row["pred_sql"] = sql
        row.pop("prompt", None)
        final_rows.append(row)

    # 4) write output: json map or jsonl
    if args.output_path.lower().endswith(".json"):
        write_index_json_map(args.output_path, pred_sql_list)
    else:
        write_jsonl(args.output_path, final_rows)

    print(f"[Done] {len(final_rows)} predictions -> {args.output_path}")
    if args.raw_output_path:
        print(f"[Raw ] {len(final_rows)} raw -> {args.raw_output_path}")

if __name__ == "__main__":
    main()
