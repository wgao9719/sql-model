#!/usr/bin/env python
"""
Client-mode replacement for vllm_infer.py. Hits a running vLLM OpenAI
server instead of loading the model in-process.

Reuses the SAME prompt JSONL schema and output format (raw JSONL + index-
keyed final JSON) as vllm_infer.py, so downstream eval scripts don't change.
"""
import os, re, json, time, argparse, asyncio
from typing import List, Dict
import jsonlines
from openai import AsyncOpenAI


def load_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for obj in jsonlines.Reader(f):
            items.append(obj)
    return items


def write_jsonl(path: str, items: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def write_index_json_map(path: str, pred_sql_list: List[str]):
    data = {str(i): s for i, s in enumerate(pred_sql_list)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# Same extractor as vllm_infer.py — catches ```sqlite, ```sql, ```mysql, ```postgresql.
_PATS = [
    re.compile(r"```[ \t]*sqlite\s*([\s\S]*?)```", re.IGNORECASE),
    re.compile(r"```[ \t]*sql\s*([\s\S]*?)```", re.IGNORECASE),
    re.compile(r"```[ \t]*mysql\s*([\s\S]*?)```", re.IGNORECASE),
    re.compile(r"```[ \t]*postgresql\s*([\s\S]*?)```", re.IGNORECASE),
]


def sql_response_extract(s: str) -> str:
    for p in _PATS:
        m = p.search(s)
        if m:
            return m.group(1).strip()
    return s.replace("```sql", "").replace("```sqlite", "").replace("```", "")


async def _one(client, model, prompt, max_tokens, temperature, sem):
    async with sem:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
        )
        return resp.choices[0].message.content or ""


async def run(args):
    items = load_jsonl(args.prompt_path)
    assert items and "prompt" in items[0]

    client = AsyncOpenAI(
        base_url=f"http://{args.host}:{args.port}/v1",
        api_key="EMPTY",
        timeout=args.timeout,
        max_retries=3,
    )
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()

    async def task(i, it):
        txt = await _one(client, args.model, it["prompt"],
                         args.max_tokens, args.temperature, sem)
        if (i + 1) % max(1, len(items) // 20) == 0 or i == len(items) - 1:
            print(f"  [{i+1}/{len(items)}]  {time.time()-t0:.1f}s", flush=True)
        return txt

    raw_texts = await asyncio.gather(*(task(i, it) for i, it in enumerate(items)))

    if args.raw_output_path:
        write_jsonl(args.raw_output_path,
                    [{**it, "response": r} for it, r in zip(items, raw_texts)])

    pred_sql_list = []
    final_rows = []
    for it, txt in zip(items, raw_texts):
        sql = sql_response_extract(txt) or ""
        pred_sql_list.append(sql)
        row = {**it, "response": txt, "pred_sql": sql}
        row.pop("prompt", None)
        final_rows.append(row)

    if args.output_path.lower().endswith(".json"):
        write_index_json_map(args.output_path, pred_sql_list)
    else:
        write_jsonl(args.output_path, final_rows)

    print(f"[Done] {len(final_rows)} predictions in {time.time()-t0:.1f}s -> {args.output_path}")


def main():
    ap = argparse.ArgumentParser("vLLM server client for BIRD-SQL inference")
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="qwen3-coder-30b")
    ap.add_argument("--prompt_path", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--raw_output_path", default="")
    ap.add_argument("--max_tokens", type=int, default=3000)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
