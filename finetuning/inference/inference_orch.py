#!/usr/bin/env python
"""Toggleable BIRD-SQL inference orchestrator (linking / hints / values / mode)."""

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BIRD_SEP = "\t----- bird -----\t"


def _execute_sqlite(sql: str, db_path: str, timeout: float) -> Dict:
    import sqlite3
    from func_timeout import func_timeout, FunctionTimedOut
    if not sql.strip():
        return {"ok": False, "rows": [], "error": "empty sql"}

    def _run():
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            cur.execute(sql)
            return cur.fetchall()
        finally:
            conn.close()

    try:
        rows = func_timeout(timeout, _run)
        return {"ok": True, "rows": rows, "error": None}
    except FunctionTimedOut:
        return {"ok": False, "rows": [], "error": f"timeout after {timeout}s"}
    except Exception as e:
        return {"ok": False, "rows": [], "error": str(e)[:500]}


def _result_hash(rows) -> str:
    normalized = sorted(repr(tuple(r)) for r in rows)
    return hashlib.sha1("|".join(normalized).encode()).hexdigest()


def _db_path(db_root: str, db_id: str, dialect: str) -> str:
    if dialect != "SQLite":
        raise NotImplementedError(
            f"dialect={dialect} requires wiring evaluation_utils.connect_db"
        )
    return os.path.join(db_root, db_id, f"{db_id}.sqlite")


def _apply_chat(tok, prompt: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _generate(
    llm, prompts: List[str], temperature: float, n: int, max_tokens: int
) -> List[List[str]]:
    import torch
    from vllm import SamplingParams
    tok = llm.get_tokenizer()
    sampling = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        n=n,
        max_tokens=max_tokens,
        stop_token_ids=[tok.eos_token_id],
    )
    conversations = [_apply_chat(tok, p) for p in prompts]
    with torch.no_grad():
        outs = llm.generate(conversations, sampling)
    return [[o.text for o in out.outputs] for out in outs]


def _build_packet(row: Dict, linking: str, hints: bool, values: bool) -> str:
    if linking != "static" or hints or values:
        raise NotImplementedError("linking/hints/values require PLAN.md Phases A+B")
    return row["prompt"]


def _repair_loop(
    llm,
    base_prompt: str,
    initial_sql: str,
    db_path: str,
    max_turns: int,
    temperature: float,
    timeout: float,
    max_tokens: int,
) -> str:
    sql = initial_sql
    prompt = base_prompt
    for _ in range(max_turns):
        res = _execute_sqlite(sql, db_path, timeout)
        if res["ok"]:
            return sql
        feedback = f"[ERROR] {res['error']}"
        prompt = (
            f"{prompt}\n\nPrevious attempt:\n```sql\n{sql}\n```\n"
            f"{feedback}\nPlease produce a corrected SQL."
        )
        from vllm_infer import sql_response_extract
        out = _generate(llm, [prompt], temperature=temperature, n=1, max_tokens=max_tokens)
        new_sql = sql_response_extract(out[0][0])
        if not new_sql:
            break
        sql = new_sql
    return sql


def _sc_vote(candidates: List[str], db_path: str, timeout: float) -> str:
    buckets: Dict[str, List[str]] = {}
    for sql in candidates:
        if not sql:
            continue
        res = _execute_sqlite(sql, db_path, timeout)
        if not res["ok"] or not res["rows"]:
            continue
        h = _result_hash(res["rows"])
        buckets.setdefault(h, []).append(sql)
    if not buckets:
        return candidates[0] if candidates else ""
    best = max(buckets.keys(), key=lambda h: (len(buckets[h]), -min(len(s) for s in buckets[h])))
    return min(buckets[best], key=len)


def main():
    ap = argparse.ArgumentParser("inference-orch")
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--prompt_path", type=str, required=True)
    ap.add_argument("--output_path", type=str, required=True)
    ap.add_argument("--raw_output_path", type=str, default="")
    ap.add_argument("--db_root", type=str, default="")
    ap.add_argument("--dialect", type=str, default="SQLite",
                    choices=["SQLite", "MySQL", "PostgreSQL"])

    ap.add_argument("--linking", choices=["static", "bm25", "sql_induced"], default="static")
    ap.add_argument("--hints", choices=["off", "on"], default="off")
    ap.add_argument("--values", choices=["off", "on"], default="off")
    ap.add_argument("--mode", choices=["single", "repair", "sc", "repair_sc"], default="single")
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--max_repair_turns", type=int, default=3)
    ap.add_argument("--sc_temperature", type=float, default=0.7)
    ap.add_argument("--exec_timeout", type=float, default=10.0)

    ap.add_argument("--gpu", type=str, default="0")
    ap.add_argument("--batch_size", type=int, default=50)
    ap.add_argument("--max_token_length", type=int, default=15000)
    ap.add_argument("--max_tokens", type=int, default=3000)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--tp_size", type=int, default=1)
    ap.add_argument("--gpu_util", type=float, default=0.9)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.mode != "single" and not args.db_root:
        ap.error(f"--db_root is required for --mode {args.mode}")

    from vllm import LLM
    from vllm_infer import load_jsonl, write_jsonl, sql_response_extract, batches

    items = load_jsonl(args.prompt_path)
    assert items and "prompt" in items[0] and "db_id" in items[0], \
        "prompt JSONL must have 'prompt' and 'db_id' fields"

    for it in items:
        it["prompt"] = _build_packet(
            it, linking=args.linking,
            hints=(args.hints == "on"), values=(args.values == "on"),
        )

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_token_length,
        disable_custom_all_reduce=True,
    )

    n = args.n_samples if args.mode in ("sc", "repair_sc") else 1
    gen_temp = args.sc_temperature if n > 1 else args.temperature

    raw_all: List[List[str]] = []
    for chunk in batches([it["prompt"] for it in items], args.batch_size):
        raw_all.extend(_generate(llm, chunk, temperature=gen_temp, n=n, max_tokens=args.max_tokens))

    final_sqls: List[str] = []
    for it, candidates in zip(items, raw_all):
        extracted = [sql_response_extract(r) or "" for r in candidates]
        db_path = _db_path(args.db_root, it["db_id"], args.dialect) if args.mode != "single" else None

        if args.mode == "single":
            sql = extracted[0]
        elif args.mode == "repair":
            sql = _repair_loop(
                llm, it["prompt"], extracted[0], db_path,
                args.max_repair_turns, args.temperature,
                args.exec_timeout, args.max_tokens,
            )
        elif args.mode == "sc":
            sql = _sc_vote(extracted, db_path, args.exec_timeout)
        elif args.mode == "repair_sc":
            repaired = [
                _repair_loop(
                    llm, it["prompt"], s, db_path,
                    args.max_repair_turns, args.temperature,
                    args.exec_timeout, args.max_tokens,
                )
                for s in extracted
            ]
            sql = _sc_vote(repaired, db_path, args.exec_timeout)
        final_sqls.append(sql)

    if args.output_path.endswith(".json"):
        payload = {
            str(i): f"{sql}{BIRD_SEP}{it['db_id']}"
            for i, (sql, it) in enumerate(zip(final_sqls, items))
        }
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    else:
        rows = []
        for sql, it in zip(final_sqls, items):
            row = {k: v for k, v in it.items() if k != "prompt"}
            row["pred_sql"] = sql
            row["pred_with_sep"] = f"{sql}{BIRD_SEP}{it['db_id']}"
            rows.append(row)
        write_jsonl(args.output_path, rows)

    if args.raw_output_path:
        raw_dump = [
            {**{k: v for k, v in it.items() if k != "prompt"}, "candidates": cands}
            for it, cands in zip(items, raw_all)
        ]
        write_jsonl(args.raw_output_path, raw_dump)

    print(f"[inference-orch] {len(final_sqls)} predictions -> {args.output_path}")
    print(
        f"  linking={args.linking} hints={args.hints} values={args.values} "
        f"mode={args.mode} n={n}"
    )


if __name__ == "__main__":
    main()
