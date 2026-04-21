"""Modal SC + Repair inference orchestrator for BIRD-SQL (base Qwen3-Coder)."""

import json
import re
import sys
from pathlib import Path

import modal

MODEL_ID = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
MODEL_DIR = "/weights/Qwen3-Coder-30B-A3B-Instruct"
GPU = "A100-80GB"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.11.2",
        "torch==2.9.0",
        "transformers>=4.46",
        "huggingface_hub>=0.26",
        "jsonlines",
    )
)
weights_vol = modal.Volume.from_name("qwen3-coder-weights", create_if_missing=True)
app = modal.App("qwen3-coder-bird-sc", image=image)


_PATS = [
    re.compile(r"```[ \t]*sqlite\s*([\s\S]*?)```", re.IGNORECASE),
    re.compile(r"```[ \t]*sql\s*([\s\S]*?)```", re.IGNORECASE),
    re.compile(r"```[ \t]*mysql\s*([\s\S]*?)```", re.IGNORECASE),
    re.compile(r"```[ \t]*postgresql\s*([\s\S]*?)```", re.IGNORECASE),
]


def _extract_sql(s: str) -> str:
    for p in _PATS:
        m = p.search(s)
        if m:
            return m.group(1).strip()
    return s.replace("```sql", "").replace("```sqlite", "").replace("```", "")


@app.cls(
    gpu=GPU,
    volumes={"/weights": weights_vol},
    scaledown_window=600,
    timeout=60 * 60,
    memory=64 * 1024,
)
class Qwen3SC:
    @modal.enter()
    def load(self):
        from vllm import LLM, SamplingParams
        print("[enter] loading vLLM on", GPU)
        self.llm = LLM(
            model=MODEL_DIR,
            dtype="bfloat16",
            gpu_memory_utilization=0.92,
            max_model_len=16000,
            trust_remote_code=True,
        )
        self._SamplingParams = SamplingParams
        print("[enter] vLLM ready")

    @modal.method()
    def generate_sc(self, prompts, n: int, max_tokens: int, temperature: float):
        sp = self._SamplingParams(
            temperature=temperature, top_p=1.0, n=n, max_tokens=max_tokens,
        )
        msgs = [[{"role": "user", "content": p}] for p in prompts]
        outs = self.llm.chat(msgs, sampling_params=sp)
        return [[o.text for o in out.outputs] for out in outs]


@app.local_entrypoint()
def eval_sc(
    prompt_file: str,
    out_file: str,
    db_root: str,
    raw_out: str = "",
    n_samples: int = 8,
    max_tokens: int = 3000,
    temperature: float = 0.7,
    exec_timeout: float = 10.0,
    num_workers: int = 8,
):
    """Generate n candidates per prompt on A100, vote locally using real SQLite DBs."""
    import jsonlines
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from inference_orch import _sc_vote, _db_path, BIRD_SEP

    prompts_path = Path(prompt_file).resolve()
    out_path = Path(out_file).resolve()
    db_root_abs = str(Path(db_root).resolve())
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = []
    with jsonlines.open(prompts_path) as r:
        for obj in r:
            items.append(obj)
    assert items and "prompt" in items[0] and "db_id" in items[0], \
        "prompt JSONL must have 'prompt' and 'db_id'"
    prompts = [it["prompt"] for it in items]
    print(f"[eval_sc] {len(prompts)} prompts  n={n_samples}  T={temperature}")

    srv = Qwen3SC()
    raw_all = srv.generate_sc.remote(
        prompts, n=n_samples, max_tokens=max_tokens, temperature=temperature,
    )
    print(f"[eval_sc] got {len(raw_all)} x {n_samples} candidates")

    if raw_out:
        raw_path = Path(raw_out).resolve()
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with open(raw_path, "w") as f:
            for it, cands in zip(items, raw_all):
                row = {**{k: v for k, v in it.items() if k != "prompt"}, "candidates": cands}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[eval_sc] raw -> {raw_path}")

    vote_jobs = []
    for it, cands in zip(items, raw_all):
        extracted = [_extract_sql(c) or "" for c in cands]
        db_path = _db_path(db_root_abs, it["db_id"], "SQLite")
        vote_jobs.append((extracted, db_path, exec_timeout))

    from concurrent.futures import ProcessPoolExecutor
    print(f"[eval_sc] voting with {num_workers} workers over {len(vote_jobs)} questions")
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        final_sqls = list(pool.map(_vote_worker, vote_jobs))

    from collections import Counter
    bucket_sizes = Counter()
    for sql, (cands, _, _) in zip(final_sqls, vote_jobs):
        bucket_sizes[cands.count(sql) if sql in cands else 0] += 1
    print(f"[eval_sc] winning-bucket size histogram: {dict(sorted(bucket_sizes.items()))}")

    payload = {
        str(i): f"{sql}{BIRD_SEP}{it['db_id']}"
        for i, (sql, it) in enumerate(zip(final_sqls, items))
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[eval_sc] {len(final_sqls)} predictions -> {out_path}")


def _vote_worker(args):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from inference_orch import _sc_vote
    extracted, db_path, exec_timeout = args
    return _sc_vote(extracted, db_path, exec_timeout)


def _exec_worker(args):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from inference_orch import _execute_sqlite
    sql, db_path, timeout = args
    return _execute_sqlite(sql, db_path, timeout)


@app.local_entrypoint()
def eval_repair_sc(
    prompt_file: str,
    raw_in: str,
    out_file: str,
    db_root: str,
    raw_out: str = "",
    max_repair_turns: int = 2,
    max_tokens: int = 3000,
    repair_temperature: float = 0.3,
    exec_timeout: float = 5.0,
    num_workers: int = 8,
):
    """Repair + SC: load raw SC candidates, repair exec_error ones, vote."""
    import jsonlines
    from concurrent.futures import ProcessPoolExecutor

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from inference_orch import _sc_vote, _db_path, BIRD_SEP

    prompts_path = Path(prompt_file).resolve()
    raw_path = Path(raw_in).resolve()
    out_path = Path(out_file).resolve()
    db_root_abs = str(Path(db_root).resolve())
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = []
    with jsonlines.open(prompts_path) as r:
        for obj in r:
            items.append(obj)
    base_prompts = [it["prompt"] for it in items]
    db_paths = [_db_path(db_root_abs, it["db_id"], "SQLite") for it in items]
    n_items = len(items)

    raw_rows = []
    with open(raw_path) as f:
        for line in f:
            line = line.strip()
            if line:
                raw_rows.append(json.loads(line))
    assert len(raw_rows) == n_items, \
        f"raw ({len(raw_rows)}) and prompt ({n_items}) length mismatch"
    cand_texts = [list(r["candidates"]) for r in raw_rows]
    n_samples = len(cand_texts[0])
    print(f"[repair_sc] {n_items} questions x {n_samples} candidates  max_repair_turns={max_repair_turns}")

    cand_sqls = [[_extract_sql(c) or "" for c in row] for row in cand_texts]

    srv = Qwen3SC()

    for turn in range(1, max_repair_turns + 1):
        flat_args = [
            (cand_sqls[q][c], db_paths[q], exec_timeout)
            for q in range(n_items) for c in range(n_samples)
        ]
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            flat_results = list(pool.map(_exec_worker, flat_args, chunksize=16))

        failures = [
            (q, c)
            for q in range(n_items) for c in range(n_samples)
            if not flat_results[q * n_samples + c]["ok"]
        ]
        print(f"[repair_sc] turn {turn}: {len(failures)} error-candidates across {n_items} questions")
        if not failures:
            break

        repair_prompts = []
        for q, c in failures:
            idx = q * n_samples + c
            err = flat_results[idx]["error"]
            repair_prompts.append(
                f"{base_prompts[q]}\n\nPrevious attempt:\n```sql\n{cand_sqls[q][c]}\n```\n"
                f"[ERROR] {err}\nPlease produce a corrected SQL."
            )
        regen = srv.generate_sc.remote(
            repair_prompts, n=1, max_tokens=max_tokens, temperature=repair_temperature,
        )
        for (q, c), reg in zip(failures, regen):
            new_sql = _extract_sql(reg[0]) or ""
            cand_texts[q][c] = reg[0]
            if new_sql:
                cand_sqls[q][c] = new_sql

    print(f"[repair_sc] voting with {num_workers} workers", flush=True)
    vote_jobs = [
        (cand_sqls[q], db_paths[q], exec_timeout)
        for q in range(n_items)
    ]
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        final_sqls = list(pool.map(_vote_worker, vote_jobs, chunksize=4))

    payload = {
        str(i): f"{final_sqls[i]}{BIRD_SEP}{items[i]['db_id']}"
        for i in range(n_items)
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[repair_sc] {n_items} predictions -> {out_path}")

    if raw_out:
        raw_outp = Path(raw_out).resolve()
        raw_outp.parent.mkdir(parents=True, exist_ok=True)
        with open(raw_outp, "w") as f:
            for it, cands in zip(items, cand_texts):
                row = {**{k: v for k, v in it.items() if k != "prompt"}, "candidates": cands}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[repair_sc] raw -> {raw_outp}")


@app.local_entrypoint()
def eval_repair(
    prompt_file: str,
    out_file: str,
    db_root: str,
    raw_out: str = "",
    max_repair_turns: int = 3,
    max_tokens: int = 3000,
    temperature: float = 0.0,
    exec_timeout: float = 10.0,
    num_workers: int = 8,
):
    """Single-sample generation with multi-turn execution-feedback repair."""
    import jsonlines
    from concurrent.futures import ProcessPoolExecutor

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from inference_orch import _db_path, BIRD_SEP

    prompts_path = Path(prompt_file).resolve()
    out_path = Path(out_file).resolve()
    db_root_abs = str(Path(db_root).resolve())
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = []
    with jsonlines.open(prompts_path) as r:
        for obj in r:
            items.append(obj)
    assert items and "prompt" in items[0] and "db_id" in items[0], \
        "prompt JSONL must have 'prompt' and 'db_id'"
    n_items = len(items)
    base_prompts = [it["prompt"] for it in items]
    db_paths = [_db_path(db_root_abs, it["db_id"], "SQLite") for it in items]
    print(f"[eval_repair] {n_items} prompts  max_turns={max_repair_turns}  T={temperature}")

    srv = Qwen3SC()
    turn0 = srv.generate_sc.remote(
        base_prompts, n=1, max_tokens=max_tokens, temperature=temperature,
    )
    current_sqls = [_extract_sql(r[0]) or "" for r in turn0]
    trace = [[turn0[i][0]] for i in range(n_items)]

    for turn in range(1, max_repair_turns + 1):
        exec_args = [(current_sqls[i], db_paths[i], exec_timeout) for i in range(n_items)]
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            exec_results = list(pool.map(_exec_worker, exec_args))

        fail_idxs = [i for i, r in enumerate(exec_results) if not r["ok"]]
        print(f"[eval_repair] turn {turn}: {len(fail_idxs)}/{n_items} need repair")
        if not fail_idxs:
            break

        repair_prompts = []
        for i in fail_idxs:
            r = exec_results[i]
            feedback = f"[ERROR] {r['error']}"
            repair_prompts.append(
                f"{base_prompts[i]}\n\nPrevious attempt:\n```sql\n{current_sqls[i]}\n```\n"
                f"{feedback}\nPlease produce a corrected SQL."
            )
        regen = srv.generate_sc.remote(
            repair_prompts, n=1, max_tokens=max_tokens, temperature=temperature,
        )
        for i, reg in zip(fail_idxs, regen):
            new_sql = _extract_sql(reg[0]) or ""
            trace[i].append(reg[0])
            if new_sql:
                current_sqls[i] = new_sql

    payload = {
        str(i): f"{current_sqls[i]}{BIRD_SEP}{items[i]['db_id']}"
        for i in range(n_items)
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[eval_repair] {n_items} predictions -> {out_path}")

    if raw_out:
        raw_path = Path(raw_out).resolve()
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with open(raw_path, "w") as f:
            for it, t in zip(items, trace):
                row = {**{k: v for k, v in it.items() if k != "prompt"}, "turns": t}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[eval_repair] traces -> {raw_path}")
