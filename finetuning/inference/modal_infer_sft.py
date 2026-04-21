"""Modal runner: merge verl-SFT FSDP checkpoints and evaluate on BIRD Mini-Dev."""

import json
import re
from pathlib import Path

import modal

GPU = "A100-80GB"
BIRD_SEP = "\t----- bird -----\t"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.11.2",
        "torch==2.9.0",
        "transformers>=4.46",
        "huggingface_hub>=0.26",
        "jsonlines",
        "verl",
        "accelerate",
    )
)

ckpts_vol = modal.Volume.from_name("bird-sql-ckpts", create_if_missing=False)
app = modal.App("qwen3-sft-eval", image=image)

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


@app.function(
    cpu=8,
    memory=128 * 1024,
    timeout=60 * 60,
    volumes={"/ckpts": ckpts_vol},
)
def merge_ckpt(ckpt_subdir: str, force: bool = False):
    """FSDP shards → HF safetensors (stored in the same volume at /hf_merged)."""
    import os
    import subprocess
    import shutil

    src = f"/ckpts/{ckpt_subdir}"
    tgt = f"{src}/hf_merged"
    index_json = os.path.join(tgt, "model.safetensors.index.json")

    if (not force) and os.path.exists(index_json):
        print(f"[merge] already merged at {tgt}, skipping")
        return tgt

    os.makedirs(tgt, exist_ok=True)

    # Idempotent patch: verl 0.7's LoRA merger crashes on peft>=0.15 string task_type.
    _bmm = "/usr/local/lib/python3.11/site-packages/verl/model_merger/base_model_merger.py"
    if os.path.exists(_bmm):
        import pathlib
        src_text = pathlib.Path(_bmm).read_text()
        needle = 'peft_config["task_type"] = peft_config["task_type"].value if peft_config["task_type"] else None'
        repl = (
            '_tt = peft_config["task_type"]\n'
            '        peft_config["task_type"] = _tt.value if hasattr(_tt, "value") else (_tt if _tt else None)'
        )
        if needle in src_text:
            pathlib.Path(_bmm).write_text(src_text.replace(needle, repl))
            print("[merge] patched verl model_merger for peft>=0.15", flush=True)

    print(f"[merge] FSDP merge: {src} -> {tgt}")
    subprocess.check_call([
        "python", "-m", "verl.model_merger", "merge",
        "--backend", "fsdp",
        "--local_dir", src,
        "--target_dir", tgt,
        "--trust-remote-code",
    ])

    hf_extras = os.path.join(src, "huggingface")
    if os.path.isdir(hf_extras):
        for fn in os.listdir(hf_extras):
            s = os.path.join(hf_extras, fn)
            d = os.path.join(tgt, fn)
            if not os.path.exists(d):
                shutil.copy2(s, d)

    ckpts_vol.commit()
    print(f"[merge] committed {tgt}")
    return tgt


@app.cls(
    gpu=GPU,
    volumes={"/ckpts": ckpts_vol},
    scaledown_window=600,
    timeout=60 * 60,
    memory=64 * 1024,
)
class QwenSFT:
    model_dir: str = modal.parameter(default="")

    @modal.enter()
    def load(self):
        from vllm import LLM, SamplingParams
        assert self.model_dir, "model_dir must be passed via .with_options(...)"
        print(f"[enter] loading vLLM from {self.model_dir} on {GPU}")
        self.llm = LLM(
            model=self.model_dir,
            dtype="bfloat16",
            gpu_memory_utilization=0.92,
            max_model_len=16000,
            trust_remote_code=True,
        )
        self._SamplingParams = SamplingParams
        print("[enter] vLLM ready")

    @modal.method()
    def generate(self, prompts, max_tokens: int = 3000, temperature: float = 0.0):
        sp = self._SamplingParams(
            temperature=temperature, top_p=1.0, max_tokens=max_tokens,
        )
        msgs = [[{"role": "user", "content": p}] for p in prompts]
        outs = self.llm.chat(msgs, sampling_params=sp)
        return [o.outputs[0].text for o in outs]

    @modal.method()
    def generate_raw(self, prompts, max_tokens: int = 3000, temperature: float = 0.0):
        """Raw prompt path (no chat template) — for train/inference template diffing."""
        sp = self._SamplingParams(
            temperature=temperature, top_p=1.0, max_tokens=max_tokens,
        )
        outs = self.llm.generate(prompts, sampling_params=sp)
        return [o.outputs[0].text for o in outs]

    @modal.method()
    def generate_sc(self, prompts, n: int, max_tokens: int, temperature: float):
        sp = self._SamplingParams(
            temperature=temperature, top_p=1.0, n=n, max_tokens=max_tokens,
        )
        msgs = [[{"role": "user", "content": p}] for p in prompts]
        outs = self.llm.chat(msgs, sampling_params=sp)
        return [[o.text for o in out.outputs] for out in outs]


def _vote_worker(args):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from inference_orch import _sc_vote
    extracted, db_path, timeout = args
    return _sc_vote(extracted, db_path, timeout)


def _exec_worker(args):
    """Exec worker that swallows all exceptions — one bad SQL can't kill the pool."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from inference_orch import _execute_sqlite
    sql, db_path, timeout = args
    try:
        return _execute_sqlite(sql, db_path, timeout)
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:500], "rows": None}


@app.local_entrypoint()
def eval_sc_repair(
    prompt_file: str,
    out_file: str,
    ckpt_subdir: str,
    db_root: str,
    raw_out: str = "",
    n_samples: int = 8,
    max_repair_turns: int = 2,
    max_tokens: int = 3000,
    initial_temperature: float = 0.7,
    repair_temperature: float = 0.3,
    exec_timeout: float = 5.0,
    num_workers: int = 8,
):
    """SC + Repair on the SFT checkpoint (single A100)."""
    import sys, jsonlines
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
    assert items and "prompt" in items[0] and "db_id" in items[0]
    base_prompts = [it["prompt"] for it in items]
    db_paths = [_db_path(db_root_abs, it["db_id"], "SQLite") for it in items]
    n_items = len(items)
    print(f"[sc_repair] {n_items} prompts  n={n_samples}  max_repair={max_repair_turns}")

    print(f"[sc_repair] ensuring HF merge for {ckpt_subdir}")
    merged_dir = merge_ckpt.remote(ckpt_subdir)

    srv = QwenSFT.with_options()(model_dir=merged_dir)

    cand_texts = srv.generate_sc.remote(
        base_prompts, n=n_samples, max_tokens=max_tokens, temperature=initial_temperature,
    )
    print(f"[sc_repair] turn 0: generated {n_items} x {n_samples} candidates")
    cand_sqls = [[_extract_sql(c) or "" for c in row] for row in cand_texts]

    # Persist turn-0 before exec/vote so a crash doesn't lose the generation work.
    if raw_out:
        raw_outp = Path(raw_out).resolve()
        raw_outp.parent.mkdir(parents=True, exist_ok=True)
        turn0_path = raw_outp.with_suffix(".turn0.jsonl")
        with open(turn0_path, "w") as f:
            for it, cands in zip(items, cand_texts):
                row = {**{k: v for k, v in it.items() if k != "prompt"}, "candidates": cands}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[sc_repair] raw turn-0 persisted -> {turn0_path}")

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
        print(f"[sc_repair] turn {turn}: {len(failures)} error-candidates")
        if not failures:
            break

        repair_prompts = []
        for q, c in failures:
            err = flat_results[q * n_samples + c]["error"]
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

    print(f"[sc_repair] voting with {num_workers} workers")
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
    print(f"[sc_repair] {n_items} predictions -> {out_path}")

    if raw_out:
        raw_outp = Path(raw_out).resolve()
        raw_outp.parent.mkdir(parents=True, exist_ok=True)
        with open(raw_outp, "w") as f:
            for it, cands in zip(items, cand_texts):
                row = {**{k: v for k, v in it.items() if k != "prompt"}, "candidates": cands}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[sc_repair] raw -> {raw_outp}")


@app.local_entrypoint()
def eval_lora_sc(
    prompt_file: str,
    out_file: str,
    ckpt_subdir: str,
    raw_out: str,
    n_samples: int = 8,
    max_tokens: int = 3000,
    temperature: float = 0.7,
):
    """SC-only inference on a LoRA checkpoint; writes raw candidates for diagnostics."""
    import jsonlines
    from pathlib import Path as _P
    prompts_path = _P(prompt_file).resolve()
    raw_path = _P(raw_out).resolve()
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    items = []
    with jsonlines.open(prompts_path) as r:
        for obj in r:
            items.append(obj)
    prompts = [it["prompt"] for it in items]
    print(f"[eval_lora_sc] {len(prompts)} prompts  n={n_samples}  T={temperature}")

    merged_dir = merge_ckpt.remote(ckpt_subdir)
    print(f"[eval_lora_sc] merged dir: {merged_dir}")

    srv = QwenSFT.with_options()(model_dir=merged_dir)
    cand_texts = srv.generate_sc.remote(
        prompts, n=n_samples, max_tokens=max_tokens, temperature=temperature,
    )
    print(f"[eval_lora_sc] {len(cand_texts)} x {n_samples} candidates")

    with open(raw_path, "w") as f:
        for it, cands in zip(items, cand_texts):
            row = {**{k: v for k, v in it.items() if k != "prompt"}, "candidates": list(cands)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[eval_lora_sc] raw -> {raw_path}")


@app.local_entrypoint()
def eval_raw(
    prompt_file: str,
    out_file: str,
    ckpt_subdir: str,
    raw_out: str = "",
    max_tokens: int = 3000,
    temperature: float = 0.0,
):
    """Single-shot, raw prompt (no chat template) — train/inference template diff test."""
    import jsonlines
    prompts_path = Path(prompt_file).resolve()
    out_path = Path(out_file).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = []
    with jsonlines.open(prompts_path) as r:
        for obj in r:
            items.append(obj)
    prompts = [it["prompt"] for it in items]
    print(f"[eval_raw] {len(prompts)} prompts (no chat template)")

    merged_dir = merge_ckpt.remote(ckpt_subdir)
    print(f"[eval_raw] merged dir: {merged_dir}")

    srv = QwenSFT.with_options()(model_dir=merged_dir)
    raw_texts = srv.generate_raw.remote(
        prompts, max_tokens=max_tokens, temperature=temperature,
    )
    print(f"[eval_raw] got {len(raw_texts)} responses")

    if raw_out:
        raw_p = Path(raw_out).resolve()
        raw_p.parent.mkdir(parents=True, exist_ok=True)
        with open(raw_p, "w") as f:
            for it, txt in zip(items, raw_texts):
                f.write(json.dumps({**it, "response": txt}, ensure_ascii=False) + "\n")
        print(f"[eval_raw] raw -> {raw_p}")

    payload = {}
    for i, (it, txt) in enumerate(zip(items, raw_texts)):
        sql = _extract_sql(txt) or ""
        payload[str(i)] = f"{sql}{BIRD_SEP}{it['db_id']}"
    with open(out_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[eval_raw] {len(items)} predictions -> {out_path}")


@app.local_entrypoint()
def eval(
    prompt_file: str,
    out_file: str,
    ckpt_subdir: str,
    raw_out: str = "",
    max_tokens: int = 3000,
    temperature: float = 0.0,
):
    """Baseline-style single-shot inference with the SFT checkpoint."""
    import jsonlines

    prompts_path = Path(prompt_file).resolve()
    out_path = Path(out_file).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = []
    with jsonlines.open(prompts_path) as r:
        for obj in r:
            items.append(obj)
    prompts = [it["prompt"] for it in items]
    print(f"[eval] {len(prompts)} prompts <- {prompts_path}")

    print(f"[eval] ensuring HF merge for {ckpt_subdir}")
    merged_dir = merge_ckpt.remote(ckpt_subdir)
    print(f"[eval] merged dir: {merged_dir}")

    srv = QwenSFT.with_options()(model_dir=merged_dir)
    raw_texts = srv.generate.remote(
        prompts, max_tokens=max_tokens, temperature=temperature,
    )
    print(f"[eval] got {len(raw_texts)} responses")

    if raw_out:
        raw_path = Path(raw_out).resolve()
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with open(raw_path, "w") as f:
            for it, txt in zip(items, raw_texts):
                f.write(json.dumps({**it, "response": txt}, ensure_ascii=False) + "\n")
        print(f"[eval] raw -> {raw_path}")

    payload = {}
    for i, (it, txt) in enumerate(zip(items, raw_texts)):
        sql = _extract_sql(txt) or ""
        payload[str(i)] = f"{sql}{BIRD_SEP}{it['db_id']}"
    with open(out_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[eval] {len(items)} predictions -> {out_path}")
