"""Modal inference runner for Qwen3-Coder-30B-A3B-Instruct (baseline, single-shot T=0)."""

import json
import re
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
        # LoRA inference path needs verl's FSDP→HF merger and peft's adapter format.
        "verl>=0.7",
        "peft>=0.19",
        "accelerate>=1.13",
    )
)

weights_vol = modal.Volume.from_name("qwen3-coder-weights", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("bird-sql-ckpts", create_if_missing=False)

app = modal.App("qwen3-coder-bird-sql", image=image)


# SQL extraction — mirrors vllm_infer.py for schema-compatible eval output.
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
    cpu=4,
    memory=16 * 1024,
    timeout=60 * 60,
    volumes={"/weights": weights_vol},
)
def download_weights():
    """One-shot: snapshot Qwen3-Coder-30B-A3B-Instruct into the volume."""
    import os
    from huggingface_hub import snapshot_download

    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    index_json = os.path.join(MODEL_DIR, "model.safetensors.index.json")
    if os.path.exists(index_json):
        shards = [f for f in os.listdir(MODEL_DIR)
                  if f.startswith("model-") and f.endswith(".safetensors")]
        if len(shards) == 16:
            print(f"[download] already have 16 shards at {MODEL_DIR}")
            return

    print(f"[download] pulling {MODEL_ID} -> {MODEL_DIR}")
    snapshot_download(repo_id=MODEL_ID, local_dir=MODEL_DIR, max_workers=8)
    weights_vol.commit()
    print("[download] committed to volume")


@app.cls(
    gpu=GPU,
    volumes={"/weights": weights_vol},
    scaledown_window=600,
    timeout=60 * 60,
    memory=64 * 1024,
)
class Qwen3:
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
    def generate(self, prompts, max_tokens: int = 3000, temperature: float = 0.0):
        sp = self._SamplingParams(
            temperature=temperature, top_p=1.0, max_tokens=max_tokens,
        )
        msgs = [[{"role": "user", "content": p}] for p in prompts]
        outs = self.llm.chat(msgs, sampling_params=sp)
        return [o.outputs[0].text for o in outs]


@app.function(
    gpu=GPU,
    volumes={"/weights": weights_vol, "/ckpts": ckpt_vol},
    scaledown_window=600,
    timeout=60 * 60,
    memory=64 * 1024,
)
def infer_lora(
    prompts: list,
    run_name: str,
    step: int,
    max_tokens: int = 3000,
    temperature: float = 0.0,
) -> list:
    """Merge a verl LoRA FSDP checkpoint to a PEFT adapter, attach to base via vLLM, generate."""
    import os, subprocess
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    ckpt_dir = f"/ckpts/{run_name}/global_step_{step}"
    adapter_dir = f"/tmp/adapter_{run_name}_s{step}"
    assert os.path.isdir(ckpt_dir), f"no checkpoint at {ckpt_dir}"

    # Idempotent patch: verl 0.7's LoRA merger crashes on peft>=0.15 string task_type.
    _bmm = "/usr/local/lib/python3.11/site-packages/verl/model_merger/base_model_merger.py"
    if os.path.exists(_bmm):
        import pathlib
        src = pathlib.Path(_bmm).read_text()
        needle = 'peft_config["task_type"] = peft_config["task_type"].value if peft_config["task_type"] else None'
        repl = (
            '_tt = peft_config["task_type"]\n'
            '        peft_config["task_type"] = _tt.value if hasattr(_tt, "value") else (_tt if _tt else None)'
        )
        if needle in src:
            pathlib.Path(_bmm).write_text(src.replace(needle, repl))
            print("[infer_lora] patched verl model_merger for peft>=0.15", flush=True)

    # verl writes the full-merged model at adapter_dir root + a PEFT adapter at lora_adapter/.
    lora_dir = os.path.join(adapter_dir, "lora_adapter")

    adapter_safetensors = os.path.join(lora_dir, "adapter_model.safetensors")
    adapter_bin = os.path.join(lora_dir, "adapter_model.bin")
    if not (os.path.exists(adapter_safetensors) or os.path.exists(adapter_bin)):
        os.makedirs(adapter_dir, exist_ok=True)
        print(f"[infer_lora] merging {ckpt_dir} -> {adapter_dir}", flush=True)
        subprocess.run([
            "python", "-m", "verl.model_merger", "merge",
            "--backend", "fsdp",
            "--local_dir", ckpt_dir,
            "--target_dir", adapter_dir,
        ], check=True)
        print(f"[infer_lora] merger top-level: {sorted(os.listdir(adapter_dir))}", flush=True)
        if os.path.isdir(lora_dir):
            print(f"[infer_lora] lora_adapter/ contents: {sorted(os.listdir(lora_dir))}", flush=True)

    assert os.path.exists(os.path.join(lora_dir, "adapter_config.json")), \
        f"expected adapter_config.json at {lora_dir}; contents={os.listdir(lora_dir) if os.path.isdir(lora_dir) else 'missing'}"

    print("[infer_lora] loading base + LoRA via vLLM", flush=True)
    llm = LLM(
        model=MODEL_DIR,
        dtype="bfloat16",
        # 0.88 leaves headroom for the adapters on top of base.
        gpu_memory_utilization=0.88,
        max_model_len=16000,
        trust_remote_code=True,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=64,
    )
    lora_req = LoRARequest(f"bird_lora_s{step}", 1, lora_dir)

    sp = SamplingParams(temperature=temperature, top_p=1.0, max_tokens=max_tokens)
    msgs = [[{"role": "user", "content": p}] for p in prompts]
    print(f"[infer_lora] generating for {len(prompts)} prompts", flush=True)
    outs = llm.chat(msgs, sampling_params=sp, lora_request=lora_req)
    return [o.outputs[0].text for o in outs]


@app.local_entrypoint()
def eval_lora(
    prompt_file: str,
    out_file: str,
    run_name: str,
    step: int,
    raw_out: str = "",
    max_tokens: int = 3000,
    temperature: float = 0.0,
):
    """Load prompts locally, run base + LoRA adapter on Modal, write results locally."""
    import jsonlines

    prompts_path = Path(prompt_file).resolve()
    out_path = Path(out_file).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = []
    with jsonlines.open(prompts_path) as r:
        for obj in r:
            items.append(obj)
    prompts = [it["prompt"] for it in items]
    print(f"[eval_lora] {len(prompts)} prompts <- {prompts_path}")
    print(f"[eval_lora] adapter: bird-sql-ckpts:/{run_name}/global_step_{step}")

    raw_texts = infer_lora.remote(
        prompts=prompts, run_name=run_name, step=step,
        max_tokens=max_tokens, temperature=temperature,
    )

    if raw_out:
        raw_path = Path(raw_out).resolve()
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with open(raw_path, "w") as f:
            for it, r in zip(items, raw_texts):
                f.write(json.dumps({**it, "response": r}, ensure_ascii=False) + "\n")
        print(f"[eval_lora] raw -> {raw_path}")

    pred_sqls = [_extract_sql(t) or "" for t in raw_texts]

    if str(out_path).lower().endswith(".json"):
        # Inject BIRD separator so evaluation_ex.py parses per-row db_id.
        mapping = {
            str(i): f"{s}\t----- bird -----\t{it.get('db_id','')}"
            for i, (it, s) in enumerate(zip(items, pred_sqls))
        }
        with open(out_path, "w") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
    else:
        with open(out_path, "w") as f:
            for it, r, s in zip(items, raw_texts, pred_sqls):
                row = {**it, "response": r, "pred_sql": s}
                row.pop("prompt", None)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[eval_lora] {len(pred_sqls)} predictions -> {out_path}")


@app.local_entrypoint()
def eval(
    prompt_file: str,
    out_file: str,
    raw_out: str = "",
    max_tokens: int = 3000,
    temperature: float = 0.0,
):
    """Read prompts locally, call remote Qwen3 server, write results locally."""
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

    srv = Qwen3()
    raw_texts = srv.generate.remote(
        prompts, max_tokens=max_tokens, temperature=temperature
    )
    print(f"[eval] got {len(raw_texts)} responses")

    if raw_out:
        raw_path = Path(raw_out).resolve()
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with open(raw_path, "w") as f:
            for it, r in zip(items, raw_texts):
                f.write(json.dumps({**it, "response": r}, ensure_ascii=False) + "\n")
        print(f"[eval] raw -> {raw_path}")

    pred_sqls = [_extract_sql(t) or "" for t in raw_texts]

    if str(out_path).lower().endswith(".json"):
        mapping = {str(i): s for i, s in enumerate(pred_sqls)}
        with open(out_path, "w") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
    else:
        with open(out_path, "w") as f:
            for it, r, s in zip(items, raw_texts, pred_sqls):
                row = {**it, "response": r, "pred_sql": s}
                row.pop("prompt", None)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[eval] {len(pred_sqls)} predictions -> {out_path}")
