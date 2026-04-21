"""Modal runner for BIRD-SQL SFT on Qwen3-Coder-30B-A3B-Instruct (full-FT and LoRA)."""

import modal

LOCAL_MODEL_PATH = "/weights/Qwen3-Coder-30B-A3B-Instruct"
TIMEOUT_H = 12

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.9.0",
        "vllm==0.11.2",
        "verl>=0.7",
        "transformers>=4.57",
        "tokenizers>=0.22",
        "accelerate>=1.13",
        "datasets>=4.8",
        "peft>=0.19",
        "huggingface_hub>=0.36",
        "safetensors>=0.7",
        "sentencepiece",
        "wandb",
        "tensorboard",
    )
)

app = modal.App("bird-sql-sft", image=image)

weights_vol = modal.Volume.from_name("qwen3-coder-weights", create_if_missing=False)
data_vol = modal.Volume.from_name("bird-sql-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("bird-sql-ckpts", create_if_missing=True)


def _base_cmd(nproc, model_path, train_parquet, val_parquet,
              train_batch_size, max_length,
              out_dir, epochs, run_name, lr,
              save_freq=28, test_freq=6,
              train_max_samples=-1, val_max_samples=-1,
              max_token_len_per_gpu=8192,
              optimizer_offload=False,
              activation_offload=False):
    cmd = [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={nproc}",
        "-m", "verl.trainer.sft_trainer",
        f"data.train_files=/data/{train_parquet}",
        f"data.val_files=/data/{val_parquet}",
        f"data.max_length={max_length}",
        # Left-trunc preserves the response tail; right-trunc silently zero-gradient'd ~9% of rows.
        "data.truncation=left",
        f"data.train_batch_size={train_batch_size}",
        f"data.train_max_samples={train_max_samples}",
        f"data.val_max_samples={val_max_samples}",
        # Packing token cap per micro-batch under use_dynamic_bsz=True.
        f"data.max_token_len_per_gpu={max_token_len_per_gpu}",
        f"model.path={model_path}",
        "model.enable_gradient_checkpointing=true",
        # flash_attn isn't in the image — force SDPA and disable FA2-only remove-padding.
        "+model.override_config.attn_implementation=sdpa",
        "model.use_remove_padding=false",
        "engine.model_dtype=bfloat16",
        f"trainer.default_local_dir={out_dir}",
        "trainer.project_name=bird_sql_sft",
        f"trainer.experiment_name={run_name}",
        f"trainer.total_epochs={epochs}",
        f"trainer.n_gpus_per_node={nproc}",
        "trainer.logger=[console,wandb]",
        f"trainer.save_freq={save_freq}",
        f"trainer.test_freq={test_freq}",
        f"optim.lr={lr}",
        "optim.weight_decay=0.01",
        "optim.lr_scheduler_type=cosine",
        "optim.lr_warmup_steps_ratio=0.03",
    ]
    if optimizer_offload:
        # verl's FSDP engine requires optim and param offload together.
        cmd.append("engine.optimizer_offload=true")
        cmd.append("engine.param_offload=true")
    if activation_offload:
        # Needed to fit 8k-token sequences at batch=128 on 4x A100-80GB with offload on.
        cmd.append("engine.enable_activation_offload=true")
    return cmd


def _prep_env(run_name_prefix: str):
    import os
    from datetime import datetime
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["WANDB_PROJECT"] = "bird_sql_sft"
    return f"{run_name_prefix}-{datetime.utcnow():%Y%m%d_%H%M}"


@app.function(
    # 4 GPUs with param+optimizer offload to CPU: ~3× slower than 8 GPUs without
    # offload (~100 s/step vs ~25 s/step), but 8xA100-80GB is less reliably schedulable
    # Smoke-validated on 4 GPUs; switch to :8 and drop offload when 8-GPU capacity is easy.
    gpu="A100-80GB:4",
    timeout=TIMEOUT_H * 3600,
    volumes={
        "/weights": weights_vol,
        "/data": data_vol,
        "/ckpts": ckpt_vol,
    },
    secrets=[modal.Secret.from_name("wandb")],
)
def train(
    model_path: str = LOCAL_MODEL_PATH,
    epochs: int = 4,
    train_parquet: str = "train_bird.parquet",
    val_parquet: str = "val_bird.parquet",
    train_batch_size: int = 128,
    max_length: int = 8192,
    lr: float = 1e-5,
    run_name: str | None = None,
):
    """Full-parameter SFT on 4xA100-80GB with FSDP + CPU offload."""
    import os, subprocess
    assert os.path.exists(os.path.join(model_path, "model.safetensors.index.json")), \
        f"weights not found at {model_path}"

    # Pre-flight: Modal sometimes attaches 0 GPUs; surface that before NCCL does.
    smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    print(f"[modal_sft full] nvidia-smi -L:\n{smi.stdout}\n{smi.stderr}", flush=True)
    gpu_count = len([l for l in smi.stdout.splitlines() if l.startswith("GPU ")])
    assert gpu_count >= 4, f"expected >=4 GPUs, got {gpu_count}; Modal failed to attach"

    run_name = run_name or _prep_env("Qwen3-Coder-30B-A3B-full")
    out_dir = f"/ckpts/{run_name}"
    os.makedirs(out_dir, exist_ok=True)

    cmd = _base_cmd(
        nproc=4,
        model_path=model_path,
        train_parquet=train_parquet, val_parquet=val_parquet,
        train_batch_size=train_batch_size,
        max_length=max_length,
        out_dir=out_dir, epochs=epochs, run_name=run_name, lr=lr,
        # 8k seq previously OOM'd on 4xA100 with param/optim offload alone; needs activation offload too.
        max_token_len_per_gpu=8192,
        optimizer_offload=True,
        activation_offload=True,
    )
    print("[modal_sft full] launching:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    ckpt_vol.commit()
    print(f"[modal_sft full] checkpoints committed to bird-sql-ckpts:/{run_name}")


@app.function(
    gpu="A100-80GB:4",
    timeout=TIMEOUT_H * 3600,
    volumes={
        "/weights": weights_vol,
        "/data": data_vol,
        "/ckpts": ckpt_vol,
    },
    secrets=[modal.Secret.from_name("wandb")],
)
def train_smoke(
    model_path: str = LOCAL_MODEL_PATH,
    train_parquet: str = "train_bird.parquet",
    val_parquet: str = "val_bird.parquet",
    run_name: str | None = None,
):
    """Smoke test on 4xA100-80GB: 1024 samples, 1 epoch — validates E2E before the real run."""
    import os, subprocess
    assert os.path.exists(os.path.join(model_path, "model.safetensors.index.json")), \
        f"weights not found at {model_path}"
    run_name = run_name or _prep_env("Qwen3-Coder-30B-A3B-smoke")
    out_dir = f"/ckpts/{run_name}"
    os.makedirs(out_dir, exist_ok=True)

    cmd = _base_cmd(
        nproc=4,
        model_path=model_path,
        train_parquet=train_parquet, val_parquet=val_parquet,
        train_batch_size=32, max_length=8192,
        out_dir=out_dir, epochs=1, run_name=run_name, lr=1e-5,
        save_freq=10, test_freq=2,
        train_max_samples=1024, val_max_samples=64,
        max_token_len_per_gpu=8192,
        optimizer_offload=True,
        activation_offload=True,
    )
    print("[modal_sft smoke] launching:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    ckpt_vol.commit()
    print(f"[modal_sft smoke] checkpoint committed to bird-sql-ckpts:/{run_name}")


@app.function(
    # attn-only LoRA: 2xA100 is enough (~37 GB peak per GPU); 4x is overkill.
    gpu="A100-80GB:2",
    timeout=TIMEOUT_H * 3600,
    volumes={
        "/weights": weights_vol,
        "/data": data_vol,
        "/ckpts": ckpt_vol,
    },
    secrets=[modal.Secret.from_name("wandb")],
)
def train_lora(
    model_path: str = LOCAL_MODEL_PATH,
    epochs: int = 2,
    train_parquet: str = "train_bird.parquet",
    val_parquet: str = "val_bird.parquet",
    train_batch_size: int = 64,
    max_length: int = 4096,
    lr: float = 1e-4,
    lora_rank: int = 64,
    lora_alpha: int | None = None,
    run_name: str | None = None,
):
    """LoRA SFT on 2xA100; attention-only targets (q/k/v/o) to avoid the 128-expert FFN blow-up."""
    import os, subprocess
    assert os.path.exists(os.path.join(model_path, "model.safetensors.index.json")), \
        f"weights not found at {model_path}"
    if lora_alpha is None:
        lora_alpha = lora_rank
    run_name = run_name or _prep_env(f"Qwen3-Coder-30B-A3B-lora{lora_rank}-attn")
    out_dir = f"/ckpts/{run_name}"
    os.makedirs(out_dir, exist_ok=True)

    cmd = _base_cmd(
        nproc=2,
        model_path=model_path,
        train_parquet=train_parquet, val_parquet=val_parquet,
        train_batch_size=train_batch_size,
        max_length=max_length,
        out_dir=out_dir, epochs=epochs, run_name=run_name, lr=lr,
    )
    cmd += [
        f"model.lora_rank={lora_rank}",
        f"model.lora_alpha={lora_alpha}",
        "model.target_modules=[q_proj,k_proj,v_proj,o_proj]",
    ]
    print("[modal_sft lora] launching:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    ckpt_vol.commit()
    print(f"[modal_sft lora] adapter committed to bird-sql-ckpts:/{run_name}")


@app.function(
    gpu="A100-80GB:4",
    timeout=TIMEOUT_H * 3600,
    volumes={
        "/weights": weights_vol,
        "/data": data_vol,
        "/ckpts": ckpt_vol,
    },
    secrets=[modal.Secret.from_name("wandb")],
)
def train_lora_smoke(
    model_path: str = LOCAL_MODEL_PATH,
    train_parquet: str = "train_bird.parquet",
    val_parquet: str = "val_bird.parquet",
    max_length: int = 4096,
    run_name: str | None = None,
):
    """Smoke test for attention-only LoRA; validates peft + Qwen3-MoE integration."""
    import os, subprocess
    assert os.path.exists(os.path.join(model_path, "model.safetensors.index.json")), \
        f"weights not found at {model_path}"
    run_name = run_name or _prep_env(f"Qwen3-Coder-30B-A3B-lora-smoke-len{max_length}")
    out_dir = f"/ckpts/{run_name}"
    os.makedirs(out_dir, exist_ok=True)

    cmd = _base_cmd(
        nproc=4,
        model_path=model_path,
        train_parquet=train_parquet, val_parquet=val_parquet,
        train_batch_size=32, max_length=max_length,
        out_dir=out_dir, epochs=1, run_name=run_name, lr=1e-4,
        save_freq=5, test_freq=2,
        train_max_samples=512, val_max_samples=32,
        # Cap per-micro-batch tokens at max_length so a full-length sample fits alone.
        max_token_len_per_gpu=max_length,
    )
    cmd += [
        "model.lora_rank=64",
        "model.lora_alpha=64",
        "model.target_modules=[q_proj,k_proj,v_proj,o_proj]",
    ]
    print("[modal_sft lora_smoke] launching:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    ckpt_vol.commit()
    print(f"[modal_sft lora_smoke] adapter committed to bird-sql-ckpts:/{run_name}")


@app.local_entrypoint()
def main(epochs: int = 4, run_name: str | None = None):
    """Full-SFT entrypoint (8xA100)."""
    train.remote(epochs=epochs, run_name=run_name)


@app.local_entrypoint()
def main_smoke(run_name: str | None = None):
    """Smoke test on 4xA100 — 1024 samples, 1 epoch, optimizer offload."""
    train_smoke.remote(run_name=run_name)


@app.local_entrypoint()
def main_lora_smoke(max_length: int = 4096, run_name: str | None = None):
    """LoRA smoke test on 4xA100 — 512 samples, 1 epoch, rank 64 attn-only (--max-length for long ctx)."""
    train_lora_smoke.remote(max_length=max_length, run_name=run_name)


@app.local_entrypoint()
def main_lora(
    epochs: int = 2,
    lora_rank: int = 64,
    lora_alpha: int | None = None,
    lr: float = 1e-4,
    run_name: str | None = None,
):
    """LoRA-SFT entrypoint (4xA100). Attention-only targets; alpha defaults to rank."""
    train_lora.remote(
        epochs=epochs,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lr=lr,
        run_name=run_name,
    )
