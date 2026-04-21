# BIRD Mini-Dev — Qwen3-Coder experiments

Fork of [BIRD-SQL Mini-Dev](https://github.com/bird-bench/mini_dev) extended with our inference-time orchestration, SFT, and LoRA work on **Qwen3-Coder-30B-A3B-Instruct**.

## Repository layout

```
evaluation/                 BIRD EX / R-VES / soft-F1 metrics
live_sql_bench_sqlite/      separate LiveSQLBench harness (not our focus)
llm/                        ICL baseline with Azure OpenAI (upstream)
finetuning/
  data_processing/          profile_db, describe_columns, rebuild_schema, split_data
  inference/                modal_infer, modal_infer_orch, modal_infer_sft, inference_orch,
                            diagnose_collapse, vote_from_raw
  training/                 modal_sft (full-FT + LoRA), sft_bird_sql.sh, lora_bird_sql.sh
eval_result/                per-run EX tables (one .txt per experiment)
notes.txt                   scoreboard of every run (baseline → SFT → SC+Repair → profilemeta)
PLAN.md                     inference-time plan (separator fix, SC, repair, ablation grid)
```

## Data

Data is **not** checked in. Download as follows:

| Artifact | Source | Local path |
|---|---|---|
| Mini-Dev eval set (500 questions, SQLite/MySQL/Postgres) | BIRD HuggingFace / Drive | `mini_dev_data/` |
| SQLite DBs (11 dev DBs) | BIRD Drive | `mini_dev_data/dev_databases/` |
| Filtered train set (6,601 rows) | `birdsql/bird23-train-filtered` | `finetuning/train_data/` |
| Train DBs for metadata profiling | BIRD | `finetuning/train_data/train_databases/` |

Derived artifacts (regenerated from the above):
- `finetuning/data_processing/profiles/{db_id}.json` — per-column stats (`profile_db.py`)
- `finetuning/data_processing/descriptions/{db_id}.json` — one-line column descriptions (`describe_columns.py`)
- `finetuning/inference/mini_dev_prompt.jsonl` — rendered prompts (committed, 7.7 MB)
- `finetuning/inference/mini_dev_prompt_profilemeta.jsonl` — prompts with profile-injected descriptions
- `finetuning/inference/mini_dev_prompt_no_evidence.jsonl` — evidence-stripped ablation prompts
- `finetuning/inference/outputs/` — raw jsonl + final JSON predictions (gitignored)

## Checkpoints

All large weights live on **modal.com volumes**, not git.

| Model | Where | Notes |
|---|---|---|
| Qwen3-Coder-30B-A3B-Instruct (base) | volume `qwen3-coder-weights` | HF safetensors, 16 shards |
| Full-SFT checkpoints | volume `bird-sql-ckpts/<run_name>/global_step_*/` | FSDP-sharded `.pt`, merge with `verl.model_merger` |
| LoRA (attn-only) checkpoints | volume `bird-sql-ckpts/<run_name>/global_step_*/` | same format + `lora_train_meta.json` |

Local mirror lives under `checkpoints/` (gitignored). The merged HF-format copy sits next to each FSDP ckpt at `hf_merged/`.

## Environment

- **Python 3.11.5**, conda env at `/n/home06/willgao/envs/bird` — has vLLM 0.11.2, torch 2.9, verl 0.7, peft 0.19, accelerate, modal, wandb, func_timeout.
- **Modal**: `modal setup` + `modal secret create wandb WANDB_API_KEY=...`
- **`.env`**: copy `.env.example` → `.env` and fill in `WANDB_API_KEY` + `HF_HOME`.
- **MySQL / PostgreSQL creds**: still hardcoded in `evaluation/evaluation_utils.py` (upstream quirk) — edit locally.

## Quickstart — inference

```bash
# Single-shot baseline (modal, A100, ~2 min + 1 min cold start)
modal run --detach finetuning/inference/modal_infer.py::eval \
  --prompt-file finetuning/inference/mini_dev_prompt.jsonl \
  --out-file    finetuning/inference/outputs/baseline.json

# SC + Repair (strongest no-training config, ~30 min, ~$2)
modal run --detach finetuning/inference/modal_infer_orch.py::eval_repair_sc \
  --prompt-file finetuning/inference/mini_dev_prompt.jsonl \
  --raw-in      finetuning/inference/outputs/<sc_raw>.jsonl \
  --out-file    finetuning/inference/outputs/sc_repair.json \
  --db-root     mini_dev_data/dev_databases

# SFT checkpoint (auto-merges FSDP → HF on first call)
modal run --detach finetuning/inference/modal_infer_sft.py::eval \
  --prompt-file finetuning/inference/mini_dev_prompt.jsonl \
  --out-file    finetuning/inference/outputs/sft.json \
  --ckpt-subdir Qwen3-Coder-30B-A3B-full-20260420_0911/global_step_55

# Score (local, bird env)
cd evaluation && python evaluation_ex.py \
  --db_root_path ../mini_dev_data/dev_databases \
  --predicted_sql_path ../finetuning/inference/outputs/sc_repair.json \
  --ground_truth_path ../mini_dev_data/mini_dev_sqlite_gold.sql \
  --diff_json_path ../mini_dev_data/mini_dev_sqlite.jsonl \
  --sql_dialect SQLite --num_cpus 8
```

## Quickstart — training

```bash
# Full SFT (4× A100 via modal, ~3 h for 4 epochs)
modal run --detach finetuning/training/modal_sft.py::main

# LoRA (2× A100, ~1.5 h)
modal run --detach finetuning/training/modal_sft.py::main_lora
```

See `finetuning/training/modal_sft.py` for hyperparameters; the shell scripts
(`sft_bird_sql.sh`, `lora_bird_sql.sh`) are the on-cluster (non-modal) variants.

## Current scoreboard (SQLite, 500 tasks)

See `notes.txt` for the full table. Summary:

| Config | EX total |
|---|---|
| Base single-shot (T=0) | 58.80 |
| Base + SC (n=8, T=0.7, vote) | 63.40 |
| Base + SC + Repair (T=0.3) | **67.00** |
| Profilemeta prompts + SC | 65.40 |
| SFT s55 single-shot | 62.10 |
| SFT s55 + SC + Repair | 62.30 |

Full-FT SFT at 55 steps kills the SC lift — see `notes.txt`, `PLAN.md`, and `finetuning/inference/diagnose_collapse.py` for the diagnostic workflow.
