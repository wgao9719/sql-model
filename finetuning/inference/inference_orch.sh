#!/bin/bash
# Editable wrapper for inference_orch.py — fill paths/toggles, then: bash inference_orch.sh
set -e

MODEL_PATH="/path/to/Qwen3-Coder-7B-Instruct"
PROMPT_PATH="/path/to/mini_dev_prompt.jsonl"
OUTPUT_PATH="/path/to/preds.json"
RAW_OUTPUT_PATH=""
DB_ROOT="/path/to/dev_databases"
DIALECT="SQLite"

LINKING="static"          # static | bm25 | sql_induced  (bm25/sql_induced need Phase A+B)
HINTS="off"               # off | on                     (on needs Phase A)
VALUES="off"              # off | on                     (on needs Phase A)
MODE="single"             # single | repair | sc | repair_sc
N_SAMPLES=8
MAX_REPAIR_TURNS=3
SC_TEMPERATURE=0.7
EXEC_TIMEOUT=10.0

GPU="${CUDA_VISIBLE_DEVICES:-0}"
BATCH_SIZE=50
MAX_TOKEN_LENGTH=15000

python "$(dirname "$0")/inference_orch.py" \
  --model_path "$MODEL_PATH" \
  --prompt_path "$PROMPT_PATH" \
  --output_path "$OUTPUT_PATH" \
  --raw_output_path "$RAW_OUTPUT_PATH" \
  --db_root "$DB_ROOT" \
  --dialect "$DIALECT" \
  --linking "$LINKING" \
  --hints "$HINTS" \
  --values "$VALUES" \
  --mode "$MODE" \
  --n_samples "$N_SAMPLES" \
  --max_repair_turns "$MAX_REPAIR_TURNS" \
  --sc_temperature "$SC_TEMPERATURE" \
  --exec_timeout "$EXEC_TIMEOUT" \
  --gpu "$GPU" \
  --batch_size "$BATCH_SIZE" \
  --max_token_length "$MAX_TOKEN_LENGTH"
