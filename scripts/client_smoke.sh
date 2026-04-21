#!/usr/bin/env bash
# Run the 5-prompt smoke test against a running vLLM server.
# Reads host/port from logs/vllm_server_info.json.
set -euo pipefail

REPO=/net/holy-isilon/ifs/rc_labs/ydu_lab/will/workspace/mini_dev
PY=/n/home06/willgao/envs/bird/bin/python
INFO=$REPO/logs/vllm_server_info.json

HOST=$($PY -c "import json; print(json.load(open('$INFO'))['host'])")
PORT=$($PY -c "import json; print(json.load(open('$INFO'))['port'])")
MODEL=$($PY -c "import json; print(json.load(open('$INFO'))['model'])")

echo "target: http://$HOST:$PORT/v1  model=$MODEL"

OUT_DIR=$REPO/finetuning/inference/outputs/smoke5
mkdir -p "$OUT_DIR"

$PY $REPO/finetuning/inference/vllm_client_infer.py \
    --host "$HOST" --port "$PORT" --model "$MODEL" \
    --prompt_path  "$REPO/finetuning/inference/smoke5_prompt.jsonl" \
    --raw_output_path "$OUT_DIR/smoke5_raw.jsonl" \
    --output_path     "$OUT_DIR/smoke5_final.json" \
    --concurrency 5 --max_tokens 3000 --temperature 0.0
