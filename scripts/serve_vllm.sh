#!/usr/bin/env bash
# Long-running vLLM OpenAI-compatible server for Qwen3-Coder-30B-A3B-Instruct.
# Submit via sbatch to a GPU partition. Writes server_info.json so clients
# can find the host:port.
set -euo pipefail

REPO=/net/holy-isilon/ifs/rc_labs/ydu_lab/will/workspace/mini_dev
export PATH=/n/home06/willgao/envs/bird/bin:$PATH
export HF_HOME=/net/holy-isilon/ifs/rc_labs/ydu_lab/will/hf_cache
unset HF_HUB_ENABLE_HF_TRANSFER

MODEL_PATH=$REPO/checkpoints/Qwen3-Coder-30B-A3B-Instruct
PORT=${PORT:-8765}
SERVED_NAME=qwen3-coder-30b
INFO_FILE=$REPO/logs/vllm_server_info.json

# Publish connection info so clients on login node / other jobs can find us.
python - <<PY
import json, socket, os
info = {
    "host": socket.gethostname(),
    "port": $PORT,
    "model": "$SERVED_NAME",
    "model_path": "$MODEL_PATH",
    "slurm_job_id": os.environ.get("SLURM_JOB_ID",""),
}
with open("$INFO_FILE", "w") as f: json.dump(info, f, indent=2)
print("server info ->", "$INFO_FILE", info)
PY

echo "=== [$(date)] vLLM server starting on $(hostname):$PORT ==="
nvidia-smi --query-gpu=name,memory.total --format=csv | head -5

exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.92 \
    --max-model-len 16000 \
    --trust-remote-code \
    --disable-custom-all-reduce
