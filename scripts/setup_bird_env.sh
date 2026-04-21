#!/usr/bin/env bash
# Run this FROM A COMPUTE NODE — login node SIGKILLs network-heavy jobs.
# Example: srun --pty --time=2:00:00 --mem=32G bash
#   then: bash scripts/setup_bird_env.sh
set -euo pipefail

BIRD_ENV=/n/home06/willgao/envs/bird
PY="$BIRD_ENV/bin/python"
MODEL_DIR=/net/holy-isilon/ifs/rc_labs/ydu_lab/will/workspace/mini_dev/checkpoints/Qwen3-Coder-30B-A3B-Instruct
LOG=/net/holy-isilon/ifs/rc_labs/ydu_lab/will/workspace/mini_dev/logs/setup_bird_env.log

export HF_HOME=/net/holy-isilon/ifs/rc_labs/ydu_lab/will/hf_cache
unset HF_HUB_ENABLE_HF_TRANSFER  # buggy on this cluster — silent partial downloads

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "=== [$(date)] setup_bird_env starting ==="
echo "host=$(hostname) python=$("$PY" --version)"

# 1. Install heavy ML packages. torch 2.7.1 matches vllm 0.11.x wheels;
#    cu124 works on H100/A100. Adjust if your driver needs cu121/cu128.
echo "--- installing torch 2.7.1 (cu128) ---"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install --index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.7.1" "torchvision" "torchaudio"

echo "--- installing vllm + transformers stack (PyPI) ---"
"$PY" -m pip install \
    "vllm>=0.10,<0.12" \
    "transformers>=4.46" \
    "tokenizers>=0.20" \
    "safetensors" \
    "sentencepiece" \
    "accelerate>=1.0" \
    "datasets>=3.0" \
    "peft>=0.13" \
    "jsonlines"

echo "--- installing verl (for SFT) ---"
"$PY" -m pip install "verl"

echo "--- version check ---"
"$PY" -c "
import importlib.metadata as im
for p in ['torch','transformers','vllm','verl','peft','accelerate','datasets','huggingface_hub','jsonlines']:
    try: print(f'  {p}={im.version(p)}')
    except Exception: print(f'  {p}=MISSING')
"

# 2. Resume / complete Qwen3-Coder download. snapshot_download will skip
#    files already in the local dir and resume .incomplete partials.
echo "--- resuming Qwen3-Coder-30B-A3B-Instruct download ---"
"$PY" -u -c "
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id='Qwen/Qwen3-Coder-30B-A3B-Instruct',
    local_dir='$MODEL_DIR',
    max_workers=4,
)
print('DONE:', p)
"

# 3. Verify all 16 shards present
echo "--- verification ---"
SHARDS=$(ls "$MODEL_DIR"/model-*.safetensors 2>/dev/null | wc -l)
SIZE=$(du -sh "$MODEL_DIR" | awk '{print $1}')
echo "shards=$SHARDS/16 size=$SIZE"
if [ "$SHARDS" != "16" ]; then
    echo "!! incomplete — re-run this script"
    exit 1
fi
echo "=== [$(date)] setup_bird_env complete ==="
