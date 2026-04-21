#!/usr/bin/env bash
# 5-prompt smoke test for vLLM + Qwen3-Coder-30B-A3B-Instruct.
# Submit via sbatch onto a GPU node with 2 GPUs.
set -euo pipefail

REPO=/net/holy-isilon/ifs/rc_labs/ydu_lab/will/workspace/mini_dev
export PATH=/n/home06/willgao/envs/bird/bin:$PATH
export HF_HOME=/net/holy-isilon/ifs/rc_labs/ydu_lab/will/hf_cache
unset HF_HUB_ENABLE_HF_TRANSFER

MODEL_PATH=$REPO/checkpoints/Qwen3-Coder-30B-A3B-Instruct
PROMPT_JSONL=$REPO/finetuning/inference/smoke5_prompt.jsonl
OUT_DIR=$REPO/finetuning/inference/outputs/smoke5

mkdir -p "$OUT_DIR"
BASE=$(basename "$PROMPT_JSONL" .jsonl)
RAW_OUT=$OUT_DIR/${BASE}_raw.jsonl
FINAL_OUT=$OUT_DIR/${BASE}_final.json

echo "=== [$(date)] smoke infer starting on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv | head -5
echo "python: $(which python)"
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"

python "$REPO/finetuning/inference/vllm_infer.py" \
  --model_path "$MODEL_PATH" \
  --prompt_path "$PROMPT_JSONL" \
  --raw_output_path "$RAW_OUT" \
  --output_path "$FINAL_OUT" \
  --gpu "0" \
  --batch_size 5 \
  --max_token_length 16000 \
  --temperature 0.0

echo "=== [$(date)] smoke infer complete ==="
echo "--- raw sample ---"
head -1 "$RAW_OUT" | python -c "
import json, sys
r = json.loads(sys.stdin.read())
print('q:', r.get('question','')[:120])
print('pred_sql:', r.get('pred_sql',''))
print('raw response (first 500):', (r.get('response','') or '')[:500])
"
