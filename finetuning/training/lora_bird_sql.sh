#!/bin/bash
# LoRA variant of sft_bird_sql.sh. Trains QLoRA-style adapters on top of a
# frozen base (all-linear target modules: attention + MoE experts + gate).
# Intended as a fast pre-flight before full SFT, or as a cheaper iteration loop.




nproc_per_node=8

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# wandb — run `wandb login` once on this node before the first training run,
# or export WANDB_API_KEY=... here. WANDB_PROJECT groups runs in the UI.
export WANDB_PROJECT=${WANDB_PROJECT:-bird_sql_lora}
current_date="$(date +%Y_%m_%d)"
REPO=/net/holy-isilon/ifs/rc_labs/ydu_lab/will/workspace/mini_dev
# Use the bird env's python (has the GLIBC-patched vllm + torch 2.9 + verl)
export PATH=/n/home06/willgao/envs/bird/bin:$PATH

# Load repo-local secrets (WANDB_API_KEY, etc.) if present. See .env.example.
if [ -f "$REPO/.env" ]; then
  set -a; . "$REPO/.env"; set +a
fi

# LoRA hyperparams
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-128}
LORA_TARGETS=${LORA_TARGETS:-all-linear}

# Path configuration
MODEL_PATH="$REPO/checkpoints/Qwen3-Coder-30B-A3B-Instruct"
MODEL_NAME=$(basename "$MODEL_PATH")
TRAIN_DATA="$REPO/finetuning/train_data/train_bird.parquet"
DEV_DATA="$REPO/finetuning/train_data/val_bird.parquet"
OUTPUT_DIR="$REPO/checkpoints/lora/${MODEL_NAME}_r${LORA_RANK}/${current_date}"
LOG_DIR="$REPO/logs/lora/${MODEL_NAME}_r${LORA_RANK}/${current_date}"
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"
# Generate log file name with current timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"

echo "Starting LoRA training, log will be saved to: $LOG_FILE"


# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=== Starting BIRD SQL LoRA training ==="
echo "GPU count: $nproc_per_node"
echo "Model path: $MODEL_PATH"
echo "Training data: $TRAIN_DATA"
echo "Output directory: $OUTPUT_DIR"
echo "LoRA: rank=$LORA_RANK alpha=$LORA_ALPHA targets=$LORA_TARGETS"

# Check if files exist
if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ Training data does not exist: $TRAIN_DATA"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Model path does not exist: $MODEL_PATH"
    exit 1
fi
if [ ! -f "$DEV_DATA" ]; then
    echo "❌ Validation data does not exist: $DEV_DATA"
    exit 1
fi

{
  echo "GPU count: $nproc_per_node"
  echo "Model path: $MODEL_PATH"
  echo "Training data: $TRAIN_DATA"
  echo "Validation data: $DEV_DATA"
  echo "Output directory: $OUTPUT_DIR"
  echo "LoRA: rank=$LORA_RANK alpha=$LORA_ALPHA targets=$LORA_TARGETS"
  echo "✅ All file checks passed, starting LoRA training..."

  python -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node=$nproc_per_node \
    -m verl.trainer.sft_trainer \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$DEV_DATA" \
    data.max_length=4096 \
    data.train_batch_size=128 \
    model.path="$MODEL_PATH" \
    model.enable_gradient_checkpointing=true \
    model.lora_rank=$LORA_RANK \
    model.lora_alpha=$LORA_ALPHA \
    model.target_modules=$LORA_TARGETS \
    engine.model_dtype=bfloat16 \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.project_name=bird_sql_lora \
    trainer.experiment_name="${MODEL_NAME}_lora_r${LORA_RANK}" \
    trainer.total_epochs=2 \
    trainer.n_gpus_per_node=$nproc_per_node \
    trainer.logger=['console','tensorboard','wandb'] \
    trainer.save_freq=55 \
    trainer.test_freq=25 \
    optim.lr=1e-4 \
    optim.weight_decay=0.0 \
    optim.lr_scheduler_type=cosine \
    optim.lr_warmup_steps_ratio=0.03 \
    "$@"

  RET=$?
  if [ $RET -eq 0 ]; then
    echo "✅ LoRA training completed, adapters saved to: $OUTPUT_DIR"
    echo "Latest checkpoints:"
    ls -1t "$OUTPUT_DIR" | head -5
    echo ""
    echo "LoRA training successful!"
  else
    echo "❌ LoRA training failed"
  fi
} 2>&1 | tee -a "$LOG_FILE"
