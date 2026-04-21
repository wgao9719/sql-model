#!/bin/bash
# set -x




nproc_per_node=${NPROC_PER_NODE:-8}

# Under SLURM, CUDA_VISIBLE_DEVICES is set by --gres; only pin manually outside SLURM.
if [ -z "${SLURM_JOB_ID:-}" ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# wandb: `wandb login` once or export WANDB_API_KEY; WANDB_PROJECT groups the runs.
export WANDB_PROJECT=${WANDB_PROJECT:-bird_sql_sft}
current_date="$(date +%Y_%m_%d)"
REPO=/net/holy-isilon/ifs/rc_labs/ydu_lab/will/workspace/mini_dev
# bird env: GLIBC-patched vllm + torch 2.9 + verl.
export PATH=/n/home06/willgao/envs/bird/bin:$PATH

# Load repo-local secrets (WANDB_API_KEY, etc.) if present.
if [ -f "$REPO/.env" ]; then
  set -a; . "$REPO/.env"; set +a
fi

MODEL_PATH="$REPO/checkpoints/Qwen3-Coder-30B-A3B-Instruct"
MODEL_NAME=$(basename "$MODEL_PATH")
TRAIN_DATA="$REPO/finetuning/train_data/train_bird.parquet"
DEV_DATA="$REPO/finetuning/train_data/val_bird.parquet"
OUTPUT_DIR="$REPO/checkpoints/sft/${MODEL_NAME}/${current_date}"
LOG_DIR="$REPO/logs/sft/${MODEL_NAME}/${current_date}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"

echo "Starting training, log will be saved to: $LOG_FILE"

echo "=== Starting BIRD SQL SFT training ==="
echo "GPU count: $nproc_per_node"
echo "Model path: $MODEL_PATH"
echo "Training data: $TRAIN_DATA"
echo "Output directory: $OUTPUT_DIR"

# Pre-flight: required files must exist.
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
  echo "✅ All file checks passed, starting SFT training..."
  
  python -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node=$nproc_per_node \
    -m verl.trainer.sft_trainer \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$DEV_DATA" \
    data.max_length=4096 \
    data.train_batch_size=128 \
    model.path="$MODEL_PATH" \
    model.enable_gradient_checkpointing=true \
    engine.model_dtype=bfloat16 \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.project_name=bird_sql_sft \
    trainer.experiment_name="${MODEL_NAME}_bird_sql_sft" \
    trainer.total_epochs=4 \
    trainer.n_gpus_per_node=$nproc_per_node \
    trainer.logger=['console','tensorboard','wandb'] \
    trainer.save_freq=55 \
    trainer.test_freq=25 \
    optim.lr=1e-5 \
    optim.weight_decay=0.01 \
    optim.lr_scheduler_type=cosine \
    optim.lr_warmup_steps_ratio=0.03 \
    "$@"
  
  RET=$?
  if [ $RET -eq 0 ]; then
    echo "✅ SFT training completed, checkpoints saved to: $OUTPUT_DIR"
    echo "Latest checkpoints:"
    ls -1t "$OUTPUT_DIR" | head -5
    echo ""
    echo "SFT training successful! Now you can proceed with GRPO training."
  else
    echo "❌ SFT training failed"
  fi
} 2>&1 | tee -a "$LOG_FILE"


# Phi-4-mini: copy the model files into each checkpoint so it loads standalone.
if [[ "$MODEL_NAME" == "Phi-4-mini-instruct" ]]; then
 ORIGINAL_PHI4_PATH="/path/to/Phi-4-mini-instruct"
 for checkpoint_dir in "$OUTPUT_DIR"/global_step_*; do
   if [ -d "$checkpoint_dir" ]; then
     echo "Processing checkpoint: $(basename "$checkpoint_dir")"
     cp "$ORIGINAL_PHI4_PATH/configuration_phi3.py" "$checkpoint_dir/" 2>/dev/null || echo "configuration_phi3.py not found"
     cp "$ORIGINAL_PHI4_PATH/modeling_phi3.py" "$checkpoint_dir/" 2>/dev/null || echo "modeling_phi3.py not found"
   fi
 done
fi