#!/bin/bash
#SBATCH --job-name=sft_smoke
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_ydu_lab
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=01:00:00
#SBATCH --output=/net/holy-isilon/ifs/rc_labs/ydu_lab/will/workspace/mini_dev/logs/slurm/%x_%j.out
#SBATCH --error=/net/holy-isilon/ifs/rc_labs/ydu_lab/will/workspace/mini_dev/logs/slurm/%x_%j.err

set -euo pipefail

REPO=/net/holy-isilon/ifs/rc_labs/ydu_lab/will/workspace/mini_dev
cd "$REPO"

echo "=== SMOKE RUN ==="
echo "host: $(hostname)"
echo "slurm job: $SLURM_JOB_ID"
echo "gpus: $CUDA_VISIBLE_DEVICES"
echo

NPROC_PER_NODE=4 \
bash "$REPO/finetuning/training/sft_bird_sql.sh" \
  data.train_max_samples=1024 \
  data.val_max_samples=64 \
  data.train_batch_size=32 \
  trainer.total_epochs=1 \
  trainer.save_freq=20 \
  trainer.test_freq=10 \
  trainer.experiment_name="smoke_${SLURM_JOB_ID}" \
  trainer.default_local_dir="$REPO/checkpoints/sft/smoke_${SLURM_JOB_ID}" \
  engine.optimizer_offload=true
