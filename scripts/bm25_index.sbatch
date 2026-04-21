#!/usr/bin/env bash
#SBATCH --job-name=bird_bm25
#SBATCH --partition=seas_compute
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/net/holy-isilon/ifs/rc_labs/ydu_lab/will/workspace/mini_dev/logs/bm25_index_build.%j.out
set -euo pipefail

MINIDEV=/net/holy-isilon/ifs/rc_labs/ydu_lab/will/workspace/mini_dev
ARCTIC=$MINIDEV/ArcticTraining/projects/arctic_text2sql_r1
BIRD=/n/home06/willgao/envs/bird

export JAVA_HOME=/n/sw/helmod-rocky8/apps/Core/jdk/21.0.2-fasrc01
export PATH=$BIRD/bin:$JAVA_HOME/bin:$PATH

cd "$ARCTIC"
echo "host=$(hostname) java=$(java -version 2>&1 | head -1)"

python -u data_preprocessing/build_contents_index.py \
    --db-root   "$MINIDEV/finetuning/train_data/train_databases" \
    --index-root "$MINIDEV/finetuning/train_data/db_contents_index" \
    --temp-dir  "$TMPDIR/bm25_tmp" \
    --threads 16
