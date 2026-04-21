#!/usr/bin/env bash
#SBATCH --job-name=bird_process
#SBATCH --partition=seas_compute
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/net/holy-isilon/ifs/rc_labs/ydu_lab/will/workspace/mini_dev/logs/process_dataset.%j.out
set -euo pipefail

MINIDEV=/net/holy-isilon/ifs/rc_labs/ydu_lab/will/workspace/mini_dev
ARCTIC=$MINIDEV/ArcticTraining/projects/arctic_text2sql_r1
BIRD=/n/home06/willgao/envs/bird
TD=$MINIDEV/finetuning/train_data

export JAVA_HOME=/n/sw/helmod-rocky8/apps/Core/jdk/21.0.2-fasrc01
export PATH=$BIRD/bin:$JAVA_HOME/bin:$PATH

cd "$ARCTIC"
echo "host=$(hostname) java=$(java -version 2>&1 | head -1)"

# Run train + val in parallel inside one allocation — val is small, plenty of RAM for both.
python -u data_preprocessing/process_dataset.py \
    --input_data_file "$TD/split_train.json" \
    --output_data_file "$TD/train_bird_raw.json" \
    --db_path "$TD/train_databases/" \
    --tables "$TD/train_tables.json" \
    --source bird --mode dev --value_limit_num 2 \
    --db_content_index_path "$TD/db_contents_index" \
    > "$MINIDEV/logs/process_dataset_train.out" 2>&1 &
TRAIN_PID=$!

python -u data_preprocessing/process_dataset.py \
    --input_data_file "$TD/split_val.json" \
    --output_data_file "$TD/val_bird_raw.json" \
    --db_path "$TD/train_databases/" \
    --tables "$TD/train_tables.json" \
    --source bird --mode dev --value_limit_num 2 \
    --db_content_index_path "$TD/db_contents_index" \
    > "$MINIDEV/logs/process_dataset_val.out" 2>&1 &
VAL_PID=$!

echo "train pid=$TRAIN_PID  val pid=$VAL_PID"
wait $TRAIN_PID; TRAIN_RC=$?
wait $VAL_PID;   VAL_RC=$?
echo "train_rc=$TRAIN_RC  val_rc=$VAL_RC"
[ $TRAIN_RC -eq 0 ] && [ $VAL_RC -eq 0 ]
