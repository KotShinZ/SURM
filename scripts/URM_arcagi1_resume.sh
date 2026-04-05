#!/usr/bin/env bash
set -euo pipefail

run_name="${RUN_NAME:-URM-arcagi1}"
checkpoint_path="${CHECKPOINT_PATH:-checkpoints/${run_name}}"
nproc_per_node="${NPROC_PER_NODE:-1}"

if [ ! -d "$checkpoint_path" ]; then
    echo "Error: checkpoint directory not found: $checkpoint_path" >&2
    exit 1
fi

latest_checkpoint_name="$(find "$checkpoint_path" -maxdepth 1 -type f -printf '%f\n' | grep -E '^step_[0-9]+\.pt$' | sort -V | tail -n 1)"
if [ -z "$latest_checkpoint_name" ]; then
    echo "Error: no checkpoint file (step_*.pt) found in $checkpoint_path" >&2
    exit 1
fi

latest_checkpoint="${checkpoint_path}/${latest_checkpoint_name}"

echo "Resuming $run_name from $latest_checkpoint"

torchrun --nproc-per-node "$nproc_per_node" pretrain.py \
    +resume_from_checkpoint_dir="$checkpoint_path" \
    +run_name="$run_name" \
    +checkpoint_path="$checkpoint_path" \
    +load_checkpoint=latest
