#!/bin/bash
# Energy-Based Transformer (EBT) training on Sudoku
# Hyperparameters match URM sudoku (scripts/URM_sudoku.sh) where applicable.
#
# Prerequisites:
#   python -m data.build_sudoku_dataset \
#     --output-dir data/sudoku-extreme-1k-aug-1000 \
#     --subsample-size 1000 --num-aug 1000

run_name="EBT-sudoku"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path

torchrun --nproc-per-node 1 pretrain.py \
data_path=data/sudoku-extreme-1k-aug-1000 \
evaluators="[]" \
arch=ebt \
arch.mcmc_num_steps=12 arch.mcmc_step_size=100.0 arch.loops=16 \
epochs=50000 \
eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 global_batch_size=128 \
+run_name=$run_name \
+checkpoint_path=$checkpoint_path \
+ema=True
