run_name="URM-sudoku-base"
checkpoint_path="checkpoints/${run_name}" 
mkdir -p $checkpoint_path

torchrun --nproc-per-node 1 pretrain.py \
data_path=data/sudoku-extreme-1k-aug-all \
arch=urm arch.loops=32 arch.H_cycles=6 arch.L_cycles=4 arch.num_layers=1 \
arch.grid_height=9 arch.grid_width=9 \
evaluators="[]" \
epochs=15 \
data_fraction=1 \
eval_interval=1 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 global_batch_size=768 \
+run_name=$run_name \
+checkpoint_path=$checkpoint_path \
+ema=True
