run_name="URM-sudoku-base"
checkpoint_path="checkpoints/${run_name}" 
mkdir -p $checkpoint_path

torchrun --nproc-per-node 1 pretrain.py \
data_path=data/sudoku-extreme-1k-aug-1000 \
arch=urm arch.loops=32 arch.H_cycles=1 arch.L_cycles=6 arch.num_layers=4 \
arch.grid_height=9 arch.grid_width=9 \
arch.use_act=True arch.norm_diff_max=0.1 arch.norm_diff_min=0.001 +arch.halt_norm_in_use_act=True \
+arch.patch_io_enabled=True +arch.patch_height=2 +arch.patch_width=2 +arch.patch_pre_embedding_size=3 \
evaluators="[]" \
epochs=50000 \
data_fraction=1 \
eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 global_batch_size=768 \
+run_name=$run_name \
+checkpoint_path=$checkpoint_path \
+ema=True
