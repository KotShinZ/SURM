run_name="URM_1d_arc"
checkpoint_path="checkpoints/${run_name}" 
mkdir -p $checkpoint_path

torchrun --nproc-per-node 1 pretrain.py \
data_path=data/1d-arc-aug \
arch=urm arch.loops=32 arch.H_cycles=2 arch.L_cycles=6 arch.num_layers=4 \
arch.grid_height=0 arch.grid_width=0 \
arch.use_act=False arch.norm_diff_max=0.1 arch.norm_diff_min=0.001 +arch.halt_norm_in_use_act=False \
evaluators="[]" \
epochs=10000 \
data_fraction=1 \
eval_interval=200 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 global_batch_size=768 \
+run_name=$run_name \
+checkpoint_path=$checkpoint_path \
+ema=True
