run_name="URM-maze"
checkpoint_path="checkpoints/${run_name}" 
mkdir -p $checkpoint_path

torchrun --nproc-per-node 1 pretrain.py \
data_path=data/maze \
arch=urm arch.loops=32 arch.H_cycles=1 arch.L_cycles=6 arch.num_layers=4 \
arch.use_act=False arch.norm_diff_max=0.1 arch.norm_diff_min=0.001 \
arch.grid_height=30 arch.grid_width=30 \
evaluators="[]" \
epochs=5000 \
data_fraction=1 \
eval_interval=200 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 global_batch_size=128 \
+run_name=$run_name \
+checkpoint_path=$checkpoint_path \
+ema=True
