run_name="URM-arcagi1"
checkpoint_path="checkpoints/${run_name}" 
mkdir -p $checkpoint_path

torchrun --nproc-per-node 1 pretrain.py \
data_path=data/arc1concept-aug-1000-unpadded \
arch=urm arch.loops=32 arch.H_cycles=1 arch.L_cycles=1 arch.num_layers=4 \
arch.use_act=False arch.norm_diff_max=0.1 arch.norm_diff_min=0.001 \
arch.grid_height=30 arch.grid_width=30 \
global_batch_size=128 \
grad_accum_steps=6 \
epochs=200000 \
eval_interval=2000 \
puzzle_emb_lr=1e-4 \
weight_decay=0.1 \
+run_name=$run_name \
+checkpoint_path=$checkpoint_path \
+ema=True \
evaluators="[]"
