run_name="URM-arcagi1"
checkpoint_path="checkpoints/${run_name}" 
mkdir -p $checkpoint_path

torchrun --nproc-per-node 1 pretrain.py \
data_path=data/arc1concept-aug-1000 \
arch=urm arch.loops=32 arch.H_cycles=1 arch.L_cycles=6 arch.num_layers=4 \
arch.grid_height=30 arch.grid_width=30 \
global_batch_size=128 \
evaluators="[]" \
grad_accum_steps=1 \
epochs=200000 \
eval_interval=20000 \
puzzle_emb_lr=1e-4 \
weight_decay=0.1 \
+run_name=$run_name \
+checkpoint_path=$checkpoint_path \
+ema=True
