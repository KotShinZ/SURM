run_name="URM-arcagi1-gen"
checkpoint_path="checkpoints/${run_name}" 
mkdir -p $checkpoint_path

torchrun --nproc-per-node 1 pretrain.py \
data_path=data/arc1gen3-aug-1000 \
arch=urm arch.loops=32 arch.H_cycles=1 arch.L_cycles=1 arch.num_layers=4 \
arch.use_act=False arch.norm_diff_max=0.1 arch.norm_diff_min=0.001 \
arch.grid_height=30 arch.grid_width=30 \
global_batch_size=768 \
grad_accum_steps=1 \
epochs=200000 \
eval_interval=2000 \
torch_compile=True \
puzzle_emb_lr=1e-2 \
weight_decay=0.1 \
+run_name=$run_name \
+checkpoint_path=$checkpoint_path \
+ema=True \
evaluators="[]"
