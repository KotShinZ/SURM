# Example: WANDB_MODE=offline bash scripts/URM_arcagi1_H100_full.sh
run_name="URM-arcagi1-full"
checkpoint_path="checkpoints/${run_name}" 
mkdir -p $checkpoint_path

torchrun --nproc-per-node 1 pretrain.py \
data_path=data/arc1concept-full-aug-1000-nopadding-13_2 \
arch=urm arch.loops=32 arch.H_cycles=1 arch.L_cycles=1 arch.num_layers=4 \
arch.use_act=False arch.norm_diff_max=0.1 arch.norm_diff_min=0.001 \
global_batch_size=256 \
grad_accum_steps=3 \
epochs=20000 \
eval_interval=200 \
puzzle_emb_lr=1e-2 \
weight_decay=0.1 \
+run_name=$run_name \
+checkpoint_path=$checkpoint_path \
+ema=True \
evaluators="[]" 
# --load_checkpoint_file checkpoints/URM-nca2d/step_13020.pt
# arch.grid_depth=25 arch.grid_height=30 arch.grid_width=30 \
