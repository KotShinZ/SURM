# Example: WANDB_MODE=offline bash scripts/URM_arcagi1_H100_full.sh
run_name="URM-arcagi1-full-gen-cross-l8-CN0"
checkpoint_path="checkpoints/${run_name}" 
mkdir -p $checkpoint_path

torchrun --nproc-per-node 1 pretrain.py \
data_path=data/arc1withgen-aug-1000 \
arch=urm arch.loops=32 arch.H_cycles=1 arch.L_cycles=1 arch.num_layers=8 \
+arch.answer_only=True +arch.answer_only_context_layers=0 arch.input_injection_enabled=True \
arch.use_act=False arch.norm_diff_max=0.1 arch.norm_diff_min=0.001 \
arch.loss.label_mask=0.0 \
+arch.num_memory_tokens=0 \
global_batch_size=256 \
grad_accum_steps=3 \
epochs=100000 \
eval_interval=2000 \
eval_first=False \
puzzle_emb_lr=1e-2 \
weight_decay=0.1 \
+run_name=$run_name \
+checkpoint_path=$checkpoint_path \
+ema=True \
evaluators="[]" \
+mask_full_training=True \
full_answer_initial_mode="noised_label" \
full_answer_initial_gamma_distribution="uniform" \
full_answer_initial_gamma_min=0.0 \
full_answer_initial_gamma_max=0.0
# --load_checkpoint_file checkpoints/URM-nca2d/step_13020.pt
# arch.grid_depth=25 arch.grid_height=30 arch.grid_width=30 \
