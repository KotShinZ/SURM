# Example: WANDB_MODE=offline bash scripts/URM_arcagi1_H100_full.sh
run_name="URM-arcagi1-full-gen-l4-casual"
checkpoint_path="checkpoints/${run_name}" 
mkdir -p $checkpoint_path

torchrun --nproc-per-node 1 pretrain.py \
data_path=data/arc1withgen-aug-1000 \
arch=urm arch.loops=16 arch.H_cycles=1 arch.L_cycles=1 arch.num_layers=4 arch.hidden_size=512 \
arch.forward_mode=casual arch.answer_only_context_layers=0 arch.input_injection_enabled=True \
arch.use_act=False arch.norm_diff_max=0.1 arch.norm_diff_min=0.001 \
arch.loss.label_mask=0.0 \
+arch.num_memory_tokens=0 \
global_batch_size=128 \
grad_accum_steps=6 \
epochs=100000 \
eval_interval=10000 \
eval_first=False \
eval_batch_size=32 \
autoregressive_eval_cache_chunk_size=64 \
puzzle_emb_lr=1e-2 \
weight_decay=0.1 \
+run_name=$run_name \
+checkpoint_path=$checkpoint_path \
+ema=True \
evaluators="[]" \
+mask_full_training=True \
+padding=False \
full_answer_initial_mode="noised_label" \
use_muon=False \
# --load_checkpoint_file checkpoints/URM-nca2d/step_13020.pt
# arch.grid_depth=25 arch.grid_height=30 arch.grid_width=30 \
