run_name="TRM-arcagi1-noeval"
checkpoint_path="checkpoints/${run_name}" 
mkdir -p $checkpoint_path

torchrun --nproc-per-node 1 pretrain.py \
data_path=data/arc1concept-aug-1000-noeval-pad \
arch=urm \
arch.loops=16 \
arch.H_cycles=3 arch.L_cycles=4 arch.grad_H_cycles=1 \
arch.num_layers=2 arch.L_layers=2 arch.H_layers=2 \
arch.is_ConvSwiGLU=False \
arch.use_act=True arch.halt_exploration_prob=0.1 \
arch.norm_diff_max=0.1 arch.norm_diff_min=0.001 \
global_batch_size=256 \
grad_accum_steps=3 \
epochs=200000 \
eval_interval=2000 \
torch_compile=True \
puzzle_emb_lr=1e-2 \
weight_decay=0.1 \
+run_name=$run_name \
+checkpoint_path=$checkpoint_path \
+ema=True \
evaluators="[]" \
+arch.loop_type="trm"

# +padding=True \
# arch.grid_height=30 arch.grid_width=30 \
