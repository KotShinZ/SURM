run_name="URM-energy-sudoku"
checkpoint_path="checkpoints/${run_name}"
mkdir -p $checkpoint_path

DISABLE_COMPILE=1 torchrun --nproc-per-node 1 pretrain.py \
data_path=data/sudoku-extreme-1k-aug-1000 \
arch=urm_energy arch.loops=16 arch.H_cycles=4 arch.L_cycles=12 arch.num_layers=2 \
evaluators="[]" \
arch.mcmc_step_size=100.0 \
epochs=50000 \
eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 global_batch_size=768 \
+run_name=$run_name \
+checkpoint_path=$checkpoint_path \
+ema=True
