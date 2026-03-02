run_name="URM-sudoku-base"
checkpoint_path="checkpoints/${run_name}" 
mkdir -p $checkpoint_path

torchrun --nproc-per-node 1 pretrain.py \
data_path=data/sudoku-extreme-1k-aug-1000 \
arch=urm arch.loops=16 arch.H_cycles=4 arch.L_cycles=6 arch.num_layers=4 \
evaluators="[]" \
epochs=500000 \
data_fraction=0.1 \
eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 global_batch_size=768 \
+run_name=$run_name \
+checkpoint_path=$checkpoint_path \
+ema=True
