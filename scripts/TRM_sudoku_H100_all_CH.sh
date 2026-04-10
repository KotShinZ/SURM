run_name="pretrain_att_sudoku_all_CH"
checkpoint_path="checkpoints/${run_name}" 
mkdir -p $checkpoint_path

torchrun --nproc-per-node 1 pretrain.py \
arch=trm \
arch.use_act=False arch.norm_diff_max=0.1 arch.norm_diff_min=0.001 \
data_path=data/sudoku-extreme-1k-aug-all \
evaluators="[]" \
epochs=15 eval_interval=1 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} \
+checkpoint_path=${checkpoint_path} \
+ema=True
