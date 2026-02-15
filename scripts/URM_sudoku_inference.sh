#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# --- 設定 (学習時と同じ変数を使用) ---
run_name="URM-sudoku-base"
checkpoint_dir="checkpoints/${run_name}"
data_path="data/sudoku-extreme-1k-aug-1000"

# 推論に使用するデータセットの分割 (test, val, train)
split="test"

# 推論時のバッチサイズ (GPUメモリに合わせて調整してください)
# 学習時は768でしたが、推論時は勾配を保持しないため、より大きくできる場合があります
batch_size=4096

# --- チェックポイントの自動検出 ---
# ディレクトリ内の "step_XXXX.pt" の中で数字が最も大きいものを探します
if [ -d "$checkpoint_dir" ]; then
    checkpoint_path=$(ls -v ${checkpoint_dir}/step_*.pt | tail -n 1)
else
    echo "Error: Checkpoint directory not found: $checkpoint_dir"
    exit 1
fi

if [ -z "$checkpoint_path" ]; then
    echo "Error: No checkpoint file (step_*.pt) found in $checkpoint_dir"
    exit 1
fi

echo "Found latest checkpoint: $checkpoint_path"

# 出力ファイル名
output_file="${checkpoint_dir}/inference_results_${split}.pt"

# --- 推論実行 ---
# Pythonスクリプト名は前回作成したものを指定 (例: inference.py)
python inference.py \
    --checkpoint "$checkpoint_path" \
    --data_path "$data_path" \
    --split "$split" \
    --batch_size "$batch_size" \
    --output "$output_file"

echo "Inference completed. Results saved to: $output_file"