# Universal Reasoning Model

[![paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2512.14693)

Universal transformers (UTs) have been widely used for complex reasoning tasks such as ARC-AGI and Sudoku, yet the specific sources of their performance gains remain underexplored. In this work, we systematically analyze UTs variants and show that improvements on ARC-AGI primarily arise from the recurrent inductive bias and strong nonlinear components of Transformer, rather than from elaborate architectural designs. Motivated by this finding, we propose the Universal Reasoning Model (URM), which enhances the UT with short convolution and truncated backpropagation. Our approach substantially improves reasoning performance, achieving state-of-the-art 53.8% pass@1 on ARC-AGI 1 and 16.0% pass@1 on ARC-AGI 2.

## ⚠️ For Question Regarding Sudoku Score
The reported score of 87.4% in the TRM paper is obtained using an MLP model, which we believe it is completely different from the TRM architecture in ARC-AGI task. Therefore, **_for fair comparison_**, when reproducing the results, we unified the architectures for ARC-AGI 1, ARC-AGI 2, and Sudoku to be exactly the same, which means **the architecture used to reproduce Sudoku is the same TRM architecture used to run ARC-AGI**.

Reproducing the correct TRM Sudoku score:
```bash
git clone https://github.com/SamsungSAILMontreal/TinyRecursiveModels
cd TinyRecursiveModels
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000

run_name="pretrain_att_sudoku"
python pretrain.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True
```

Results:

<img width="400" height="250" alt="image" src="https://github.com/user-attachments/assets/c0699d98-64d1-4f41-9c8f-818f924ad77b" />


## Installation
```bash
uv venv
uv pip install -r requirements.txt
uv pip install flash-attn --no-build-isolation 
```

## Login Wandb
```bash
wandb login YOUR_API_KEY
```

## Preparing Data
```bash
# ARC-AGI-1
python -m data.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation

# ARC-AGI-1-full
python -m data.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1withgen-aug-1000-noeos \
  --subsets training evaluation concept \
  --test-set-name evaluation \
  --no-padding \
  --sources ARC-AGI1 ARC-GEN1 \
  --no-eos

# pair ごとの最小キャンバスで詰め、EOS は残す
python -m data.build_arc_dataset_full \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-full-aug-1000-pair-eos \
  --subsets training evaluation concept \
  --test-set-name evaluation \
  --no-padding \
  --no-padding-mode pair_eos

# pair ごとの最小キャンバスで詰め、EOS も入れない
python -m data.build_arc_dataset_full \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-full-aug-1000-pair-no-eos \
  --subsets training evaluation concept \
  --test-set-name evaluation \
  --no-padding \
  --no-padding-mode pair_no_eos

# ARC-AGI-2
python -m data.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2

# Sudoku
python -m data.build_sudoku_dataset \
  --output-dir data/sudoku-extreme-1k-aug-1000  \
  --subsample-size 1000 \
  --num-aug 1000

# Maze
python -m data.build_maze_dataset \
  --output-dir data/maze-30x30-hard-1k-aug-1000 \
  --subsample-size 1000 \
  --num-aug 1000 \
  --rebuild

# NCA-1D
python data/build_nca1d_dataset.py \
  --output-dir data/nca1d-9x9 \
  --train-size 100000 \
  --test-size 2000 \
  --state-height 9 \
  --time-start 30 \
  --time-end 38

# NCA-2D
python data/build_nca2d_dataset.py \
  --output-dir data/nca2d-1k \
  --train-size 1000000 \
  --test-size 10000 \
  --state-height 12 \
  --state-height-min 8 \
  --state-height-max 12 \
  --state-width 12 \
  --state-width-min 8 \
  --state-width-max 12 \
  --num-colors 4 \
  --patch-size 1 \
  --answer-steps 1 \
  --counts 3 \
  --counts-min 2 \
  --counts-max 4 \
  --time-start 24 \
  --time-span 2 \
  --batch-candidate-size 256 \
  --max-sampling-rounds 20 \
  --max-cells-per-candidate-batch 4000000 \
  --save-dtype int32 \
  --gzip-threshold-low 0.025 \
  --gzip-threshold-high 0.1


# upload ARC-AGI-1
export HF_TOKEN=YOUR_HF_TOKEN
python -m data.upload_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --subsets training evaluation \
  --test-set-name evaluation \
  --hf-repo-id "your-username/arc-agi-augmented" \
  --hf-token $HF_TOKEN \
  --num-aug 100

# upload ARC-AGI-2
export HF_TOKEN=YOUR_HF_TOKEN
python -m data.upload_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --subsets training2 evaluation2 \
  --test-set-name evaluation2 \
  --hf-repo-id "your-username/arc-agi-augmented" \
  --hf-token $HF_TOKEN
```

## Reproducing ARC-AGI 1 Score
```bash
bash scripts/URM_arcagi1.sh
```

## Reproducing ARC-AGI 2 Score
```bash
bash scripts/URM_arcagi2.sh
```

## Reproducing Sudoku Score
```bash
bash scripts/URM_sudoku.sh
```

## evalate Sudoku Score
```bash
python evaluate_trained_model.py --checkpoint checkpoints/URM-Sudoku-base --max_problems 4096 --loops 32 --batch_size 4096 --hidden_diff_threshold 0.1
```

## evalate ARC-AGI Score
```bash
python evaluate_trained_model.py --checkpoint checkpoints/URM-arcagi1 --max_problems 4096 --loops 32 --batch_size 4096 --hidden_diff_threshold 0.1
```

## continue training with checkpoint
```bash
torchrun --nproc-per-node 1 pretrain.py +resume_from_checkpoint_dir=checkpoints/URM-arcagi1 +run_name=URM-arcagi1 +checkpoint_path=checkpoints/URM-arcagi1 +load_checkpoint=latest
```


### Citation
```
@misc{gao2025universalreasoningmodel,
      title={Universal Reasoning Model}, 
      author={Zitian Gao and Lynx Chen and Yihao Xiao and He Xing and Ran Tao and Haoming Luo and Joey Zhou and Bryan Dai},
      year={2025},
      eprint={2512.14693},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2512.14693}, 
}
```
