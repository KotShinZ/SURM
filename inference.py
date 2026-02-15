import argparse
import os
import copy
import re
import yaml
import json
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm

# 元のコードで使用されているモジュールをインポート
from models.muon import Muon
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from utils import load_model_class

from typing import List, Optional, Any, Dict
import pydantic
import dataclasses

class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str

class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str
    loss: LossConfig

class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str

class PretrainConfig(pydantic.BaseModel):
    arch: ArchConfig
    data_path: str
    evaluators: List[EvaluatorConfig] = []
    global_batch_size: int
    epochs: int
    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int
    weight_decay: float
    beta1: float
    beta2: float
    target_q_update_every: int
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float
    grad_accum_steps: int = 1
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None
    load_checkpoint: Optional[str] = None
    load_strict: bool = True
    load_optimizer_state: bool = True
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    eval_save_outputs: List[str] = []
    loop_deltas: List[str] = []
    ema: bool = False
    ema_rate: float = 0.999
    use_muon: bool = False

# --- Helper Functions ---

def _resolve_checkpoint_path(path: str) -> Optional[str]:
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        pattern = re.compile(r"step_(\d+)(?:\.pt)?$")
        candidates = []
        for file_name in os.listdir(path):
            match = pattern.match(file_name)
            if match:
                candidates.append((int(match.group(1)), os.path.join(path, file_name)))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[-1][1]
    return None

def load_config_from_checkpoint_path(checkpoint_path: str) -> Optional[PretrainConfig]:
    resolved_path = _resolve_checkpoint_path(checkpoint_path)
    if resolved_path is None:
        return None
    checkpoint_dir = Path(resolved_path).parent
    candidates = [
        checkpoint_dir / "config.json",
        checkpoint_dir / "config.yaml",
        checkpoint_dir / "all_config.yaml",
        checkpoint_dir / ".hydra" / "config.yaml"
    ]
    for candidate in candidates:
        if not candidate.exists(): continue
        try:
            conf = OmegaConf.load(candidate)
            as_dict = OmegaConf.to_container(conf, resolve=True)
            if isinstance(as_dict, dict): return PretrainConfig(**as_dict)
        except Exception: pass
        try:
            with open(candidate, "r") as f:
                config_dict = json.load(f) if candidate.suffix == '.json' else yaml.safe_load(f)
            if isinstance(config_dict, dict): return PretrainConfig(**config_dict)
        except Exception: pass
    return None

def create_model_for_inference(config: PretrainConfig, train_metadata, device="cuda"):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,
        batch_size=1,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False,
    )
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    print("Initializing model on CPU...")
    model = model_cls(model_cfg)
    model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)
    
    print(f"Moving model to {device}...")
    model = model.to(device)
    return model

# --- Device Helper ---

def recursive_to_device(obj, device):
    """Recursively move tensors in a nested structure (dict, list, dataclass) to device"""
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, list):
        return [recursive_to_device(x, device) for x in obj]
    if isinstance(obj, tuple):
        return tuple(recursive_to_device(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: recursive_to_device(v, device) for k, v in obj.items()}
    
    # Handle Dataclasses
    if hasattr(obj, "__dataclass_fields__"):
        changes = {
            f: recursive_to_device(getattr(obj, f), device) 
            for f in obj.__dataclass_fields__
        }
        return type(obj)(**changes)
        
    return obj

def inference(model, dataloader, output_keys: List[str], device="cuda"):
    model.eval()
    results = {}
    targets = {} # 正解ラベル(labels)を格納する辞書

    print(f"Starting inference on {len(dataloader)} batches...")

    with torch.inference_mode():
        for i, (set_name, batch, global_batch_size) in enumerate(tqdm(dataloader)):
            # Move batch to device
            batch_gpu = {k: v.to(device) for k, v in batch.items()}

            # Initial carry
            carry = model.initial_carry(batch_gpu)
            
            # CarryをGPUへ転送
            carry = recursive_to_device(carry, device)

            inference_steps = 0
            while True:
                carry, _, _, preds, all_finish = model(
                    carry=carry, batch=batch_gpu, return_keys=output_keys
                )
                inference_steps += 1
                if all_finish:
                    break
            
            # 推論結果を収集
            for k, v in preds.items():
                if k not in results: results[k] = []
                results[k].append(v.cpu())
            
            # 正解ラベル(labels)を収集
            if "labels" in batch:
                if "labels" not in targets: targets["labels"] = []
                targets["labels"].append(batch["labels"]) # batchは元々CPUにあるのでそのまま使う

            # テスト用：長すぎる場合は途中で打ち切る（必要に応じて調整してください）
            if i >= 100 and len(dataloader) > 100:
                print("Reached 100 batches, stopping early for quick evaluation.")
                break

    # テンソルを結合
    final_results = {k: torch.cat(v, dim=0) for k, v in results.items()}
    final_targets = {k: torch.cat(v, dim=0) for k, v in targets.items()}
    
    return final_results, final_targets

def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Accuracy計算関数
    predictions: [N, SeqLen] (indices) or [N, SeqLen, Vocab] (logits)
    targets: [N, SeqLen] (indices)
    """
    print("\n--- Accuracy Calculation ---")
    
    # logitsの場合はargmaxを取る
    if predictions.dim() == targets.dim() + 1:
        print("Predictions contain logits, taking argmax...")
        predictions = predictions.argmax(dim=-1)
    
    if predictions.shape != targets.shape:
        print(f"Warning: Shape mismatch! Preds: {predictions.shape}, Targets: {targets.shape}")
        # 形状が合わない場合の簡易的なリサイズやスライス（必要に応じて）
        min_len = min(predictions.shape[1], targets.shape[1])
        predictions = predictions[:, :min_len]
        targets = targets[:, :min_len]

    # 正誤判定行列 (True/False)
    correct_matrix = (predictions == targets)

    # 1. 平均正答率 (Average Accuracy): 全マスのうち合っているマスの割合
    avg_acc = correct_matrix.float().mean().item()

    # 2. 完全正答率 (Exact Accuracy): 1つも間違いがない行の割合
    # dim=1 (各パズル) 方向に対して all() をとる
    exact_acc = correct_matrix.all(dim=1).float().mean().item()

    print(f"Element-wise Accuracy (Average): {avg_acc * 100:.2f}%")
    print(f"Exact Match Accuracy (All Correct): {exact_acc * 100:.2f}%")
    print("----------------------------\n")
    
    return avg_acc, exact_acc

def main():
    parser = argparse.ArgumentParser(description="Inference script for Puzzle Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--data_path", type=str, default=None, help="Path to dataset")
    parser.add_argument("--output", type=str, default="inference_results.pt", help="Output file")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading config from checkpoint: {args.checkpoint}")
    config = load_config_from_checkpoint_path(args.checkpoint)
    if config is None: raise ValueError("Could not load config.")
    if args.data_path: config.data_path = args.data_path
    
    print(f"Loading dataset split: {args.split}")
    ds_config = PuzzleDatasetConfig(
        seed=config.seed, dataset_path=config.data_path, rank=0, num_replicas=1,
        global_batch_size=config.global_batch_size, test_set_mode=True, epochs_per_iter=1
    )
    dataset = PuzzleDataset(ds_config, split=args.split)
    metadata = dataset.metadata
    dataloader = DataLoader(dataset, batch_size=None, num_workers=1, prefetch_factor=2, pin_memory=True)

    print("Creating model...")
    model = create_model_for_inference(config, metadata, device=device)

    resolved_ckpt_path = _resolve_checkpoint_path(args.checkpoint)
    print(f"Loading weights from {resolved_ckpt_path}")
    checkpoint = torch.load(resolved_ckpt_path, map_location=device)
    
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    # Prefix removal
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            new_state_dict[key[len("_orig_mod."):]] = value
        else:
            new_state_dict[key] = value
    state_dict = new_state_dict

    try:
        model.load_state_dict(state_dict, strict=True)
        print("Successfully loaded state_dict (strict=True).")
    except RuntimeError as e:
        print(f"Strict loading failed. Retrying with strict=False. Error: {e}")
        model.load_state_dict(state_dict, strict=False)

    # outputsキーを確実に取得する
    output_keys = config.eval_save_outputs if config.eval_save_outputs else []
    if "outputs" not in output_keys:
        output_keys.append("outputs")
    print(f"Output keys to capture: {output_keys}")
    
    # 推論実行
    results, targets = inference(model, dataloader, output_keys, device=device)

    # 結果保存
    print(f"Saving results to {args.output}")
    # targetsも保存しておくと後で分析に便利です
    save_data = {"predictions": results, "targets": targets}
    torch.save(save_data, args.output)

    # 正答率計算
    if "outputs" in results and "labels" in targets:
        calculate_accuracy(results["outputs"], targets["labels"])
    else:
        print("Cannot calculate accuracy: 'outputs' or 'labels' missing.")

    print("Done.")

if __name__ == "__main__":
    main()