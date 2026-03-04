import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math

@dataclass
class WelfordStats:
    count: float = 0.0
    mean: float = 0.0
    m2: float = 0.0  # Sum of squares of differences from the mean

    @staticmethod
    def from_tensor(x: torch.Tensor):
        """Tensorから統計量を作成"""
        # xは任意のshape。フラットにして計算
        x_flat = x.detach().view(-1).float()
        count = x_flat.numel()
        if count == 0:
            return WelfordStats(0, 0.0, 0.0)
        mean = x_flat.mean().item()
        # M2 = sum((x - mean)^2)
        m2 = torch.sum((x_flat - mean) ** 2).item()
        return WelfordStats(float(count), mean, m2)

def merge_welford(stats_a: WelfordStats, stats_b: WelfordStats) -> WelfordStats:
    """2つのWelfordStats（統計量）を結合する"""
    if stats_a.count == 0: return stats_b
    if stats_b.count == 0: return stats_a

    new_count = stats_a.count + stats_b.count
    delta = stats_b.mean - stats_a.mean
    
    # 新しい平均
    new_mean = stats_a.mean + delta * (stats_b.count / new_count)
    
    # 新しいM2 (Chan et al. の並列アルゴリズムに基づく結合式)
    # M2_ab = M2_a + M2_b + delta^2 * (n_a * n_b / n_ab)
    new_m2 = stats_a.m2 + stats_b.m2 + (delta ** 2) * (stats_a.count * stats_b.count / new_count)
    
    return WelfordStats(new_count, new_mean, new_m2)

class WandbLogger:
    _instance = None
    is_log = True
    store_count = {}
    data = {}
    table_dict = {}

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def store(self, key, value):
        """合計値を一時保存する
        value: int, float, tensor(1D)
        """
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if key in self.data:
                if self.data[key].shape[0] < value.shape[0]:
                    value = value[:self.data[key].shape[0]]
        if key not in self.data:
            self.data[key] = value
            self.store_count[key] = 1
        else:
            self.data[key] += value
            self.store_count[key] += 1
        
    def get_log_dict(self, global_step):
        """保存するための辞書を取得する"""
        log_dict = {}
            
        for key, value in self.data.items():
            self.data[key] = value / self.store_count[key]
            if isinstance(self.data[key], torch.Tensor) and self.data[key].ndim > 0 and self.data[key].shape[0] > 1:
                if self.table_dict.get(key) is None:
                    self.table_dict[key] = []
                for i, d in enumerate(self.data[key].tolist()):
                    self.table_dict[key].append([global_step, i, d])
                log_dict[key] = wandb.Table(columns=["global_step", "internal_step", key], data=self.table_dict[key])
            else: 
                log_dict[key] = self.data[key]
        self.data = {}
        self.store_count = {}
        return log_dict

# グローバルインスタンス
global_logger = WandbLogger.get_instance()