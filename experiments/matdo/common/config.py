"""MATDO实验配置

包含所有实验共享的常数和参数
"""

import torch
from dataclasses import dataclass
from typing import Tuple


@dataclass
class MATDOConfig:
    """MATDO实验配置"""
    
    # 模型配置
    model_name: str = "meta-llama/Llama-2-7b-hf"
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    
    # MATDO维度配置
    R_options: Tuple[int, ...] = (2, 4, 8)  # 量化比特
    R_min: int = 2  # 最小量化
    S: int = 4  # 每层块数
    N_block: int = 1024  # 每块token数
    
    # 误差系数（从Phase 1系统辨识获得）
    alpha: float = 0.85  # Space误差系数
    beta: float = 12.4   # Scope误差系数
    gamma: float = 3.1   # Specificity误差系数
    delta: float = 0.23  # Space-Scope耦合
    epsilon: float = 0.08  # Scope-Spec耦合
    
    # 硬件成本系数（FLOPs）
    c_R: float = 1.2e3   # HBM访问成本
    c_M: float = 2.5e3   # SRAM访问成本
    c_T: float = 8.0e4   # 计算成本
    
    # KV Cache配置
    C_KV: int = 80 * 1024**3  # 80GB (A100)
    C_unit: float = None  # 将在__post_init__计算
    
    # SLA配置
    E_target: float = 0.05  # 目标误差5%
    
    # 计算预算
    B_max: float = 1e12  # 最大FLOPs预算
    
    def __post_init__(self):
        # 计算每token-bit字节数: 2 * d / 8 (Key+Value, FP16转字节)
        self.C_unit = 2 * self.d_model / 8
        
    def compute_T_max(self) -> float:
        """计算最大可行T（受计算预算限制）"""
        return (self.B_max - self.c_R * self.R_min * self.d_model) / (self.c_T * self.d_model ** 2)
    
    def compute_M_at_rho(self, rho: float, R: int = None) -> int:
        """在给定fill rate下计算可行M"""
        if R is None:
            R = self.R_min
        M_float = self.C_KV * (1 - rho) / (self.N_block * R * self.C_unit)
        return max(1, int(M_float))
    
    def compute_M_min(self) -> float:
        """计算满足SLA的最小M"""
        return self.beta / (self.S * self.E_target)
    
    def compute_rho_collapse(self) -> float:
        """计算信息坍缩点"""
        M_min = self.compute_M_min()
        return 1 - (M_min * self.N_block * self.R_min * self.C_unit) / self.C_KV


# 全局配置实例
config = MATDOConfig()
