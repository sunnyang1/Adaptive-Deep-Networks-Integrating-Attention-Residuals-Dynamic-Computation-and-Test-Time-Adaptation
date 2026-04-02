"""
US6: 在线系统辨识

使用递归最小二乘法(RLS)在线估计耦合系数(δ, ε)
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.matdo.common.config import config


@dataclass
class RLSState:
    """RLS算法状态"""
    theta: np.ndarray  # 参数估计 [δ, ε]
    P: np.ndarray      # 协方差矩阵
    lambda_: float     # 遗忘因子


def rls_update(
    state: RLSState,
    x_t: np.ndarray,
    y_t: float
) -> RLSState:
    """
    递归最小二乘法更新
    
    标准RLS算法:
    K_t = P_{t-1} x_t / (λ + x_t^T P_{t-1} x_t)
    θ_t = θ_{t-1} + K_t (y_t - x_t^T θ_{t-1})
    P_t = (P_{t-1} - K_t x_t^T P_{t-1}) / λ
    
    Args:
        state: 当前RLS状态
        x_t: 特征向量 [2^{-2R}/M, ln(M)/T]
        y_t: 观测误差（耦合项的贡献）
    
    Returns:
        new_state: 更新后的状态
    """
    # 计算增益
    denom = state.lambda_ + x_t.T @ state.P @ x_t
    K_t = state.P @ x_t / denom
    
    # 更新参数
    prediction = x_t.T @ state.theta
    error = y_t - prediction
    theta_new = state.theta + K_t * error
    
    # 更新协方差
    P_new = (state.P - np.outer(K_t, x_t.T @ state.P)) / state.lambda_
    
    return RLSState(theta=theta_new, P=P_new, lambda_=state.lambda_)


def simulate_online_queries(
    num_queries: int,
    true_delta: float = 0.23,
    true_epsilon: float = 0.08
) -> List[Tuple[np.ndarray, float]]:
    """
    模拟在线查询序列
    
    生成(R, M, T)配置和观测误差的序列
    
    Args:
        num_queries: 查询数量
        true_delta: 真实的δ值
        true_epsilon: 真实的ε值
    
    Returns:
        data: [(x_t, y_t), ...] 列表
    """
    data = []
    
    for t in range(num_queries):
        # 随机生成配置（模拟系统运行时的变化）
        R = np.random.choice([2, 4, 8])
        M = np.random.randint(8, 64)
        T = np.random.choice([4, 8, 16, 32])
        
        # 计算特征
        x1 = (2 ** (-2 * R)) / M  # Space-Scope耦合特征
        x2 = np.log(M) / T         # Scope-Spec耦合特征
        x_t = np.array([x1, x2])
        
        # 生成观测（真实耦合项 + 噪声）
        y_t = true_delta * x1 + true_epsilon * x2
        y_t += np.random.normal(0, 0.001)  # 观测噪声
        
        data.append((x_t, y_t))
    
    return data


def run_online_identification(
    num_queries: int = 100,
    lambda_: float = 0.95,
    output_dir: Optional[Path] = None
) -> dict:
    """
    运行在线系统辨识实验
    
    Args:
        num_queries: 模拟查询数
        lambda_: 遗忘因子
        output_dir: 输出目录
    
    Returns:
        results: 辨识结果
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("US6: 在线系统辨识")
    print("=" * 70)
    print(f"查询数: {num_queries}")
    print(f"遗忘因子 λ: {lambda_}")
    print()
    
    # 真实值（用于评估）
    true_delta = config.delta
    true_epsilon = config.epsilon
    
    print(f"真实值: δ={true_delta:.4f}, ε={true_epsilon:.4f}")
    print()
    
    # 初始化RLS
    theta_0 = np.array([0.0, 0.0])  # 初始猜测
    P_0 = np.eye(2) * 10  # 初始协方差（较大不确定性）
    state = RLSState(theta=theta_0, P=P_0, lambda_=lambda_)
    
    # 生成数据
    data = simulate_online_queries(num_queries, true_delta, true_epsilon)
    
    # 记录估计历史
    history = {
        'delta_est': [],
        'epsilon_est': [],
        'errors': []
    }
    
    print("运行RLS更新...")
    convergence_step = None
    
    for t, (x_t, y_t) in enumerate(data):
        # RLS更新
        state = rls_update(state, x_t, y_t)
        
        # 记录
        delta_est, epsilon_est = state.theta
        history['delta_est'].append(float(delta_est))
        history['epsilon_est'].append(float(epsilon_est))
        
        # 计算与真实值的误差
        error = np.sqrt((delta_est - true_delta)**2 + (epsilon_est - true_epsilon)**2)
        history['errors'].append(float(error))
        
        # 检查收敛（误差<5%）
        if convergence_step is None and error < 0.05 * np.sqrt(true_delta**2 + true_epsilon**2):
            convergence_step = t
        
        if (t + 1) % 20 == 0:
            print(f"  Step {t+1}: δ={delta_est:.4f}, ε={epsilon_est:.4f}, error={error:.4f}")
    
    # 最终结果
    final_delta, final_epsilon = state.theta
    final_error = np.sqrt((final_delta - true_delta)**2 + (final_epsilon - true_epsilon)**2)
    relative_error = final_error / np.sqrt(true_delta**2 + true_epsilon**2)
    
    print()
    print("最终结果:")
    print(f"  估计 δ: {final_delta:.4f} (真实: {true_delta:.4f})")
    print(f"  估计 ε: {final_epsilon:.4f} (真实: {true_epsilon:.4f})")
    print(f"  相对误差: {relative_error*100:.2f}%")
    if convergence_step:
        print(f"  收敛步数: {convergence_step}")
    
    # 验收标准
    converged = convergence_step is not None and convergence_step < 100
    error_small = relative_error < 0.05
    
    print()
    print("验收标准:")
    print(f"  收敛步数 < 100: {convergence_step if convergence_step else 'N/A'} {'✅' if converged else '❌'}")
    print(f"  相对误差 < 5%: {relative_error*100:.2f}% {'✅' if error_small else '❌'}")
    
    # 保存结果
    results = {
        'parameters': {
            'num_queries': num_queries,
            'lambda': lambda_,
            'true_delta': float(true_delta),
            'true_epsilon': float(true_epsilon)
        },
        'estimates': {
            'delta': float(final_delta),
            'epsilon': float(final_epsilon)
        },
        'history': history,
        'metrics': {
            'convergence_step': convergence_step,
            'final_relative_error': float(relative_error)
        },
        'acceptance': {
            'converged': converged,
            'error_small': error_small,
            'overall_pass': converged and error_small
        }
    }
    
    output_file = output_dir / "rls_identification_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果保存至: {output_file}")
    
    # 最终结论
    print()
    print("=" * 70)
    if results['acceptance']['overall_pass']:
        print("✅ US6 PASSED: 在线系统辨识成功")
        print(f"   收敛步数: {convergence_step}")
        print(f"   相对误差: {relative_error*100:.2f}%")
    else:
        print("❌ US6 FAILED: 未通过验收标准")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    np.random.seed(42)
    results = run_online_identification()
