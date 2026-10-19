import random
import numpy as np
import torch
from typing import Union

def set_global_seed(seed: int):
    """设置全局随机种子，保证训练和模拟复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def kge(sim: Union[np.ndarray, torch.Tensor], obs: Union[np.ndarray, torch.Tensor], return_tensor=False):
    """
    计算 Kling-Gupta Efficiency，支持 numpy 或 torch。
    自动跳过观测值为 NaN 的时段。
    
    Args:
        sim: 模拟值
        obs: 观测值
        return_tensor: 是否返回tensor（用于梯度计算），默认False返回float
    """
    if isinstance(sim, np.ndarray):
        sim = torch.tensor(sim, dtype=torch.float32)
    if isinstance(obs, np.ndarray):
        obs = torch.tensor(obs, dtype=torch.float32)

    # 自动掩膜NaN：只使用观测值和模拟值都有效的时段
    valid_mask = ~torch.isnan(obs) & ~torch.isnan(sim)
    if valid_mask.sum() < 10:
        # 有效数据太少，无法计算有意义的KGE
        if return_tensor:
            return torch.tensor(float('nan'), requires_grad=True)
        else:
            return float('nan')
    
    sim = sim[valid_mask]
    obs = obs[valid_mask]

    sim_mean = torch.mean(sim, dim=0)
    obs_mean = torch.mean(obs, dim=0)
    sim_std = torch.std(sim, dim=0)
    obs_std = torch.std(obs, dim=0)
    r = torch.sum((sim - sim_mean) * (obs - obs_mean), dim=0) / (
        torch.sqrt(torch.sum((sim - sim_mean) ** 2, dim=0))
        * torch.sqrt(torch.sum((obs - obs_mean) ** 2, dim=0))
        + 1e-6
    )
    alpha = sim_std / (obs_std + 1e-6)
    beta = sim_mean / (obs_mean + 1e-6)
    kge_val = 1 - torch.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    
    if return_tensor:
        return kge_val.mean()
    else:
        return kge_val.mean().item()


def nse(sim: Union[np.ndarray, torch.Tensor], obs: Union[np.ndarray, torch.Tensor]):
    """
    计算 Nash-Sutcliffe Efficiency，支持 numpy 或 torch。
    
    NSE = 1 - SS_res / SS_tot
    其中：
    SS_res = sum((obs - sim)^2)  # 残差平方和
    SS_tot = sum((obs - obs_mean)^2)  # 总平方和
    
    NSE范围: (-∞, 1]
    NSE = 1: 完美拟合
    NSE = 0: 模型等同于观测平均值
    NSE < 0: 模型比观测平均值更差
    """
    if isinstance(sim, np.ndarray):
        sim = torch.tensor(sim, dtype=torch.float32)
    if isinstance(obs, np.ndarray):
        obs = torch.tensor(obs, dtype=torch.float32)
    
    # 自动掩膜NaN
    valid_mask = ~torch.isnan(obs) & ~torch.isnan(sim)
    if valid_mask.sum() < 10:
        return float('nan')
    
    sim = sim[valid_mask]
    obs = obs[valid_mask]
    
    obs_mean = torch.mean(obs, dim=0)
    
    # 残差平方和
    ss_res = torch.sum((obs - sim) ** 2, dim=0)
    
    # 总平方和
    ss_tot = torch.sum((obs - obs_mean) ** 2, dim=0)
    
    # NSE
    nse_val = 1 - ss_res / (ss_tot + 1e-6)
    
    return nse_val.mean().item()


def rmse(sim: Union[np.ndarray, torch.Tensor], obs: Union[np.ndarray, torch.Tensor]):
    """
    计算 均方根误差 (Root Mean Square Error)，支持 numpy 或 torch。
    
    RMSE = sqrt(mean((obs - sim)^2))
    
    RMSE范围: [0, +∞)
    RMSE = 0: 完美拟合
    RMSE越小越好
    """
    if isinstance(sim, np.ndarray):
        sim = torch.tensor(sim, dtype=torch.float32)
    if isinstance(obs, np.ndarray):
        obs = torch.tensor(obs, dtype=torch.float32)
    
    # 自动掩膜NaN
    valid_mask = ~torch.isnan(obs) & ~torch.isnan(sim)
    if valid_mask.sum() < 10:
        return float('nan')
    
    sim = sim[valid_mask]
    obs = obs[valid_mask]
    
    # 均方误差
    mse = torch.mean((obs - sim) ** 2, dim=0)
    
    # 均方根误差
    rmse_val = torch.sqrt(mse)
    
    return rmse_val.mean().item()


def bias(sim: Union[np.ndarray, torch.Tensor], obs: Union[np.ndarray, torch.Tensor]):
    """
    计算 偏差 (Bias)，支持 numpy 或 torch。
    
    Bias = mean(sim - obs)
    相对偏差 (百分比) = Bias / mean(obs) * 100%
    
    Bias范围: (-∞, +∞)
    Bias = 0: 无偏差
    Bias > 0: 高估
    Bias < 0: 低估
    """
    if isinstance(sim, np.ndarray):
        sim = torch.tensor(sim, dtype=torch.float32)
    if isinstance(obs, np.ndarray):
        obs = torch.tensor(obs, dtype=torch.float32)
    
    # 自动掩膜NaN
    valid_mask = ~torch.isnan(obs) & ~torch.isnan(sim)
    if valid_mask.sum() < 10:
        return float('nan')
    
    sim = sim[valid_mask]
    obs = obs[valid_mask]
    
    # 相对偏差（百分比）
    obs_mean = torch.mean(obs, dim=0)
    bias_val = torch.mean(sim - obs, dim=0) / (obs_mean + 1e-6) * 100.0

    return bias_val.mean().item()
