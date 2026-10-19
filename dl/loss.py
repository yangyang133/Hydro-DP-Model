import torch
import torch.nn as nn

# -----------------------------
# KGE 损失示例
# -----------------------------
class KGE_Loss(nn.Module):
    """
    Kling-Gupta Efficiency 损失
    """
    def __init__(self):
        super().__init__()

    def forward(self, sim, obs):
        sim_mean = torch.mean(sim, dim=0)
        obs_mean = torch.mean(obs, dim=0)
        sim_std = torch.std(sim, dim=0)
        obs_std = torch.std(obs, dim=0)
        r = torch.sum((sim - sim_mean)*(obs - obs_mean), dim=0) / (torch.sqrt(torch.sum((sim - sim_mean)**2, dim=0)) * torch.sqrt(torch.sum((obs - obs_mean)**2, dim=0)) + 1e-6)
        alpha = sim_std / (obs_std + 1e-6)
        beta = sim_mean / (obs_mean + 1e-6)
        kge = 1 - torch.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        return 1 - kge.mean()  # 作为 loss 越小越好


# 兼容旧命名
KGELoss = KGE_Loss
