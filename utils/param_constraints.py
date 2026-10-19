"""
参数约束处理模块

用于处理水文模型中的参数约束关系，特别是神经网络训练时的约束处理。

各模型参数约束关系总结：

1. XAJ 新安江模型
   - 三水源响应约束: ki、kg 分别限制在标准取值范围内，未流出的自由水保留在 S 中
   - 衰减保留系数约束: cr、ci、cg ∈ (0, 1)
   - 土壤张力水容量: wum、wlm、wdm 独立给定，wm 为三者之和

2. GR4J 模型
   - 当前实验保留 x2 参数，但固定为 0（关闭地下水交换）

3. SACSMA 模型
   - 张力水分配: uztwm + lztwm 为总张力水容量
   - 自由水分配: uzfwm + lzfpm + lzfsm 为总自由水容量  
   - 下渗分配: pfree + (1-pfree) = 1

4. Tank 模型
   - 各层蓄水容量独立，但建议 S1 < S2 < S3 < S4

5. HBV 模型
   - 衰减系数约束: k0 > k1 > k2 (快 > 中 > 慢)
   - LP ≤ 1 (蒸散临界比)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ParameterConstraints:
    """参数约束处理类"""
    
    @staticmethod
    def softmax_split(raw_params: torch.Tensor, n_parts: int = 3) -> torch.Tensor:
        """
        使用 softmax 将原始参数转换为和为1的分配系数
        
        适用于：需要和为 1 的通用分配参数
        
        Args:
            raw_params: 原始参数 [batch, n_parts] 或 [n_parts]
            n_parts: 分配部分数量
            
        Returns:
            约束后的参数，和为1
            
        Example:
            >>> raw = torch.randn(3)  # 任意值
            >>> constrained = ParameterConstraints.softmax_split(raw, 3)
            >>> constrained.sum()  # = 1.0
        """
        return F.softmax(raw_params, dim=-1)
    
    @staticmethod
    def sigmoid_range(raw_param: torch.Tensor, low: float, high: float) -> torch.Tensor:
        """
        使用 sigmoid 将参数约束到 [low, high] 范围
        
        适用于：衰减系数 cr, ci, cg, k0, k1, k2 等
        
        Args:
            raw_param: 原始参数
            low: 下界
            high: 上界
            
        Returns:
            约束到 [low, high] 的参数
        """
        return low + (high - low) * torch.sigmoid(raw_param)
    
    @staticmethod
    def ordered_params(raw_params: torch.Tensor, ascending: bool = True) -> torch.Tensor:
        """
        将参数转换为有序序列（通过累积 softmax）
        
        适用于：HBV 的 k0 > k1 > k2 约束
        
        Args:
            raw_params: 原始参数 [batch, n] 或 [n]
            ascending: True 为升序，False 为降序
            
        Returns:
            有序参数序列
        """
        # 先用 softplus 确保正值，然后累积
        positive = F.softplus(raw_params)
        if ascending:
            return torch.cumsum(positive, dim=-1)
        else:
            # 降序：反转累积
            return torch.flip(torch.cumsum(torch.flip(positive, [-1]), dim=-1), [-1])
    
    @staticmethod
    def sum_constraint(raw_params: torch.Tensor, total: float = 1.0) -> torch.Tensor:
        """
        约束参数和为指定值（通过 softmax 缩放）
        
        适用于：需要总量上限的通用比例参数
        
        Args:
            raw_params: 原始参数
            total: 目标和值
            
        Returns:
            和为 total 的参数
        """
        weights = F.softmax(raw_params, dim=-1)
        return weights * total


class XAJConstraints(nn.Module):
    """
    XAJ 模型参数约束层
    
    约束关系：
    1. ki、kg ∈ [0.10, 0.50]，自由水剩余量保留在 S 中
    2. cr、ci、cg 为线性水库退水保留系数，限制在标准范围内
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, raw_params: dict) -> dict:
        """
        Args:
            raw_params: dict，包含原始网络输出
                - 'ki', 'kg': 壤中流/地下水出流系数原始值
                - 'cr', 'ci', 'cg': 保留系数原始值
                
        Returns:
            约束后的参数 dict
        """
        constrained = {}
        
        # 1. 自由水出流系数，不再做三水源和为 1 的归一化
        if 'ki' in raw_params:
            constrained['ki'] = 0.10 + 0.40 * torch.sigmoid(raw_params['ki'])
        if 'kg' in raw_params:
            constrained['kg'] = 0.10 + 0.40 * torch.sigmoid(raw_params['kg'])
        
        # 2. 三水源响应水库保留系数
        if 'cr' in raw_params:
            constrained['cr'] = 0.010 + 0.989 * torch.sigmoid(raw_params['cr'])
        if 'ci' in raw_params:
            constrained['ci'] = 0.010 + 0.989 * torch.sigmoid(raw_params['ci'])
        if 'cg' in raw_params:
            constrained['cg'] = 0.050 + 0.949 * torch.sigmoid(raw_params['cg'])
        
        return constrained

class HBVConstraints(nn.Module):
    """
    HBV 模型参数约束层
    
    约束关系：
    1. k0 > k1 > k2 (衰减系数递减)
    2. LP ∈ (0.3, 0.9)
    3. beta ∈ (0.5, 2.0)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, raw_params: dict) -> dict:
        constrained = {}
        
        # 1. k0 > k1 > k2 约束
        if 'k_raw' in raw_params:
            # 使用 softplus 确保正值，然后累积形成降序
            k_raw = raw_params['k_raw']  # [batch, 3]
            k_positive = F.softplus(k_raw) * 0.1  # 缩放到合理范围
            
            # k2 最小，k1 = k2 + delta1, k0 = k1 + delta2
            constrained['k2'] = k_positive[..., 0] + 0.001  # 最小值
            constrained['k1'] = constrained['k2'] + k_positive[..., 1] + 0.01
            constrained['k0'] = constrained['k1'] + k_positive[..., 2] + 0.1
        
        # 2. LP 约束
        if 'lp' in raw_params:
            constrained['lp'] = 0.3 + 0.6 * torch.sigmoid(raw_params['lp'])
        
        # 3. beta 约束
        if 'beta' in raw_params:
            constrained['beta'] = 0.5 + 1.5 * torch.sigmoid(raw_params['beta'])
        
        return constrained


# =============================================================================
# 示例：带约束的参数网络
# =============================================================================

class ConstrainedParamNet(nn.Module):
    """
    带参数约束的神经网络示例
    
    用于从输入特征预测水文模型参数，同时满足物理约束
    """
    
    def __init__(self, input_dim: int, model_type: str = 'xaj'):
        super().__init__()
        self.model_type = model_type
        
        # 共享特征提取层
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # 根据模型类型创建参数输出头和约束层
        if model_type == 'xaj':
            self.param_heads = nn.ModuleDict({
                'ki': nn.Linear(32, 1),
                'kg': nn.Linear(32, 1),
                'cr': nn.Linear(32, 1),
                'ci': nn.Linear(32, 1),
                'cg': nn.Linear(32, 1),
            })
            self.constraint = XAJConstraints()
            
        elif model_type == 'hbv':
            self.param_heads = nn.ModuleDict({
                'k_raw': nn.Linear(32, 3),  # k0, k1, k2
                'lp': nn.Linear(32, 1),
                'beta': nn.Linear(32, 1),
            })
            self.constraint = HBVConstraints()
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: 输入特征 [batch, input_dim]
            
        Returns:
            约束后的参数字典
        """
        # 特征提取
        features = self.feature_net(x)
        
        # 预测原始参数
        raw_params = {}
        for name, head in self.param_heads.items():
            raw_params[name] = head(features)
        
        # 应用约束
        constrained_params = self.constraint(raw_params)
        
        return constrained_params


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("参数约束处理示例")
    print("=" * 60)
    
    # 1. 测试 XAJ 自由水出流系数约束
    print("\n1. XAJ 自由水出流系数约束")
    raw_ki = torch.randn(1)
    raw_kg = torch.randn(1)
    ki = 0.10 + 0.40 * torch.sigmoid(raw_ki)
    kg = 0.10 + 0.40 * torch.sigmoid(raw_kg)
    print(f"   ki={ki.item():.4f}, kg={kg.item():.4f}")
    
    # 2. 测试范围约束
    print("\n2. 范围约束 (sigmoid)")
    raw_cr = torch.randn(1)
    cr = torch.sigmoid(raw_cr)
    print(f"   原始值: {raw_cr.item():.4f}")
    print(f"   cr (0-1): {cr.item():.4f}")
    
    # 3. 测试通用总量约束
    print("\n3. 通用总量约束")
    raw_split = torch.randn(2)
    split = F.softmax(raw_split, dim=-1) * 0.9
    print(f"   原始值: {raw_split.numpy()}")
    print(f"   约束后: {split.numpy()}, 和={split.sum().item():.4f}")
    
    # 4. 测试 HBV k0 > k1 > k2 约束
    print("\n4. HBV 衰减系数有序约束")
    raw_k = torch.randn(3)
    k_positive = F.softplus(raw_k) * 0.1
    k2 = k_positive[0] + 0.001
    k1 = k2 + k_positive[1] + 0.01
    k0 = k1 + k_positive[2] + 0.1
    print(f"   k0={k0.item():.4f} > k1={k1.item():.4f} > k2={k2.item():.4f}")
    
    # 5. 测试完整的参数网络
    print("\n5. 带约束的参数网络")
    net = ConstrainedParamNet(input_dim=10, model_type='xaj')
    x = torch.randn(4, 10)  # batch=4
    params = net(x)
    print(f"   输入: [4, 10]")
    print(f"   输出参数:")
    for k, v in params.items():
        print(f"     {k}: shape={list(v.shape)}, range=[{v.min().item():.4f}, {v.max().item():.4f}]")
    
    # 验证约束
    print(f"   ki 范围: [{params['ki'].min().item():.4f}, {params['ki'].max().item():.4f}]")
    print(f"   kg 范围: [{params['kg'].min().item():.4f}, {params['kg'].max().item():.4f}]")
    
    print("\n" + "=" * 60)


def apply_constraints(param_tensors: dict, param_info: dict):
    """
    应用参数约束
    
    Args:
        param_tensors: 参数张量字典
        param_info: 参数信息字典
    """
    # 对于训练过程，主要使用clamp确保参数在合理范围内
    # XAJ参数约束（与 subbasin_runoff.XAJModel 当前范围保持一致）
    xaj_bounds = {
        'kc': (0.5, 2.0),
        'wum': (5.0, 30.0),
        'wlm': (40.0, 90.0),
        'wdm': (35.0, 80.0),
        'c': (0.09, 0.20),
        'im': (0.005, 0.50),
        'b': (0.1, 0.4),
        'sm': (5.0, 50.0),
        'ex': (1.0, 2.0),
        'kg': (0.1, 0.5),
        'ki': (0.1, 0.5),
        'ci': (0.010, 0.999),
        'cg': (0.050, 0.999),
        'cr': (0.010, 0.999),
        'lag': (0.0, 5.0)
    }
    
    # Muskingum参数约束
    musk_bounds = {
        'K': (0.1, 10.0),
        'X': (0.0, 0.5)
    }
    
    # 应用约束
    for key, tensor in param_tensors.items():
        parts = key.split('_')
        param_name = parts[-1]
        
        # 检查是否是XAJ参数
        if param_name in xaj_bounds:
            bounds = xaj_bounds[param_name]
            tensor.data.clamp_(bounds[0], bounds[1])
        
        # 检查是否是Muskingum参数
        elif param_name in musk_bounds:
            bounds = musk_bounds[param_name]
            tensor.data.clamp_(bounds[0], bounds[1])
