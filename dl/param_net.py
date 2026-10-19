"""
参数区域化神经网络模块
从流域/河道特征预测水文模型参数
"""
import torch
import torch.nn as nn


class SingleParameterNetwork(nn.Module):
    """
    单参数网络：为每个参数创建独立的小网络
    便于单独保存、加载和冻结
    """
    
    def __init__(self, input_dim: int, param_name: str, param_range: tuple, hidden_dims=[16, 8]):
        """
        Args:
            input_dim: 输入特征维度
            param_name: 参数名称
            param_range: 参数范围 (min, max)
            hidden_dims: 隐藏层维度列表
        """
        super().__init__()
        self.param_name = param_name
        self.param_range = param_range
        self.min_val, self.max_val = param_range
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # 输出0-1范围
        self.network = nn.Sequential(*layers)
        
        # Xavier初始化
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        """
        前向传播：特征 -> 归一化参数 -> 缩放到实际范围
        
        Args:
            x: [batch, input_dim] 输入特征
            
        Returns:
            scaled: [batch, 1] 缩放后的参数
        """
        normalized = self.network(x)  # [batch, 1]
        scaled = normalized * (self.max_val - self.min_val) + self.min_val
        return scaled


class ParameterNetwork(nn.Module):
    """
    参数区域化网络：将流域特征映射到水文模型参数
    使用Sigmoid确保输出在[0,1]，然后线性缩放到参数实际范围
    """
    
    def __init__(self, input_dim, param_names, param_ranges, hidden_dims=[32, 16]):
        """
        Args:
            input_dim: 输入特征维度
            param_names: 参数名称列表
            param_ranges: 参数范围字典 {name: (min, max)}
            hidden_dims: 隐藏层维度列表
        """
        super().__init__()
        self.param_names = param_names
        self.param_ranges = param_ranges
        output_dim = len(param_names)
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())  # 输出0-1范围
        self.network = nn.Sequential(*layers)
        
        # Xavier初始化
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        """
        前向传播：特征 -> 归一化参数 -> 缩放到实际范围
        
        Args:
            x: [batch, input_dim] 输入特征
            
        Returns:
            scaled: [batch, n_params] 缩放后的参数
        """
        normalized = self.network(x)
        
        scaled = torch.zeros_like(normalized)
        for i, param_name in enumerate(self.param_names):
            min_val, max_val = self.param_ranges[param_name]
            scaled[:, i] = normalized[:, i] * (max_val - min_val) + min_val
        
        return scaled

