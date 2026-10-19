"""
动态参数预测器
基于参数分类系统自动构建和管理参数预测网络
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd

from dl.parameter_registry import get_all_parameters
from dl.network_config import NetworkConfig, get_network_save_path


class ParameterNetwork(nn.Module):
    """通用参数预测网络（改进版：BatchNorm + 更好的初始化）"""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 hidden_dims: List[int] = [64, 32],
                 dropout: float = 0.1,
                 network_name: str = "param_net"):
        """
        Args:
            input_dim: 输入特征维度
            output_dim: 输出参数数量
            hidden_dims: 隐藏层维度列表
            dropout: Dropout率
            network_name: 网络名称（用于日志）
        """
        super().__init__()
        self.network_name = network_name
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))  # 添加BatchNorm提高稳定性
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        # 改进的权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """改进的权重初始化策略"""
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                # 使用较小的初始化范围，避免梯度爆炸
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: [n_items, input_dim] 输入特征
        
        Returns:
            params: [n_items, output_dim] 原始参数预测值（带梯度保护）
        """
        output = self.network(features)
        # 梯度裁剪保护
        output = torch.clamp(output, min=-10.0, max=10.0)
        return output


class SingleParamPredictor(nn.Module):
    """单参数预测器（从预训练权重加载）"""
    
    def __init__(self, param_data: dict):
        """
        Args:
            param_data: 从 .pt 文件加载的数据，包含 state_dict, bounds 等
        """
        super().__init__()
        self.param_name = param_data['param_name']
        self.bounds = param_data['bounds']
        self.input_dim = param_data['input_dim']
        
        # 从 state_dict 推断网络结构
        state_dict = param_data['state_dict']
        
        # 重建网络结构（与 ParameterNetwork 相同，但 output_dim=1）
        hidden_dims = [64, 32]  # 默认隐藏层
        dropout = 0.1
        
        layers = []
        prev_dim = self.input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))  # 单参数输出
        self.network = nn.Sequential(*layers)
        
        # 加载权重
        self.load_state_dict(state_dict)
        
        # 冻结参数
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        预测单个参数
        
        Args:
            features: [n_items, input_dim]
        Returns:
            [n_items] 约束后的参数值
        """
        raw = self.network(features)
        raw = torch.clamp(raw, min=-10.0, max=10.0)
        
        # 应用约束
        lower, upper = float(self.bounds[0]), float(self.bounds[1])
        normalized = torch.sigmoid(raw.squeeze(-1))
        normalized = 0.01 + 0.98 * normalized
        return lower + (upper - lower) * normalized


class DynamicParameterPredictor:
    """
    动态参数预测器
    根据模型和参数分类系统自动构建所需的网络
    
    核心逻辑：
    - 参数由其所属模型(model字段)的param_mode决定是trainable还是pretrained
    - 同一个网络可能同时包含来自不同模型的参数
    - trainable参数构建到训练网络中
    - pretrained参数直接加载对应的.pt文件
    
    例如：当sphy=trainable, d8_travel_time=pretrained时：
    - sphy的参数 → 构建到grid_runoff网络训练
    - d8_travel_time的velocity → 加载pretrained/d8_travel_time/velocity.pt
    - d8_travel_time的K → 加载pretrained/d8_travel_time/K.pt
    """
    
    def __init__(self, model_configs: Dict[str, dict] = None,
                 hidden_dims: List[int] = None, dropout: float = 0.1,
                 data_dir: str = 'data/dl_input',
                 enable_cold_region: bool = False,
                 enable_precip_correct: bool = False,
                 verbose: bool = True):
        """
        Args:
            model_configs: 各模型的配置
                {
                    'xaj': {'param_mode': 'trainable', 'pretrained_path': ''},
                    'muskingum': {'param_mode': 'pretrained', 'pretrained_path': 'results/...'},
                    'd8_travel_time': {'param_mode': 'pretrained', 'pretrained_path': 'results/...'},
                }
            hidden_dims: 自定义隐藏层维度（None则使用默认）
            dropout: Dropout率
            data_dir: 数据目录（用于自动推断特征维度）
            enable_cold_region: 是否启用冰冻圈模式
            enable_precip_correct: 是否启用降水校正系数
            verbose: 是否打印构建信息
        """
        self.model_configs = model_configs or {}
        self.enable_cold_region = enable_cold_region
        self.enable_precip_correct = enable_precip_correct
        self.verbose = verbose
        self._passthrough_printed = set()  # 记录已打印过的passthrough信息，避免重复
        self._nan_output_printed = set()  # 记录已打印过的NaN输出警告，避免重复
        
        # 自动推断特征维度
        NetworkConfig.infer_feature_dims_from_data(data_dir)
        
        # 获取网络配置
        if hidden_dims is None:
            structure = NetworkConfig.get_network_structure()
            hidden_dims = structure['hidden_dims']
            dropout = structure['dropout']
        
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        
        # 存储网络和预测器
        self.networks = {}              # {network_type: ParameterNetwork} trainable网络
        self.pretrained_predictors = {} # {param_name: {...}} pretrained单参数预测器
        self.trainable_params_by_network = {}  # {network_type: [param_info, ...]}
        self.param_to_network_idx = {}  # {param_name: (network_type, idx)} 参数在网络中的位置
        
        self._build_networks()
    
    def _get_feature_type(self, network_type: str) -> str:
        """获取网络类型对应的特征类型"""
        mapping = {
            'subbasin_runoff': 'basin',
            'hillslope_routing': 'hillslope',
            'river_routing': 'river',
            'grid_runoff': 'basin',
        }
        return mapping.get(network_type, 'basin')
    
    def _get_input_dim(self, network_type: str) -> int:
        """获取网络的输入维度"""
        feature_type = self._get_feature_type(network_type)
        return NetworkConfig.FEATURE_DIMS.get(feature_type, 10)
    
    def _build_networks(self):
        """
        根据模型配置动态构建网络
        
        核心逻辑：
        1. 获取所有配置模型的参数
        2. 按参数所属模型的param_mode分类为trainable或pretrained
        3. trainable参数按network_mapping分组，构建训练网络
        4. pretrained参数直接加载对应的.pt文件
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("动态网络构建（按模型param_mode区分）")
            print("=" * 80 + "\n")
        
        # 获取所有参数
        all_params = get_all_parameters(enable_cold_region=self.enable_cold_region, enable_precip_correct=self.enable_precip_correct)
        
        # 按模型的param_mode分类参数
        trainable_params = []  # 需要构建网络的参数
        pretrained_params = [] # 需要加载.pt的参数
        
        for param_info in all_params:
            model_name = param_info['model']
            
            # 检查该模型是否在配置中
            if model_name not in self.model_configs:
                continue
            
            model_config = self.model_configs[model_name]
            param_mode = model_config.get('param_mode', 'trainable')
            
            if param_mode == 'trainable':
                trainable_params.append(param_info)
            elif param_mode == 'pretrained':
                pretrained_params.append({
                    **param_info,
                    'pretrained_path': model_config.get('pretrained_path', '')
                })
            # fixed模式不处理，由外部提供参数
        
        # 按网络类型分组trainable参数
        trainable_by_network = {}  # {network_type: [param_info, ...]}
        for param_info in trainable_params:
            network_type = param_info['network_mapping']
            if network_type == 'none':
                continue
            if network_type not in trainable_by_network:
                trainable_by_network[network_type] = []
            trainable_by_network[network_type].append(param_info)
        
        # 构建trainable网络
        for network_type, params in trainable_by_network.items():
            if not params:
                continue
            
            input_dim = self._get_input_dim(network_type)
            output_dim = len(params)
            
            # 按模型分组显示
            models = set(p['model'] for p in params)
            if self.verbose:
                print(f"[TRAINABLE] {network_type} 网络:")
                for model in sorted(models):
                    model_params = [p['name'] for p in params if p['model'] == model]
                    print(f"  {model}: {model_params}")
            
            # 构建网络
            network = ParameterNetwork(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=self.hidden_dims,
                dropout=self.dropout,
                network_name=f"{network_type}_net"
            )
            self.networks[network_type] = network
            self.trainable_params_by_network[network_type] = params
            
            # 记录参数位置
            for idx, param_info in enumerate(params):
                self.param_to_network_idx[param_info['name']] = (network_type, idx)
            
            if self.verbose:
                param_count = sum(p.numel() for p in network.parameters())
                print(f"  [Network] {input_dim} -> {output_dim} params, {param_count} weights\n")
        
        # 加载pretrained参数
        if pretrained_params:
            # 按模型分组
            pretrained_by_model = {}
            for param_info in pretrained_params:
                model_name = param_info['model']
                if model_name not in pretrained_by_model:
                    pretrained_by_model[model_name] = []
                pretrained_by_model[model_name].append(param_info)
            
            for model_name, params in pretrained_by_model.items():
                pretrained_path = params[0].get('pretrained_path', '')
                if self.verbose:
                    print(f"[PRETRAINED] {model_name}:")
                    print(f"  参数: {[p['name'] for p in params]}")
                
                self._load_pretrained_params_for_model(model_name, params, pretrained_path)
        
        # 打印总结
        if self.verbose:
            print("\n" + "=" * 80)
            trainable_count = len(self.networks)
            pretrained_count = len(self.pretrained_predictors)
            if trainable_count > 0:
                total_weights = sum(
                    sum(p.numel() for p in net.parameters()) 
                    for net in self.networks.values()
                )
                print(f"Trainable 网络: {trainable_count} 个, 共 {total_weights} 权重")
            if pretrained_count > 0:
                print(f"Pretrained 参数: {pretrained_count} 个 (已冻结)")
            print("=" * 80 + "\n")
    
    def _load_pretrained_params_for_model(self, model_name: str, params: list, pretrained_path: str):
        """加载指定模型的预训练参数"""
        if not pretrained_path:
            if self.verbose:
                print(f"  [Warning] 未指定 pretrained_path，跳过")
            return
        
        pretrained_dir = Path(pretrained_path) / 'pretrained' / model_name
        loaded_params = []
        
        for param_info in params:
            param_name = param_info['name']
            network_type = param_info['network_mapping']
            param_file = pretrained_dir / f'{param_name}.pt'
            
            if not param_file.exists():
                if self.verbose:
                    print(f"  [Warning] 未找到 {param_file}")
                continue
            
            # 加载单参数网络
            param_data = torch.load(param_file, weights_only=False)
            predictor = SingleParamPredictor(param_data)
            
            self.pretrained_predictors[param_name] = {
                'predictor': predictor,
                'network_type': network_type,
                'feature_type': self._get_feature_type(network_type),
                'model': model_name,
                'passthrough': param_info.get('passthrough', param_info.get('init', 0.5)),
            }
            loaded_params.append(param_name)
        
        if self.verbose and loaded_params:
            print(f"  [Loaded] {loaded_params}\n")

    def predict_params(self, features: Dict[str, torch.Tensor], training: bool = False) -> Dict[str, torch.Tensor]:
        """
        预测所有参数
        
        Args:
            features: {feature_type: tensor} 特征字典
                     如 {'basin': [n,10], 'hillslope': [n,5], 'river': [n,5]}
            training: 是否在训练模式（如果为True，保留梯度）
        
        Returns:
            dict: {param_name: predicted_values}
        """
        all_params = {}
        
        # 网络类型到特征类型的映射
        network_to_feature = {
            'subbasin_runoff': 'basin',
            'hillslope_routing': 'hillslope',
            'river_routing': 'river',
            'grid_runoff': 'basin',
        }
        
        # 确定 n_units 和 device（用于特征缺失时生成 passthrough 参数）
        n_units = None
        device = torch.device('cpu')
        for feat in features.values():
            n_units = feat.shape[0]
            device = feat.device
            break
        
        # 1. 从 trainable 网络预测
        for network_type, network in self.networks.items():
            feature_type = network_to_feature.get(network_type, 'basin')
            
            if feature_type not in features or features[feature_type].shape[-1] == 0:
                # 特征缺失或为空：所有单元使用无延时无调蓄的 passthrough 参数值
                if n_units is not None and network_type in self.trainable_params_by_network:
                    params_info = self.trainable_params_by_network[network_type]
                    for param_info in params_info:
                        passthrough_val = param_info.get('passthrough', param_info.get('init', 0.5))
                        all_params[param_info['name']] = torch.full(
                            (n_units,), float(passthrough_val), device=device
                        )
                    _pt_key = f'empty_{network_type}'
                    if _pt_key not in self._passthrough_printed:
                        param_strs = ', '.join(
                            f'{p["name"]}={p.get("passthrough", p.get("init", 0.5))}' for p in params_info
                        )
                        print(f"[Passthrough] {network_type}: 特征为空，使用无延时无调蓄参数 {{{param_strs}}}")
                        self._passthrough_printed.add(_pt_key)
                else:
                    print(f"[Warning] Missing features for {feature_type}, skipping {network_type}")
                continue
            
            # 预测（训练模式下保留梯度）
            feat = features[feature_type]
            if training:
                raw_params = network(feat)
            else:
                with torch.no_grad():
                    raw_params = network(feat)

            if not torch.isfinite(raw_params).all():
                warn_key = f"nan_{network_type}"
                if self.verbose and warn_key not in self._nan_output_printed:
                    print(f"[Warning] {network_type} raw params contain non-finite values; applying nan_to_num")
                    self._nan_output_printed.add(warn_key)
                raw_params = torch.nan_to_num(raw_params, nan=0.0, posinf=10.0, neginf=-10.0)
            
            # 应用约束
            params_info = self.trainable_params_by_network[network_type]
            constrained = self._apply_constraints(raw_params, params_info)
            
            # 保存到字典
            for i, param_info in enumerate(params_info):
                all_params[param_info['name']] = constrained[:, i]
            
            # 逐行passthrough：对特征含NaN的单元（上游无河网/小面积），替换为passthrough值
            nan_mask_key = f'{feature_type}_nan_mask'
            if nan_mask_key in features:
                nan_mask = features[nan_mask_key].to(device)  # [n_units] bool
                nan_count = nan_mask.sum().item()
                if nan_count > 0:
                    for param_info in params_info:
                        pname = param_info['name']
                        passthrough_val = float(param_info.get('passthrough', param_info.get('init', 0.5)))
                        # 用 where 替换：NaN行用passthrough，其余保留网络预测值
                        all_params[pname] = torch.where(
                            nan_mask, 
                            torch.tensor(passthrough_val, device=device),
                            all_params[pname]
                        )
                    _pt_key = f'nan_{network_type}'
                    if self.verbose and _pt_key not in self._passthrough_printed:
                        param_strs = ', '.join(
                            f'{p["name"]}={p.get("passthrough", p.get("init", 0.5))}' for p in params_info
                        )
                        print(f"[Passthrough] {network_type}: {nan_count}个单元（上游/小面积）"
                              f"使用passthrough参数 {{{param_strs}}}")
                        self._passthrough_printed.add(_pt_key)
        
        # 2. 从 pretrained 单参数网络预测
        for param_name, info in self.pretrained_predictors.items():
            predictor_net = info['predictor']
            feature_type = info['feature_type']
            
            if feature_type not in features or features[feature_type].shape[-1] == 0:
                # 特征缺失或为空：使用 passthrough 参数值
                if n_units is not None:
                    passthrough_val = info.get('passthrough', 0.5)
                    all_params[param_name] = torch.full(
                        (n_units,), float(passthrough_val), device=device
                    )
                    _pt_key = f'pretrained_{param_name}'
                    if _pt_key not in self._passthrough_printed:
                        print(f"[Passthrough] {param_name}: 特征为空，使用无延时无调蓄参数 = {passthrough_val}")
                        self._passthrough_printed.add(_pt_key)
                else:
                    print(f"[Warning] Missing features for {feature_type}, skipping {param_name}")
                continue
            
            feat = features[feature_type]
            predictor_net.to(feat.device)
            
            with torch.no_grad():
                all_params[param_name] = predictor_net(feat)
            
            # 逐行passthrough：对特征含NaN的单元，替换为passthrough值
            nan_mask_key = f'{feature_type}_nan_mask'
            if nan_mask_key in features:
                nan_mask = features[nan_mask_key].to(device)
                if nan_mask.any():
                    passthrough_val = float(info.get('passthrough', 0.5))
                    all_params[param_name] = torch.where(
                        nan_mask,
                        torch.tensor(passthrough_val, device=device),
                        all_params[param_name]
                    )
        
        return all_params
    
    def _apply_constraints(self, raw_params: torch.Tensor, 
                          params_info: List[Dict]) -> torch.Tensor:
        """应用参数约束（改进版：更稳定的sigmoid映射）"""
        constrained = torch.zeros_like(raw_params)
        
        for i, param_info in enumerate(params_info):
            bounds = param_info['bounds']
            lower, upper = float(bounds[0]), float(bounds[1])
            
            # 裁剪原始输出避免sigmoid溢出
            clipped = torch.clamp(raw_params[:, i], min=-10.0, max=10.0)
            
            # sigmoid映射到[0, 1]，添加小偏移避免边界值
            normalized = torch.sigmoid(clipped)
            # 映射到[0.01, 0.99]避免极端值
            normalized = 0.01 + 0.98 * normalized
            
            # 缩放到[lower, upper]
            constrained[:, i] = lower + (upper - lower) * normalized
        
        return constrained
    
    def save_networks(self, output_dir: str):
        """
        保存所有网络权重
        
        Args:
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for network_type, network in self.networks.items():
            save_path = get_network_save_path(output_dir, network_type)
            torch.save(network.state_dict(), save_path)
            
            if self.verbose:
                param_count = sum(p.numel() for p in network.parameters())
                print(f"[Saved] {network_type}_net.pt: {param_count} weights")
        
        # 同时保存到 pretrained/模型名/ 目录，供 pretrained 模式使用
        self._save_pretrained_networks(output_dir)
    
    def _save_pretrained_networks(self, output_dir: str):
        """
        保存网络到 pretrained/模型名/ 目录结构
        
        每个参数保存到其所属模型的目录下
        
        目录结构示例：
        pretrained/
        ├── xaj/                              # XAJ模型
        │   ├── wum.pt, wlm.pt, ...           # 产流参数
        │   └── cr.pt, ci.pt, cg.pt           # 坡面汇流参数
        ├── sphy/                             # SPHY模型
        │   └── ...
        ├── d8_travel_time/                   # D8坡面汇流模型
        │   ├── velocity.pt
        │   └── K.pt
        └── muskingum/                        # 马斯京根模型
            └── K.pt, X.pt
        """
        output_path = Path(output_dir) / 'pretrained'
        
        for network_type, network in self.networks.items():
            if network_type not in self.trainable_params_by_network:
                continue
            
            params_info = self.trainable_params_by_network[network_type]
            
            # 按模型分组参数
            params_by_model = {}
            for i, param_info in enumerate(params_info):
                model_name = param_info['model']
                if model_name not in params_by_model:
                    params_by_model[model_name] = []
                params_by_model[model_name].append((i, param_info))
            
            # 为每个模型保存参数
            for model_name, model_params in params_by_model.items():
                model_dir = output_path / model_name
                model_dir.mkdir(parents=True, exist_ok=True)
                
                self._save_param_networks(network, model_params, model_dir, model_name)
    
    def _save_param_networks(self, network: nn.Module, indexed_params: list, model_dir: Path, model_name: str = None):
        """
        拆分网络，为每个参数保存独立的网络权重
        
        原理：将完整网络的最后一层拆分，每个输出神经元对应一个参数。
        保存内容包括：共享的隐藏层 + 该参数对应的输出权重。
        
        Args:
            network: 完整的参数网络
            indexed_params: [(index, param_info), ...] 带索引的参数信息列表
            model_dir: 保存目录
            model_name: 模型名称（用于日志显示）
        """
        state_dict = network.state_dict()
        
        # 获取网络结构信息
        # 最后一个 Linear 层的权重形状为 [output_dim, hidden_dim]
        # 需要找到最后一个 Linear 层
        last_linear_weight_key = None
        last_linear_bias_key = None
        
        for key in state_dict.keys():
            if 'weight' in key and 'network' in key:
                last_linear_weight_key = key
            if 'bias' in key and 'network' in key:
                last_linear_bias_key = key
        
        if last_linear_weight_key is None:
            return
        
        # 提取最后一层权重
        last_weight = state_dict[last_linear_weight_key]  # [output_dim, hidden_dim]
        last_bias = state_dict.get(last_linear_bias_key)  # [output_dim] or None
        
        # 构建共享层的 state_dict（除了最后一层）
        shared_state = {}
        for key, value in state_dict.items():
            if key != last_linear_weight_key and key != last_linear_bias_key:
                shared_state[key] = value
        
        # 为每个参数保存独立的网络权重
        saved_files = []
        for idx, param_info in indexed_params:
            param_name = param_info['name']
            
            # 构建该参数的完整 state_dict
            param_state = shared_state.copy()
            
            # 添加该参数对应的输出层权重（保持原始形状用于兼容）
            # 保存为单输出网络格式：[1, hidden_dim]
            param_state[last_linear_weight_key] = last_weight[idx:idx+1, :]  # [1, hidden_dim]
            if last_bias is not None:
                param_state[last_linear_bias_key] = last_bias[idx:idx+1]  # [1]
            
            # 文件名：直接用参数名
            filename = f'{param_name}.pt'
            
            # 同时保存参数信息（bounds）
            param_data = {
                'state_dict': param_state,
                'param_name': param_name,
                'param_index': idx,
                'bounds': param_info['bounds'],
                'input_dim': network.input_dim,
                'output_dim': 1,  # 单参数网络
                'model_name': model_name,
            }
            
            save_path = model_dir / filename
            torch.save(param_data, save_path)
            saved_files.append(param_name)
        
        if self.verbose:
            print(f"[Pretrained] Saved to {model_name}/: {saved_files}")
    
    def load_networks(self, input_dir: str):
        """
        加载所有网络权重（从根目录）
        
        Args:
            input_dir: 输入目录
        """
        for network_type, network in self.networks.items():
            load_path = get_network_save_path(input_dir, network_type)
            
            if not Path(load_path).exists():
                print(f"[Warning] {load_path} not found, skipping")
                continue
            
            network.load_state_dict(torch.load(load_path, weights_only=True))
            network.eval()
            
            if self.verbose:
                print(f"[Loaded] {network_type}_net.pt")
    
    def load_pretrained_network(self, network_type: str, pretrained_path: str, model_name: str = None):
        """
        从 pretrained 目录加载单个网络
        
        Args:
            network_type: 网络类型 ('subbasin_runoff', 'river_routing', 'hillslope_routing')
            pretrained_path: pretrained 根目录
            model_name: 模型名称（如 'xaj', 'muskingum'），如果为 None 则自动推断
        """
        if network_type not in self.networks:
            print(f"[Warning] Network '{network_type}' not found in predictor")
            return False
        
        # 自动推断模型名
        if model_name is None:
            if network_type == 'subbasin_runoff':
                model_name = self.model_name
            elif network_type == 'river_routing':
                model_name = self.river_model_name
            else:
                model_name = network_type
        
        # 构建加载路径
        load_path = Path(pretrained_path) / 'pretrained' / model_name / f'{network_type}_net.pt'
        
        if not load_path.exists():
            # 尝试直接从根目录加载（兼容旧格式）
            load_path = Path(pretrained_path) / f'{network_type}_net.pt'
        
        if load_path.exists():
            try:
                state_dict = torch.load(load_path, map_location='cpu', weights_only=True)
                self.networks[network_type].load_state_dict(state_dict)
                self.networks[network_type].eval()
                if self.verbose:
                    print(f"[Pretrained] Loaded {network_type} from: {load_path}")
                return True
            except Exception as e:
                print(f"[Warning] Failed to load {load_path}: {e}")
                return False
        else:
            print(f"[Warning] Pretrained network not found: {load_path}")
            return False
    
    def get_network_summary(self) -> Dict:
        """获取网络构建摘要"""
        return {
            'model': self.model_name,
            'river_model': self.river_model_name,
            'networks_built': list(self.networks.keys()),
            'params_by_network': {
                net_type: [p['name'] for p in params]
                for net_type, params in self.params_by_network.items()
            }
        }


def load_features(data_dir: str = 'data/dl_input') -> Dict[str, torch.Tensor]:
    """
    加载所有特征数据
    
    Args:
        data_dir: 数据目录
    
    Returns:
        dict: {feature_type: tensor, ...}
              当特征文件不存在或特征列为空时，该feature_type不会包含在返回字典中。
              对于存在NaN行的特征（如上游无河网、小面积单元），会额外生成:
                '{feature_type}_nan_mask': bool tensor [n_units]
              标记哪些行原本为NaN（已填0），后续预测时这些行使用passthrough参数。
    """
    features = {}
    feature_files = NetworkConfig.get_feature_files()
    
    for feature_type, filename in feature_files.items():
        file_path = Path(data_dir) / Path(filename).name
        
        if not file_path.exists():
            print(f"[Warning] {file_path} not found, skipping {feature_type} features")
            continue
        
        df = pd.read_csv(file_path)
        
        # 跳过第一列（假设是ID）
        feature_array = df.iloc[:, 1:].values.astype(np.float64)
        
        # 检查特征是否为空（无特征列或行数为0）
        if feature_array.shape[1] == 0 or feature_array.shape[0] == 0:
            print(f"[Warning] {file_path} has no feature data (shape={feature_array.shape}), "
                  f"skipping {feature_type} features")
            continue
        
        # 检测逐行NaN：标记整行或部分列为NaN的单元
        # 这些单元属于上游无河网或面积很小的子流域，后续使用passthrough参数
        row_has_nan = np.isnan(feature_array).any(axis=1)  # [n_units]
        nan_count = row_has_nan.sum()
        if nan_count > 0:
            print(f"[Info] {file_path}: {nan_count}/{len(feature_array)} 个单元特征含NaN"
                  f"（上游无河网/小面积），将使用passthrough参数")
            # 填0使tensor有效（这些行后续会被passthrough覆盖，填什么不影响）
            feature_array = np.nan_to_num(feature_array, nan=0.0)
            features[f'{feature_type}_nan_mask'] = torch.tensor(row_has_nan, dtype=torch.bool)
        
        features[feature_type] = torch.tensor(feature_array, dtype=torch.float32)
        
        print(f"[Data] {feature_type}: {features[feature_type].shape}")
    
    return features


if __name__ == "__main__":
    print("=== 动态参数预测器测试 ===\n")
    
    # 测试1: 创建预测器
    print("测试1: 创建XAJ + Muskingum预测器")
    predictor = DynamicParameterPredictor('xaj', 'muskingum')
    
    print("\n摘要:")
    summary = predictor.get_network_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # 测试2: 加载特征并预测
    print("\n\n测试2: 加载特征并预测参数")
    features = load_features()
    
    if features:
        params = predictor.predict_params(features)
        print(f"\n预测的参数: {list(params.keys())}")
        print(f"总共 {len(params)} 个参数")
