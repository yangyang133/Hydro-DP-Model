"""
分布式水文模型核心流水线
支持多子流域产流 + 树状河网汇流
可扩展至网格产流和坡面汇流
支持三种参数模式：fixed/network/trainable
"""
import json
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Union
from collections import OrderedDict
from pathlib import Path


class HydroModel:
    """
    分布式水文模型
    
    特点：
    - 多子流域并行产流计算
    - 树状河网拓扑汇流
    - 模块化设计，易于扩展
    - 支持网格产流和坡面汇流（预留接口）
    - 支持三种参数模式：
      * fixed: 固定参数（传统水文模型）
      * network: 神经网络预测参数
      * trainable: 可训练优化参数
    """

    def __init__(self, model_attr_path: str, config: Optional[Dict] = None):
        """
        初始化模型
        
        Args:
            model_attr_path: 模型属性配置文件路径
            config: 完整配置字典（包含参数模式设置）
        """
        with open(model_attr_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 保存model_attr用于观测数据加载
        self.model_attr = self.config
        
        # 保存完整配置（用于参数模式）
        self.full_config = config or {}
        
        # 基本信息（units = 子流域+河道 一一对应）
        units_config = self.config['units']
        self.n_subbasins = units_config['count']
        self.subbasin_ids = [f"sub_{uid}" for uid in units_config['ids']]
        self.unit_ids = units_config['ids']
        
        # 河网拓扑信息
        self.topology = units_config['topology']
        self.outlet_reach = f"r{self.topology['outlet']}"
        
        # 构建reaches字典（兼容旧代码）
        self.reaches = {}
        for uid in self.unit_ids:
            upstream = self.topology.get(uid, [])
            self.reaches[f"r{uid}"] = {
                'upstream_reaches': [f"r{u}" for u in upstream]
            }
        
        # 观测站配置
        self.gauges = units_config.get('gauges', {})
        
        # 子流域面积 (km2)，用于 mm/day → m3/s 转换
        area_km2_list = units_config.get('attributes', {}).get('area_km2', None)
        if area_km2_list is not None and len(area_km2_list) == self.n_subbasins:
            self.subbasin_areas = {
                f"sub_{uid}": area
                for uid, area in zip(self.unit_ids, area_km2_list)
            }
            print(f"[Model] Loaded area_km2 for {len(self.subbasin_areas)} subbasins, "
                  f"total={sum(area_km2_list):.1f} km2")
        else:
            self.subbasin_areas = None
            print("[Model] WARNING: area_km2 not found, runoff will stay in mm/day")
        
        # 冰川比例（从 model_attr.json 的 glacier_ratio 读取）
        glacier_ratio_list = units_config.get('attributes', {}).get('glacier_ratio', None)
        if glacier_ratio_list is not None and len(glacier_ratio_list) == self.n_subbasins:
            self.glacier_ratios = {
                f"sub_{uid}": ratio
                for uid, ratio in zip(self.unit_ids, glacier_ratio_list)
            }
            n_nonzero = sum(1 for r in glacier_ratio_list if r > 0)
            print(f"[Model] Loaded glacier_ratio for {len(self.glacier_ratios)} subbasins, "
                  f"{n_nonzero} have glaciers (max={max(glacier_ratio_list):.4f})")
        else:
            self.glacier_ratios = None
            print("[Model] WARNING: glacier_ratio not found in model_attr, glacier melt will be 0")
        
        # 参数模式配置
        self._init_param_modes()
        
        # 计算河道拓扑顺序
        self.reach_order = self._compute_topology()
        self.reach_levels = self._compute_topology_levels()

        # 强迫数据张量缓存：避免训练时每个epoch重复 numpy -> tensor 转换
        self._forcing_tensor_cache: Dict[tuple, tuple] = {}
        
        # 初始化子流域产流模型
        self._init_subbasin_models()
        
        # 初始化河道汇流模型
        self._init_river_models()
        
        # 加载固定参数（如果param_mode=fixed）
        self._load_fixed_params()
        
        # 初始化参数网络（如果需要）
        self._init_parameter_networks()
        
        print(f"[Model] Initialized: {self.n_subbasins} units")

    def _compute_topology(self) -> List[str]:
        """计算河道拓扑顺序（从上游到下游）"""
        in_degree = {rid: 0 for rid in self.reaches.keys()}
        
        for rid, rinfo in self.reaches.items():
            in_degree[rid] = len(rinfo.get('upstream_reaches', []))
        
        order = []
        queue = [rid for rid, deg in in_degree.items() if deg == 0]
        temp_degree = in_degree.copy()
        
        while queue:
            current = queue.pop(0)
            order.append(current)
            
            for rid, rinfo in self.reaches.items():
                if current in rinfo.get('upstream_reaches', []):
                    temp_degree[rid] -= 1
                    if temp_degree[rid] == 0:
                        queue.append(rid)
        
        if len(order) != len(self.reaches):
            raise ValueError("River network has cycles!")
        
        return order

    def _compute_topology_levels(self) -> List[List[str]]:
        """按拓扑层级分组河道（同层河道可并行计算）"""
        in_degree = {rid: len(rinfo.get('upstream_reaches', [])) for rid, rinfo in self.reaches.items()}
        levels: List[List[str]] = []
        current_level = [rid for rid, deg in in_degree.items() if deg == 0]

        while current_level:
            levels.append(current_level)
            next_level: List[str] = []

            for current in current_level:
                for rid, rinfo in self.reaches.items():
                    if current in rinfo.get('upstream_reaches', []):
                        in_degree[rid] -= 1
                        if in_degree[rid] == 0:
                            next_level.append(rid)

            current_level = next_level

        flat_count = sum(len(level) for level in levels)
        if flat_count != len(self.reaches):
            raise ValueError("River network has cycles (topology levels invalid)!")

        return levels

    def _to_tensor_if_needed(self, value: Any, device: torch.device) -> Optional[torch.Tensor]:
        """将标量/张量统一转为指定设备上的 float32 张量"""
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=torch.float32)
        return torch.tensor(float(value), dtype=torch.float32, device=device)

    def _get_forcing_tensors(self, forcing: Dict[str, Any], device: torch.device):
        """获取强迫数据张量（支持缓存）"""
        P = forcing['P']
        E = forcing['E']
        T_data = forcing.get('T', None)

        # 若传入已是tensor，直接使用（必要时搬运到目标设备）
        if isinstance(P, torch.Tensor) and isinstance(E, torch.Tensor):
            P_tensor = P.to(device=device, dtype=torch.float32)
            E_tensor = E.to(device=device, dtype=torch.float32)
            T_tensor = T_data.to(device=device, dtype=torch.float32) if isinstance(T_data, torch.Tensor) else None
            if T_data is not None and not isinstance(T_data, torch.Tensor):
                T_tensor = torch.tensor(T_data, dtype=torch.float32, device=device)
            return P_tensor, E_tensor, T_tensor

        # 使用对象id作为缓存键：训练期forcing数组对象稳定复用
        cache_key = (id(P), id(E), id(T_data), str(device))
        if cache_key in self._forcing_tensor_cache:
            return self._forcing_tensor_cache[cache_key]

        P_tensor = torch.tensor(P, dtype=torch.float32, device=device)
        E_tensor = torch.tensor(E, dtype=torch.float32, device=device)
        T_tensor = torch.tensor(T_data, dtype=torch.float32, device=device) if T_data is not None else None

        self._forcing_tensor_cache[cache_key] = (P_tensor, E_tensor, T_tensor)
        return P_tensor, E_tensor, T_tensor

    def _route_reaches_batched(
        self,
        T: int,
        subbasin_runoff: Dict[str, torch.Tensor],
        params: Dict,
        state: Dict,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """按拓扑层进行河道汇流，同层河道批量并行计算"""
        reach_flow: Dict[str, torch.Tensor] = {}

        for level_reaches in self.reach_levels:
            q_in_by_reach: Dict[str, torch.Tensor] = {}
            q_lateral_by_reach: Dict[str, torch.Tensor] = {}
            to_route: List[str] = []

            for reach_id in level_reaches:
                reach_info = self.reaches[reach_id]
                upstream_inflow = [reach_flow[up] for up in reach_info.get('upstream_reaches', []) if up in reach_flow]
                if upstream_inflow:
                    Q_in = torch.stack(upstream_inflow).sum(dim=0)
                else:
                    Q_in = torch.zeros(T, dtype=torch.float32, device=device)
                q_in_by_reach[reach_id] = Q_in

                sub_id = f"sub_{reach_id[1:]}"
                if sub_id in subbasin_runoff:
                    Q_lateral = subbasin_runoff[sub_id]
                    if self.subbasin_areas is not None and sub_id in self.subbasin_areas:
                        Q_lateral = Q_lateral * self.subbasin_areas[sub_id] / 86.4
                else:
                    Q_lateral = torch.zeros(T, dtype=torch.float32, device=device)
                q_lateral_by_reach[reach_id] = Q_lateral

                if len(reach_info.get('upstream_reaches', [])) > 0:
                    to_route.append(reach_id)
                else:
                    reach_flow[reach_id] = Q_lateral
                    state['rivers'][reach_id] = state['rivers'].get(reach_id, {})

            if not to_route:
                continue

            # 仅在同层模型类型一致时启用批量路由，否则回退单河段
            first_model = self.river_models[to_route[0]]
            homogeneous = all(isinstance(self.river_models[rid], type(first_model)) for rid in to_route)

            if homogeneous:
                Q_in_batch = torch.stack([q_in_by_reach[rid] for rid in to_route], dim=0)  # [n_reaches, T]

                K_vals = []
                X_vals = []
                q_out_prev_vals = []
                q_in_prev_vals = []

                for idx, reach_id in enumerate(to_route):
                    reach_params = params.get('rivers', {}).get(reach_id, {})
                    static_params = reach_params.get('static', {})
                    K_vals.append(self._to_tensor_if_needed(static_params.get('K', self.river_models[reach_id].K), device))
                    X_vals.append(self._to_tensor_if_needed(static_params.get('X', self.river_models[reach_id].X), device))

                    reach_state = state['rivers'].get(reach_id, {})
                    q_out_prev = reach_state.get('Q_out_prev', Q_in_batch[idx:idx + 1, 0:1])
                    q_in_prev = reach_state.get('Q_in_prev', Q_in_batch[idx:idx + 1, 0:1])

                    q_out_prev_t = self._to_tensor_if_needed(q_out_prev, device)
                    q_in_prev_t = self._to_tensor_if_needed(q_in_prev, device)
                    q_out_prev_vals.append(q_out_prev_t.reshape(-1)[0])
                    q_in_prev_vals.append(q_in_prev_t.reshape(-1)[0])

                static_params_batch = {
                    'K': torch.stack(K_vals),
                    'X': torch.stack(X_vals),
                }
                batch_state = {
                    'Q_out_prev': torch.stack(q_out_prev_vals).unsqueeze(1),
                    'Q_in_prev': torch.stack(q_in_prev_vals).unsqueeze(1),
                }

                routed_outputs, batch_state = first_model.forward(
                    inputs={'Q_in': Q_in_batch},
                    static_params=static_params_batch,
                    dynamic_params={},
                    state=batch_state,
                )
                Q_routed_batch = routed_outputs['Q']

                for idx, reach_id in enumerate(to_route):
                    reach_flow[reach_id] = Q_routed_batch[idx] + q_lateral_by_reach[reach_id]
                    state['rivers'][reach_id] = {
                        'Q_out_prev': batch_state['Q_out_prev'][idx:idx + 1, :],
                        'Q_in_prev': batch_state['Q_in_prev'][idx:idx + 1, :],
                    }
            else:
                for reach_id in to_route:
                    reach_params = params.get('rivers', {}).get(reach_id, {})
                    static_params = reach_params.get('static', {})
                    dynamic_params = reach_params.get('dynamic', {})

                    routed_outputs, state['rivers'][reach_id] = self.river_models[reach_id].forward(
                        inputs={'Q_in': q_in_by_reach[reach_id]},
                        static_params=static_params,
                        dynamic_params=dynamic_params,
                        state=state['rivers'][reach_id],
                    )
                    reach_flow[reach_id] = routed_outputs['Q'] + q_lateral_by_reach[reach_id]

        return reach_flow
    
    def _get_component_config(self, component_name: str) -> dict:
        """获取组件配置
        
        Args:
            component_name: 组件名称 (subbasin_runoff, grid_runoff, hillslope_routing, river_routing)
        
        Returns:
            dict: 组件配置 {model, param_mode}
        """
        components = self.full_config.get('model', {}).get('components', {})
        return components.get(component_name, {'model': 'xaj', 'param_mode': 'trainable'})
    
    def _init_param_modes(self):
        """初始化参数模式配置"""
        self.param_modes = {}
        self.network_configs = {}
        
        # 从components配置中读取参数模式
        components = self.full_config.get('model', {}).get('components', {})
        
        for comp_name, comp_config in components.items():
            param_mode = comp_config.get('param_mode', 'trainable')
            self.param_modes[comp_name] = param_mode
            self.network_configs[comp_name] = {}
        
        print(f"[Model] Parameter modes: {self.param_modes}")
    
    def _get_params_dir(self) -> str:
        """获取参数文件目录（从model_attr_path推断）"""
        # model_attr_path 格式：data/流域名称/model_attr/model_attr.json
        # params目录：data/流域名称/params/
        override = self.full_config.get('_params_dir_override')
        if override:
            return str(override)

        import os
        model_attr_path = self.full_config.get('_model_attr_path', '')
        if model_attr_path:
            base_dir = os.path.dirname(os.path.dirname(model_attr_path))
            return os.path.join(base_dir, 'params')
        return ''
    
    def _load_fixed_params(self):
        """加载固定参数（fixed模式）
        
        参数文件命名规则：{模型名称}_params.csv
        例如：xaj_params.csv, muskingum_params.csv
        
        优先从 data/流域名称/params/ 读取，如果不存在则使用默认值
        """
        import pandas as pd
        from pathlib import Path
        
        params_dir = Path(self._get_params_dir())
        
        # 获取模型名称
        subbasin_config = self._get_component_config('subbasin_runoff')
        river_config = self._get_component_config('river_routing')
        subbasin_model_name = subbasin_config.get('model', 'xaj')
        river_model_name = river_config.get('model', 'muskingum')
        
        # 加载子流域固定参数
        subbasin_mode = self.param_modes.get('subbasin_runoff', 'trainable')
        if subbasin_mode == 'fixed':
            params_file = params_dir / f'{subbasin_model_name}_params.csv'
            if params_file.exists():
                df = pd.read_csv(params_file)
                print(f"[Model] Loading fixed params from: {params_file}")
                
                # 为每个子流域设置参数
                for _, row in df.iterrows():
                    unit_id = str(row['unit_id'])
                    sub_id = f"sub_{unit_id}"
                    
                    if sub_id in self.subbasin_models:
                        model = self.subbasin_models[sub_id]
                        for col in df.columns:
                            if col != 'unit_id' and hasattr(model, col):
                                setattr(model, col, float(row[col]))
                
                print(f"[Model] Loaded {subbasin_model_name} params for {len(df)} subbasins")
            else:
                print(f"[Model] No {subbasin_model_name}_params.csv found, using default values")
        
        # 加载河道固定参数
        river_mode = self.param_modes.get('river_routing', 'trainable')
        if river_mode == 'fixed':
            params_file = params_dir / f'{river_model_name}_params.csv'
            if params_file.exists():
                df = pd.read_csv(params_file)
                print(f"[Model] Loading fixed params from: {params_file}")
                
                # 为每个河道设置参数
                for _, row in df.iterrows():
                    reach_id = row['reach_id']
                    
                    if reach_id in self.river_models:
                        model = self.river_models[reach_id]
                        for col in df.columns:
                            if col != 'reach_id' and hasattr(model, col):
                                setattr(model, col, float(row[col]))
                
                print(f"[Model] Loaded {river_model_name} params for {len(df)} reaches")
            else:
                print(f"[Model] No {river_model_name}_params.csv found, using default values")

    def _init_subbasin_models(self):
        """初始化各子流域产流模型（根据配置选择模型类型）"""
        from models.components.subbasin_runoff import (
            XAJModel,
            GR4JModel,
            TankModel,
            HBVModel,
            SACSMAModel,
        )
        
        # 从components配置获取模型类型
        subbasin_config = self._get_component_config('subbasin_runoff')
        model_type = subbasin_config.get('model', 'xaj')
        
        # 从全局配置读取冰冻圈设置
        cold_region_config = self.full_config.get('model', {}).get('cold_region', {})
        enable_cold_region = cold_region_config.get('enabled', False)
        
        # 从全局配置读取降水校正设置
        precip_correct_config = self.full_config.get('model', {}).get('precip_correct', {})
        enable_precip_correct = precip_correct_config.get('enabled', False)
        
        # 模型类映射
        model_classes = {
            'xaj': XAJModel,
            'gr4j': GR4JModel,
            'tank': TankModel,
            'hbv': HBVModel,
            'sacsma': SACSMAModel,
        }
        
        ModelClass = model_classes.get(model_type, XAJModel)
        print(f"[Model] Using subbasin runoff model: {model_type}, cold_region: {enable_cold_region}, precip_correct: {enable_precip_correct}")
        
        # 获取模型特定的默认参数（如果存在）
        model_params_key = f'{model_type}_params'
        default_params = {
            'enable_cold_region': enable_cold_region,
            'enable_precip_correct': enable_precip_correct,
        }
        
        # 添加冰冻圈默认参数
        if enable_cold_region:
            default_params.update({
                't_snow': cold_region_config.get('t_snow', 0.0),
                't_melt': cold_region_config.get('t_melt', 0.0),
                'ddf_snow': cold_region_config.get('ddf_snow', 3.0),
                'ddf_ice': cold_region_config.get('ddf_ice', 6.0),
            })
        
        # 添加降水校正默认参数
        if enable_precip_correct:
            default_params.update({
                'kp': precip_correct_config.get('kp', 1.0),
            })
        
        if model_params_key in self.config:
            default_params.update(self.config[model_params_key].get('default', {}))
        
        self.subbasin_models = {}
        for sub_id in self.subbasin_ids:
            sub_params = default_params.copy()
            # 从 model_attr.json 注入每个子流域的冰川比例
            if self.glacier_ratios is not None and sub_id in self.glacier_ratios:
                sub_params['glacier_frac'] = self.glacier_ratios[sub_id]
            self.subbasin_models[sub_id] = ModelClass(**sub_params)

    def _init_river_models(self):
        """初始化各河道汇流模型"""
        from models.components.river_routing import Muskingum
        
        # 从full_config读取河道参数配置
        river_config = self.full_config.get('model', {}).get('river_routing', {})
        musk_params = river_config.get('muskingum_params', {})
        
        self.river_models = {}
        for reach_id in self.reaches.keys():
            params = musk_params.get(reach_id, {'K': 1.0, 'X': 0.2})
            self.river_models[reach_id] = Muskingum(**params)
    
    def _init_parameter_networks(self):
        """初始化参数网络（network模式）- 加载已训练的网络"""
        self.param_networks = {}
        
        # 子流域参数网络
        subbasin_mode = self.param_modes.get('subbasin_runoff', 'trainable')
        if subbasin_mode == 'network':
            from dl.predictor import ParameterNetwork
            
            # 加载子流域特征
            subbasin_features = self._load_subbasin_features()
            network_config = self.network_configs.get('subbasin_runoff', {})
            
            # 获取参数信息
            model_instance = list(self.subbasin_models.values())[0]
            param_info = model_instance.param_info
            n_params = len(param_info['static']['names'])
            n_features = next(iter(subbasin_features.values())).shape[0] if subbasin_features else 10
            
            # 创建网络
            self.param_networks['subbasin'] = ParameterNetwork(
                input_dim=n_features,
                output_dim=n_params,
                hidden_dims=network_config.get('hidden_dims', [64, 32]),
                network_name='subbasin_runoff'
            )
            
            # 加载已训练的网络权重
            network_params_file = network_config.get('network_params_file', None)
            if network_params_file:
                try:
                    self.param_networks['subbasin'].load_state_dict(torch.load(network_params_file))
                    self.param_networks['subbasin'].eval()
                    print(f"[Model] Loaded subbasin network from: {network_params_file}")
                except Exception as e:
                    print(f"[Error] Failed to load network: {e}")
                    raise ValueError(f"network模式必须指定有效的network_params_file")
            else:
                raise ValueError(f"network模式必须在config中指定network_params_file路径")
            print(f"[Model] Created subbasin parameter network")
        
        # 河道参数网络
        river_mode = self.param_modes.get('river_routing', 'trainable')
        if river_mode == 'network':
            from dl.predictor import ParameterNetwork
            
            # 加载河道特征
            river_features = self._load_river_features()
            network_config = self.network_configs.get('river_routing', {})
            
            # Muskingum 有2个参数
            n_params = 2
            n_features = next(iter(river_features.values())).shape[0] if river_features else 5
            
            # 创建网络
            self.param_networks['river'] = ParameterNetwork(
                input_dim=n_features,
                output_dim=n_params,
                hidden_dims=network_config.get('hidden_dims', [32, 16]),
                network_name='river_routing'
            )
            
            # 加载已训练的网络权重
            network_params_file = network_config.get('network_params_file', None)
            if network_params_file:
                try:
                    self.param_networks['river'].load_state_dict(torch.load(network_params_file))
                    self.param_networks['river'].eval()
                    print(f"[Model] Loaded river network from: {network_params_file}")
                except Exception as e:
                    print(f"[Error] Failed to load network: {e}")
                    raise ValueError(f"network模式必须指定有效的network_params_file")
            else:
                raise ValueError(f"network模式必须在config中指定network_params_file路径")
            print(f"[Model] Created river parameter network")
    
    def _load_subbasin_features(self) -> Dict[str, torch.Tensor]:
        """加载子流域特征"""
        features = {}
        
        # 从配置中读取特征（如果有）
        subbasin_data = self.config.get('subbasins', {}).get('features', {})
        
        for sub_id in self.subbasin_ids:
            # 如果配置中有特征数据，使用之
            if sub_id in subbasin_data:
                feature_dict = subbasin_data[sub_id]
                feature_list = []
                
                # 按照network_config中指定的顺序提取特征
                feature_names = self.network_configs.get('subbasin_runoff', {}).get('input_features', [])
                for feat_name in feature_names:
                    feature_list.append(feature_dict.get(feat_name, 0.0))
                
                features[sub_id] = torch.tensor(feature_list, dtype=torch.float32)
            else:
                # 否则使用默认值（示例）
                # TODO: 从GIS数据或其他来源加载
                features[sub_id] = torch.randn(3)  # 假设3个特征
                print(f"[Warning] Using random features for {sub_id}, please provide real features")
        
        return features
    
    def _load_river_features(self) -> Dict[str, torch.Tensor]:
        """加载河道特征（从dl_input加载）"""
        features = {}
        
        # 从dl_input/river.csv加载特征
        river_csv = Path(self.full_config.get('data', {}).get('dl_input', 'data/dl_input')) / 'river.csv'
        
        if river_csv.exists():
            import pandas as pd
            river_df = pd.read_csv(river_csv, index_col=0)
            
            for reach_id in self.reaches.keys():
                if reach_id in river_df.index:
                    features[reach_id] = torch.tensor(river_df.loc[reach_id].values, dtype=torch.float32)
                else:
                    features[reach_id] = torch.randn(3)
                    print(f"[Warning] Using random features for {reach_id}")
        else:
            # 使用默认值
            for reach_id in self.reaches.keys():
                features[reach_id] = torch.randn(3)
            print(f"[Warning] River features file not found: {river_csv}")
        
        return features

    def _prepare_params(
        self,
        params: Optional[Dict],
        device: torch.device
    ) -> Dict:
        """
        根据参数模式准备参数
        
        Args:
            params: 外部提供的参数（trainable模式）
            device: 计算设备
            
        Returns:
            完整的参数字典
        """
        if params is None:
            params = {'subbasins': {}, 'rivers': {}}
        
        # 子流域参数
        subbasin_mode = self.param_modes.get('subbasin_runoff', 'trainable')
        
        if subbasin_mode == 'fixed':
            # fixed模式：使用模型默认参数
            for sub_id in self.subbasin_ids:
                if sub_id not in params.get('subbasins', {}):
                    if 'subbasins' not in params:
                        params['subbasins'] = {}
                    params['subbasins'][sub_id] = {
                        'static': {},  # 使用模型自带的默认值
                        'dynamic': {}
                    }
        
        elif subbasin_mode == 'network':
            # network模式：使用神经网络预测参数
            if 'subbasin' in self.param_networks:
                parameterizer = self.param_networks['subbasin']
                for sub_id in self.subbasin_ids:
                    # 从网络预测参数
                    predicted_params = parameterizer.predict_params(sub_id, device)
                    
                    if 'subbasins' not in params:
                        params['subbasins'] = {}
                    if sub_id not in params['subbasins']:
                        params['subbasins'][sub_id] = {'static': {}, 'dynamic': {}}
                    
                    # 合并预测的参数
                    params['subbasins'][sub_id]['static'].update(predicted_params)
        
        # trainable模式：直接使用传入的params（无需额外处理）
        
        # 河道参数
        river_mode = self.param_modes.get('river_routing', 'trainable')
        
        if river_mode == 'fixed':
            # fixed模式：使用模型默认参数
            for reach_id in self.reaches.keys():
                if reach_id not in params.get('rivers', {}):
                    if 'rivers' not in params:
                        params['rivers'] = {}
                    params['rivers'][reach_id] = {
                        'static': {},
                        'dynamic': {}
                    }
        
        elif river_mode == 'network':
            # network模式：使用神经网络预测参数
            if 'river' in self.param_networks:
                parameterizer = self.param_networks['river']
                for reach_id in self.reaches.keys():
                    # 从网络预测参数
                    predicted_params = parameterizer.predict_params(reach_id, device)
                    
                    if 'rivers' not in params:
                        params['rivers'] = {}
                    if reach_id not in params['rivers']:
                        params['rivers'][reach_id] = {'static': {}, 'dynamic': {}}
                    
                    # 合并预测的参数
                    params['rivers'][reach_id]['static'].update(predicted_params)
        
        return params
    
    def _compute_subbasin_runoff_vectorized(
        self,
        P_tensor: torch.Tensor,
        E_tensor: torch.Tensor,
        T_tensor: torch.Tensor,
        params: Dict,
        state: Dict,
        device: torch.device,
        simulate: bool = False
    ) -> tuple:
        """
        空间向量化计算所有子流域产流（时间维度循环）
        
        设计理念：
        - 空间向量化：同时计算所有子流域（批量计算）
        - 时间循环：由于产流需要上一步状态，时间维度保持循环
        
        Args:
            P_tensor: [T, n_subbasins] 降水
            E_tensor: [T, n_subbasins] 蒸发
            T_tensor: [T, n_subbasins] 温度 (可为None)
            params: 参数字典
            state: 状态字典
            device: 计算设备
            simulate: 是否输出详细中间物理过程
            
        Returns:
            subbasin_runoff: {sub_id: runoff_tensor} 字典
            details: 详细输出（模拟模式）或空字典（训练模式）
        """
        T, n_subs = P_tensor.shape
        
        # 获取参数信息
        param_info = self.get_param_info()
        param_names = param_info['subbasins']['params_per_subbasin']['static']['names']
        
        # 准备批量参数 [n_subbasins]
        batch_params = {}
        for param_name in param_names:
            param_values = []
            for sub_id in self.subbasin_ids:
                sub_params = params.get('subbasins', {}).get(sub_id, {}).get('static', {})
                value = sub_params.get(param_name, getattr(self.subbasin_models[sub_id], param_name))
                if isinstance(value, torch.Tensor):
                    param_values.append(value)
                else:
                    param_values.append(torch.tensor(float(value), dtype=torch.float32, device=device))
            batch_params[param_name] = torch.stack(param_values)  # [n_subbasins]
        
        # 准备批量状态 [n_subbasins]
        batch_state = {}
        first_model = self.subbasin_models[self.subbasin_ids[0]]
        state_keys = list(first_model.init_state().keys())
        
        for state_key in state_keys:
            state_values = []
            for sub_id in self.subbasin_ids:
                value = state['subbasins'][sub_id].get(state_key, 0.0)
                if isinstance(value, torch.Tensor):
                    state_values.append(value)
                else:
                    # 初始化状态
                    if state_key in ['wu', 'wl', 'wd']:
                        init_val = batch_params[state_key + 'm'].mean().item() / 2
                    else:
                        init_val = 0.0
                    state_values.append(torch.tensor(float(init_val), dtype=torch.float32, device=device))
            batch_state[state_key] = torch.stack(state_values)  # [n_subbasins]
        
        # 预分配输出数组
        runoff_output = torch.zeros(n_subs, T, dtype=torch.float32, device=device)
        
        # 模拟模式：预分配详细输出存储
        details = {}
        if simulate:
            # 状态变量时序 [T, n_subbasins]
            details['states'] = {state_name: torch.zeros(T, n_subs, dtype=torch.float32, device=device) 
                                 for state_name in state_keys}
            # 产流分量存储（动态添加）
            details['fluxes'] = {}
        
        # 准备冰川比例向量 [n_subbasins]（时间不变，提前构建）
        glacier_frac_tensor = None
        if self.glacier_ratios is not None:
            glacier_frac_values = [
                self.glacier_ratios.get(sub_id, 0.0)
                for sub_id in self.subbasin_ids
            ]
            glacier_frac_tensor = torch.tensor(glacier_frac_values, dtype=torch.float32, device=device)
        
        # 时间循环（空间批量计算）
        for t in range(T):
            # 准备当前时间步输入 [n_subbasins]
            batch_inputs = {
                'P': P_tensor[t, :],
                'E': E_tensor[t, :]
            }
            if T_tensor is not None:
                batch_inputs['T'] = T_tensor[t, :]
            if glacier_frac_tensor is not None:
                batch_inputs['glacier_frac'] = glacier_frac_tensor
            
            # 批量调用模型（空间向量化）
            batch_outputs, batch_state = first_model.forward(
                inputs=batch_inputs,
                static_params=batch_params,
                dynamic_params={},
                state=batch_state,
                return_details=simulate,
            )
            
            # 存储当前时间步的径流
            runoff_output[:, t] = batch_outputs['runoff']
            
            # 模拟模式：存储所有中间物理过程
            if simulate:
                def _to_subbasin_series(value: torch.Tensor) -> Optional[torch.Tensor]:
                    """将模型输出/状态转换为 [n_subs] 序列用于保存。"""
                    if not isinstance(value, torch.Tensor):
                        return None
                    if value.ndim == 0:
                        return value.expand(n_subs)
                    if value.ndim == 1:
                        return value if value.shape[0] == n_subs else None
                    if value.ndim >= 2 and value.shape[0] == n_subs:
                        # 对多维状态（如GR4J的UH队列）按非子流域维求和，保留每个子流域的总量
                        reduce_dims = tuple(range(1, value.ndim))
                        return value.sum(dim=reduce_dims)
                    return None

                # 存储状态变量
                for state_name in state_keys:
                    if state_name in batch_state:
                        state_series = _to_subbasin_series(batch_state[state_name])
                        if state_series is not None:
                            details['states'][state_name][t, :] = state_series
                
                # 存储模型输出的所有中间变量（others字典）
                others = batch_outputs.get('others', {})
                for key, value in others.items():
                    if isinstance(value, torch.Tensor):
                        flux_series = _to_subbasin_series(value)
                        if flux_series is None:
                            continue
                        if key not in details['fluxes']:
                            details['fluxes'][key] = torch.zeros(T, n_subs, dtype=torch.float32, device=device)
                        details['fluxes'][key][t, :] = flux_series
        
        # 转换回字典格式
        subbasin_runoff = {}
        for i, sub_id in enumerate(self.subbasin_ids):
            subbasin_runoff[sub_id] = runoff_output[i, :]  # [T]
            
            # 更新最终状态
            for state_key in state_keys:
                state['subbasins'][sub_id][state_key] = batch_state[state_key][i]
        
        return subbasin_runoff, details

    def init_state(self) -> Dict:
        """初始化模型状态"""
        state = {
            'subbasins': {},
            'rivers': {}
        }
        
        for sub_id in self.subbasin_ids:
            state['subbasins'][sub_id] = self.subbasin_models[sub_id].init_state()
        
        for reach_id in self.reaches.keys():
            state['rivers'][reach_id] = {}
        
        return state

    def forward(
        self,
        forcing: Dict[str, Union[np.ndarray, torch.Tensor]],
        params: Optional[Dict] = None,
        state: Optional[Dict] = None,
        device: Optional[torch.device] = None,
        simulate: bool = False
    ) -> tuple:
        """
        运行模型前向传播
        
        Args:
            forcing: 强迫数据 {'P': [T, n_sub], 'E': [T, n_sub]}
            params: 参数字典（trainable模式使用）
            state: 状态字典
            device: 计算设备
            simulate: 是否输出详细中间过程
                - False: 训练模式，只输出流量序列（节省内存）
                - True: 模拟模式，输出完整中间物理过程
            
        Returns:
            outputs: 输出字典
            state: 更新后的状态
        """
        device = device or torch.device('cpu')
        
        if state is None:
            state = self.init_state()
        
        # 根据参数模式准备参数
        params = self._prepare_params(params, device)
        
        # 获取/缓存强迫数据张量
        P_tensor, E_tensor, T_tensor = self._get_forcing_tensors(forcing, device)
        T = P_tensor.shape[0]
        
        # =====================================================
        # 步骤1: 子流域产流（向量化批量计算）
        # =====================================================
        # 检查是否所有子流域使用相同类型的模型
        model_type = type(self.subbasin_models[self.subbasin_ids[0]])
        use_vectorized = all(isinstance(self.subbasin_models[sid], model_type) 
                            for sid in self.subbasin_ids)
        
        details = {}  # 用于存储模拟模式下的详细输出
        
        if use_vectorized:
            # 向量化计算：一次计算所有子流域
            subbasin_runoff, runoff_details = self._compute_subbasin_runoff_vectorized(
                P_tensor, E_tensor, T_tensor, params, state, device, simulate
            )
            if simulate:
                details['runoff'] = runoff_details
        else:
            # 回退到循环计算（用于混合模型类型）
            subbasin_runoff = {}
            for i, sub_id in enumerate(self.subbasin_ids):
                P_sub = P_tensor[:, i]
                E_sub = E_tensor[:, i]
                
                # 获取参数
                sub_params = params.get('subbasins', {}).get(sub_id, {})
                static_params = sub_params.get('static', {})
                dynamic_params = sub_params.get('dynamic', {})
                
                # 运行产流模型
                inputs = {'P': P_sub, 'E': E_sub}
                if T_tensor is not None:
                    inputs['T'] = T_tensor[:, i]
                if self.glacier_ratios is not None and sub_id in self.glacier_ratios:
                    inputs['glacier_frac'] = self.glacier_ratios[sub_id]
                outputs, state['subbasins'][sub_id] = self.subbasin_models[sub_id].forward(
                    inputs=inputs,
                    static_params=static_params,
                    dynamic_params=dynamic_params,
                    state=state['subbasins'][sub_id],
                    return_details=simulate,
                )
                
                subbasin_runoff[sub_id] = outputs['runoff']  # [T]
        
        # =====================================================
        # 步骤2: 河道汇流（按拓扑层批量并行）
        # =====================================================
        reach_flow = self._route_reaches_batched(T, subbasin_runoff, params, state, device)
        
        # 整理输出
        outputs = {
            'subbasin_runoff': subbasin_runoff,
            'reach_flow': reach_flow,
            'river_flows': reach_flow,  # 别名，用于多观测点训练
            'outlet_flow': reach_flow[self.outlet_reach]
        }
        
        # 模拟模式：添加详细输出
        if simulate and details:
            outputs['details'] = details
        
        return outputs, state

    def get_trainable_params(self, device=None) -> Dict[str, torch.nn.Parameter]:
        """
        获取所有可训练参数（动态从模型获取参数名称）
        
        Args:
            device: torch设备
            
        Returns:
            参数字典 {param_key: tensor}
        """
        device = device or torch.device('cpu')
        params = {}
        
        # 动态获取参数信息
        param_info = self.get_param_info()
        
        # 子流域参数
        subbasin_param_names = param_info['subbasins']['params_per_subbasin']['static']['names']
        
        for sub_id in self.subbasin_ids:
            model = self.subbasin_models[sub_id]
            for param_name in subbasin_param_names:
                value = getattr(model, param_name)
                tensor_val = torch.tensor(float(value), dtype=torch.float32,
                                         device=device, requires_grad=True)
                params[f"{sub_id}_{param_name}"] = tensor_val
        
        # 河道参数
        river_param_names = param_info['rivers']['params_per_reach']['static']['names']
        
        for reach_id in self.reaches.keys():
            model = self.river_models[reach_id]
            for param_name in river_param_names:
                value = getattr(model, param_name)
                tensor_val = torch.tensor(float(value), dtype=torch.float32,
                                         device=device, requires_grad=True)
                params[f"{reach_id}_{param_name}"] = tensor_val
        
        return params

    def update_params_from_tensors(self, param_tensors: Dict[str, torch.Tensor]):
        """从张量字典更新模型参数"""
        for key, tensor in param_tensors.items():
            parts = key.split('_')
            
            if key.startswith('sub_'):
                # 子流域参数
                sub_id = '_'.join(parts[:-1])
                param_name = parts[-1]
                if sub_id in self.subbasin_models:
                    setattr(self.subbasin_models[sub_id], param_name,
                           float(tensor.detach().cpu().item()))
            
            elif key.startswith('r'):
                # 河道参数
                reach_id = '_'.join(parts[:-1])
                param_name = parts[-1]
                if reach_id in self.river_models:
                    setattr(self.river_models[reach_id], param_name,
                           float(tensor.detach().cpu().item()))

    def get_param_info(self):
        """获取参数信息"""
        xaj_info = self.subbasin_models[self.subbasin_ids[0]].get_param_info()
        musk_info = self.river_models[self.reach_order[0]].get_param_info()
        
        return {
            'subbasins': {
                'count': self.n_subbasins,
                'params_per_subbasin': xaj_info
            },
            'rivers': {
                'count': len(self.reaches),
                'params_per_reach': musk_info
            }
        }
    
    def get_trainable_components(self) -> Dict:
        """
        获取需要训练的组件
        
        Returns:
            字典包含：
            - 'tensors': 可训练的参数tensor字典（trainable模式）
            - 'networks': 可训练的神经网络（network模式）
        """
        components = {
            'tensors': {},
            'networks': {}
        }
        
        # 子流域参数
        subbasin_mode = self.param_modes.get('subbasin_runoff', 'trainable')
        if subbasin_mode == 'network' and 'subbasin' in self.param_networks:
            components['networks']['subbasin'] = self.param_networks['subbasin'].get_network()
        
        # 河道参数
        river_mode = self.param_modes.get('river_routing', 'trainable')
        if river_mode == 'network' and 'river' in self.param_networks:
            components['networks']['river'] = self.param_networks['river'].get_network()
        
        return components
    
    def save_parameter_networks(self, save_dir: str):
        """保存参数网络"""
        from pathlib import Path
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if 'subbasin' in self.param_networks:
            self.param_networks['subbasin'].save_network(
                str(save_path / 'subbasin_param_net.pt')
            )
            print(f"[Save] Subbasin parameter network saved")
        
        if 'river' in self.param_networks:
            self.param_networks['river'].save_network(
                str(save_path / 'river_param_net.pt')
            )
            print(f"[Save] River parameter network saved")
    
    def load_parameter_networks(self, load_dir: str):
        """加载参数网络"""
        from pathlib import Path
        load_path = Path(load_dir)
        
        if 'subbasin' in self.param_networks:
            net_file = load_path / 'subbasin_param_net.pt'
            if net_file.exists():
                self.param_networks['subbasin'].load_network(str(net_file))
                print(f"[Load] Subbasin parameter network loaded")
        
        if 'river' in self.param_networks:
            net_file = load_path / 'river_param_net.pt'
            if net_file.exists():
                self.param_networks['river'].load_network(str(net_file))
                print(f"[Load] River parameter network loaded")
