"""
分布式网格水文模型
支持网格产流 + 坡面汇流 + 河道汇流
用于SPHY等分布式模型的训练和模拟
"""
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path

from models.hydro_model import HydroModel


class GridHydroModel(HydroModel):
    """
    分布式网格水文模型
    
    继承自HydroModel，扩展支持网格产流模式：
    - grid_runoff: 网格产流（如SPHY、VIC、XAJ网格版）
    - hillslope_routing: 坡面汇流（D8 travel time等）
    - river_routing: 河道汇流（Muskingum）
    
    与子流域模式的区别：
    - 输入数据为(T, H, W)的网格数据而非(T, n_subbasins)
    - 参数为(H, W)的网格参数而非标量
    - 需要额外的静态数据：D8流向、子流域掩膜
    """
    
    def __init__(
        self,
        model_attr_path: str,
        config: Optional[Dict] = None,
        grid_info: Optional[Dict] = None,
    ):
        """
        初始化网格模型
        
        Args:
            model_attr_path: 模型属性配置文件路径
            config: 完整配置字典
            grid_info: 网格信息字典 {
                'H': int, 'W': int,
                'cellsize': float,
                'n_basins': int,
                'n_features': int,  # 网格特征数量
            }
        """
        # 保存网格信息（在父类初始化前）
        self.grid_info = grid_info or {}
        self.H = self.grid_info.get('H', 50)
        self.W = self.grid_info.get('W', 50)
        self.cellsize = self.grid_info.get('cellsize', 1000.0)
        self.n_basins = self.grid_info.get('n_basins', 1)
        self.n_grid_features = self.grid_info.get('n_features', 4)
        
        # 模式标记
        self.use_grid_mode = True
        
        # 网格模型和坡面汇流模型
        self.grid_model = None
        self.hillslope_model = None
        
        # 网格参数网络
        self.grid_param_network = None
        self.hillslope_param_network = None
        
        # 固定参数缓存（fixed模式时使用）
        self.fixed_grid_params = {}
        
        # 调用父类初始化
        super().__init__(model_attr_path, config)
        
        # 初始化网格相关模型
        self._init_grid_models()
        self._init_hillslope_models()
        self._init_grid_parameter_networks()
        
        # 加载固定网格参数
        self._load_fixed_grid_params()
    
    def _init_grid_models(self):
        """初始化网格产流模型"""
        from models.components.grid_runoff import SPHYModel, XAJGridModel, VICModel
        from models.registry import GRID_REGISTRY
        
        # 从components配置获取模型类型
        components = self.full_config.get('model', {}).get('components', {})
        grid_config = components.get('grid_runoff', {'model': 'sphy', 'param_mode': 'trainable'})
        model_type = grid_config.get('model', 'sphy')
        
        # 从全局配置读取冰冻圈设置
        cold_region_config = self.full_config.get('model', {}).get('cold_region', {})
        enable_cold_region = cold_region_config.get('enabled', False)
        
        # 模型类映射
        model_classes = {
            'sphy': SPHYModel,
            'xaj_grid': XAJGridModel,
            'vic': VICModel,
        }
        
        # 尝试从注册表获取
        if model_type in GRID_REGISTRY:
            ModelClass = GRID_REGISTRY[model_type]
        else:
            ModelClass = model_classes.get(model_type, SPHYModel)
        
        self.grid_model = ModelClass(enable_cold_region=enable_cold_region)
        self.grid_model_type = model_type
        
        print(f"[GridModel] Using grid runoff model: {model_type}")
    
    def _init_hillslope_models(self):
        """初始化坡面汇流模型"""
        from models.registry import HILLSLOPE_REGISTRY
        
        # 从components配置获取模型类型
        components = self.full_config.get('model', {}).get('components', {})
        hillslope_config = components.get('hillslope_routing', {'model': 'd8_travel_time', 'param_mode': 'trainable'})
        model_type = hillslope_config.get('model', 'd8_travel_time')
        
        if model_type in HILLSLOPE_REGISTRY:
            self.hillslope_model = HILLSLOPE_REGISTRY[model_type]()
            print(f"[GridModel] Using hillslope routing model: {model_type}")
        else:
            print(f"[Warning] Hillslope model '{model_type}' not found in registry")
            self.hillslope_model = None
    
    def _init_grid_parameter_networks(self):
        """初始化网格参数网络"""
        from dl.param_net import ParameterNetwork
        
        # 检查是否需要网格参数网络
        grid_mode = self.param_modes.get('grid_runoff', 'trainable')
        
        if grid_mode in ['network', 'trainable']:
            # 获取网格产流模型的参数信息
            grid_param_info = self.grid_model._define_parameters()
            grid_params = grid_param_info['static']
            
            # 筛选属于grid_runoff的参数
            grid_param_names = []
            grid_param_bounds = []
            network_mapping = grid_params.get('network_mapping', {})
            
            for i, name in enumerate(grid_params['names']):
                mapping = network_mapping.get(name, 'grid_runoff')
                if mapping == 'grid_runoff':
                    grid_param_names.append(name)
                    grid_param_bounds.append(grid_params['bounds'][i])
            
            # 添加hillslope的velocity参数（由grid_runoff网络预测）
            if self.hillslope_model is not None:
                hs_param_info = self.hillslope_model._define_parameters()
                hs_params = hs_param_info['static']
                hs_mapping = hs_params.get('network_mapping', {})
                
                for i, name in enumerate(hs_params['names']):
                    if hs_mapping.get(name) == 'grid_runoff':
                        grid_param_names.append(name)
                        grid_param_bounds.append(hs_params['bounds'][i])
            
            if grid_param_names:
                # 创建参数范围字典
                param_ranges = {}
                for name, bounds in zip(grid_param_names, grid_param_bounds):
                    param_ranges[name] = tuple(bounds)
                
                # 创建网格参数网络
                # 输入：网格特征 (n_features)
                # 输出：网格参数 (n_params)
                self.grid_param_network = ParameterNetwork(
                    input_dim=self.n_grid_features,
                    param_names=grid_param_names,
                    param_ranges=param_ranges,
                    hidden_dims=[64, 32],
                )
                
                print(f"[GridModel] Created grid parameter network:")
                print(f"  - Input features: {self.n_grid_features}")
                print(f"  - Output params: {grid_param_names}")
        
        # 坡面汇流参数网络（hillslope_routing的非网格参数）
        hs_mode = self.param_modes.get('hillslope_routing', 'trainable')
        
        if hs_mode in ['network', 'trainable'] and self.hillslope_model is not None:
            hs_param_info = self.hillslope_model._define_parameters()
            hs_params = hs_param_info['static']
            hs_mapping = hs_params.get('network_mapping', {})
            
            hs_param_names = []
            hs_param_bounds = []
            
            for i, name in enumerate(hs_params['names']):
                if hs_mapping.get(name) == 'hillslope_routing':
                    hs_param_names.append(name)
                    hs_param_bounds.append(hs_params['bounds'][i])
            
            if hs_param_names:
                param_ranges = {}
                for name, bounds in zip(hs_param_names, hs_param_bounds):
                    param_ranges[name] = tuple(bounds)
                
                # 坡面汇流参数可以是标量，使用简单网络
                self.hillslope_param_network = ParameterNetwork(
                    input_dim=self.n_basins,  # 使用子流域特征
                    param_names=hs_param_names,
                    param_ranges=param_ranges,
                    hidden_dims=[16],
                )
                
                print(f"[GridModel] Created hillslope parameter network: {hs_param_names}")
    
    def _load_fixed_grid_params(self):
        """
        加载固定网格参数（仅fixed模式）
        
        支持两种参数来源（param_source）：
        - uniform: 按子流域设置，从 {model_name}_params.csv 读取
                   文件格式：unit_id, param1, param2, ...（与子流域参数格式一致）
                   根据子流域掩码将参数映射到对应网格
        - spatial: 空间分布，从 params/grid/{model_name}/{param_name}.tif 读取
        
        文件位置：
        - uniform模式: data/流域名称/params/{model_name}_params.csv
        - spatial模式: data/流域名称/params/grid/{model_name}/{param_name}.tif
        
        冰冻圈参数：
        - 由 cold_region.enabled 控制是否启用
        - fixed模式从 data/流域名称/params/cold_region_params.csv 读取
        - 同样支持 uniform（按子流域）和 spatial（tif）两种模式
        
        参数文件应用失败时使用默认参数
        """
        import pandas as pd
        from pathlib import Path
        
        # 检查是否为fixed模式
        grid_config = self._get_component_config('grid_runoff')
        grid_mode = self.param_modes.get('grid_runoff', 'trainable')
        
        if grid_mode != 'fixed':
            return
        
        params_dir = Path(self._get_params_dir())
        model_name = grid_config.get('model', 'sphy')
        param_source = grid_config.get('param_source', 'uniform')
        
        # 获取模型参数信息
        param_info = self.grid_model._define_parameters()
        static_params = param_info.get('static', {})
        param_names = static_params.get('names', [])
        param_bounds = static_params.get('bounds', [])
        
        # 冰冻圈参数名称
        cold_region_params = {
            't_snow', 't_melt', 'ddf_snow', 'k_snow',
            'ddf_ice', 't_ice_melt', 'glacier_runoff_frac',
            'fi_alpha', 'fi_beta'
        }
        
        # 分离模型参数和冰冻圈参数
        model_param_names = []
        model_param_bounds = []
        cold_param_names = []
        cold_param_bounds = []
        
        for name, bounds in zip(param_names, param_bounds):
            if name in cold_region_params:
                cold_param_names.append(name)
                cold_param_bounds.append(bounds)
            else:
                model_param_names.append(name)
                model_param_bounds.append(bounds)
        
        # 加载模型参数
        self._load_grid_params_from_source(
            param_source, params_dir, model_name,
            model_param_names, model_param_bounds, is_cold_region=False
        )
        
        # 加载冰冻圈参数（如果启用）
        cold_region_config = self.full_config.get('model', {}).get('cold_region', {})
        if cold_region_config.get('enabled', False) and cold_param_names:
            self._load_grid_params_from_source(
                param_source, params_dir, 'cold_region',
                cold_param_names, cold_param_bounds, is_cold_region=True
            )
    
    def _load_grid_params_from_source(
        self,
        param_source: str,
        params_dir: Path,
        file_prefix: str,
        param_names: list,
        param_bounds: list,
        is_cold_region: bool = False
    ):
        """
        从指定来源加载网格参数
        
        Args:
            param_source: 'uniform' 或 'spatial'
            params_dir: 参数目录
            file_prefix: 文件前缀（模型名称或'cold_region'）
            param_names: 参数名称列表
            param_bounds: 参数范围列表
            is_cold_region: 是否为冰冻圈参数
        """
        import pandas as pd
        from pathlib import Path
        
        param_type = "cold_region" if is_cold_region else "model"
        
        if param_source == 'uniform':
            # 从CSV读取按子流域设置的参数
            params_file = params_dir / f'{file_prefix}_params.csv'
            if params_file.exists():
                try:
                    df = pd.read_csv(params_file)
                    print(f"[GridModel] Loading {param_type} params from: {params_file}")
                    
                    # CSV格式：unit_id, param1, param2, ...
                    # 存储为字典：{param_name: {unit_id: value}}
                    for col in df.columns:
                        if col != 'unit_id' and col in param_names:
                            param_dict = {}
                            for _, row in df.iterrows():
                                unit_id = int(row['unit_id'])
                                param_dict[unit_id] = float(row[col])
                            self.fixed_grid_params[col] = param_dict
                    
                    print(f"[GridModel] Loaded {param_type} params: {[n for n in param_names if n in self.fixed_grid_params]}")
                except Exception as e:
                    print(f"[GridModel] Failed to load {params_file}: {e}, using default values")
                    self._set_default_params(param_names, param_bounds)
            else:
                print(f"[GridModel] No {file_prefix}_params.csv found, using default values")
                self._set_default_params(param_names, param_bounds)
                    
        elif param_source == 'spatial':
            # 从tif文件读取空间分布
            grid_dir = params_dir / 'grid' / file_prefix
            if grid_dir.exists():
                print(f"[GridModel] Loading spatial {param_type} params from: {grid_dir}")
                loaded_count = 0
                
                for i, name in enumerate(param_names):
                    tif_file = grid_dir / f'{name}.tif'
                    if tif_file.exists():
                        try:
                            param_array = self._load_tif(str(tif_file))
                            if param_array is not None:
                                self.fixed_grid_params[name] = param_array
                                loaded_count += 1
                                print(f"  - Loaded {name}.tif: shape={param_array.shape}")
                            else:
                                self._set_single_default(name, param_bounds[i])
                        except Exception as e:
                            print(f"  - Error loading {name}.tif: {e}")
                            self._set_single_default(name, param_bounds[i])
                    else:
                        self._set_single_default(name, param_bounds[i])
                        print(f"  - {name}.tif not found, using default")
                
                print(f"[GridModel] Loaded {loaded_count}/{len(param_names)} spatial {param_type} params")
            else:
                print(f"[GridModel] Grid params dir not found: {grid_dir}, using default values")
                self._set_default_params(param_names, param_bounds)
        else:
            print(f"[Warning] Unknown param_source: {param_source}, using defaults")
            self._set_default_params(param_names, param_bounds)
    
    def _set_default_params(self, param_names: list, param_bounds: list):
        """设置默认参数值（范围中值）"""
        for name, bounds in zip(param_names, param_bounds):
            self.fixed_grid_params[name] = (bounds[0] + bounds[1]) / 2.0
    
    def _set_single_default(self, name: str, bounds: list):
        """设置单个参数的默认值"""
        self.fixed_grid_params[name] = (bounds[0] + bounds[1]) / 2.0
    
    def _load_tif(self, filepath: str) -> Optional[np.ndarray]:
        """
        加载tif文件为numpy数组
        
        Args:
            filepath: tif文件路径
            
        Returns:
            numpy数组，或None（如果加载失败）
        """
        try:
            try:
                import rasterio
                with rasterio.open(filepath) as src:
                    return src.read(1).astype(np.float32)
            except ImportError:
                # 如果没有rasterio，尝试使用gdal
                try:
                    from osgeo import gdal
                    ds = gdal.Open(filepath)
                    if ds is not None:
                        arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
                        ds = None
                        return arr
                except ImportError:
                    print(f"[Warning] Neither rasterio nor gdal available, cannot load {filepath}")
                    return None
        except Exception as e:
            print(f"[Error] Failed to load tif: {filepath}, error: {e}")
            return None

    def forward_grid(
        self,
        forcing: Dict[str, torch.Tensor],
        static_data: Dict[str, torch.Tensor],
        grid_features: torch.Tensor,
        params: Optional[Dict] = None,
        state: Optional[Dict] = None,
        device: Optional[torch.device] = None,
        simulate: bool = False,
    ) -> tuple:
        """
        网格模式前向传播
        
        Args:
            forcing: 强迫数据 {
                'P': (T, H, W) 降水,
                'E': (T, H, W) 蒸发,
                'T': (T, H, W) 温度（可选）
            }
            static_data: 静态数据 {
                'flow_dir': (H, W) D8流向,
                'basin_mask': (H, W) 子流域掩膜,
            }
            grid_features: (C, H, W) 网格特征
            params: 参数字典（可选）
            state: 状态字典（可选）
            device: 计算设备
            simulate: 是否输出详细中间物理过程
                - False: 训练模式，只输出流量（节省内存）
                - True: 模拟模式，输出完整中间过程
            
        Returns:
            outputs: 输出字典
            state: 更新后的状态
        """
        device = device or torch.device('cpu')
        
        P = forcing['P']  # (T, H, W)
        E = forcing['E']  # (T, H, W)
        T_data = forcing.get('T', None)  # (T, H, W)
        
        T_steps, H, W = P.shape
        
        # 初始化状态
        if state is None:
            state = self._init_grid_state(H, W, device)
        
        # =====================================================
        # 步骤1: 预测网格参数（使用参数网络）
        # =====================================================
        basin_mask = static_data.get('basin_mask', None)
        grid_params = self._predict_grid_params(grid_features, device, basin_mask)
        
        # =====================================================
        # 步骤2: 网格产流计算
        # =====================================================
        runoff_grid, grid_details = self._compute_grid_runoff(
            P, E, T_data, grid_params, state, device, simulate
        )  # (T, H, W)
        
        # =====================================================
        # 步骤3: 坡面汇流
        # =====================================================
        if self.hillslope_model is not None:
            Q_lateral = self._compute_hillslope_routing(
                runoff_grid,
                static_data,
                grid_params,
                state,
                device
            )  # (n_basins, T)
        else:
            # 无坡面汇流：直接按子流域求和
            Q_lateral = self._aggregate_runoff_to_basins(
                runoff_grid, static_data['basin_mask']
            )  # (n_basins, T)
        
        # =====================================================
        # 步骤4: 河道汇流
        # =====================================================
        # 准备参数
        if params is None:
            params = {'rivers': {}}
        params = self._prepare_params(params, device)
        
        reach_flow = self._compute_river_routing(
            Q_lateral, params, state, device
        )
        
        # 整理输出
        outputs = {
            'runoff_grid': runoff_grid,
            'Q_lateral': Q_lateral,
            'reach_flow': reach_flow,
            'river_flows': reach_flow,
            'outlet_flow': reach_flow[self.outlet_reach],
        }
        
        # 模拟模式：添加详细输出
        if simulate and grid_details:
            outputs['details'] = {'grid': grid_details}
        
        return outputs, state
    
    def _init_grid_state(self, H: int, W: int, device: torch.device) -> Dict:
        """初始化网格状态"""
        state = {
            'grid': self.grid_model.init_state(),
            'hillslope': {},
            'rivers': {},
        }
        
        # 将标量状态扩展为网格
        for key, val in state['grid'].items():
            if isinstance(val, (int, float)):
                state['grid'][key] = torch.full(
                    (H, W), val, dtype=torch.float32, device=device
                )
            elif isinstance(val, torch.Tensor) and val.ndim == 0:
                state['grid'][key] = val.expand(H, W).clone().to(device)
        
        # 河道状态
        for reach_id in self.reaches.keys():
            state['rivers'][reach_id] = {}
        
        return state
    
    def _predict_grid_params(
        self,
        grid_features: torch.Tensor,
        device: torch.device,
        basin_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        使用参数网络预测网格参数
        
        Args:
            grid_features: (C, H, W) 网格特征
            device: 计算设备
            basin_mask: (H, W) 子流域掩码，uniform模式需要
            
        Returns:
            grid_params: {param_name: (H, W) tensor}
        """
        grid_params = {}
        
        C, H, W = grid_features.shape
        
        # 检查是否为fixed模式
        grid_mode = self.param_modes.get('grid_runoff', 'trainable')
        
        if grid_mode == 'fixed' and self.fixed_grid_params:
            # Fixed模式：使用加载的固定参数
            for name, value in self.fixed_grid_params.items():
                if isinstance(value, np.ndarray):
                    # spatial模式：tif加载的数组
                    param_tensor = torch.from_numpy(value).to(device).float()
                    # 如果尺寸不匹配，需要调整大小
                    if param_tensor.shape != (H, W):
                        import torch.nn.functional as F
                        param_tensor = param_tensor.unsqueeze(0).unsqueeze(0)
                        param_tensor = F.interpolate(
                            param_tensor, size=(H, W), mode='bilinear', align_corners=False
                        )
                        param_tensor = param_tensor.squeeze(0).squeeze(0)
                    grid_params[name] = param_tensor
                elif isinstance(value, dict):
                    # uniform模式：按子流域掩码映射参数
                    # value = {unit_id: param_value}
                    if basin_mask is not None:
                        param_tensor = torch.zeros((H, W), dtype=torch.float32, device=device)
                        mask_np = basin_mask.cpu().numpy() if isinstance(basin_mask, torch.Tensor) else basin_mask
                        
                        for unit_id, param_val in value.items():
                            unit_mask = (mask_np == unit_id)
                            param_tensor[unit_mask] = param_val
                        
                        grid_params[name] = param_tensor
                    else:
                        # 无掩码时使用所有子流域的均值
                        mean_val = sum(value.values()) / len(value) if value else 0.5
                        grid_params[name] = torch.full(
                            (H, W), mean_val, dtype=torch.float32, device=device
                        )
                else:
                    # 标量值（默认值情况）
                    grid_params[name] = torch.full(
                        (H, W), float(value), dtype=torch.float32, device=device
                    )
        elif self.grid_param_network is not None:
            # Trainable模式：使用神经网络预测
            # 重塑特征: (C, H, W) -> (H*W, C)
            features_flat = grid_features.permute(1, 2, 0).reshape(-1, C)
            
            # 网络预测: (H*W, C) -> (H*W, n_params)
            params_flat = self.grid_param_network(features_flat)
            
            # 重塑回网格: (H*W, n_params) -> (n_params, H, W)
            n_params = params_flat.shape[1]
            params_grid = params_flat.reshape(H, W, n_params).permute(2, 0, 1)
            
            # 转换为字典
            for i, name in enumerate(self.grid_param_network.param_names):
                grid_params[name] = params_grid[i]  # (H, W)
        else:
            # 无网络且非fixed模式：使用默认值
            param_info = self.grid_model._define_parameters()
            static_params = param_info.get('static', {})
            for name, bounds in zip(static_params.get('names', []), static_params.get('bounds', [])):
                default_val = (bounds[0] + bounds[1]) / 2.0
                grid_params[name] = torch.full(
                    (H, W), default_val, dtype=torch.float32, device=device
                )
        
        return grid_params
    
    def _compute_grid_runoff(
        self,
        P: torch.Tensor,
        E: torch.Tensor,
        T_data: Optional[torch.Tensor],
        grid_params: Dict[str, torch.Tensor],
        state: Dict,
        device: torch.device,
        simulate: bool = False,
    ) -> tuple:
        """
        计算网格产流（空间向量化，时间循环）
        
        Args:
            P: (T, H, W) 降水
            E: (T, H, W) 蒸发
            T_data: (T, H, W) 温度（可选）
            grid_params: 网格参数
            state: 状态字典
            device: 计算设备
            simulate: 是否输出详细中间物理过程
            
        Returns:
            runoff: (T, H, W) 产流
            details: 详细输出（模拟模式）或空字典（训练模式）
        """
        T_steps, H, W = P.shape
        runoff_list = []
        
        grid_state = state['grid']
        
        # 模拟模式：准备详细输出存储
        details = {}
        if simulate:
            state_keys = list(grid_state.keys())
            details['states'] = {key: [] for key in state_keys}
            details['fluxes'] = {}
        
        for t in range(T_steps):
            inputs = {
                'P': P[t],  # (H, W)
                'E': E[t],  # (H, W)
            }
            if T_data is not None:
                inputs['T'] = T_data[t]  # (H, W)
            
            result, grid_state = self.grid_model.forward(
                inputs=inputs,
                static_params=grid_params,
                dynamic_params={},
                state=grid_state,
            )
            
            runoff_list.append(result['runoff'])
            
            # 模拟模式：存储详细输出
            if simulate:
                # 存储状态变量
                for key in state_keys:
                    if key in grid_state:
                        details['states'][key].append(grid_state[key].clone())
                
                # 存储产流分量和其他变量
                others = result.get('others', {})
                for key, value in others.items():
                    if isinstance(value, torch.Tensor):
                        if key not in details['fluxes']:
                            details['fluxes'][key] = []
                        details['fluxes'][key].append(value.clone())
        
        state['grid'] = grid_state
        
        # 模拟模式：将列表转换为张量
        if simulate:
            for key in list(details['states'].keys()):
                if details['states'][key]:
                    details['states'][key] = torch.stack(details['states'][key], dim=0)  # (T, H, W)
            for key in list(details['fluxes'].keys()):
                if details['fluxes'][key]:
                    details['fluxes'][key] = torch.stack(details['fluxes'][key], dim=0)  # (T, H, W)
        
        return torch.stack(runoff_list, dim=0), details  # (T, H, W)
    
    def _compute_hillslope_routing(
        self,
        runoff: torch.Tensor,
        static_data: Dict[str, torch.Tensor],
        grid_params: Dict[str, torch.Tensor],
        state: Dict,
        device: torch.device,
    ) -> torch.Tensor:
        """
        计算坡面汇流
        
        Args:
            runoff: (T, H, W) 网格产流
            static_data: 静态数据
            grid_params: 网格参数（包含velocity）
            state: 状态字典
            device: 计算设备
            
        Returns:
            Q_lateral: (n_basins, T) 各子流域侧向入流
        """
        # 准备hillslope输入
        inputs = {
            'runoff': runoff,
            'flow_dir': static_data['flow_dir'],
            'basin_mask': static_data['basin_mask'],
            'cellsize': self.cellsize,
            'dt': 1.0,
        }
        
        # 提取hillslope参数
        hs_params = {}
        
        # velocity从grid_params获取
        if 'velocity' in grid_params:
            hs_params['velocity'] = grid_params['velocity']
        
        # K等参数从hillslope_param_network获取
        if self.hillslope_param_network is not None:
            # 使用子流域特征预测K
            # 简化：使用固定特征
            basin_features = torch.ones(1, self.n_basins, device=device)
            hs_scalar_params = self.hillslope_param_network(basin_features)
            
            for i, name in enumerate(self.hillslope_param_network.param_names):
                hs_params[name] = hs_scalar_params[0, i]
        
        # 调用坡面汇流模型
        hs_state = state.get('hillslope', {})
        result, hs_state = self.hillslope_model.forward(
            inputs=inputs,
            static_params=hs_params,
            dynamic_params={},
            state=hs_state,
        )
        state['hillslope'] = hs_state
        
        return result['Q_lateral']  # (n_basins, T)
    
    def _aggregate_runoff_to_basins(
        self,
        runoff: torch.Tensor,
        basin_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        将网格产流聚合到子流域
        
        Args:
            runoff: (T, H, W) 网格产流
            basin_mask: (H, W) 子流域掩膜
            
        Returns:
            Q_lateral: (n_basins, T)
        """
        T_steps, H, W = runoff.shape
        device = runoff.device
        
        n_basins = int(basin_mask[basin_mask >= 0].max().item()) + 1
        Q_lateral = torch.zeros(n_basins, T_steps, device=device)
        
        for b in range(n_basins):
            mask = (basin_mask == b)
            Q_lateral[b] = runoff[:, mask].sum(dim=1)
        
        return Q_lateral
    
    def _compute_river_routing(
        self,
        Q_lateral: torch.Tensor,
        params: Dict,
        state: Dict,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        计算河道汇流
        
        Args:
            Q_lateral: (n_basins, T) 各子流域侧向入流
            params: 参数字典
            state: 状态字典
            device: 计算设备
            
        Returns:
            reach_flow: {reach_id: (T,) tensor}
        """
        n_basins, T_steps = Q_lateral.shape
        reach_flow = {}
        
        for reach_id in self.reach_order:
            reach_info = self.reaches[reach_id]
            
            # ===== 1. 计算上游河道入流 Q_in =====
            upstream_inflow = []
            for upstream_reach in reach_info.get('upstream_reaches', []):
                if upstream_reach in reach_flow:
                    upstream_inflow.append(reach_flow[upstream_reach])
            
            if upstream_inflow:
                Q_in = torch.stack(upstream_inflow).sum(dim=0)
            else:
                Q_in = torch.zeros(T_steps, dtype=torch.float32, device=device)
            
            # ===== 2. 获取对应子流域的产流（侧向入流） =====
            # 河道与子流域一一对应：ri ↔ sub_i
            sub_id = f"sub_{reach_id[1:]}"  # r0 -> sub_0, r14 -> sub_14
            if sub_id in self.subbasin_ids:
                sub_idx = self.subbasin_ids.index(sub_id)
                if sub_idx < n_basins:
                    Q_lateral_sub = Q_lateral[sub_idx]
                else:
                    Q_lateral_sub = torch.zeros(T_steps, dtype=torch.float32, device=device)
            else:
                Q_lateral_sub = torch.zeros(T_steps, dtype=torch.float32, device=device)
            
            # ===== 3. 河道汇流计算 =====
            reach_params = params.get('rivers', {}).get(reach_id, {})
            static_params = reach_params.get('static', {})
            
            has_upstream = len(reach_info.get('upstream_reaches', [])) > 0
            
            if has_upstream:
                # 有上游河道：对Q_in进行Muskingum演算
                inputs = {'Q_in': Q_in}
                outputs, state['rivers'][reach_id] = self.river_models[reach_id].forward(
                    inputs=inputs,
                    static_params=static_params,
                    dynamic_params={},
                    state=state['rivers'][reach_id],
                )
                reach_flow[reach_id] = outputs['Q'] + Q_lateral_sub
            else:
                # 源头河道：直接输出子流域产流
                reach_flow[reach_id] = Q_lateral_sub
                state['rivers'][reach_id] = state['rivers'].get(reach_id, {})
        
        return reach_flow
    
    def get_trainable_params_grid(self, device=None) -> Dict[str, torch.nn.Parameter]:
        """
        获取网格模式的可训练参数
        
        Returns:
            包含网络参数的字典
        """
        device = device or torch.device('cpu')
        params = {}
        
        # 网格参数网络
        if self.grid_param_network is not None:
            for name, param in self.grid_param_network.named_parameters():
                params[f'grid_net.{name}'] = param
        
        # 坡面汇流参数网络
        if self.hillslope_param_network is not None:
            for name, param in self.hillslope_param_network.named_parameters():
                params[f'hillslope_net.{name}'] = param
        
        # 河道参数（复用父类）
        river_params = super().get_trainable_params(device)
        for key, val in river_params.items():
            if key.startswith('r'):  # 河道参数
                params[key] = val
        
        return params
    
    def save_networks(self, save_dir: str):
        """保存所有网络"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if self.grid_param_network is not None:
            torch.save(
                self.grid_param_network.state_dict(),
                save_path / 'grid_param_net.pt'
            )
            print(f"[Save] Grid parameter network saved")
        
        if self.hillslope_param_network is not None:
            torch.save(
                self.hillslope_param_network.state_dict(),
                save_path / 'hillslope_param_net.pt'
            )
            print(f"[Save] Hillslope parameter network saved")
        
        # 保存河道参数网络
        super().save_parameter_networks(save_dir)
    
    def load_networks(self, load_dir: str):
        """加载所有网络"""
        load_path = Path(load_dir)
        
        if self.grid_param_network is not None:
            net_file = load_path / 'grid_param_net.pt'
            if net_file.exists():
                self.grid_param_network.load_state_dict(torch.load(net_file))
                print(f"[Load] Grid parameter network loaded")
        
        if self.hillslope_param_network is not None:
            net_file = load_path / 'hillslope_param_net.pt'
            if net_file.exists():
                self.hillslope_param_network.load_state_dict(torch.load(net_file))
                print(f"[Load] Hillslope parameter network loaded")
        
        # 加载河道参数网络
        super().load_parameter_networks(load_dir)
