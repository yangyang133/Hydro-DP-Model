"""
网格数据加载模块（Grid Data Loader）

本模块负责加载分布式水文模型所需的网格数据，包括：
1. 逐日强迫数据（TIF格式）：降水P、蒸发E、气温T
2. 静态地形数据：D8流向、子流域掩膜、高程、坡度等
3. 网格特征数据：用于参数网络学习

支持的数据格式：
- 逐日数据：按日期命名的TIF文件（如 P_20000101.tif）
- 静态数据：单个TIF文件

目录结构示例：
    data/grid/
        forcing/
            P/
                P_20000101.tif
                P_20000102.tif
                ...
            E/
                E_20000101.tif
                ...
            T/
                T_20000101.tif
                ...
        static/
            flow_dir.tif      # D8流向
            basin_mask.tif    # 子流域编号掩膜
            dem.tif           # 高程（可选）
            slope.tif         # 坡度（可选）
        features/
            grid_features.tif # 网格特征（多波段，用于参数学习）
"""

import os
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import pandas as pd


# ============================================================================
# 条件导入：支持有无rasterio的环境
# ============================================================================
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("[Warning] rasterio not installed, TIF support disabled. Using numpy fallback.")


# ============================================================================
# TIF文件读取工具
# ============================================================================

def read_tif(filepath: str) -> Tuple[np.ndarray, dict]:
    """
    读取TIF文件
    
    Args:
        filepath: TIF文件路径
        
    Returns:
        data: numpy数组 (H, W) 或 (C, H, W)
        meta: 元数据字典（transform, crs, nodata等）
    """
    if HAS_RASTERIO:
        with rasterio.open(filepath) as src:
            data = src.read()
            meta = {
                'transform': src.transform,
                'crs': src.crs,
                'nodata': src.nodata,
                'height': src.height,
                'width': src.width,
                'count': src.count,
                'dtype': src.dtypes[0],
            }
            # 单波段时压缩维度
            if data.shape[0] == 1:
                data = data[0]
            return data, meta
    else:
        # 无rasterio时的fallback：尝试用numpy加载npy文件
        npy_path = filepath.replace('.tif', '.npy')
        if os.path.exists(npy_path):
            data = np.load(npy_path)
            meta = {'height': data.shape[-2], 'width': data.shape[-1]}
            return data, meta
        else:
            raise ImportError(
                f"rasterio未安装，无法读取TIF文件: {filepath}\n"
                "请安装rasterio: pip install rasterio\n"
                "或将数据转换为.npy格式"
            )


def save_tif(filepath: str, data: np.ndarray, meta: dict = None):
    """
    保存TIF文件
    
    Args:
        filepath: 输出文件路径
        data: numpy数组
        meta: 元数据字典（可选）
    """
    if HAS_RASTERIO:
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        
        count = data.shape[0]
        height, width = data.shape[1], data.shape[2]
        
        profile = {
            'driver': 'GTiff',
            'dtype': data.dtype,
            'width': width,
            'height': height,
            'count': count,
            'crs': meta.get('crs') if meta else None,
            'transform': meta.get('transform') if meta else None,
            'nodata': meta.get('nodata', -9999),
        }
        
        with rasterio.open(filepath, 'w', **profile) as dst:
            dst.write(data)
    else:
        # fallback: 保存为npy
        npy_path = filepath.replace('.tif', '.npy')
        np.save(npy_path, data)
        print(f"[Info] Saved as npy (rasterio not available): {npy_path}")


# ============================================================================
# 网格数据加载类
# ============================================================================

class GridDataLoader:
    """
    网格数据加载器
    
    负责加载和管理分布式水文模型所需的所有网格数据。
    
    支持新的目录结构：
    - forcing_dir: data/forcing/grid
    - flow_dir/basin_mask: data/model_attr/
    - grid_features: data/dl_input/grid.tif
    
    Attributes:
        forcing_dir: 网格强迫数据目录
        H, W: 网格尺寸
        cellsize: 网格边长（米）
        device: 计算设备
    """
    
    def __init__(
        self,
        forcing_dir: str = None,
        flow_dir_path: str = None,
        basin_mask_path: str = None,
        grid_features_path: str = None,
        device: torch.device = None,
        cellsize: float = 1000.0,
        nodata: float = -9999.0,
        config: dict = None,
    ):
        """
        初始化网格数据加载器
        
        Args:
            forcing_dir: 网格强迫数据目录
            flow_dir_path: D8流向文件路径
            basin_mask_path: 子流域掩膜文件路径
            grid_features_path: 网格特征文件路径
            device: 计算设备
            cellsize: 网格边长（米）
            nodata: 无效值标记
            config: 配置字典（可代替单独参数）
        """
        self.device = device or torch.device('cpu')
        self.cellsize = cellsize
        self.nodata = nodata
        
        # 从配置获取路径（如果提供）
        if config is not None:
            grid_config = config.get('data', {}).get('grid', {})
            forcing_dir = forcing_dir or grid_config.get('forcing_dir', 'data/forcing/grid')
            flow_dir_path = flow_dir_path or grid_config.get('flow_dir', 'data/model_attr/flow_dir.tif')
            basin_mask_path = basin_mask_path or grid_config.get('basin_mask', 'data/model_attr/basin_mask.tif')
            grid_features_path = grid_features_path or grid_config.get('grid_features', 'data/dl_input/grid.tif')
            self.cellsize = grid_config.get('cellsize', cellsize)
        
        # 数据路径
        self.forcing_dir = Path(forcing_dir) if forcing_dir else Path('data/forcing/grid')
        self.flow_dir_path = Path(flow_dir_path) if flow_dir_path else Path('data/model_attr/flow_dir.tif')
        self.basin_mask_path = Path(basin_mask_path) if basin_mask_path else Path('data/model_attr/basin_mask.tif')
        self.grid_features_path = Path(grid_features_path) if grid_features_path else Path('data/dl_input/grid.tif')
        
        # 缓存
        self._static_cache = {}
        self._grid_meta = None
        self._H = None
        self._W = None
        
        # 初始化：加载静态数据获取网格尺寸
        self._init_grid_info()
    
    def _init_grid_info(self):
        """从静态数据初始化网格信息"""
        # 尝试读取basin_mask或flow_dir获取网格尺寸
        for fpath in [self.basin_mask_path, self.flow_dir_path]:
            if fpath.exists():
                data, meta = read_tif(str(fpath))
                self._H = data.shape[0]
                self._W = data.shape[1]
                self._grid_meta = meta
                print(f"[GridLoader] Grid size: {self._H} x {self._W}")
                return
        
        # 如果没有静态数据，尝试从forcing数据推断
        for var in ['P', 'E', 'T']:
            var_dir = self.forcing_dir / var
            if var_dir.exists():
                files = list(var_dir.glob('*.tif')) + list(var_dir.glob('*.npy'))
                if files:
                    data, meta = read_tif(str(files[0]))
                    self._H = data.shape[0]
                    self._W = data.shape[1]
                    self._grid_meta = meta
                    print(f"[GridLoader] Grid size inferred from {var}: {self._H} x {self._W}")
                    return
        
        print("[Warning] Could not determine grid size, will infer from data")
    
    @property
    def H(self) -> int:
        return self._H
    
    @property
    def W(self) -> int:
        return self._W
    
    # ==========================================================================
    # 静态数据加载
    # ==========================================================================
    
    def load_flow_dir(self) -> torch.Tensor:
        """
        加载D8流向数据
        
        Returns:
            flow_dir: (H, W) D8流向编码
        """
        if 'flow_dir' in self._static_cache:
            return self._static_cache['flow_dir']
        
        if not self.flow_dir_path.exists():
            raise FileNotFoundError(f"Flow direction file not found: {self.flow_dir_path}")
        
        data, _ = read_tif(str(self.flow_dir_path))
        flow_dir = torch.from_numpy(data.astype(np.int32)).to(self.device)
        self._static_cache['flow_dir'] = flow_dir
        
        return flow_dir
    
    def load_basin_mask(self) -> torch.Tensor:
        """
        加载子流域掩膜
        
        Returns:
            basin_mask: (H, W) 子流域编号（0, 1, 2, ...）对应河道节点r0, r1, r2...
        """
        if 'basin_mask' in self._static_cache:
            return self._static_cache['basin_mask']
        
        if not self.basin_mask_path.exists():
            raise FileNotFoundError(f"Basin mask file not found: {self.basin_mask_path}")
        
        data, _ = read_tif(str(self.basin_mask_path))
        # 处理nodata
        data = np.where(data == self.nodata, -1, data)
        basin_mask = torch.from_numpy(data.astype(np.int32)).to(self.device)
        self._static_cache['basin_mask'] = basin_mask
        
        # 计算子流域数量
        n_basins = int(basin_mask[basin_mask >= 0].max().item()) + 1
        print(f"[GridLoader] Loaded basin_mask: {n_basins} basins (corresponding to r0-r{n_basins-1})")
        
        return basin_mask
    
    def load_static_raster(self, name: str) -> torch.Tensor:
        """
        加载通用静态栅格数据
        
        Args:
            name: 文件名（不含扩展名）
            
        Returns:
            data: (H, W) 或 (C, H, W) tensor
        """
        if name in self._static_cache:
            return self._static_cache[name]
        
        fpath = self.static_dir / f'{name}.tif'
        if not fpath.exists():
            fpath = self.static_dir / f'{name}.npy'
        
        if not fpath.exists():
            raise FileNotFoundError(f"Static raster not found: {name}")
        
        data, _ = read_tif(str(fpath))
        data = np.where(data == self.nodata, np.nan, data)
        tensor = torch.from_numpy(data.astype(np.float32)).to(self.device)
        self._static_cache[name] = tensor
        
        return tensor
    
    def load_grid_features(self) -> torch.Tensor:
        """
        加载网格特征数据（用于参数网络）
        
        Returns:
            features: (C, H, W) 特征张量，C为特征数量
        """
        if 'grid_features' in self._static_cache:
            return self._static_cache['grid_features']
        
        if not self.grid_features_path.exists():
            # 如果没有特征文件，返回占位特征
            print(f"[Warning] Grid features file not found: {self.grid_features_path}, using placeholder")
            if self._H and self._W:
                features = torch.ones(1, self._H, self._W, device=self.device)
            else:
                features = None
            self._static_cache['grid_features'] = features
            return features
        
        data, _ = read_tif(str(self.grid_features_path))
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        
        features = torch.from_numpy(data.astype(np.float32)).to(self.device)
        self._static_cache['grid_features'] = features
        
        return features
    
    def _construct_grid_features(self) -> torch.Tensor:
        """从静态数据构建网格特征"""
        feature_layers = []
        
        # 尝试加载各种可能的特征
        for name in ['dem', 'slope', 'aspect', 'twi', 'land_cover', 'soil_type']:
            try:
                data = self.load_static_raster(name)
                if data.ndim == 2:
                    data = data.unsqueeze(0)
                feature_layers.append(data)
            except FileNotFoundError:
                continue
        
        if feature_layers:
            features = torch.cat(feature_layers, dim=0)
        else:
            # 如果没有任何特征，创建占位特征
            features = torch.ones(1, self._H, self._W, device=self.device)
            print("[Warning] No static features found, using placeholder")
        
        self._static_cache['grid_features'] = features
        return features
    
    def load_static_data(self) -> Dict[str, torch.Tensor]:
        """
        加载所有静态数据
        
        Returns:
            static_data: {
                'flow_dir': (H, W) D8流向编码,
                'basin_mask': (H, W) 子流域掩膜,
                'grid_features': (C, H, W) 网格特征,
            }
        """
        static_data = {}
        
        # 必需的静态数据
        try:
            static_data['flow_dir'] = self.load_flow_dir()
        except FileNotFoundError:
            print("[Warning] flow_dir not found")
        
        try:
            static_data['basin_mask'] = self.load_basin_mask()
        except FileNotFoundError:
            print("[Warning] basin_mask not found")
        
        # 网格特征
        try:
            static_data['grid_features'] = self.load_grid_features()
        except Exception as e:
            print(f"[Warning] Failed to load grid_features: {e}")
        
        return static_data
    
    # ==========================================================================
    # 逐日强迫数据加载
    # ==========================================================================
    
    def _get_date_string(self, date: Union[str, datetime, pd.Timestamp]) -> str:
        """将日期转换为字符串格式 YYYYMMDD"""
        if isinstance(date, str):
            date = pd.to_datetime(date)
        return date.strftime('%Y%m%d')
    
    def load_forcing_single(
        self,
        var: str,
        date: Union[str, datetime, pd.Timestamp]
    ) -> torch.Tensor:
        """
        加载单日强迫数据
        
        Args:
            var: 变量名（P, E, T）
            date: 日期
            
        Returns:
            data: (H, W) tensor
        """
        date_str = self._get_date_string(date)
        
        # 尝试不同的文件命名格式
        var_dir = self.forcing_dir / var
        possible_names = [
            f'{var}_{date_str}.tif',
            f'{date_str}.tif',
            f'{var}_{date_str}.npy',
            f'{date_str}.npy',
        ]
        
        for fname in possible_names:
            fpath = var_dir / fname
            if fpath.exists():
                data, _ = read_tif(str(fpath))
                # 处理nodata
                data = np.where(data == self.nodata, 0.0, data)
                return torch.from_numpy(data.astype(np.float32)).to(self.device)
        
        # 如果找不到文件，返回零值
        print(f"[Warning] Forcing not found: {var}/{date_str}, using zeros")
        return torch.zeros(self._H, self._W, device=self.device)
    
    def load_forcing_period(
        self,
        var: str,
        start_date: Union[str, datetime, pd.Timestamp],
        end_date: Union[str, datetime, pd.Timestamp],
    ) -> torch.Tensor:
        """
        加载时间段内的强迫数据
        
        Args:
            var: 变量名（P, E, T）
            start_date: 起始日期
            end_date: 结束日期
            
        Returns:
            data: (T, H, W) tensor, T为天数
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start, end, freq='D')
        
        data_list = []
        for date in dates:
            data = self.load_forcing_single(var, date)
            data_list.append(data)
        
        return torch.stack(data_list, dim=0)
    
    def load_all_forcing(
        self,
        start_date: Union[str, datetime, pd.Timestamp],
        end_date: Union[str, datetime, pd.Timestamp],
        variables: List[str] = ['P', 'E', 'T'],
    ) -> Dict[str, torch.Tensor]:
        """
        加载时间段内的所有强迫数据
        
        Args:
            start_date: 起始日期
            end_date: 结束日期
            variables: 变量列表
            
        Returns:
            forcing: {var: (T, H, W) tensor}
        """
        forcing = {}
        for var in variables:
            print(f"[GridLoader] Loading {var} from {start_date} to {end_date}...")
            forcing[var] = self.load_forcing_period(var, start_date, end_date)
            print(f"[GridLoader] Loaded {var}: shape={forcing[var].shape}")
        
        return forcing
    
    # ==========================================================================
    # 观测数据
    # ==========================================================================
    
    def load_observations(
        self,
        obs_file: str,
        start_date: Union[str, datetime, pd.Timestamp],
        end_date: Union[str, datetime, pd.Timestamp],
    ) -> Dict[str, torch.Tensor]:
        """
        加载观测流量数据
        
        Args:
            obs_file: 观测数据文件路径（CSV格式）
            start_date: 起始日期
            end_date: 结束日期
            
        Returns:
            observations: {station_id: (T,) tensor}
        """
        df = pd.read_csv(obs_file, index_col=0, parse_dates=True)
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        mask = (df.index >= start) & (df.index <= end)
        df_period = df[mask]
        
        observations = {}
        for col in df_period.columns:
            observations[col] = torch.from_numpy(
                df_period[col].values.astype(np.float32)
            ).to(self.device)
        
        return observations
    
    # ==========================================================================
    # 数据集创建
    # ==========================================================================
    
    def create_dataset(
        self,
        start_date: Union[str, datetime, pd.Timestamp],
        end_date: Union[str, datetime, pd.Timestamp],
        obs_file: Optional[str] = None,
        variables: List[str] = ['P', 'E', 'T'],
    ) -> Dict:
        """
        创建完整的训练/验证数据集
        
        Args:
            start_date: 起始日期
            end_date: 结束日期
            obs_file: 观测数据文件（可选）
            variables: 强迫变量列表
            
        Returns:
            dataset: {
                'forcing': {var: (T, H, W)},
                'static': {
                    'flow_dir': (H, W),
                    'basin_mask': (H, W),
                    'grid_features': (C, H, W),
                },
                'observations': {station: (T,)},  # 如果提供了obs_file
                'dates': DatetimeIndex,
                'cellsize': float,
            }
        """
        dataset = {
            'forcing': self.load_all_forcing(start_date, end_date, variables),
            'static': {
                'flow_dir': self.load_flow_dir(),
                'basin_mask': self.load_basin_mask(),
                'grid_features': self.load_grid_features(),
            },
            'dates': pd.date_range(start_date, end_date, freq='D'),
            'cellsize': self.cellsize,
        }
        
        if obs_file:
            dataset['observations'] = self.load_observations(obs_file, start_date, end_date)
        
        return dataset


# ============================================================================
# 便捷函数
# ============================================================================

def load_grid_data(
    config: dict,
    data_type: str = 'train',
    device: torch.device = None,
) -> Dict:
    """
    根据配置加载网格数据
    
    Args:
        config: 配置字典
        data_type: 'train' 或 'valid'
        device: 计算设备
        
    Returns:
        dataset: 数据集字典
    """
    data_config = config.get('data', {})
    time_config = config.get('time', {})
    
    grid_dir = data_config.get('grid_dir', 'data/grid')
    cellsize = data_config.get('cellsize', 1000.0)
    obs_file = data_config.get('observation', {}).get('Q_outlet', None)
    
    loader = GridDataLoader(grid_dir, device=device, cellsize=cellsize)
    
    # 解析时间段
    periods_key = f'{data_type}_periods'
    if periods_key not in time_config:
        raise ValueError(f"Missing '{periods_key}' in time config")
    
    start_year = time_config['start_year']
    end_year = time_config['end_year']
    
    # 简化：使用整年数据（可根据需要扩展为按月段）
    # 这里假设train_periods格式如 ["3-4", "7-9"]
    all_data = {'forcing': {}, 'static': None, 'dates': [], 'cellsize': cellsize}
    
    for period_str in time_config[periods_key]:
        if '-' in period_str:
            start_month, end_month = map(int, period_str.split('-'))
        else:
            start_month = end_month = int(period_str)
        
        for year in range(start_year, end_year + 1):
            start_date = f'{year}-{start_month:02d}-01'
            if end_month == 12:
                end_date = f'{year}-12-31'
            else:
                # 获取月末
                next_month = datetime(year, end_month + 1, 1) - timedelta(days=1)
                end_date = next_month.strftime('%Y-%m-%d')
            
            dataset = loader.create_dataset(
                start_date, end_date,
                obs_file=obs_file,
                variables=['P', 'E', 'T']
            )
            
            # 合并数据
            if all_data['static'] is None:
                all_data['static'] = dataset['static']
            
            for var, data in dataset['forcing'].items():
                if var not in all_data['forcing']:
                    all_data['forcing'][var] = []
                all_data['forcing'][var].append(data)
            
            all_data['dates'].extend(dataset['dates'].tolist())
    
    # 拼接时间序列
    for var in all_data['forcing']:
        all_data['forcing'][var] = torch.cat(all_data['forcing'][var], dim=0)
    
    all_data['dates'] = pd.DatetimeIndex(all_data['dates'])
    
    if obs_file:
        # 重新加载完整观测数据
        all_data['observations'] = loader.load_observations(
            obs_file,
            all_data['dates'][0],
            all_data['dates'][-1]
        )
    
    return all_data


# ============================================================================
# 示例数据生成（用于测试）
# ============================================================================

def generate_sample_grid_data(
    output_dir: str,
    H: int = 50,
    W: int = 50,
    n_basins: int = 5,
    start_date: str = '2000-01-01',
    end_date: str = '2000-12-31',
):
    """
    生成示例网格数据用于测试
    
    Args:
        output_dir: 输出目录
        H, W: 网格尺寸
        n_basins: 子流域数量
        start_date, end_date: 日期范围
    """
    output_path = Path(output_dir)
    
    # 创建目录结构
    (output_path / 'forcing' / 'P').mkdir(parents=True, exist_ok=True)
    (output_path / 'forcing' / 'E').mkdir(parents=True, exist_ok=True)
    (output_path / 'forcing' / 'T').mkdir(parents=True, exist_ok=True)
    (output_path / 'static').mkdir(parents=True, exist_ok=True)
    
    # 生成静态数据
    # D8流向（简化：都流向东南）
    flow_dir = np.full((H, W), 2, dtype=np.int32)  # 2 = SE
    np.save(str(output_path / 'static' / 'flow_dir.npy'), flow_dir)
    
    # 子流域掩膜
    basin_mask = np.zeros((H, W), dtype=np.int32)
    rows_per_basin = H // n_basins
    for i in range(n_basins):
        basin_mask[i*rows_per_basin:(i+1)*rows_per_basin, :] = i
    np.save(str(output_path / 'static' / 'basin_mask.npy'), basin_mask)
    
    # 高程（简化：从北到南递减）
    dem = np.linspace(3000, 1000, H).reshape(-1, 1).repeat(W, axis=1)
    np.save(str(output_path / 'static' / 'dem.npy'), dem.astype(np.float32))
    
    # 生成逐日强迫数据
    dates = pd.date_range(start_date, end_date, freq='D')
    
    print(f"Generating {len(dates)} days of forcing data...")
    for date in dates:
        date_str = date.strftime('%Y%m%d')
        
        # 降水（带季节变化的随机值）
        doy = date.dayofyear
        P = np.random.exponential(5.0, (H, W)) * (1 + 0.5 * np.sin(2 * np.pi * doy / 365))
        P = np.where(np.random.random((H, W)) > 0.7, P, 0)  # 70%的日子无降水
        np.save(str(output_path / 'forcing' / 'P' / f'P_{date_str}.npy'), P.astype(np.float32))
        
        # 蒸发（与温度相关的正弦变化）
        E = 3.0 + 2.0 * np.sin(2 * np.pi * doy / 365) + np.random.normal(0, 0.5, (H, W))
        E = np.clip(E, 0.5, 8.0)
        np.save(str(output_path / 'forcing' / 'E' / f'E_{date_str}.npy'), E.astype(np.float32))
        
        # 温度（季节变化 + 高程递减）
        T_base = 15 + 15 * np.sin(2 * np.pi * (doy - 80) / 365)
        T = T_base - 0.006 * dem + np.random.normal(0, 2, (H, W))
        np.save(str(output_path / 'forcing' / 'T' / f'T_{date_str}.npy'), T.astype(np.float32))
    
    print(f"Sample grid data generated at: {output_path}")
    print(f"  Grid size: {H} x {W}")
    print(f"  Basins: {n_basins}")
    print(f"  Date range: {start_date} to {end_date}")


if __name__ == '__main__':
    # 生成示例数据
    generate_sample_grid_data(
        output_dir='data/grid',
        H=50,
        W=50,
        n_basins=5,
        start_date='2000-01-01',
        end_date='2000-03-31'
    )
