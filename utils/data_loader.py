"""
数据加载工具模块
统一处理强迫数据、观测数据和流域特征的加载
"""
import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple, List


def parse_time_periods(periods_config: List[str], start_year: int, end_year: int) -> List[Tuple]:
    """
    解析时间段配置
    
    Args:
        periods_config: 时间段配置列表，如 ["3-4", "7-9"]
        start_year: 起始年份
        end_year: 结束年份
        
    Returns:
        periods: [(start_date, end_date), ...]
    """
    periods = []
    
    for period_str in periods_config:
        if '-' in period_str:
            start_month, end_month = map(int, period_str.split('-'))
        else:
            start_month = end_month = int(period_str)
        
        for year in range(start_year, end_year + 1):
            start_date = pd.Timestamp(year=year, month=start_month, day=1)
            
            if end_month == 12:
                end_date = pd.Timestamp(year=year, month=12, day=31)
            else:
                end_date = pd.Timestamp(year=year, month=end_month + 1, day=1) - pd.Timedelta(days=1)
            
            periods.append((start_date, end_date))
    
    return periods


def load_forcing_data(config: dict, data_type: str = 'train') -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    加载强迫数据（降水P和蒸发E）
    
    Args:
        config: 配置字典
        data_type: 'train' 或 'valid'
        
    Returns:
        P: 降水数组 [n_days, n_subbasins]
        E: 蒸发数组 [n_days, n_subbasins]
        dates: 日期索引
    """
    data_config = config['data']
    time_config = config['time']
    
    # 读取数据文件
    P_df = pd.read_csv(data_config['forcing']['precipitation'], index_col=0, parse_dates=True)
    E_df = pd.read_csv(data_config['forcing']['evaporation'], index_col=0, parse_dates=True)
    
    # 解析时间段
    periods_key = f'{data_type}_periods'
    if periods_key in time_config:
        periods = parse_time_periods(
            time_config[periods_key],
            time_config['start_year'],
            time_config['end_year']
        )
    else:
        raise ValueError(f"Missing '{periods_key}' in time config")
    
    # 提取时间段数据
    P_list, E_list, date_list = [], [], []
    for start_date, end_date in periods:
        mask = (P_df.index >= start_date) & (P_df.index <= end_date)
        P_list.append(P_df[mask].values)
        E_list.append(E_df[mask].values)
        date_list.extend(P_df[mask].index.tolist())
    
    P = np.vstack(P_list)
    E = np.vstack(E_list)
    dates = pd.DatetimeIndex(date_list)
    
    print(f"[Data] Loaded {data_type} forcing: {P.shape[0]} days, {P.shape[1]} subbasins")
    
    return P, E, dates


def load_observations(model_attr: dict, dates: pd.DatetimeIndex) -> Dict:
    """
    加载观测站点数据
    
    Args:
        model_attr: 模型属性字典
        dates: 日期索引
        
    Returns:
        observations: {reach_id: (obs_data, weight), ...}
    """
    units_config = model_attr.get('units', {})
    gauges = units_config.get('gauges', {})
    
    obs_file = gauges.get('obs_file')
    if not obs_file:
        raise ValueError("No obs_file found in model_attr!")
    
    # 读取观测数据文件
    Q_obs_df = pd.read_csv(obs_file, index_col=0, parse_dates=True)
    Q_obs_df = Q_obs_df.loc[dates]
    
    observations = {}
    for key, gauge_name in gauges.items():
        if key in ['obs_file', 'comment']:
            continue
        
        unit_id = key
        reach_id = f"r{unit_id}"
        column = gauge_name
        weight = 1.0
        
        if column not in Q_obs_df.columns:
            print(f"[Warning] Column '{column}' not found, skipping gauge '{gauge_name}'")
            continue
        
        Q_obs = Q_obs_df[column].values
        observations[reach_id] = (Q_obs, weight)
        print(f"[Data] Loaded gauge '{gauge_name}' at reach {reach_id}: {len(Q_obs)} days")
    
    if not observations:
        raise ValueError("No valid gauge observations loaded!")
    
    return observations


def load_features(device=None) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    加载流域和河道特征
    
    Args:
        device: torch设备
        
    Returns:
        basin_features: [n_subbasins, n_basin_features]
        river_features: [n_reaches, n_river_features]
        n_subbasins: 子流域数量
        n_reaches: 河道数量
    """
    import pandas as pd
    
    basin_features_df = pd.read_csv('data/dl_input/basin.csv')
    river_features_df = pd.read_csv('data/dl_input/river.csv')
    
    n_subbasins = len(basin_features_df)
    n_reaches = len(river_features_df)
    
    # 转换为tensor
    device = device or torch.device('cpu')
    basin_features = torch.tensor(basin_features_df.values, dtype=torch.float32, device=device)
    river_features = torch.tensor(river_features_df.iloc[:, 1:].values, dtype=torch.float32, device=device)
    
    print(f"[Data] Basin features: {n_subbasins} x {basin_features.shape[1]}")
    print(f"[Data] River features: {n_reaches} x {river_features.shape[1]}")
    
    return basin_features, river_features, n_subbasins, n_reaches
