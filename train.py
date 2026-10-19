"""
水文模型统一训练脚本

支持两种模式（根据mode配置自动选择）：
1. subbasin模式: 子流域产流(XAJ/GR4J等) + 马斯京根河道汇流
2. grid模式: 网格产流(SPHY/VIC等) + D8坡面汇流 + 马斯京根河道汇流

支持单流域和多流域训练：
- basins配置中只有1个流域 → 单流域训练
- basins配置中有多个流域 → 多流域联合训练

Usage:
    python train.py                          # 使用默认配置
    python train.py --config my_config.yaml  # 指定配置文件
"""
from __future__ import annotations

import os
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import argparse
import shutil
import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

from utils.seed import kge, nse, set_global_seed
from utils.param_constraints import apply_constraints
from dl.predictor import DynamicParameterPredictor, load_features
from dl.network_config import NetworkConfig


# ==============================================================================
# 配置相关
# ==============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train hydrological model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    return parser.parse_args()


def load_config(config_path: str = 'config.yaml') -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_config_to_output(config: dict, output_dir: Path):
    """保存配置副本到输出目录"""
    config_copy_path = output_dir / 'config.yaml'
    with open(config_copy_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"[Config] Saved configuration to {config_copy_path}")


def get_mode(config: dict) -> str:
    """获取运行模式"""
    return config.get('mode', 'subbasin')


def is_grid_mode(config: dict) -> bool:
    """是否为网格模式"""
    return get_mode(config) == 'grid'


def get_enabled_basins(config: dict) -> List[dict]:
    """获取所有启用的流域配置"""
    basins = config.get('basins', [])
    return [b for b in basins if b.get('enabled', True)]


def get_basin_data_paths(basin_name: str) -> dict:
    """根据流域名称获取数据路径
    
    数据目录结构：data/流域名称/...
    """
    base_dir = f"data/{basin_name}"
    return {
        'dl_input': f"{base_dir}/dl_input",
        'model_attr': f"{base_dir}/model_attr/model_attr.json",
        'forcing': {
            'precipitation': f"{base_dir}/forcing/P.csv",
            'evaporation': f"{base_dir}/forcing/E.csv",
            'temperature': f"{base_dir}/forcing/T.csv",
        },
        'observation': f"{base_dir}/observation/Q.csv",
    }


def save_named_config_to_output(config: dict, output_dir: Path, filename: str):
    """Save a config copy to a specific YAML filename under the output directory."""
    config_copy_path = output_dir / filename
    with open(config_copy_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"[Config] Saved configuration to {config_copy_path}")


def get_active_component_names(config: dict) -> List[str]:
    """Return active model-component names for the selected run mode."""
    mode = get_mode(config)
    if mode == 'grid':
        return ['grid_runoff', 'hillslope_routing', 'river_routing']
    return ['subbasin_runoff', 'river_routing']


def _normalize_strategy_name(category: str, strategy: str) -> str:
    normalized = str(strategy).strip().lower()
    if category == 'dl':
        aliases = {
            'from_scratch': 'scratch',
            'from-scratch': 'scratch',
            'resume': 'continue',
            'continued': 'continue',
        }
        return aliases.get(normalized, normalized)

    aliases = {
        'de': 'differential_evolution',
        'differential-evolution': 'differential_evolution',
        'manual_calibration': 'manual',
        'manual-calibration': 'manual',
        'hand_tuned': 'manual',
        'hand-tuned': 'manual',
    }
    return aliases.get(normalized, normalized)


def resolve_component_param_modes(config: dict) -> Dict[str, ResolvedParamMode]:
    """Parse and validate the new param_mode schema for active components only."""
    components = config.get('model', {}).get('components', {})
    resolved: Dict[str, ResolvedParamMode] = {}

    for component_name in get_active_component_names(config):
        component_cfg = components.get(component_name)
        if not component_cfg:
            raise ValueError(f"Missing component config for '{component_name}'")

        model_name = str(component_cfg.get('model', component_name))
        param_mode_cfg = component_cfg.get('param_mode')
        if not isinstance(param_mode_cfg, dict):
            raise ValueError(
                f"Component '{component_name}' must use the new param_mode mapping, "
                "for example {category: dl, strategy: scratch}."
            )

        category = str(param_mode_cfg.get('category', '')).strip().lower()
        strategy = _normalize_strategy_name(category, param_mode_cfg.get('strategy', ''))

        if category not in {'dl', 'traditional'}:
            raise ValueError(
                f"Component '{component_name}' has invalid param_mode.category={category!r}. "
                "Expected 'dl' or 'traditional'."
            )

        checkpoint_dir = param_mode_cfg.get('checkpoint_dir')
        runtime_param_mode = 'fixed'

        if category == 'dl':
            if strategy not in {'scratch', 'continue'}:
                raise ValueError(
                    f"Component '{component_name}' has invalid dl strategy={strategy!r}. "
                    "Expected 'scratch' or 'continue'."
                )
            if strategy == 'continue' and not checkpoint_dir:
                raise ValueError(
                    f"Component '{component_name}' uses dl/continue but checkpoint_dir is missing."
                )
            runtime_param_mode = 'trainable'
        else:
            if strategy not in {'manual', 'differential_evolution'}:
                raise ValueError(
                    f"Component '{component_name}' has invalid traditional strategy={strategy!r}. "
                    "Expected 'manual' or 'differential_evolution'."
                )
            runtime_param_mode = 'fixed' if strategy == 'manual' else 'trainable'

        resolved[component_name] = ResolvedParamMode(
            component=component_name,
            model_name=model_name,
            category=category,
            strategy=strategy,
            runtime_param_mode=runtime_param_mode,
            checkpoint_dir=str(checkpoint_dir) if checkpoint_dir else None,
        )

    return resolved


def build_runtime_config(config: dict, resolved_modes: Dict[str, ResolvedParamMode]) -> dict:
    """Build a low-level runtime config that current model code can consume."""
    runtime_config = copy.deepcopy(config)
    components = runtime_config.setdefault('model', {}).setdefault('components', {})

    for component_name, resolved in resolved_modes.items():
        component_cfg = components.setdefault(component_name, {})
        component_cfg['param_mode'] = resolved.runtime_param_mode

    runtime_config.setdefault('training', {}).pop('resume', None)
    return runtime_config


def summarize_param_modes(resolved_modes: Dict[str, ResolvedParamMode]) -> None:
    """Print a concise summary of the normalized training modes."""
    print("\n[ParamMode] Active component plan")
    for component_name, resolved in resolved_modes.items():
        extra = ""
        if resolved.checkpoint_dir:
            extra = f", checkpoint={resolved.checkpoint_dir}"
        print(
            f"  - {component_name}: {resolved.category}/{resolved.strategy} "
            f"-> runtime={resolved.runtime_param_mode}{extra}"
        )


def determine_training_runner(
    config: dict,
    resolved_modes: Dict[str, ResolvedParamMode],
) -> Tuple[str, Optional[str]]:
    """Choose the top-level runner for the current configuration."""
    active_modes = [resolved_modes[name] for name in get_active_component_names(config)]
    traditional_optimizers = [
        mode for mode in active_modes
        if mode.category == 'traditional' and mode.strategy != 'manual'
    ]

    if traditional_optimizers:
        if get_mode(config) != 'subbasin':
            raise ValueError(
                "Optimization-based traditional calibration is currently supported in subbasin mode only."
            )
        if any(mode.category == 'dl' for mode in active_modes):
            raise ValueError(
                "Mixing optimization-based traditional calibration with DL-updated components "
                "in one train run is not supported yet."
            )
        if any(mode.category == 'traditional' and mode.strategy == 'manual' for mode in active_modes):
            raise ValueError(
                "Mixing manual fixed parameters with optimization-based traditional calibration "
                "in one train run is not supported yet."
            )

        strategies = {mode.strategy for mode in traditional_optimizers}
        if len(strategies) != 1:
            raise ValueError(
                "All optimization-based traditional components must use the same strategy."
            )
        return 'traditional', strategies.pop()

    if all(mode.category == 'traditional' and mode.strategy == 'manual' for mode in active_modes):
        return 'manual_only', None

    return 'dl', None


def get_traditional_method_script(method: str) -> Path:
    """Resolve the per-method traditional calibration script path."""
    method_map = {
        'manual': Path(__file__).with_name('traditional_methods') / 'manual.py',
        'differential_evolution': Path(__file__).with_name('traditional_methods') / 'differential_evolution.py',
    }
    script_path = method_map.get(method)
    if script_path is None:
        raise ValueError(f"Unsupported traditional calibration method: {method!r}")
    if not script_path.exists():
        raise FileNotFoundError(f"Traditional method script not found: {script_path}")
    return script_path


# ==============================================================================
# 数据结构
# ==============================================================================

@dataclass
class BasinData:
    """单个流域的数据容器"""
    name: str
    description: str
    weight: float
    role: str  # both/train/valid
    
    # 模型和属性
    model: Any  # HydroModel
    model_attr: dict
    features: Dict[str, torch.Tensor]
    
    # 训练数据
    P_train: np.ndarray
    E_train: np.ndarray
    T_train: Optional[np.ndarray] = None
    dates_train: Optional[pd.DatetimeIndex] = None
    obs_train: Optional[Dict[str, Any]] = None
    
    # 验证数据
    P_valid: Optional[np.ndarray] = None
    E_valid: Optional[np.ndarray] = None
    T_valid: Optional[np.ndarray] = None
    dates_valid: Optional[pd.DatetimeIndex] = None
    obs_valid: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ResolvedParamMode:
    """Normalized param-mode settings for one active component."""

    component: str
    model_name: str
    category: str
    strategy: str
    runtime_param_mode: str
    checkpoint_dir: Optional[str] = None


# ==============================================================================
# 数据加载
# ==============================================================================

def parse_time_periods(years_config):
    """解析时间段配置（按年）"""
    periods = []
    for year in years_config:
        start_date = pd.Timestamp(year=year, month=1, day=1)
        end_date = pd.Timestamp(year=year, month=12, day=31)
        periods.append((start_date, end_date))
    return periods


def load_forcing_data_for_basin(basin_config: dict, global_config: dict, data_type: str = 'train'):
    """加载单个流域的强迫数据"""
    basin_name = basin_config['name']
    data_paths = get_basin_data_paths(basin_name)
    time_config = basin_config.get('time', {})
    evap_config = global_config.get('model', {}).get('evaporation', {})
    
    # 读取降水数据
    P_df = pd.read_csv(data_paths['forcing']['precipitation'], index_col=0, parse_dates=True)
    
    # 读取温度数据（如果存在）
    T_df = None
    temp_file = data_paths['forcing']['temperature']
    if os.path.exists(temp_file):
        T_df = pd.read_csv(temp_file, index_col=0, parse_dates=True)
    
    # 处理蒸发数据
    use_input_evap = evap_config.get('use_input', True)
    if use_input_evap:
        E_df = pd.read_csv(data_paths['forcing']['evaporation'], index_col=0, parse_dates=True)
    else:
        pet_method = evap_config.get('pet_method', 'hargreaves')
        if T_df is None:
            raise ValueError(f"Basin {basin_name}: Temperature data required for PET calculation!")
        from models.physics.pet import compute_pet
        E_df = compute_pet(T_df, method=pet_method)
    
    # 解析时间段
    if data_type == 'train':
        periods = parse_time_periods(time_config['train_years'])
    else:
        periods = parse_time_periods(time_config['valid_years'])

    # 训练期和验证期都前置 warmup 段，避免首年被统一 warmup 裁剪影响。
    warmup_days = int(time_config.get('warmup_days', 365))
    if periods and warmup_days > 0:
        first_start, first_end = periods[0]
        periods[0] = (first_start - pd.Timedelta(days=warmup_days), first_end)
    
    # 提取时间段数据
    P_list, E_list, T_list, date_list = [], [], [], []
    
    for start_date, end_date in periods:
        mask = (P_df.index >= start_date) & (P_df.index <= end_date)
        P_list.append(P_df[mask].values)
        E_list.append(E_df[mask].values)
        if T_df is not None:
            T_list.append(T_df[mask].values)
        date_list.extend(P_df[mask].index.tolist())
    
    P = np.vstack(P_list)
    E = np.vstack(E_list)
    T = np.vstack(T_list) if T_list else None
    dates = pd.DatetimeIndex(date_list)
    
    print(f"  {data_type.capitalize()}: {P.shape[0]} days, {P.shape[1]} subbasins")
    if len(dates) > 0:
        target_years = time_config.get('train_years', []) if data_type == 'train' else time_config.get('valid_years', [])
        warmup_days = int(time_config.get('warmup_days', 365))
        print(f"  {data_type.capitalize()} target years: {target_years}")
        print(
            f"  {data_type.capitalize()} loaded period (with warmup={warmup_days}d): "
            f"{dates[0].date()} -> {dates[-1].date()}"
        )
    
    return P, E, T, dates


def load_observations_for_basin(model_attr: dict, dates: pd.DatetimeIndex, 
                                basin_config: dict, data_type: str = 'train'):
    """加载单个流域的观测数据
    
    自动处理：
    1. 站点在config中定义但Q.csv中不存在 → 跳过
    2. 站点数据部分缺测（NaN）→ 保留，在计算loss时自动掩膜
    3. 站点在该时段全部缺测 → 跳过
    """
    try:
        units_config = model_attr.get('units', {})
        gauges = units_config.get('gauges', {})
        
        obs_file = gauges.get('obs_file')
        if not obs_file:
            return None
        
        if not os.path.exists(obs_file):
            print(f"  [Warning] Observation file not found: {obs_file}")
            return None
        
        Q_obs_df = pd.read_csv(obs_file, index_col=0, parse_dates=True)
        available_columns = set(Q_obs_df.columns)
        
        # 只提取dates范围内的数据（用reindex确保日期完全匹配，缺失自动填NaN）
        Q_obs_df = Q_obs_df.reindex(dates)
        
        observations = {}
        skipped_no_column = []
        skipped_all_nan = []
        
        for key, gauge_name in gauges.items():
            if key in ['obs_file', 'comment']:
                continue
            
            unit_id = key
            reach_id = f"r{unit_id}"
            
            # 检查站点是否在Q.csv中
            if gauge_name not in available_columns:
                skipped_no_column.append(gauge_name)
                continue
            
            Q_obs = Q_obs_df[gauge_name].values.astype(np.float64)
            n_total = len(Q_obs)
            n_valid = np.sum(~np.isnan(Q_obs))
            n_nan = n_total - n_valid
            
            # 该时段全部缺测 → 跳过
            if n_valid == 0:
                skipped_all_nan.append(gauge_name)
                continue
            
            observations[reach_id] = (Q_obs, 1.0)
            
            if n_nan > 0:
                valid_pct = n_valid / n_total * 100
                print(f"    Loaded gauge '{gauge_name}' at {reach_id}: "
                      f"{n_valid}/{n_total} valid days ({valid_pct:.1f}%), "
                      f"{n_nan} NaN will be auto-masked")
            else:
                print(f"    Loaded gauge '{gauge_name}' at {reach_id}: {n_total} days (complete)")
        
        if skipped_no_column:
            print(f"  [Info] {len(skipped_no_column)} gauge(s) not in {obs_file}: "
                  f"{', '.join(skipped_no_column)}")
        if skipped_all_nan:
            print(f"  [Info] {len(skipped_all_nan)} gauge(s) have no valid data in "
                  f"{data_type} period: {', '.join(skipped_all_nan)}")
        
        return observations if observations else None
    except Exception as e:
        print(f"  [Warning] Failed to load observation data: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_basin_data(basin_config: dict, global_config: dict) -> BasinData:
    """加载单个流域的完整数据"""
    from models.hydro_model import HydroModel
    
    basin_name = basin_config['name']
    description = basin_config.get('description', '')
    weight = basin_config.get('weight', 1.0)
    role = str(basin_config.get('role', 'both')).lower()
    if role not in {'both', 'train', 'valid'}:
        raise ValueError(f"Basin {basin_name}: invalid role '{role}', expected one of both/train/valid")
    
    print(f"\n{'='*60}")
    print(f"Loading basin: {basin_name}")
    print(f"{'='*60}")
    
    # 获取数据路径
    data_paths = get_basin_data_paths(basin_name)
    
    # 创建模型（传入model_attr_path用于fixed模式加载参数）
    model_config = global_config.copy()
    model_config['_model_attr_path'] = data_paths['model_attr']
    model = HydroModel(data_paths['model_attr'], config=model_config)
    print(f"[Model] Created HydroModel with {model.n_subbasins} subbasins")
    
    # 加载特征
    features = load_features(data_paths['dl_input'])
    print(f"[Features] Loaded from {data_paths['dl_input']}")
    for name, tensor in features.items():
        if name.endswith('_nan_mask'):
            continue
        print(f"  {name}: {tensor.shape}")
    
    # 加载训练数据
    print(f"\n[Data] Loading training data...")
    P_train, E_train, T_train, dates_train = load_forcing_data_for_basin(
        basin_config, global_config, 'train'
    )
    obs_train = load_observations_for_basin(
        model.model_attr, dates_train, basin_config, 'train'
    )
    if obs_train:
        print(f"  Observation gauges: {len(obs_train)}")
    
    # 加载验证数据
    print(f"\n[Data] Loading validation data...")
    P_valid, E_valid, T_valid, dates_valid = load_forcing_data_for_basin(
        basin_config, global_config, 'valid'
    )
    obs_valid = load_observations_for_basin(
        model.model_attr, dates_valid, basin_config, 'valid'
    )
    
    return BasinData(
        name=basin_name,
        description=description,
        weight=weight,
        role=role,
        model=model,
        model_attr=model.model_attr,
        features=features,
        P_train=P_train,
        E_train=E_train,
        T_train=T_train,
        dates_train=dates_train,
        obs_train=obs_train,
        P_valid=P_valid,
        E_valid=E_valid,
        T_valid=T_valid,
        dates_valid=dates_valid,
        obs_valid=obs_valid
    )


# ==============================================================================
# 模型创建
# ==============================================================================

def create_model(config: dict, basin_name: str = None):
    """创建水文模型
    
    Args:
        config: 配置字典
        basin_name: 流域名称，如果为None则使用第一个启用的流域
    """
    if basin_name is None:
        enabled_basins = get_enabled_basins(config)
        if enabled_basins:
            basin_name = enabled_basins[0]['name']
        else:
            raise ValueError("No enabled basins found!")
    
    data_paths = get_basin_data_paths(basin_name)
    mode = get_mode(config)
    
    if mode == 'grid':
        from models.grid_hydro_model import GridHydroModel
        model = GridHydroModel(data_paths['model_attr'], config=config)
        print(f"[Model] Created GridHydroModel")
    else:
        from models.hydro_model import HydroModel
        model = HydroModel(data_paths['model_attr'], config=config)
        print(f"[Model] Created HydroModel with {model.n_subbasins} subbasins")
    
    return model


# ==============================================================================
# 训练核心
# ==============================================================================

def _save_run_configs(raw_config: dict, runtime_config: dict, output_dir: Path):
    """Persist both the user-facing config and the generated runtime config."""
    save_named_config_to_output(raw_config, output_dir, 'config_raw.yaml')
    save_config_to_output(runtime_config, output_dir)


def _load_continue_checkpoints(
    predictor: DynamicParameterPredictor,
    resolved_modes: Dict[str, ResolvedParamMode],
) -> None:
    """Load per-component DL checkpoints for components configured as dl/continue."""
    loaded_count = 0
    for component_name, resolved in resolved_modes.items():
        if resolved.category != 'dl' or resolved.strategy != 'continue':
            continue
        if component_name not in predictor.networks:
            print(f"[Resume] Network '{component_name}' not built; skip checkpoint loading")
            continue

        checkpoint_dir = Path(resolved.checkpoint_dir)
        net_file = checkpoint_dir / f"{component_name}_net.pt"
        if not net_file.exists():
            raise FileNotFoundError(
                f"Checkpoint file not found for component '{component_name}': {net_file}"
            )

        predictor.networks[component_name].load_state_dict(
            torch.load(str(net_file), weights_only=True)
        )
        loaded_count += 1
        print(f"[Resume] Loaded {net_file}")

    if loaded_count > 0:
        print(f"[Resume] Loaded {loaded_count} component network(s) for continuation")


def _build_model_configs(
    runtime_config: dict,
    resolved_modes: Dict[str, ResolvedParamMode],
) -> Dict[str, dict]:
    """Build predictor model configs from resolved component runtime modes."""
    model_configs = {}
    components = runtime_config.get('model', {}).get('components', {})

    for component_name in get_active_component_names(runtime_config):
        component_cfg = components.get(component_name, {})
        model_name = str(component_cfg.get('model', component_name))
        runtime_mode = resolved_modes[component_name].runtime_param_mode

        if runtime_mode == 'fixed':
            print(f"[Config] {component_name} ({model_name}): fixed/manual")
            continue

        model_configs[model_name] = {
            'param_mode': 'trainable',
            'pretrained_path': '',
        }
        print(
            f"[Config] {component_name} ({model_name}): "
            f"{resolved_modes[component_name].category}/{resolved_modes[component_name].strategy}"
        )

    return model_configs


def train_dl(
    raw_config: dict,
    runtime_config: dict,
    resolved_modes: Dict[str, ResolvedParamMode],
):
    """统一训练入口 - 支持单流域和多流域训练
    
    Args:
        config: 配置字典
        
    Returns:
        predictor: 训练后的参数预测器
        history: 训练历史
    """
    # 设置随机种子
    config = runtime_config
    set_global_seed(config['training']['seed'])
    
    # 获取启用的流域
    enabled_basins = get_enabled_basins(config)
    
    if not enabled_basins:
        raise ValueError("No enabled basins found in configuration!")
    
    # 显示模式
    if len(enabled_basins) == 1:
        print(f"\n[Train] Running in SINGLE-BASIN mode: {enabled_basins[0]['name']}")
    else:
        print(f"\n[Train] Running in MULTI-BASIN mode ({len(enabled_basins)} basins)")
    print("=" * 80)
    
    # 输出目录
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    _save_run_configs(raw_config, runtime_config, output_dir)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() and config['training']['use_gpu'] else 'cpu')
    print(f"[Train] Using device: {device}")
    
    # 加载所有流域数据
    basins: Dict[str, BasinData] = {}
    for basin_config in enabled_basins:
        basin_data = load_basin_data(basin_config, config)
        basins[basin_data.name] = basin_data

    train_basins: Dict[str, BasinData] = {
        name: data for name, data in basins.items()
        if data.role in {'both', 'train'}
    }
    valid_basins: Dict[str, BasinData] = {
        name: data for name, data in basins.items()
        if data.role in {'both', 'valid'}
    }

    if not train_basins:
        raise ValueError("No basins available for training (role in {'both', 'train'})")
    if not valid_basins:
        raise ValueError("No basins available for validation (role in {'both', 'valid'})")
    
    print(f"\n{'='*60}")
    print(f"Loaded {len(basins)} basins for training")
    print(f"Train basins: {list(train_basins.keys())}")
    print(f"Valid basins: {list(valid_basins.keys())}")
    print(f"{'='*60}")
    
    # 创建共享参数预测器
    print("\n[Network] Creating shared parameter predictor...")
    
    # 从components配置获取模型名称和参数模式
    components = config.get('model', {}).get('components', {})
    mode = config.get('mode', 'subbasin')
    
    # 构建模型配置字典（每个模型独立配置）
    model_configs = {}
    
    if mode == 'subbasin':
        # subbasin模式：subbasin_runoff + river_routing
        subbasin_config = components.get('subbasin_runoff', {})
        river_config = components.get('river_routing', {})
        
        subbasin_model_name = subbasin_config.get('model', 'xaj')
        river_model_name = river_config.get('model', 'muskingum')
        
        # 添加产流模型配置
        if subbasin_config.get('param_mode', 'trainable') != 'fixed':
            model_configs[subbasin_model_name] = {
                'param_mode': subbasin_config.get('param_mode', 'trainable'),
                'pretrained_path': subbasin_config.get('pretrained_path', ''),
            }
        
        # 添加河道汇流模型配置
        if river_config.get('param_mode', 'trainable') != 'fixed':
            model_configs[river_model_name] = {
                'param_mode': river_config.get('param_mode', 'trainable'),
                'pretrained_path': river_config.get('pretrained_path', ''),
            }
        
        print(f"[Config] {subbasin_model_name}: {subbasin_config.get('param_mode', 'trainable')}")
        print(f"[Config] {river_model_name}: {river_config.get('param_mode', 'trainable')}")
        
    else:  # grid模式
        # grid模式：grid_runoff + hillslope_routing + river_routing
        grid_config = components.get('grid_runoff', {})
        hillslope_config = components.get('hillslope_routing', {})
        river_config = components.get('river_routing', {})
        
        grid_model_name = grid_config.get('model', 'sphy')
        hillslope_model_name = hillslope_config.get('model', 'd8_travel_time')
        river_model_name = river_config.get('model', 'muskingum')
        
        # 添加网格产流模型配置
        if grid_config.get('param_mode', 'trainable') != 'fixed':
            model_configs[grid_model_name] = {
                'param_mode': grid_config.get('param_mode', 'trainable'),
                'pretrained_path': grid_config.get('pretrained_path', ''),
            }
        
        # 添加坡面汇流模型配置
        if hillslope_config.get('param_mode', 'trainable') != 'fixed':
            model_configs[hillslope_model_name] = {
                'param_mode': hillslope_config.get('param_mode', 'trainable'),
                'pretrained_path': hillslope_config.get('pretrained_path', ''),
            }
        
        # 添加河道汇流模型配置
        if river_config.get('param_mode', 'trainable') != 'fixed':
            model_configs[river_model_name] = {
                'param_mode': river_config.get('param_mode', 'trainable'),
                'pretrained_path': river_config.get('pretrained_path', ''),
            }
        
        print(f"[Config] {grid_model_name}: {grid_config.get('param_mode', 'trainable')}")
        print(f"[Config] {hillslope_model_name}: {hillslope_config.get('param_mode', 'trainable')}")
        print(f"[Config] {river_model_name}: {river_config.get('param_mode', 'trainable')}")
    
    # 冰冻圈配置
    cold_region_config = config.get('model', {}).get('cold_region', {})
    enable_cold_region = cold_region_config.get('enabled', False)
    
    # 降水校正配置
    precip_correct_config = config.get('model', {}).get('precip_correct', {})
    enable_precip_correct = precip_correct_config.get('enabled', False)
    
    # 使用第一个流域的特征维度初始化
    first_basin = list(basins.values())[0]
    first_basin_name = enabled_basins[0]['name']
    first_basin_dl_input = f"data/{first_basin_name}/dl_input"
    
    # 创建预测器（按模型的param_mode自动构建网络或加载pretrained）
    model_configs = _build_model_configs(config, resolved_modes)
    predictor = DynamicParameterPredictor(
        model_configs=model_configs,
        data_dir=first_basin_dl_input,
        enable_cold_region=enable_cold_region,
        enable_precip_correct=enable_precip_correct,
        verbose=True
    )
    
    # 收集可训练网络参数（只有 trainable 模式的参数加入优化器）
    param_list = []
    trainable_networks = []
    
    for net_name, network in predictor.networks.items():
        network.to(device)
        param_list.extend(list(network.parameters()))
        trainable_networks.append(net_name)
    
    # pretrained 参数已经被冻结，不需要额外处理
    pretrained_params = list(predictor.pretrained_predictors.keys())
    
    if trainable_networks:
        print(f"[Network] Trainable networks: {trainable_networks}")
    if pretrained_params:
        print(f"[Network] Pretrained parameters: {len(pretrained_params)} params frozen")
    
    total_params = sum(p.numel() for p in param_list)
    print(f"[Network] Total trainable parameters: {total_params}")
    
    # 检查是否有可训练参数
    if total_params == 0 and not pretrained_params:
        print("\n" + "=" * 60)
        print("[Warning] No trainable parameters!")
        print("All components are in 'fixed' mode.")
        print("This is a simulation-only configuration, no training needed.")
    elif total_params == 0 and pretrained_params:
        print("\n" + "=" * 60)
        print("[Warning] No trainable parameters!")
        print(f"All models are in 'pretrained' mode (frozen).")
        print("No training will occur, but simulation will use pretrained networks.")
        print("Please use simulate.py for running simulations.")
        print("=" * 60)
        return predictor, {}
    
    # 断点续训：从已有权重加载（保持trainable，不冻结）
    resume_config = config['training'].get('resume', {})
    if resume_config.get('enabled', False):
        resume_path = resume_config.get('path', '')
        if resume_path and Path(resume_path).exists():
            print(f"\n[Resume] Loading weights from: {resume_path}")
            loaded_count = 0
            for net_name, network in predictor.networks.items():
                net_file = Path(resume_path) / f"{net_name}_net.pt"
                if net_file.exists():
                    network.load_state_dict(torch.load(str(net_file), weights_only=True))
                    loaded_count += 1
                    print(f"  [Resume] Loaded {net_name}_net.pt")
                else:
                    print(f"  [Resume] {net_file.name} not found, using random init")
            print(f"[Resume] Loaded {loaded_count}/{len(predictor.networks)} networks (trainable, not frozen)")
        else:
            print(f"[Resume] Path '{resume_path}' not found, training from scratch")
    
    # 优化器
    _load_continue_checkpoints(predictor, resolved_modes)
    optimizer = torch.optim.Adam(param_list, lr=config['training']['learning_rate'])
    
    # 学习率调度器
    lr_config = config['training'].get('lr_scheduler', {})
    scheduler = None
    if lr_config.get('enabled', False):
        sched_type = lr_config.get('type', 'reduce_on_plateau')
        if sched_type == 'reduce_on_plateau':
            sched_kwargs = {
                'mode': 'max',
                'factor': lr_config.get('factor', 0.5),
                'patience': lr_config.get('patience', 10),
            }
            try:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, verbose=True, **sched_kwargs
                )
            except TypeError:
                # Compatible with torch versions where ReduceLROnPlateau has no `verbose` argument.
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, **sched_kwargs
                )
            print(f"[LR] ReduceOnPlateau: factor={lr_config.get('factor', 0.5)}, patience={lr_config.get('patience', 10)}")
    
    # 训练历史
    history = {
        'epoch': [],
        'train_loss': [],
        'train_kge': {basin_name: [] for basin_name in basins.keys()},
        'valid_kge': {basin_name: [] for basin_name in basins.keys()},
        'avg_train_kge': [],
        'avg_valid_kge': [],
        # 每站（每年）KGE: list of dicts，每epoch一个dict
        'station_kge_train': [],  # [{col: val, ...}, ...]
        'station_kge_valid': [],
    }
    
    best_kge = -np.inf
    best_networks = None
    patience_counter = 0
    
    # 训练配置
    epochs = config['training']['epochs']
    valid_interval = config['training'].get('valid_interval', 10)
    warmup = first_basin.model_attr.get('units', {}).get('warmup_days', 
             config.get('time', {}).get('warmup_days', 365))
    strategy = config.get('multi_basin', {}).get('training_strategy', 'sequential')
    kge_mode = config['training'].get('kge_mode', 'series')  # 'series' 或 'yearly'
    
    print(f"\n{'='*80}")
    print(f"Starting Training")
    print(f"Strategy: {strategy}")
    print(f"KGE mode: {kge_mode}")
    print(f"Basins: {list(basins.keys())}")
    print(f"Train basins: {list(train_basins.keys())}")
    print(f"Valid basins: {list(valid_basins.keys())}")
    print(f"Epochs: {epochs}")
    print(f"{'='*80}\n")
    
    # 构建 reach_id -> 站点名的映射（用于 KGE 日志列名）
    basin_gauge_names = {}  # {basin_name: {reach_id: gauge_name}}
    for bn, bd in basins.items():
        gauges = bd.model_attr.get('units', {}).get('gauges', {})
        gn = {}
        for key, gauge_name in gauges.items():
            if key in ['obs_file', 'comment']:
                continue
            gn[f"r{key}"] = gauge_name
        basin_gauge_names[bn] = gn
    
    # 训练循环
    for epoch in range(epochs):
        # 切换到训练模式（启用dropout、batchnorm训练行为）
        for net in predictor.networks.values():
            net.train()
        
        if strategy == 'sequential':
            # 顺序训练：每个流域依次训练并更新
            total_loss = 0.0
            train_kge_per_basin = {}
            epoch_train_station_kge = {}  # 本 epoch 训练站点KGE
            updated_basins = 0
            
            for basin_name, basin_data in train_basins.items():
                # 预测参数
                predicted_params = predictor.predict_params(basin_data.features, training=True)
                
                # 构建参数字典
                params_dict = _build_params_dict(basin_data, predicted_params)
                
                # 前向传播
                forcing = {'P': basin_data.P_train, 'E': basin_data.E_train}
                if basin_data.T_train is not None:
                    forcing['T'] = basin_data.T_train
                
                outputs, _ = basin_data.model.forward(forcing, params=params_dict, device=device)
                
                # 计算损失
                basin_loss, basin_kge, train_yearly = _compute_loss(
                    outputs, basin_data.obs_train, warmup, device,
                    dates=basin_data.dates_train, kge_mode=kge_mode,
                    gauge_names=basin_gauge_names.get(basin_name, {})
                )
                
                # 收集本epoch训练站点KGE
                _collect_station_kge(epoch_train_station_kge, train_yearly, kge_mode)
                
                weighted_loss = basin_loss * basin_data.weight
                if not torch.isfinite(weighted_loss).item():
                    print(f"  [Warn] {basin_name}: non-finite loss detected, skip this basin update")
                    _log_nonfinite_diagnostics(
                        basin_name,
                        outputs,
                        basin_data.obs_train,
                        warmup,
                        predicted_params,
                    )
                    train_kge_per_basin[basin_name] = float('nan')
                    continue

                total_loss += weighted_loss.item()
                train_kge_per_basin[basin_name] = basin_kge
                updated_basins += 1
                
                # 反向传播
                optimizer.zero_grad()
                weighted_loss.backward()
                if _sanitize_grads(param_list):
                    print(f"  [Warn] {basin_name}: non-finite gradients detected, sanitized to zero")
                torch.nn.utils.clip_grad_norm_(param_list, max_norm=5.0)
                optimizer.step()
            
            avg_loss = total_loss / max(updated_basins, 1)
            history['station_kge_train'].append(epoch_train_station_kge)
            
        else:  # mixed
            # 混合训练：累积所有流域梯度后统一更新
            optimizer.zero_grad()
            total_loss = 0.0
            train_kge_per_basin = {}
            epoch_train_station_kge = {}  # 本 epoch 训练站点KGE
            updated_basins = 0
            
            for basin_name, basin_data in train_basins.items():
                predicted_params = predictor.predict_params(basin_data.features, training=True)
                params_dict = _build_params_dict(basin_data, predicted_params)
                
                forcing = {'P': basin_data.P_train, 'E': basin_data.E_train}
                if basin_data.T_train is not None:
                    forcing['T'] = basin_data.T_train
                
                outputs, _ = basin_data.model.forward(forcing, params=params_dict, device=device)
                
                basin_loss, basin_kge, train_yearly = _compute_loss(
                    outputs, basin_data.obs_train, warmup, device,
                    dates=basin_data.dates_train, kge_mode=kge_mode,
                    gauge_names=basin_gauge_names.get(basin_name, {})
                )
                
                # 收集本epoch训练站点KGE
                _collect_station_kge(epoch_train_station_kge, train_yearly, kge_mode)
                
                # 累积梯度
                weighted_loss = basin_loss * basin_data.weight / len(train_basins)
                if not torch.isfinite(weighted_loss).item():
                    print(f"  [Warn] {basin_name}: non-finite loss detected, skip this basin in mixed update")
                    _log_nonfinite_diagnostics(
                        basin_name,
                        outputs,
                        basin_data.obs_train,
                        warmup,
                        predicted_params,
                    )
                    train_kge_per_basin[basin_name] = float('nan')
                    continue
                weighted_loss.backward()
                if _sanitize_grads(param_list):
                    print(f"  [Warn] {basin_name}: non-finite gradients detected, sanitized to zero")
                
                total_loss += basin_loss.item() * basin_data.weight
                train_kge_per_basin[basin_name] = basin_kge
                updated_basins += 1
            
            if updated_basins > 0:
                torch.nn.utils.clip_grad_norm_(param_list, max_norm=5.0)
                optimizer.step()
            else:
                print("  [Warn] No valid basin update in this epoch (mixed strategy)")
            
            denom_weight = sum(b.weight for b in train_basins.values()) if updated_basins > 0 else 1.0
            avg_loss = total_loss / denom_weight
            history['station_kge_train'].append(epoch_train_station_kge)
        
        # 记录训练结果
        history['epoch'].append(epoch)
        history['train_loss'].append(avg_loss)
        avg_train_kge = _safe_mean(list(train_kge_per_basin.values()))
        history['avg_train_kge'].append(avg_train_kge)
        
        for basin_name in basins.keys():
            history['train_kge'][basin_name].append(train_kge_per_basin.get(basin_name, float('nan')))
        
        # 验证
        if (epoch + 1) % valid_interval == 0:
            # 切换到评估模式（关闭dropout、使用累积batchnorm统计）
            for net in predictor.networks.values():
                net.eval()
            
            valid_kge_per_basin = {}
            epoch_valid_station_kge = {}  # 本 epoch 验证站点KGE
            
            with torch.no_grad():
                for basin_name, basin_data in valid_basins.items():
                    predicted_params = predictor.predict_params(basin_data.features, training=False)
                    params_dict = _build_params_dict(basin_data, predicted_params)
                    
                    forcing = {'P': basin_data.P_valid, 'E': basin_data.E_valid}
                    if basin_data.T_valid is not None:
                        forcing['T'] = basin_data.T_valid
                    
                    outputs, _ = basin_data.model.forward(forcing, params=params_dict, device=device)
                    
                    _, basin_kge, basin_yearly = _compute_loss(
                        outputs, basin_data.obs_valid, warmup, device,
                        dates=basin_data.dates_valid, kge_mode=kge_mode,
                        gauge_names=basin_gauge_names.get(basin_name, {})
                    )
                    if not np.isfinite(basin_kge):
                        print(f"  [Warn] {basin_name}: non-finite validation KGE detected")
                        _log_nonfinite_diagnostics(
                            basin_name,
                            outputs,
                            basin_data.obs_valid,
                            warmup,
                            predicted_params,
                        )
                    valid_kge_per_basin[basin_name] = basin_kge
                    
                    # 收集本epoch验证站点KGE
                    _collect_station_kge(epoch_valid_station_kge, basin_yearly, kge_mode)
                    
                    # 输出每站每年KGE详情
                    if basin_yearly and kge_mode == 'yearly':
                        for station_name, year_kges in basin_yearly.items():
                            years_str = ', '.join(f"{yr}:{v:.3f}" for yr, v in sorted(year_kges.items()))
                            print(f"    {basin_name}/{station_name}: {years_str}")
            
            avg_valid_kge = _safe_mean(list(valid_kge_per_basin.values()))
            history['avg_valid_kge'].append(avg_valid_kge)
            history['station_kge_valid'].append(epoch_valid_station_kge)
            
            for basin_name in basins.keys():
                history['valid_kge'][basin_name].append(valid_kge_per_basin.get(basin_name, float('nan')))
            
            # 打印进度
            print(f"Epoch {epoch+1:4d}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Train KGE: {avg_train_kge:.4f} | Valid KGE: {avg_valid_kge:.4f}")
            
            for basin_name in basins.keys():
                train_val = train_kge_per_basin.get(basin_name, float('nan'))
                valid_val = valid_kge_per_basin.get(basin_name, float('nan'))
                train_str = f"{train_val:.4f}" if not np.isnan(train_val) else "N/A"
                valid_str = f"{valid_val:.4f}" if not np.isnan(valid_val) else "N/A"
                print(f"  {basin_name}: Train={train_str}, Valid={valid_str}")
            
            # 保存最佳模型
            if np.isfinite(avg_valid_kge) and avg_valid_kge > best_kge:
                best_kge = avg_valid_kge
                best_networks = {
                    name: copy.deepcopy(net.state_dict())
                    for name, net in predictor.networks.items()
                }
                print(f"  --> New best KGE: {best_kge:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 学习率调度
            if scheduler is not None:
                scheduler.step(avg_valid_kge)
            
            # 早停
            early_stop_config = config['training'].get('early_stopping', {})
            if early_stop_config.get('enabled', False):
                patience = early_stop_config.get('patience', 20)
                if patience_counter >= patience:
                    print(f"\n[Early Stop] No improvement for {patience} validation intervals")
                    break
    
    print(f"\n{'='*80}")
    print(f"Training completed!")
    print(f"Best validation KGE: {best_kge:.4f}")
    print(f"{'='*80}")
    
    # 恢复最佳网络
    if best_networks:
        for name, state_dict in best_networks.items():
            predictor.networks[name].load_state_dict(state_dict)
    
    # 保存网络
    print("\n[Save] Saving networks...")
    predictor.save_networks(str(output_dir))
    
    # 保存训练历史（包含每站每年KGE）
    _save_history(history, basins, output_dir)
    
    # 绘制训练曲线
    _plot_curves(history, basins, output_dir)
    
    return predictor, history


def run_manual_only(raw_config: dict, runtime_config: dict, config_path: Optional[str] = None):
    """Dispatch manual validation through the dedicated traditional-method script."""
    return run_traditional_calibration(
        raw_config=raw_config,
        runtime_config=runtime_config,
        method='manual',
        config_path=config_path,
    )


def run_traditional_calibration(
    raw_config: dict,
    runtime_config: dict,
    method: str,
    config_path: Optional[str] = None,
):
    """Dispatch traditional calibration through a per-method script."""
    output_dir = Path(runtime_config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_run_configs(raw_config, runtime_config, output_dir)

    runtime_config_path = output_dir / 'config.yaml'
    traditional_cfg = runtime_config.get('training', {}).get('traditional', {})
    method_script = get_traditional_method_script(method)

    command = [
        sys.executable,
        str(method_script),
        '--config',
        str(runtime_config_path),
        '--output-dir',
        str(output_dir),
    ]

    if method == 'differential_evolution':
        arg_mapping = {
            'maxiter': '--maxiter',
            'popsize': '--popsize',
            'mutation': '--mutation',
            'recombination': '--recombination',
            'tol': '--tol',
            'seed': '--seed',
        }
        for key, flag in arg_mapping.items():
            value = traditional_cfg.get(key)
            if value is None:
                continue
            command.extend([flag, str(value)])

        if bool(traditional_cfg.get('write_to_data_params', False)):
            command.append('--write-to-data-params')

    print("\n" + "=" * 80)
    print(f"Traditional calibration mode: {method}")
    print("=" * 80)
    if config_path:
        print(f"[Config] User config: {config_path}")
    print(f"[Config] Runtime config: {runtime_config_path}")
    print(f"[Script] {method_script}")
    print(f"[Output] {output_dir}")
    print(f"[Run] {' '.join(command)}")

    subprocess.run(command, check=True)
    return None, {}


def train(config: dict, config_path: Optional[str] = None):
    """Top-level training dispatcher for the new param_mode schema."""
    resolved_modes = resolve_component_param_modes(config)
    summarize_param_modes(resolved_modes)

    runtime_config = build_runtime_config(config, resolved_modes)
    runner_type, method = determine_training_runner(config, resolved_modes)

    if runner_type == 'manual_only':
        return run_manual_only(config, runtime_config, config_path=config_path)
    if runner_type == 'traditional':
        return run_traditional_calibration(config, runtime_config, method=method, config_path=config_path)
    return train_dl(config, runtime_config, resolved_modes)


# ==============================================================================
# 辅助函数
# ==============================================================================

def _build_params_dict(basin_data: BasinData, 
                       predicted_params: Dict[str, torch.Tensor]) -> dict:
    """构建参数字典"""
    params_dict = {'subbasins': {}, 'rivers': {}}
    
    model = basin_data.model
    param_info = model.get_param_info()
    
    subbasin_param_names = param_info['subbasins']['params_per_subbasin']['static']['names']
    river_param_names = param_info['rivers']['params_per_reach']['static']['names']
    
    # 子流域参数
    for i, sub_id in enumerate(model.subbasin_ids):
        params_dict['subbasins'][sub_id] = {'static': {}, 'dynamic': {}}
        for param_name in subbasin_param_names:
            if param_name in predicted_params:
                params_dict['subbasins'][sub_id]['static'][param_name] = predicted_params[param_name][i]
    
    # 河道参数
    def _reach_sort_key(x):
        """支持 r1, r2 和 r4070652220_up 等多种 reach ID 格式"""
        suffix = x[1:]  # 去掉 'r' 前缀
        try:
            return (int(suffix), '')
        except ValueError:
            # 对于包含非数字字符的ID，按字符串排序
            return (0, suffix)
    reach_ids_ordered = sorted(
        [r for r in model.reaches.keys() if r.startswith('r')],
        key=_reach_sort_key
    )
    for i, reach_id in enumerate(reach_ids_ordered):
        params_dict['rivers'][reach_id] = {'static': {}, 'dynamic': {}}
        for param_name in river_param_names:
            if param_name in predicted_params:
                params_dict['rivers'][reach_id]['static'][param_name] = predicted_params[param_name][i]
    
    return params_dict


def _kge_matrix(sim_mat: torch.Tensor, obs_mat: torch.Tensor) -> torch.Tensor:
    """批量计算KGE，输入形状均为 [n_gauges, n_times]。"""
    # 仅使用有限值（finite）参与统计，避免 inf/NaN 破坏 KGE。
    valid_mask = torch.isfinite(obs_mat) & torch.isfinite(sim_mat)
    sim_safe = torch.nan_to_num(sim_mat, nan=0.0, posinf=0.0, neginf=0.0)
    obs_safe = torch.nan_to_num(obs_mat, nan=0.0, posinf=0.0, neginf=0.0)
    valid_count = valid_mask.sum(dim=1)
    valid_enough = valid_count >= 10

    valid_mask_f = valid_mask.float()
    safe_count = valid_count.float().clamp(min=1.0)

    sim_masked = torch.where(valid_mask, sim_safe, torch.zeros_like(sim_safe))
    obs_masked = torch.where(valid_mask, obs_safe, torch.zeros_like(obs_safe))

    sim_mean = (sim_masked.sum(dim=1) / safe_count)
    obs_mean = (obs_masked.sum(dim=1) / safe_count)

    sim_centered = torch.where(valid_mask, sim_safe - sim_mean[:, None], torch.zeros_like(sim_safe))
    obs_centered = torch.where(valid_mask, obs_safe - obs_mean[:, None], torch.zeros_like(obs_safe))

    denom_std = (safe_count - 1.0).clamp(min=1.0)
    sim_std = torch.sqrt((sim_centered.pow(2).sum(dim=1) / denom_std).clamp(min=0.0))
    obs_std = torch.sqrt((obs_centered.pow(2).sum(dim=1) / denom_std).clamp(min=0.0))
    std_floor = 1e-3
    sim_std = torch.clamp(sim_std, min=std_floor)
    obs_std = torch.clamp(obs_std, min=std_floor)

    numerator = (sim_centered * obs_centered).sum(dim=1)
    denominator = torch.sqrt(sim_centered.pow(2).sum(dim=1)) * torch.sqrt(obs_centered.pow(2).sum(dim=1)) + 1e-6
    denominator = torch.clamp(denominator, min=std_floor)
    r = numerator / denominator

    alpha = sim_std / (obs_std + 1e-6)
    beta = sim_mean / (obs_mean + 1e-6)
    kge_vals = 1 - torch.sqrt((r - 1).pow(2) + (alpha - 1).pow(2) + (beta - 1).pow(2))

    nan_tensor = torch.full_like(kge_vals, float('nan'))
    return torch.where(valid_enough, kge_vals, nan_tensor)


def _safe_mean(values: List[float], default: float = float('nan')) -> float:
    """计算有限值均值；若无有限值则返回 default。"""
    finite_vals = [float(v) for v in values if np.isfinite(v)]
    if not finite_vals:
        return float(default)
    return float(np.mean(finite_vals))


def _tensor_nonfinite_stats(tensor: torch.Tensor) -> Dict[str, float]:
    """Return counts and min/max for finite values in a tensor."""
    t_flat = tensor.reshape(-1)
    nan_count = torch.isnan(t_flat).sum().item()
    inf_count = torch.isinf(t_flat).sum().item()
    finite_mask = torch.isfinite(t_flat)
    if finite_mask.any():
        finite_vals = t_flat[finite_mask]
        min_val = float(finite_vals.min().item())
        max_val = float(finite_vals.max().item())
    else:
        min_val = float('nan')
        max_val = float('nan')
    return {
        'nan': float(nan_count),
        'inf': float(inf_count),
        'min': min_val,
        'max': max_val,
    }


def _has_nonfinite_grads(param_list: List[torch.nn.Parameter]) -> bool:
    """Return True if any gradient has NaN/Inf."""
    for param in param_list:
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            return True
    return False


def _sanitize_grads(param_list: List[torch.nn.Parameter]) -> bool:
    """Replace non-finite gradients with zeros. Returns True if any were found."""
    had_nonfinite = False
    for param in param_list:
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            had_nonfinite = True
            param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
    return had_nonfinite


def _log_nonfinite_diagnostics(
    basin_name: str,
    outputs: Dict[str, Any],
    obs: Optional[Dict[str, Any]],
    warmup: int,
    predicted_params: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    """Log a minimal set of diagnostics when non-finite loss is detected."""
    if not isinstance(outputs, dict):
        return

    subbasin_runoff = outputs.get('subbasin_runoff', {})
    if isinstance(subbasin_runoff, dict) and subbasin_runoff:
        try:
            sub_vals = torch.cat([v[ warmup: ] for v in subbasin_runoff.values()], dim=0)
            stats = _tensor_nonfinite_stats(sub_vals)
            print(
                f"  [Diag] {basin_name} subbasin_runoff: nan={int(stats['nan'])}, "
                f"inf={int(stats['inf'])}, min={stats['min']:.3f}, max={stats['max']:.3f}"
            )
            bad_subs = []
            for sub_id, vals in subbasin_runoff.items():
                v = vals[warmup:]
                nan_count = int(torch.isnan(v).sum().item())
                if nan_count > 0:
                    bad_subs.append((sub_id, nan_count, v.numel()))
            if bad_subs:
                preview = bad_subs[:5]
                preview_str = ", ".join(
                    f"{sid}({n}/{total})" for sid, n, total in preview
                )
                more = "" if len(bad_subs) <= 5 else f" (+{len(bad_subs) - 5} more)"
                print(f"  [Diag] {basin_name} subbasin_runoff NaN subs: {preview_str}{more}")
        except Exception:
            print(f"  [Diag] {basin_name} subbasin_runoff: failed to aggregate")

    river_flows = outputs.get('river_flows', {})
    if obs and river_flows:
        reach_ids = [rid for rid in obs.keys() if rid in river_flows]
        if reach_ids:
            flow_vals = torch.cat([river_flows[rid][warmup:] for rid in reach_ids], dim=0)
            stats = _tensor_nonfinite_stats(flow_vals)
            print(
                f"  [Diag] {basin_name} river_flows: nan={int(stats['nan'])}, "
                f"inf={int(stats['inf'])}, min={stats['min']:.3f}, max={stats['max']:.3f}"
            )

            bad_reaches = []
            for rid in reach_ids:
                vals = river_flows[rid][warmup:]
                nan_count = int(torch.isnan(vals).sum().item())
                if nan_count > 0:
                    bad_reaches.append((rid, nan_count, vals.numel()))
            if bad_reaches:
                preview = bad_reaches[:5]
                preview_str = ", ".join(
                    f"{rid}({n}/{total})" for rid, n, total in preview
                )
                more = "" if len(bad_reaches) <= 5 else f" (+{len(bad_reaches) - 5} more)"
                print(f"  [Diag] {basin_name} river_flows NaN reaches: {preview_str}{more}")

    if predicted_params:
        bad_params = []
        for name, values in predicted_params.items():
            if not torch.isfinite(values).all():
                bad_params.append(name)
        if bad_params:
            print(f"  [Diag] {basin_name} non-finite params: {bad_params}")


def _compute_loss(outputs: dict, obs: dict, warmup: int, 
                  device: torch.device, dates: pd.DatetimeIndex = None,
                  kge_mode: str = 'series',
                  gauge_names: dict = None) -> Tuple[torch.Tensor, float, dict]:
    """计算损失和KGE
    
    支持两种模式：
    - 'series': 整个时间序列计算一个KGE（原始模式）
    - 'yearly': 按年计算KGE，再在多年×多站上平均
    
    Args:
        gauge_names: {reach_id: gauge_name} 映射，用于yearly_details的key
    
    Returns:
        loss: 损失值（可反向传播）
        avg_kge: 平均KGE值
        yearly_details: {station_name: {year: kge_value}} 每站每年KGE
    """
    yearly_details = {}  # {station_name: {year: kge_value}}

    if obs is None:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0, yearly_details

    if gauge_names is None:
        gauge_names = {}

    river_flows = outputs.get('river_flows', {})
    valid_reaches = [rid for rid in obs.keys() if rid in river_flows]
    if not valid_reaches:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0, yearly_details

    sim_rows = []
    obs_rows = []
    weights = []
    detail_keys = []

    for reach_id in valid_reaches:
        Q_obs, weight = obs[reach_id]
        sim_rows.append(river_flows[reach_id][warmup:])
        obs_rows.append(torch.tensor(Q_obs, dtype=torch.float32, device=device)[warmup:])
        weights.append(float(weight))
        detail_keys.append(gauge_names.get(reach_id, reach_id))

    sim_mat = torch.stack(sim_rows, dim=0)
    obs_mat = torch.stack(obs_rows, dim=0)
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    # 使用与 sim_mat 关联的安全零值，避免 inf * 0.0 产生 NaN。
    sim_safe = torch.nan_to_num(sim_mat, nan=0.0, posinf=0.0, neginf=0.0)
    loss = sim_safe.sum() * 0.0
    kge_total = 0.0
    n_gauges = 0.0

    if kge_mode == 'yearly' and dates is not None:
        years = dates[warmup:].year
        unique_years = sorted(years.unique())

        year_kge_values_by_gauge = [[] for _ in detail_keys]

        for yr in unique_years:
            yr_mask_np = np.asarray(years == yr)
            yr_mask = torch.tensor(yr_mask_np, dtype=torch.bool, device=device)
            if yr_mask.sum() == 0:
                continue

            kge_vals_yr = _kge_matrix(sim_mat[:, yr_mask], obs_mat[:, yr_mask])
            valid_mask = ~torch.isnan(kge_vals_yr)

            for i in torch.where(valid_mask)[0].tolist():
                kge_val = kge_vals_yr[i]
                year_kge_values_by_gauge[i].append(kge_val)
                yearly_details.setdefault(detail_keys[i], {})[int(yr)] = float(kge_val.detach().item())

        for i, kge_list in enumerate(year_kge_values_by_gauge):
            if not kge_list:
                continue
            avg_yearly_kge = torch.stack(kge_list).mean()
            w = float(weights[i])
            loss = loss + (1 - avg_yearly_kge) * w
            kge_total += float(avg_yearly_kge.detach().item()) * w
            n_gauges += w
    else:
        kge_vals = _kge_matrix(sim_mat, obs_mat)
        valid_mask = ~torch.isnan(kge_vals)

        if valid_mask.any():
            valid_kge_vals = kge_vals[valid_mask]
            valid_weights = weight_tensor[valid_mask]
            loss = loss + ((1 - valid_kge_vals) * valid_weights).sum()

            kge_total = float((valid_kge_vals.detach() * valid_weights).sum().item())
            n_gauges = float(valid_weights.sum().item())

            valid_indices = torch.where(valid_mask)[0].tolist()
            for i in valid_indices:
                yearly_details[detail_keys[i]] = {'series': float(kge_vals[i].detach().item())}

    if n_gauges <= 0:
        # 防止出现“无有效KGE却显示0.0并继续训练”的静默失败。
        sim_safe = torch.nan_to_num(sim_mat, nan=0.0, posinf=1e6, neginf=-1e6)
        overflow_penalty = torch.relu(torch.abs(sim_safe) - 1e4).mean() / 1e4
        loss = loss + 1.0 + overflow_penalty
        return loss, float('nan'), yearly_details

    avg_kge = kge_total / n_gauges
    return loss, avg_kge, yearly_details


def _collect_station_kge(epoch_dict: dict, yearly_details: dict, kge_mode: str):
    """收集一个epoch内每站（每年）的KGE到epoch_dict中
    
    yearly模式: key为 {station}_{year}，如 r4070636400_up_2002
    series模式: key为 {station}，如 r4070636400_up
    """
    if kge_mode == 'yearly':
        for rid, yr_kges in yearly_details.items():
            for yr, val in yr_kges.items():
                epoch_dict[f"{rid}_{yr}"] = val
    else:
        for rid, yr_kges in yearly_details.items():
            epoch_dict[rid] = yr_kges.get('series', float('nan'))


def _save_history(history: dict, basins: Dict[str, BasinData], output_dir: Path):
    """保存训练历史（包含每站每年KGE）"""
    # 一次性构建所有列数据（避免DataFrame碎片化警告）
    data_dict = {
        'epoch': history['epoch'],
        'train_loss': history['train_loss'],
        'avg_train_kge': history['avg_train_kge'],
    }
    
    # 添加各流域训练KGE列
    for basin_name in basins.keys():
        data_dict[f'train_kge_{basin_name}'] = history['train_kge'][basin_name]
    
    # 添加每站（每年）train KGE列
    if history['station_kge_train']:
        train_station_df = pd.DataFrame(history['station_kge_train'])
        for col in sorted(train_station_df.columns):
            data_dict[f'train_kge_{col}'] = train_station_df[col].values
    
    # 添加每站（每年）valid KGE列：需按epoch对齐（valid_interval可能 > 1）
    if history['station_kge_valid']:
        valid_epochs = history['avg_valid_kge']
        valid_interval = len(history['epoch']) // len(valid_epochs) if valid_epochs else 1
        all_cols = set()
        for d in history['station_kge_valid']:
            all_cols.update(d.keys())
        
        for col in sorted(all_cols):
            full = [float('nan')] * len(history['epoch'])
            for i, d in enumerate(history['station_kge_valid']):
                epoch_idx = (i + 1) * valid_interval - 1
                if epoch_idx < len(full):
                    full[epoch_idx] = d.get(col, float('nan'))
            data_dict[f'valid_kge_{col}'] = full
    
    # 一次性创建DataFrame
    history_df = pd.DataFrame(data_dict)
    
    csv_path = output_dir / 'training_history.csv'
    if csv_path.exists():
        # 追加到已有文件（跳过header），epoch编号续接
        existing_df = pd.read_csv(csv_path)
        last_epoch = int(existing_df['epoch'].max()) if len(existing_df) > 0 else 0
        history_df['epoch'] = history_df['epoch'] + last_epoch
        history_df.to_csv(csv_path, mode='a', header=False, index=False)
        print(f"[Save] Training history appended (epoch {last_epoch+1}~{history_df['epoch'].max()})")
    else:
        history_df.to_csv(csv_path, index=False)
        print(f"[Save] Training history saved")


def _plot_curves(history: dict, basins: Dict[str, BasinData], output_dir: Path):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = history['epoch']
    
    # 1. 总体损失
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 2. 平均KGE
    axes[0, 1].plot(epochs, history['avg_train_kge'], 'b-', label='Train')
    if history['avg_valid_kge']:
        valid_step = len(epochs) // len(history['avg_valid_kge']) if history['avg_valid_kge'] else 1
        valid_epochs = epochs[valid_step-1::valid_step][:len(history['avg_valid_kge'])]
        axes[0, 1].plot(valid_epochs, history['avg_valid_kge'], 'r-o', label='Valid')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('KGE')
    axes[0, 1].set_title('Average KGE (All Basins)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 3. 各流域训练KGE
    colors = plt.cm.tab10(np.linspace(0, 1, len(basins)))
    for i, basin_name in enumerate(basins.keys()):
        axes[1, 0].plot(epochs, history['train_kge'][basin_name], 
                       color=colors[i], label=basin_name)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('KGE')
    axes[1, 0].set_title('Training KGE per Basin')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 4. 各流域验证KGE
    for i, basin_name in enumerate(basins.keys()):
        if history['valid_kge'][basin_name]:
            axes[1, 1].plot(history['valid_kge'][basin_name], 
                           color=colors[i], label=basin_name)
    axes[1, 1].set_xlabel('Validation Interval')
    axes[1, 1].set_ylabel('KGE')
    axes[1, 1].set_title('Validation KGE per Basin')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    plt.close()
    print(f"[Plot] Training curves saved")


# ==============================================================================
# 主入口
# ==============================================================================

if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置
    print(f"[Config] Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # 显示模式
    mode = get_mode(config)
    print(f"[Config] Running mode: {mode}")
    
    # 训练模型
    predictor, history = train(config, config_path=args.config)
    
    print("\n[Done] Training completed successfully!")
