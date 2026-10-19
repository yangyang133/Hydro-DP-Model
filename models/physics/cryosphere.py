"""
冰冻圈过程模块（Cold Region Process Module）

包含：积雪、冰川、冻土过程的统一处理

供 grid_runoff 和 subbasin_runoff 模块共同调用
"""

import torch


# ==============================================================================
# 冰冻圈参数定义
# ==============================================================================

# 冰冻圈默认参数
COLD_REGION_DEFAULTS = {
    "t_snow": 0.0,        # 雨雪分界温度 (°C)
    "t_melt": 0.0,        # 融雪温度阈值 (°C)
    "ddf_snow": 3.0,      # 积雪度日因子 (mm/°C/day)
    "k_snow": 0.5,        # 积雪消融系数
    "glacier_frac": 0.0,  # 冰川面积比例（从输入获取）
    "ddf_ice": 6.0,       # 冰川度日因子 (mm/°C/day)
    "t_ice_melt": 0.0,    # 冰川融化温度阈值 (°C)
    "glacier_runoff_frac": 0.7,  # 冰川融水地表径流比例
    "fi_alpha": 0.9,      # 冻结指数衰减系数
    "fi_beta": 0.1,       # 冻结指数影响系数
    "adjust_et": False,   # 是否根据温度调节蒸发（输入蒸发已含温度影响时设为False）
}

# 冰冻圈参数边界
COLD_REGION_BOUNDS = {
    "t_snow": [-2, 2],
    "t_melt": [-1, 3],
    "ddf_snow": [1, 10],
    "k_snow": [0.1, 0.9],
    "ddf_ice": [3, 15],
    "t_ice_melt": [0, 2],
    "glacier_runoff_frac": [0.3, 0.9],
    "fi_alpha": [0.5, 0.99],
    "fi_beta": [0.01, 0.5],
}

# 冰冻圈参数名称列表（按顺序）
COLD_REGION_PARAM_NAMES = [
    "t_snow", "t_melt", "ddf_snow", "k_snow",
    "ddf_ice", "t_ice_melt", "glacier_runoff_frac",
    "fi_alpha", "fi_beta"
]

# 冰冻圈参数边界列表（与名称顺序对应）
COLD_REGION_PARAM_BOUNDS = [
    [-2, 2], [-1, 3], [1, 10], [0.1, 0.9],
    [3, 15], [0, 2], [0.3, 0.9],
    [0.5, 0.99], [0.01, 0.5]
]


# ==============================================================================
# 冰冻圈过程函数
# ==============================================================================

def process_cryosphere(P, T, swe, FI, glacier_frac, params):
    """
    冰冻圈过程处理函数（积雪、冰川、冻土）
    
    参数:
        P: 降水量 (mm)
        T: 温度 (°C)
        swe: 积雪水当量状态 (mm)
        FI: 冻结指数状态
        glacier_frac: 冰川覆盖比例
        params: 冰冻圈参数字典，包含:
            - t_snow: 雨雪分界温度 (°C)
            - t_melt: 融雪温度阈值 (°C)
            - ddf_snow: 积雪度日因子 (mm/°C/day)
            - k_snow: 积雪消融系数
            - ddf_ice: 冰川度日因子 (mm/°C/day)
            - t_ice_melt: 冰川融化温度阈值 (°C)
            - glacier_runoff_frac: 冰川融水地表径流比例
            - fi_alpha: 冻结指数衰减系数
            - fi_beta: 冻结指数影响系数
    
    返回:
        dict: 包含以下键值:
            - P_eff: 有效降水（液态降水 + 融雪）
            - P_rain: 液态降水
            - P_snow: 固态降水
            - swe: 更新后的积雪水当量
            - FI: 更新后的冻结指数
            - snow_melt: 融雪量
            - glacier_melt: 冰川融化量
            - glacier_melt_surface: 冰川地表融水
            - glacier_melt_ground: 冰川地下融水
            - frozen_factor: 冻土因子（减少入渗）
            - et_factor: 蒸发调节因子
    """
    # 获取参数
    t_snow = params.get("t_snow", COLD_REGION_DEFAULTS["t_snow"])
    t_melt = params.get("t_melt", COLD_REGION_DEFAULTS["t_melt"])
    ddf_snow = params.get("ddf_snow", COLD_REGION_DEFAULTS["ddf_snow"])
    k_snow = params.get("k_snow", COLD_REGION_DEFAULTS["k_snow"])
    ddf_ice = params.get("ddf_ice", COLD_REGION_DEFAULTS["ddf_ice"])
    t_ice_melt = params.get("t_ice_melt", COLD_REGION_DEFAULTS["t_ice_melt"])
    glacier_runoff_frac = params.get("glacier_runoff_frac", COLD_REGION_DEFAULTS["glacier_runoff_frac"])
    fi_alpha = params.get("fi_alpha", COLD_REGION_DEFAULTS["fi_alpha"])
    fi_beta = params.get("fi_beta", COLD_REGION_DEFAULTS["fi_beta"])
    
    # --- 1. 降水分离（雨/雪）---
    snow_frac = torch.sigmoid((t_snow - T) * 2.0)
    P_snow = P * snow_frac
    P_rain = P * (1 - snow_frac)
    
    # --- 2. 积雪累积与融化 ---
    swe = swe + P_snow
    melt_potential = ddf_snow * torch.clamp(T - t_melt, min=0.0)
    snow_melt = k_snow * torch.minimum(melt_potential, swe)
    swe = swe - snow_melt
    
    # --- 3. 冰川融化（先融雪后融冰）---
    # 只有当积雪融尽（swe趋近0）时，冰川才开始融化
    ice_melt_factor = torch.exp(-swe / 10.0)
    glacier_melt_potential = glacier_frac * ddf_ice * torch.clamp(T - t_ice_melt, min=0.0)
    glacier_melt = glacier_melt_potential * ice_melt_factor
    
    # 冰川融水分配
    glacier_melt_surface = glacier_melt * glacier_runoff_frac
    glacier_melt_ground = glacier_melt * (1 - glacier_runoff_frac)
    
    # --- 4. 冻土过程（冻结指数记忆机制）---
    FI = fi_alpha * FI + torch.relu(-T)
    frozen_factor = torch.exp(-fi_beta * FI)
    
    # --- 5. 蒸发调节因子 ---
    # 如果输入蒸发数据已包含温度影响，则仅考虑冻土因子
    # 否则同时考虑温度和冻土对蒸发的抑制
    adjust_et = params.get("adjust_et", COLD_REGION_DEFAULTS["adjust_et"])
    # et_adjust: 是否需要额外温度调节（仅当 PET 不含温度效应时）

    # 改为sigmod函数平滑过渡
    # torch.sigmoid((T - T_et) / delta_T * k) 
    # 其中 T_et 为蒸发调节温度阈值，delta_T 为过渡宽度，k 为陡峭度
    et_factor = frozen_factor * (torch.sigmoid(T/ 2.0) if adjust_et else 1.0)
    
    # 有效降水
    P_eff = P_rain + snow_melt
    
    return {
        "P_eff": P_eff,
        "P_rain": P_rain,
        "P_snow": P_snow,
        "swe": swe,
        "FI": FI,
        "snow_melt": snow_melt,
        "glacier_melt": glacier_melt,
        "glacier_melt_surface": glacier_melt_surface,
        "glacier_melt_ground": glacier_melt_ground,
        "frozen_factor": frozen_factor,
        "et_factor": et_factor,
    }


def get_cold_region_params(input_data, _t=None, defaults=None):
    """
    获取冰冻圈参数
    
    支持三种调用方式：
    1. 从模型实例获取参数定义（用于参数注册）
       get_cold_region_params(model_instance) -> (names, bounds, init_values)
    
    2. 从参数字典获取参数定义（用于参数注册）
       get_cold_region_params(defaults_dict) -> (names, bounds, init_values)
       defaults_dict 必须包含所有冰冻圈参数
    
    3. 从静态参数字典获取参数值（用于冰冻圈计算）
       get_cold_region_params(static_params, _t, defaults) -> params_dict
    
    参数:
        input_data: 模型实例 / 默认值字典 / 静态参数字典
        _t: 张量转换函数（仅方式3使用）
        defaults: 默认参数字典（仅方式3使用）
    
    返回:
        方式1/2: tuple (names, bounds, init_values)
        方式3: dict 参数字典
    """
    # 方式3：从参数字典获取参数值（用于冰冻圈计算）
    if _t is not None:
        if defaults is None:
            defaults = COLD_REGION_DEFAULTS
        return {
            "t_snow": _t(input_data.get("t_snow", defaults["t_snow"])),
            "t_melt": _t(input_data.get("t_melt", defaults["t_melt"])),
            "ddf_snow": _t(input_data.get("ddf_snow", defaults["ddf_snow"])),
            "k_snow": _t(input_data.get("k_snow", defaults["k_snow"])),
            "ddf_ice": _t(input_data.get("ddf_ice", defaults["ddf_ice"])),
            "t_ice_melt": _t(input_data.get("t_ice_melt", defaults["t_ice_melt"])),
            "glacier_runoff_frac": _t(input_data.get("glacier_runoff_frac", defaults["glacier_runoff_frac"])),
            "fi_alpha": _t(input_data.get("fi_alpha", defaults["fi_alpha"])),
            "fi_beta": _t(input_data.get("fi_beta", defaults["fi_beta"])),
        }
    
    # 方式1/2：返回参数定义
    names = COLD_REGION_PARAM_NAMES.copy()
    bounds = COLD_REGION_PARAM_BOUNDS.copy()
    
    if isinstance(input_data, dict):
        # 方式2：从字典获取初始值
        init_values = [
            input_data.get("t_snow", COLD_REGION_DEFAULTS["t_snow"]),
            input_data.get("t_melt", COLD_REGION_DEFAULTS["t_melt"]),
            input_data.get("ddf_snow", COLD_REGION_DEFAULTS["ddf_snow"]),
            input_data.get("k_snow", COLD_REGION_DEFAULTS["k_snow"]),
            input_data.get("ddf_ice", COLD_REGION_DEFAULTS["ddf_ice"]),
            input_data.get("t_ice_melt", COLD_REGION_DEFAULTS["t_ice_melt"]),
            input_data.get("glacier_runoff_frac", COLD_REGION_DEFAULTS["glacier_runoff_frac"]),
            input_data.get("fi_alpha", COLD_REGION_DEFAULTS["fi_alpha"]),
            input_data.get("fi_beta", COLD_REGION_DEFAULTS["fi_beta"]),
        ]
    else:
        # 方式1：从模型实例获取初始值
        model_instance = input_data
        init_values = [
            model_instance.t_snow, model_instance.t_melt, 
            model_instance.ddf_snow, model_instance.k_snow,
            model_instance.ddf_ice, model_instance.t_ice_melt, 
            model_instance.glacier_runoff_frac,
            model_instance.fi_alpha, model_instance.fi_beta
        ]
    
    return names, bounds, init_values


def get_cold_region_network_mapping():
    """
    获取冰冻圈参数的网络映射
    
    返回:
        dict: 参数名 -> 网络类型
    """
    return {name: "subbasin_runoff" for name in COLD_REGION_PARAM_NAMES}
