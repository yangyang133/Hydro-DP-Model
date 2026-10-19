"""
降水校正模块（Precipitation Correction Module）

对输入降水施加折损/增益系数 kp，用于补偿：
  - 降水观测误差（雨量计的系统性低估/高估）
  - 降水空间插值偏差
  - 水量平衡残差

供 grid_runoff 和 subbasin_runoff 模块共同调用
"""

import torch


# ==============================================================================
# 降水校正参数定义
# ==============================================================================

# 降水校正默认参数
PRECIP_CORRECT_DEFAULTS = {
    "kp": 1.0,  # 降水校正系数（>1 增大降水, <1 减小降水）
}

# 降水校正参数边界
PRECIP_CORRECT_BOUNDS = {
    "kp": [0.5, 1.5],
}

# 降水校正参数名称列表
PRECIP_CORRECT_PARAM_NAMES = ["kp"]

# 降水校正参数边界列表（与名称顺序对应）
PRECIP_CORRECT_PARAM_BOUNDS = [[0.5, 1.5]]


# ==============================================================================
# 降水校正函数
# ==============================================================================

def apply_precip_correction(P, kp):
    """
    对降水施加校正系数

    参数:
        P: 降水量 (mm)，标量或张量
        kp: 降水校正系数，标量或张量

    返回:
        P_corrected: 校正后降水量 (mm)
    """
    return P * kp


# ==============================================================================
# 参数注册辅助函数
# ==============================================================================

def get_precip_correct_params(input_data):
    """
    获取降水校正参数（用于参数注册）

    支持两种调用方式：
    1. 从模型实例获取：get_precip_correct_params(model_instance)
    2. 从字典获取：get_precip_correct_params(defaults_dict)

    参数:
        input_data: 模型实例或默认值字典

    返回:
        tuple: (names, bounds, init_values)
    """
    names = list(PRECIP_CORRECT_PARAM_NAMES)
    bounds = [list(b) for b in PRECIP_CORRECT_PARAM_BOUNDS]

    if isinstance(input_data, dict):
        init_values = [input_data.get("kp", PRECIP_CORRECT_DEFAULTS["kp"])]
    else:
        # 从模型实例获取
        init_values = [input_data.kp]

    return names, bounds, init_values


def get_precip_correct_network_mapping():
    """
    获取降水校正参数的网络映射

    返回:
        dict: 参数名 -> 网络类型
    """
    return {name: "subbasin_runoff" for name in PRECIP_CORRECT_PARAM_NAMES}
