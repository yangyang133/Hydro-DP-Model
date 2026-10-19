# models/physics/__init__.py
"""
物理过程模块

包含：
- cryosphere: 冰冻圈过程（积雪、冰川、冻土）
- precip_correct: 降水校正（降水折损/增益系数）
- pet: 潜在蒸散发计算
"""

from .cryosphere import (
    process_cryosphere,
    COLD_REGION_DEFAULTS,
    COLD_REGION_BOUNDS,
    COLD_REGION_PARAM_NAMES,
    COLD_REGION_PARAM_BOUNDS,
    get_cold_region_params,
    get_cold_region_network_mapping,
)

from .precip_correct import (
    apply_precip_correction,
    PRECIP_CORRECT_DEFAULTS,
    PRECIP_CORRECT_BOUNDS,
    PRECIP_CORRECT_PARAM_NAMES,
    PRECIP_CORRECT_PARAM_BOUNDS,
    get_precip_correct_params,
    get_precip_correct_network_mapping,
)

from .pet import (
    calculate_pet_hargreaves,
    calculate_pet_priestley_taylor,
    calculate_pet_penman_monteith,
    calculate_pet_thornthwaite,
    calculate_pet_simple,
)

__all__ = [
    # Cryosphere
    "process_cryosphere",
    "COLD_REGION_DEFAULTS",
    "COLD_REGION_BOUNDS",
    "COLD_REGION_PARAM_NAMES",
    "COLD_REGION_PARAM_BOUNDS",
    "get_cold_region_params",
    "get_cold_region_network_mapping",
    # Precipitation Correction
    "apply_precip_correction",
    "PRECIP_CORRECT_DEFAULTS",
    "PRECIP_CORRECT_BOUNDS",
    "PRECIP_CORRECT_PARAM_NAMES",
    "PRECIP_CORRECT_PARAM_BOUNDS",
    "get_precip_correct_params",
    "get_precip_correct_network_mapping",
    # PET
    "calculate_pet_hargreaves",
    "calculate_pet_priestley_taylor",
    "calculate_pet_penman_monteith",
    "calculate_pet_thornthwaite",
    "calculate_pet_simple",
]
