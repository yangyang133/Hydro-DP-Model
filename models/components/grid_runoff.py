"""
栅格产流模型模块

本模块实现三种常用的栅格产流模型，代码保持原有计算逻辑，仅替换并补充为清晰的中文注释：

1. XAJGridModel - 新安江栅格模型（基于蓄满产流机制）
2. SPHYModel - SPHY 高山/冰川区域模型（包含积雪、冰川、冻土过程耦合）
3. VICModel - VIC（Variable Infiltration Capacity）模型（可变入渗容量曲线）

每个模型均支持可选的冰冻圈扩展（`enable_cold_region=True`）并调用
`models.physics.cryosphere.process_cryosphere` 处理积雪/冰川/冻土相关量。

模块导出：通过注册器将模型注册到 `GRID_REGISTRY`，以便整体框架按名称使用。
"""

import torch
from models.base import BaseModel
from models.registry import GRID_REGISTRY, register
import math

# 从冰冻圈模块导入处理函数与参数定义
from models.physics.cryosphere import (
    process_cryosphere,
    COLD_REGION_DEFAULTS,
    COLD_REGION_BOUNDS,
    COLD_REGION_PARAM_NAMES,
    COLD_REGION_PARAM_BOUNDS,
    get_cold_region_params,
)

# 向后兼容常量别名
COLD_REGION_DEFAULT_VALUES = COLD_REGION_DEFAULTS


# ==============================================================================
# 1. 新安江栅格模型（Xinanjiang Grid Model）
# ==============================================================================
@register(GRID_REGISTRY, "xaj_grid")
class XAJGridModel(BaseModel):
    """
    新安江栅格模型（栅格版）

    概述：基于蓄满产流思想的分布式栅格产流实现。每个栅格单元独立计算土壤水量、入渗、产流、三水源分配及
    通过三个线性水库的汇流过程。可选地耦合冰冻圈过程（积雪、冰川融水、冻土对入渗的抑制）。

    主要输入（`inputs`）:
        - "P": 降水 (mm)
        - "E": 潜在蒸散发 (mm)
        - "T": 温度 (℃，仅在启用冰冻圈时使用)
        - "glacier_frac": 冰川覆盖率 (0-1，可选)

    返回（result, state）：result 包含 "runoff" 与若干诊断变量；state 保存模型内部状态。
    """
    def __init__(
        self,
        enable_cold_region=False,
        # 冰冻圈参数（若启用）
        t_snow=0.0, t_melt=0.0, ddf_snow=3.0, k_snow=0.5,
        glacier_frac=0.0, ddf_ice=6.0, t_ice_melt=0.0,
        glacier_runoff_frac=0.7,
        fi_alpha=0.9, fi_beta=0.1,
        # XAJ 参数
        wum=30.0, wlm=60.0, wdm=80.0,
        b=0.5, kc=1.0,
        im=0.1,
        sm=100.0, ex=1.5,
        kg=0.15, ki=0.35, ks=0.50,
        cr=0.5, ci=0.5, cg=0.1,
        modules=None
    ):
        super().__init__()
        self.enable_cold_region = enable_cold_region
        # 冰冻圈默认/初始参数
        self.t_snow = t_snow
        self.t_melt = t_melt
        self.ddf_snow = ddf_snow
        self.k_snow = k_snow
        self.glacier_frac = glacier_frac
        self.ddf_ice = ddf_ice
        self.t_ice_melt = t_ice_melt
        self.glacier_runoff_frac = glacier_runoff_frac
        self.fi_alpha = fi_alpha
        self.fi_beta = fi_beta
        # XAJ 参数赋值
        self.wum = wum
        self.wlm = wlm
        self.wdm = wdm
        self.b = b
        self.kc = kc
        self.im = im
        self.sm = sm
        self.ex = ex
        self.kg = kg
        self.ki = ki
        self.ks = ks
        self.cr = cr
        self.ci = ci
        self.cg = cg

    def forward(self, inputs, static_params, dynamic_params, state):
        """
        计算单步水文量: 接受 inputs（P, E, 可选T）和参数集合，返回 (result, state)。
        主要流程概述：
            1) 若启用冰冻圈则先调用 process_cryosphere 计算有效降水与融雪/冰川量；
            2) 计算蒸散发、入渗与超饱和产流（蓄满产流）；
            3) 将产流划分为三水源并通过线性水库计算汇流；
            4) 更新状态并返回诊断量。
        """
        P = inputs.get("P", 0.0)
        E = inputs.get("E", 0.0)
        device = P.device if isinstance(P, torch.Tensor) else None
        dtype = P.dtype if isinstance(P, torch.Tensor) else torch.float32
        def _t(x):
            return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=dtype, device=device)
        P = _t(P)
        E = _t(E)

        # 处理冰冻圈：积雪/融雪/冰川/冻土对入渗和蒸散发的影响
        snow_melt = _t(0.0)
        glacier_melt = _t(0.0)
        swe = _t(0.0)
        FI = _t(0.0)
        glacier_melt_surface = _t(0.0)
        glacier_melt_ground = _t(0.0)
        et_factor = _t(1.0)
        frozen_factor = _t(1.0)
        if self.enable_cold_region:
            T = _t(inputs.get("T", 10.0))
            swe = _t(state.get("swe", 0.0))
            FI = _t(state.get("FI", 0.0))
            glacier_frac = _t(inputs.get("glacier_frac", self.glacier_frac))
            cryo_params = {
                "t_snow": _t(static_params.get("t_snow", self.t_snow)),
                "t_melt": _t(static_params.get("t_melt", self.t_melt)),
                "ddf_snow": _t(static_params.get("ddf_snow", self.ddf_snow)),
                "k_snow": _t(static_params.get("k_snow", self.k_snow)),
                "ddf_ice": _t(static_params.get("ddf_ice", self.ddf_ice)),
                "t_ice_melt": _t(static_params.get("t_ice_melt", self.t_ice_melt)),
                "glacier_runoff_frac": _t(static_params.get("glacier_runoff_frac", self.glacier_runoff_frac)),
                "fi_alpha": _t(static_params.get("fi_alpha", self.fi_alpha)),
                "fi_beta": _t(static_params.get("fi_beta", self.fi_beta)),
            }
            cryo_result = process_cryosphere(P, T, swe, FI, glacier_frac, cryo_params)
            P_eff = cryo_result["P_eff"]
            swe = cryo_result["swe"]
            FI = cryo_result["FI"]
            snow_melt = cryo_result["snow_melt"]
            glacier_melt = cryo_result["glacier_melt"]
            glacier_melt_surface = cryo_result["glacier_melt_surface"]
            glacier_melt_ground = cryo_result["glacier_melt_ground"]
            frozen_factor = cryo_result["frozen_factor"]
            et_factor = cryo_result["et_factor"]
        else:
            P_eff = P

        # 从 static_params 中读取或使用默认参数
        wum = _t(static_params.get("wum", self.wum))
        wlm = _t(static_params.get("wlm", self.wlm))
        wdm = _t(static_params.get("wdm", self.wdm))
        wm = wum + wlm + wdm
        b  = _t(static_params.get("b",  self.b))
        kc = _t(static_params.get("kc", self.kc))
        im = _t(static_params.get("im", self.im))
        sm = _t(static_params.get("sm", self.sm))
        ex = _t(static_params.get("ex", self.ex))
        kg = _t(static_params.get("kg", self.kg))
        ki = _t(static_params.get("ki", self.ki))
        ks = _t(static_params.get("ks", self.ks))
        # 归一化三水源系数
        total_k = kg + ki + ks
        kg = kg / total_k
        ki = ki / total_k
        ks = ks / total_k
        cr_coef = _t(static_params.get("cr", self.cr))
        ci = _t(static_params.get("ci", self.ci))
        cg = _t(static_params.get("cg", self.cg))

        # 状态变量读取或初始化
        wu = _t(state.get("wu", wum / 2))
        wl = _t(state.get("wl", wlm / 2))
        wd = _t(state.get("wd", wdm / 2))
        sf = _t(state.get("sf", 0.0))
        si = _t(state.get("si", 0.0))
        sg = _t(state.get("sg", 0.0))

        # 1) 三层蒸散发（上-中-下层）
        et = E * kc * et_factor
        eu = torch.minimum(wu, et)
        wu = wu - eu
        et = et - eu
        el = torch.minimum(wl, et)
        wl = wl - el
        et = et - el
        ed = torch.minimum(wd, et)
        wd = wd - ed
        total_et = eu + el + ed
        w = wu + wl + wd

        # 2) 产流计算（蓄满产流机制）
        runoff_im = im * P_eff
        Pp = (1 - im) * P_eff
        A = wm * (1 - torch.pow(1 - w / wm, 1 + b))
        tmp = wm * (1 - torch.pow((A + Pp) / wm, 1 / (1 + b))) - w
        runoff_p_candidate = Pp - tmp
        runoff_p_alt = Pp - (wm - w)
        runoff_p_pos = torch.where((A + Pp) < wm, runoff_p_candidate, runoff_p_alt)
        runoff_p = torch.where(Pp <= 0, torch.zeros_like(Pp), runoff_p_pos)
        runoff_p = torch.clamp(runoff_p, min=0.0)
        runoff_p = torch.minimum(runoff_p, Pp)

        # 冻土抑制入渗（若启用冰冻圈）
        if self.enable_cold_region:
            potential_infil = Pp - runoff_p
            actual_infil = potential_infil * frozen_factor
            frozen_excess = potential_infil - actual_infil
            runoff_p = runoff_p + frozen_excess
            infil = actual_infil
        else:
            infil = Pp - runoff_p

        # 按土层分配入渗到上/中/下层
        wu_space = wum - wu
        wu_add = torch.minimum(wu_space, infil * (wum / wm))
        wu = wu + wu_add
        infil_remain = infil - wu_add
        wl_space = wlm - wl
        wl_add = torch.minimum(wl_space, infil_remain * (wlm / (wlm + wdm)))
        wl = wl + wl_add
        infil_remain = infil_remain - wl_add
        wd_space = wdm - wd
        wd_add = torch.minimum(wd_space, infil_remain)
        wd = wd + wd_add
        infil_remain = infil_remain - wd_add
        runoff_p = runoff_p + infil_remain

        # 3) 自由水（sf）与三水源划分
        runoff_total = runoff_im + runoff_p
        sf = sf + runoff_total
        qf_calc = torch.where(
            sf <= sm,
            sf * (1 - torch.pow(1 - sf / sm, ex)),
            sf
        )
        qf = torch.minimum(qf_calc, sf)
        sf = torch.clamp(sf - qf, min=0.0)
        qs = ks * qf
        qi = ki * qf
        qg = kg * qf

        # 4) 三个线性水库的出流计算（地表sr、中间si、地下sg）
        sr = _t(state.get("sr", 0.0))
        sr = sr + qs + glacier_melt_surface
        sr_out = torch.minimum(cr_coef * sr, sr)
        sr = torch.clamp(sr - sr_out, min=0.0)
        si = si + qi
        si_out = torch.minimum(ci * si, si)
        si = si - si_out
        sg = sg + qg + glacier_melt_ground
        sg_out = torch.minimum(cg * sg, sg)
        sg = sg - sg_out
        sr = torch.clamp(sr, min=0.0)
        si = torch.clamp(si, min=0.0)
        sg = torch.clamp(sg, min=0.0)
        q_out = sr_out + si_out + sg_out
        q_out = torch.clamp(q_out, min=0.0)

        # 更新状态并构造结果
        state.update({
            "wu": wu, "wl": wl, "wd": wd,
            "sf": sf, "sr": sr, "si": si, "sg": sg,
        })
        if self.enable_cold_region:
            state["swe"] = swe
            state["FI"] = FI
        total_storage = wu + wl + wd + sf + sr + si + sg
        if self.enable_cold_region:
            total_storage = total_storage + swe
        result = {
            "runoff": q_out,
            "others": {
                "storage": total_storage,
                "et": total_et,
                "sr_out": sr_out,
                "si_out": si_out,
                "sg_out": sg_out,
            }
        }
        if self.enable_cold_region:
            result["others"]["swe"] = swe
            result["others"]["snow_melt"] = snow_melt
            result["others"]["glacier_melt"] = glacier_melt
            result["others"]["frozen_factor"] = frozen_factor
            result["others"]["FI"] = FI
        return result, state

    def _define_parameters(self):
        names = [
            "wum", "wlm", "wdm", "b", "kc",
            "im", "sm", "ex",
            "kg", "ki", "ks",
            "cr", "ci", "cg"
        ]
        bounds = [
            [10, 100], [20, 200], [50, 300], [0.0, 1.5], [0.1, 1.0],
            [0, 0.4], [20, 500], [0.5, 2.0],
            [0.05, 0.35], [0.15, 0.55], [0.30, 0.70],
            [0.1, 0.9], [0.1, 0.9], [0.01, 0.5]
        ]
        init = [
            self.wum, self.wlm, self.wdm, self.b, self.kc,
            self.im, self.sm, self.ex,
            self.kg, self.ki, self.ks,
            self.cr, self.ci, self.cg
        ]
        network_mapping = {
            "wum": "grid_runoff", "wlm": "grid_runoff", "wdm": "grid_runoff",
            "b": "grid_runoff", "kc": "grid_runoff",
            "im": "grid_runoff", "sm": "grid_runoff", "ex": "grid_runoff",
            "kg": "grid_runoff", "ki": "grid_runoff", "ks": "grid_runoff",
            "cr": "grid_runoff", "ci": "grid_runoff", "cg": "grid_runoff"
        }
        if self.enable_cold_region:
            cold_init = [
                self.t_snow, self.t_melt, self.ddf_snow, self.k_snow,
                self.ddf_ice, self.t_ice_melt, self.glacier_runoff_frac,
                self.fi_alpha, self.fi_beta
            ]
            names = COLD_REGION_PARAM_NAMES + names
            bounds = COLD_REGION_PARAM_BOUNDS + bounds
            init = cold_init + init
            for n in COLD_REGION_PARAM_NAMES:
                network_mapping[n] = "grid_runoff"
        return {
            "static": {
                "names": names,
                "bounds": bounds,
                "init": init,
                "network_mapping": network_mapping
            },
            "dynamic": {"names": [], "bounds": []},
        }

    def init_state(self):
        state = {
            "wu": self.wum / 2,
            "wl": self.wlm / 2,
            "wd": self.wdm / 2,
            "sf": 0.0,
            "sr": 0.0,
            "si": 0.0,
            "sg": 0.0,
        }
        if self.enable_cold_region:
            state["swe"] = 0.0
            state["FI"] = 0.0
        return state


# ==============================================================================
# 2. SPHY 模型（适用于高山/冰川区域）
# ==============================================================================
@register(GRID_REGISTRY, "sphy")
class SPHYModel(BaseModel):
    """
    SPHY 模型（Spatial Processes in Hydrology）

    概述：面向高山与冰冻圈区域的分布式水文模型，包含较为精细的根域/次根域与地下水分配，
    并与冰冻圈过程耦合以模拟融雪与冰川补给。
    """
    def __init__(
        self,
        enable_cold_region=False,
        # 冰冻圈参数
        t_snow=0.0, t_melt=0.0, ddf_snow=4.0, k_snow=0.6,
        glacier_frac=0.0, ddf_ice=8.0, t_ice_melt=0.0,
        glacier_runoff_frac=0.8,
        fi_alpha=0.9, fi_beta=0.1,
        # SPHY 参数
        rootzone_depth=0.5, subzone_depth=2.0,
        ksat=100.0, field_cap=0.30, wilting_point=0.10,
        alpha_gw=0.01, delta_gw=0.5, gw_frac=0.3,
        modules=None
    ):
        super().__init__()
        self.enable_cold_region = enable_cold_region
        # 冰冻圈参数
        self.t_snow = t_snow
        self.t_melt = t_melt
        self.ddf_snow = ddf_snow
        self.k_snow = k_snow
        self.glacier_frac = glacier_frac
        self.ddf_ice = ddf_ice
        self.t_ice_melt = t_ice_melt
        self.glacier_runoff_frac = glacier_runoff_frac
        self.fi_alpha = fi_alpha
        self.fi_beta = fi_beta
        # SPHY 土壤与基流参数
        self.rootzone_depth = rootzone_depth
        self.subzone_depth = subzone_depth
        self.ksat = ksat
        self.field_cap = field_cap
        self.wilting_point = wilting_point
        self.alpha_gw = alpha_gw
        self.delta_gw = delta_gw
        self.gw_frac = gw_frac

    def forward(self, inputs, static_params, dynamic_params, state):
        P = inputs.get("P", 0.0)
        E = inputs.get("E", 0.0)
        device = P.device if isinstance(P, torch.Tensor) else None
        dtype = P.dtype if isinstance(P, torch.Tensor) else torch.float32
        def _t(x):
            return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=dtype, device=device)
        P, E = _t(P), _t(E)

        # 从参数或默认中读取土壤和基流参数
        rootzone_depth = _t(static_params.get("rootzone_depth", self.rootzone_depth))
        subzone_depth = _t(static_params.get("subzone_depth", self.subzone_depth))
        ksat = _t(static_params.get("ksat", self.ksat))
        field_cap = _t(static_params.get("field_cap", self.field_cap))
        wilting_point = _t(static_params.get("wilting_point", self.wilting_point))
        alpha_gw = _t(static_params.get("alpha_gw", self.alpha_gw))
        delta_gw = _t(static_params.get("delta_gw", self.delta_gw))
        gw_frac = _t(static_params.get("gw_frac", self.gw_frac))

        # 计算最大含水量与状态初始化
        rootzone_max = rootzone_depth * 1000 * (field_cap - wilting_point)
        subzone_max = subzone_depth * 1000 * (field_cap - wilting_point)
        rootzone = _t(state.get("rootzone", rootzone_max / 2))
        subzone = _t(state.get("subzone", subzone_max / 2))
        gw = _t(state.get("gw", 0.0))
        gw_delay = _t(state.get("gw_delay", 0.0))

        # 冰冻圈处理（若启用）
        snow_melt = _t(0.0)
        glacier_melt = _t(0.0)
        swe = _t(0.0)
        FI = _t(0.0)
        glacier_melt_surface = _t(0.0)
        glacier_melt_ground = _t(0.0)
        et_factor = _t(1.0)
        frozen_factor = _t(1.0)
        if self.enable_cold_region:
            T = _t(inputs.get("T", 10.0))
            swe = _t(state.get("swe", 0.0))
            FI = _t(state.get("FI", 0.0))
            glacier_frac = _t(inputs.get("glacier_frac", self.glacier_frac))
            cryo_params = {
                "t_snow": _t(static_params.get("t_snow", self.t_snow)),
                "t_melt": _t(static_params.get("t_melt", self.t_melt)),
                "ddf_snow": _t(static_params.get("ddf_snow", self.ddf_snow)),
                "k_snow": _t(static_params.get("k_snow", self.k_snow)),
                "ddf_ice": _t(static_params.get("ddf_ice", self.ddf_ice)),
                "t_ice_melt": _t(static_params.get("t_ice_melt", self.t_ice_melt)),
                "glacier_runoff_frac": _t(static_params.get("glacier_runoff_frac", self.glacier_runoff_frac)),
                "fi_alpha": _t(static_params.get("fi_alpha", self.fi_alpha)),
                "fi_beta": _t(static_params.get("fi_beta", self.fi_beta)),
            }
            cryo_result = process_cryosphere(P, T, swe, FI, glacier_frac, cryo_params)
            P_eff = cryo_result["P_eff"]
            swe = cryo_result["swe"]
            FI = cryo_result["FI"]
            snow_melt = cryo_result["snow_melt"]
            glacier_melt = cryo_result["glacier_melt"]
            glacier_melt_surface = cryo_result["glacier_melt_surface"]
            glacier_melt_ground = cryo_result["glacier_melt_ground"]
            frozen_factor = cryo_result["frozen_factor"]
            et_factor = cryo_result["et_factor"]
        else:
            P_eff = P

        # 1) 蒸散发：按根域可蒸散发量限制
        et_ratio = torch.clamp(rootzone / rootzone_max, min=0.0, max=1.0)
        et_actual = torch.minimum(rootzone, E * et_ratio * et_factor)
        rootzone = rootzone - et_actual

        # 2) 入渗与超饱和产生的径流
        runoff_excess = torch.clamp(P_eff - ksat, min=0.0)
        infil = P_eff - runoff_excess
        if self.enable_cold_region:
            actual_infil = infil * frozen_factor
            frozen_excess = infil - actual_infil
            runoff_excess = runoff_excess + frozen_excess
            infil = actual_infil
        rootzone_space = rootzone_max - rootzone
        rootzone_add = torch.minimum(rootzone_space, infil)
        rootzone = rootzone + rootzone_add
        infil_remain = infil - rootzone_add
        runoff_sat = infil_remain

        # 3) 下渗到下层（按根域饱和度的幂律）
        rootzone_sat = rootzone / rootzone_max
        perc = ksat * torch.pow(torch.clamp(rootzone_sat, min=1e-6), 3)
        perc = torch.minimum(perc, rootzone)
        rootzone = rootzone - perc
        subzone_add = perc * (1 - gw_frac)
        gw_recharge = perc * gw_frac
        subzone_space = subzone_max - subzone
        actual_subzone_add = torch.minimum(subzone_space, subzone_add)
        subzone_overflow = subzone_add - actual_subzone_add  # 溢出部分加入径流
        subzone = subzone + actual_subzone_add

        # 4) 壤中流（次根域到中层的流出）
        subzone_sat = subzone / subzone_max
        interflow = ksat * 0.1 * torch.pow(torch.clamp(subzone_sat, min=1e-6), 2)
        interflow = torch.minimum(interflow, subzone)
        subzone = subzone - interflow

        # 5) 基流与延迟项
        gw_delay = gw_delay + gw_recharge + glacier_melt_ground
        gw_release = delta_gw * gw_delay
        gw_delay = gw_delay - gw_release
        gw = gw + gw_release
        baseflow = alpha_gw * gw
        gw = gw - baseflow

        # 6) 汇总径流（包含subzone溢出）
        runoff = runoff_excess + runoff_sat + interflow + baseflow + glacier_melt_surface + subzone_overflow

        # 更新状态并返回
        state.update({
            "rootzone": rootzone,
            "subzone": subzone,
            "gw": gw,
            "gw_delay": gw_delay,
        })
        if self.enable_cold_region:
            state["swe"] = swe
            state["FI"] = FI
        storage = rootzone + subzone + gw + gw_delay
        if self.enable_cold_region:
            storage = storage + swe
        result = {
            "runoff": runoff,
            "others": {
                "et": et_actual,
                "surface_runoff": runoff_excess + runoff_sat + glacier_melt_surface,
                "interflow": interflow,
                "baseflow": baseflow,
                "storage": storage,
            }
        }
        if self.enable_cold_region:
            result["others"]["swe"] = swe
            result["others"]["snow_melt"] = snow_melt
            result["others"]["glacier_melt"] = glacier_melt
            result["others"]["frozen_factor"] = frozen_factor
            result["others"]["FI"] = FI
        return result, state

    def _define_parameters(self):
        names = [
            "rootzone_depth", "subzone_depth", "ksat", "field_cap", "wilting_point",
            "alpha_gw", "delta_gw", "gw_frac"
        ]
        bounds = [
            [0.1, 2.0], [0.5, 5.0], [10, 500], [0.15, 0.45], [0.05, 0.20],
            [0.001, 0.1], [0.1, 0.9], [0.1, 0.5]
        ]
        init = [
            self.rootzone_depth, self.subzone_depth, self.ksat,
            self.field_cap, self.wilting_point,
            self.alpha_gw, self.delta_gw, self.gw_frac
        ]
        network_mapping = {
            "rootzone_depth": "grid_runoff", "subzone_depth": "grid_runoff",
            "ksat": "grid_runoff", "field_cap": "grid_runoff", "wilting_point": "grid_runoff",
            "alpha_gw": "grid_runoff", "delta_gw": "grid_runoff", "gw_frac": "grid_runoff"
        }
        if self.enable_cold_region:
            cold_init = [
                self.t_snow, self.t_melt, self.ddf_snow, self.k_snow,
                self.ddf_ice, self.t_ice_melt, self.glacier_runoff_frac,
                self.fi_alpha, self.fi_beta
            ]
            names = COLD_REGION_PARAM_NAMES + names
            bounds = COLD_REGION_PARAM_BOUNDS + bounds
            init = cold_init + init
            for n in COLD_REGION_PARAM_NAMES:
                network_mapping[n] = "grid_runoff"
        return {
            "static": {
                "names": names,
                "bounds": bounds,
                "init": init,
                "network_mapping": network_mapping
            },
            "dynamic": {"names": [], "bounds": []},
        }

    def init_state(self):
        rootzone_max = self.rootzone_depth * 1000 * (self.field_cap - self.wilting_point)
        subzone_max = self.subzone_depth * 1000 * (self.field_cap - self.wilting_point)
        state = {
            "rootzone": rootzone_max / 2,
            "subzone": subzone_max / 2,
            "gw": 0.0,
            "gw_delay": 0.0,
        }
        if self.enable_cold_region:
            state["swe"] = 0.0
            state["FI"] = 0.0
        return state


# ==============================================================================
# 3. VIC 模型（Variable Infiltration Capacity）
# ==============================================================================
@register(GRID_REGISTRY, "vic")
class VICModel(BaseModel):
    """
    VIC 模型（栅格实现）

    概述：采用可变入渗容量曲线来描述空间上不同土壤含水能力，常用于大尺度流域的分布式模拟。
    本实现包含 ARNO 基流方案选择项、三层土壤储水及可选的冰冻圈耦合。
    """
    def __init__(
        self,
        enable_cold_region=False,
        # 冰冻圈参数
        t_snow=0.0, t_melt=0.0, ddf_snow=3.0, k_snow=0.5,
        glacier_frac=0.0, ddf_ice=6.0, t_ice_melt=0.0,
        glacier_runoff_frac=0.7,
        fi_alpha=0.9, fi_beta=0.1,
        # VIC 参数
        b_infil=0.2, im=0.05,
        d1=0.1, d2=0.5, d3=1.0,
        ksat=100.0, phi_s=0.45, phi_r=0.08, c=6.0,
        ds=0.1, dsmax=10.0, ws=0.5,
        modules=None
    ):
        super().__init__()
        self.enable_cold_region = enable_cold_region
        # 冰冻圈参数
        self.t_snow = t_snow
        self.t_melt = t_melt
        self.ddf_snow = ddf_snow
        self.k_snow = k_snow
        self.glacier_frac = glacier_frac
        self.ddf_ice = ddf_ice
        self.t_ice_melt = t_ice_melt
        self.glacier_runoff_frac = glacier_runoff_frac
        self.fi_alpha = fi_alpha
        self.fi_beta = fi_beta
        # VIC 参数
        self.b_infil = b_infil
        self.im = im
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.ksat = ksat
        self.phi_s = phi_s
        self.phi_r = phi_r
        self.c = c
        self.ds = ds
        self.dsmax = dsmax
        self.ws = ws

    def forward(self, inputs, static_params, dynamic_params, state):
        P = inputs.get("P", 0.0)
        E = inputs.get("E", 0.0)
        device = P.device if isinstance(P, torch.Tensor) else None
        dtype = P.dtype if isinstance(P, torch.Tensor) else torch.float32
        def _t(x):
            return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=dtype, device=device)
        P, E = _t(P), _t(E)

        # 读取 VIC 参数
        b_infil = _t(static_params.get("b_infil", self.b_infil))
        im = _t(static_params.get("im", self.im))
        d1 = _t(static_params.get("d1", self.d1))
        d2 = _t(static_params.get("d2", self.d2))
        d3 = _t(static_params.get("d3", self.d3))
        ksat = _t(static_params.get("ksat", self.ksat))
        phi_s = _t(static_params.get("phi_s", self.phi_s))
        phi_r = _t(static_params.get("phi_r", self.phi_r))
        c = _t(static_params.get("c", self.c))
        ds = _t(static_params.get("ds", self.ds))
        dsmax = _t(static_params.get("dsmax", self.dsmax))
        ws = _t(static_params.get("ws", self.ws))

        # 计算三层土壤最大可用水量（mm）
        sm1 = d1 * 1000 * (phi_s - phi_r)
        sm2 = d2 * 1000 * (phi_s - phi_r)
        sm3 = d3 * 1000 * (phi_s - phi_r)
        s1 = _t(state.get("s1", sm1 / 2))
        s2 = _t(state.get("s2", sm2 / 2))
        s3 = _t(state.get("s3", sm3 / 2))

        # 冰冻圈处理（若启用）
        snow_melt = _t(0.0)
        glacier_melt = _t(0.0)
        swe = _t(0.0)
        FI = _t(0.0)
        glacier_melt_surface = _t(0.0)
        glacier_melt_ground = _t(0.0)
        et_factor = _t(1.0)
        frozen_factor = _t(1.0)
        if self.enable_cold_region:
            T = _t(inputs.get("T", 10.0))
            swe = _t(state.get("swe", 0.0))
            FI = _t(state.get("FI", 0.0))
            glacier_frac = _t(inputs.get("glacier_frac", self.glacier_frac))
            cryo_params = {
                "t_snow": _t(static_params.get("t_snow", self.t_snow)),
                "t_melt": _t(static_params.get("t_melt", self.t_melt)),
                "ddf_snow": _t(static_params.get("ddf_snow", self.ddf_snow)),
                "k_snow": _t(static_params.get("k_snow", self.k_snow)),
                "ddf_ice": _t(static_params.get("ddf_ice", self.ddf_ice)),
                "t_ice_melt": _t(static_params.get("t_ice_melt", self.t_ice_melt)),
                "glacier_runoff_frac": _t(static_params.get("glacier_runoff_frac", self.glacier_runoff_frac)),
                "fi_alpha": _t(static_params.get("fi_alpha", self.fi_alpha)),
                "fi_beta": _t(static_params.get("fi_beta", self.fi_beta)),
            }
            cryo_result = process_cryosphere(P, T, swe, FI, glacier_frac, cryo_params)
            P_eff = cryo_result["P_eff"]
            swe = cryo_result["swe"]
            FI = cryo_result["FI"]
            snow_melt = cryo_result["snow_melt"]
            glacier_melt = cryo_result["glacier_melt"]
            glacier_melt_surface = cryo_result["glacier_melt_surface"]
            glacier_melt_ground = cryo_result["glacier_melt_ground"]
            frozen_factor = cryo_result["frozen_factor"]
            et_factor = cryo_result["et_factor"]
        else:
            P_eff = P

        # 1) 蒸散发分配
        e1 = torch.minimum(s1, E * et_factor)
        s1 = s1 - e1
        e_remain = E * et_factor - e1
        e2 = torch.minimum(s2, e_remain * 0.5)
        s2 = s2 - e2
        total_et = e1 + e2

        # 2) 入渗与地表径流（VIC 可变入渗容量曲线）
        runoff_im = im * P_eff
        P_avail = (1 - im) * P_eff
        rel_sat = (s1 + s2) / (sm1 + sm2 + 1e-6)
        rel_sat = torch.clamp(rel_sat, min=0.0, max=1.0)
        im_max = sm1 + sm2
        i_cap = im_max * (1 - torch.pow(torch.clamp(1 - rel_sat, min=1e-6), 1 + b_infil))
        runoff_excess = torch.clamp(P_avail - i_cap, min=0.0)
        infil = P_avail - runoff_excess
        if self.enable_cold_region:
            actual_infil = infil * frozen_factor
            frozen_excess = infil - actual_infil
            runoff_excess = runoff_excess + frozen_excess
            infil = actual_infil
        runoff_surface = runoff_im + runoff_excess + glacier_melt_surface

        # 3) 土壤水再分配与下渗（Campbell 公式）
        s1_space = sm1 - s1
        s1_add = torch.minimum(s1_space, infil)
        s1 = s1 + s1_add
        infil = infil - s1_add
        s2_space = sm2 - s2
        s2_add = torch.minimum(s2_space, infil)
        s2 = s2 + s2_add
        overflow = infil - s2_add
        runoff_surface = runoff_surface + overflow
        rel_sat1 = s1 / (sm1 + 1e-6)
        perc1 = ksat * torch.pow(torch.clamp(rel_sat1, min=1e-6), 2 * c + 3)
        perc1 = torch.minimum(perc1, s1)
        s1 = s1 - perc1
        s2 = s2 + perc1
        rel_sat2 = s2 / (sm2 + 1e-6)
        perc2 = ksat * 0.5 * torch.pow(torch.clamp(rel_sat2, min=1e-6), 2 * c + 3)
        perc2 = torch.minimum(perc2, s2)
        s2 = s2 - perc2
        s3 = s3 + perc2 + glacier_melt_ground

        # 4) 基流（ARNO 方案）
        rel_sat3 = s3 / (sm3 + 1e-6)
        rel_sat3 = torch.clamp(rel_sat3, min=0.0, max=1.0)
        baseflow = torch.where(
            rel_sat3 < ws,
            ds * dsmax * rel_sat3 / (ws + 1e-6),
            ds * dsmax * rel_sat3 / (ws + 1e-6) + (1 - ds) * dsmax *
            torch.pow((rel_sat3 - ws) / (1 - ws + 1e-6), 2)
        )
        baseflow = torch.minimum(baseflow, s3)
        s3 = s3 - baseflow

        # 5) 总径流
        runoff = runoff_surface + baseflow

        state.update({"s1": s1, "s2": s2, "s3": s3})
        if self.enable_cold_region:
            state["swe"] = swe
            state["FI"] = FI
        storage = s1 + s2 + s3
        if self.enable_cold_region:
            storage = storage + swe
        result = {
            "runoff": runoff,
            "others": {
                "et": total_et,
                "surface_runoff": runoff_surface,
                "baseflow": baseflow,
                "storage": storage,
            }
        }
        if self.enable_cold_region:
            result["others"]["swe"] = swe
            result["others"]["snow_melt"] = snow_melt
            result["others"]["glacier_melt"] = glacier_melt
            result["others"]["frozen_factor"] = frozen_factor
            result["others"]["FI"] = FI
        return result, state

    def _define_parameters(self):
        names = [
            "b_infil", "im", "d1", "d2", "d3",
            "ksat", "phi_s", "phi_r", "c",
            "ds", "dsmax", "ws"
        ]
        bounds = [
            [0.0, 0.5], [0.0, 0.2], [0.05, 0.3], [0.3, 1.5], [0.5, 2.5],
            [10, 500], [0.3, 0.6], [0.02, 0.15], [3, 15],
            [0.0, 1.0], [0.0, 50.0], [0.0, 1.0]
        ]
        init = [
            self.b_infil, self.im, self.d1, self.d2, self.d3,
            self.ksat, self.phi_s, self.phi_r, self.c,
            self.ds, self.dsmax, self.ws
        ]
        network_mapping = {
            "b_infil": "grid_runoff", "im": "grid_runoff",
            "d1": "grid_runoff", "d2": "grid_runoff", "d3": "grid_runoff",
            "ksat": "grid_runoff", "phi_s": "grid_runoff", "phi_r": "grid_runoff", "c": "grid_runoff",
            "ds": "grid_runoff", "dsmax": "grid_runoff", "ws": "grid_runoff"
        }
        if self.enable_cold_region:
            cold_names = [
                "t_snow", "t_melt", "ddf_snow", "k_snow",
                "ddf_ice", "t_ice_melt", "glacier_runoff_frac",
                "fi_alpha", "fi_beta"
            ]
            cold_bounds = [
                [-2, 2], [-1, 3], [1, 10], [0.1, 0.9],
                [3, 15], [0, 2], [0.3, 0.9],
                [0.5, 0.99], [0.01, 0.5]
            ]
            cold_init = [
                self.t_snow, self.t_melt, self.ddf_snow, self.k_snow,
                self.ddf_ice, self.t_ice_melt, self.glacier_runoff_frac,
                self.fi_alpha, self.fi_beta
            ]
            names = cold_names + names
            bounds = cold_bounds + bounds
            init = cold_init + init
            for n in cold_names:
                network_mapping[n] = "grid_runoff"
        return {
            "static": {
                "names": names,
                "bounds": bounds,
                "init": init,
                "network_mapping": network_mapping
            },
            "dynamic": {"names": [], "bounds": []},
        }

    def init_state(self):
        sm1 = self.d1 * 1000 * (self.phi_s - self.phi_r)
        sm2 = self.d2 * 1000 * (self.phi_s - self.phi_r)
        sm3 = self.d3 * 1000 * (self.phi_s - self.phi_r)
        state = {
            "s1": sm1 / 2,
            "s2": sm2 / 2,
            "s3": sm3 / 2,
        }
        if self.enable_cold_region:
            state["swe"] = 0.0
            state["FI"] = 0.0
        return state
