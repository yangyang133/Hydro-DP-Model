import torch
from models.base import BaseModel
from models.registry import SUBBASIN_REGISTRY, register
import math

# 从冰冻圈模块导入（与grid_runoff保持一致）
from models.physics.cryosphere import (
    process_cryosphere,
    COLD_REGION_DEFAULTS,
    COLD_REGION_BOUNDS,
    COLD_REGION_PARAM_NAMES,
    COLD_REGION_PARAM_BOUNDS,
    get_cold_region_params,
    get_cold_region_network_mapping,
)

# 从降水校正模块导入
from models.physics.precip_correct import (
    apply_precip_correction,
    PRECIP_CORRECT_DEFAULTS,
    get_precip_correct_params,
    get_precip_correct_network_mapping,
)

# 向后兼容常量别名
COLD_REGION_DEFAULT_VALUES = COLD_REGION_DEFAULTS


def _update_process_outputs(target, **kwargs):
    """只把 tensor 诊断量写入输出字典，避免影响主计算逻辑。"""
    for name, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            target[name] = value


def _append_common_inputs(target, P_input, E_input, P_corr, P_eff, T_input=None, glacier_frac_input=None):
    _update_process_outputs(
        target,
        forcing_precip_input=P_input,
        forcing_evap_input=E_input,
        forcing_precip_corrected=P_corr,
        forcing_precip_effective=P_eff,
        forcing_temperature=T_input,
        forcing_glacier_fraction=glacier_frac_input,
    )


def _append_cryosphere_outputs(target, cryo_result):
    if not cryo_result:
        return
    _update_process_outputs(
        target,
        cryosphere_rain_input=cryo_result.get("P_rain"),
        cryosphere_snow_input=cryo_result.get("P_snow"),
        cryosphere_swe=cryo_result.get("swe"),
        cryosphere_freezing_index=cryo_result.get("FI"),
        cryosphere_snow_melt=cryo_result.get("snow_melt"),
        cryosphere_glacier_melt=cryo_result.get("glacier_melt"),
        cryosphere_glacier_melt_surface=cryo_result.get("glacier_melt_surface"),
        cryosphere_glacier_melt_ground=cryo_result.get("glacier_melt_ground"),
        cryosphere_frozen_factor=cryo_result.get("frozen_factor"),
        cryosphere_et_factor=cryo_result.get("et_factor"),
    )


def _make_process_outputs(runoff, storage_total=None, evap_total=None):
    outputs = {}
    _update_process_outputs(
        outputs,
        storage_total=storage_total,
        evap_total=evap_total,
    )
    return outputs


def _route_cdf_unit_hydrograph(q_in, route_time, state, queue_name, max_len=10):
    """
    CDF-difference unit hydrograph routing for a single model time step.

    `route_time <= 1` is treated as no lag. The queue stores basin-average water
    depth that has entered the unit hydrograph but has not yet left it.
    """
    dtype = q_in.dtype
    device = q_in.device
    route_t = route_time if isinstance(route_time, torch.Tensor) else torch.tensor(route_time, dtype=dtype, device=device)
    route_t = route_t.to(dtype=dtype, device=device)
    route_t = torch.clamp(route_t, min=1.0, max=float(max_len))

    x0 = torch.arange(0, max_len, dtype=dtype, device=device)
    x1 = x0 + 1.0

    def _cdf(x, mb_value):
        x_clip = torch.minimum(torch.clamp(x, min=0.0), mb_value)
        half = 0.5 * mb_value
        lower = 2.0 * torch.pow(x_clip / mb_value, 2.0)
        upper = 1.0 - 2.0 * torch.pow((mb_value - x_clip) / mb_value, 2.0)
        return torch.where(x_clip <= half, lower, upper)

    original_shape = q_in.shape
    if q_in.numel() > 1:
        q_flat = q_in.reshape(-1)
        n_items = q_flat.numel()
        lag_queue = state.get(queue_name, None)
        if lag_queue is None or not isinstance(lag_queue, torch.Tensor) or lag_queue.dim() != 2 or lag_queue.shape != (n_items, max_len):
            lag_queue = torch.zeros(n_items, max_len, dtype=dtype, device=device)

        route_flat = route_t.reshape(-1)
        if route_flat.numel() == 1:
            mb = route_flat.reshape(1, 1).expand(n_items, 1)
            direct_mask = (route_flat.reshape(1) <= 1.0).expand(n_items)
        else:
            mb = route_flat.reshape(n_items, 1)
            direct_mask = route_flat.reshape(n_items) <= 1.0

        weights = _cdf(x1.unsqueeze(0), mb) - _cdf(x0.unsqueeze(0), mb)
        weights = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-12)

        lag_queue = lag_queue + q_flat.unsqueeze(1) * weights
        routed_flat = lag_queue[:, 0]
        shifted_queue = torch.cat([lag_queue[:, 1:], torch.zeros(n_items, 1, dtype=dtype, device=device)], dim=1)
        q_out_flat = torch.where(direct_mask, q_flat, routed_flat)
        lag_queue = torch.where(direct_mask.unsqueeze(1), torch.zeros_like(shifted_queue), shifted_queue)
        queue_storage = lag_queue.sum(dim=1)
        first_weight = torch.where(direct_mask, torch.ones_like(routed_flat), weights[:, 0])

        return (
            q_out_flat.reshape(original_shape),
            lag_queue,
            queue_storage.reshape(original_shape),
            first_weight.reshape(original_shape),
        )

    q_scalar = q_in.reshape(())
    lag_queue = state.get(queue_name, None)
    if lag_queue is None or not isinstance(lag_queue, torch.Tensor) or lag_queue.dim() != 1 or lag_queue.numel() != max_len:
        lag_queue = torch.zeros(max_len, dtype=dtype, device=device)

    route_scalar = route_t.reshape(-1)[0]
    if bool((route_scalar <= 1.0).detach().item()):
        q_out = q_scalar
        lag_queue = torch.zeros_like(lag_queue)
        first_weight = torch.ones_like(q_scalar)
    else:
        mb = route_scalar.reshape(())
        weights = _cdf(x1, mb) - _cdf(x0, mb)
        weights = weights / torch.clamp(weights.sum(), min=1e-12)
        lag_queue = lag_queue + q_scalar * weights
        q_out = lag_queue[0]
        lag_queue = torch.cat([lag_queue[1:], torch.zeros(1, dtype=dtype, device=device)])
        first_weight = weights[0]

    queue_storage = lag_queue.sum()
    return (
        q_out.reshape(original_shape),
        lag_queue,
        queue_storage.reshape(original_shape),
        first_weight.reshape(original_shape),
    )


# ==============================================================================
# 水文模型定义
# ==============================================================================

@register(SUBBASIN_REGISTRY, "xaj")

class XAJModel(BaseModel):
    """
    新安江模型（日尺度）- 严格标准产流版，支持冰冻圈开关

    标准过程：
        - 净雨 PE = P_eff - KC * E，降水优先满足蒸散需求。
        - 三层蒸散采用上/中/下层张力水标准规则，含参数 c。
        - 不透水面积 IM 先作用于净雨 PE，形成直接径流。
        - 透水区以 PE 深度进入蓄水容量曲线：WMM = WM * (1 + B)，产流后再乘 (1 - IM)。
        - 透水区 PE 扣除产流后的水量按透水区局地水深补充上层、中层、深层张力水。
        - 自由水按 ki/kg 产生 RI/RG，未流出的自由水继续保留在 S 中。
        - 三水源响应后可接 lag 滞后单位线；lag <= 1 时退化为无滞后直接输出。
        - 冷区过程保留：有效降水、冻土入渗抑制、积雪/冰川融水；蒸散不使用 et_factor。

    参数说明（保留当前代码参数名，对应常用新安江参数表）：

        | 序号 | 模型结构 | 当前参数 | 标准名称 | 参数名称 | 取值范围 | 默认值 | 有效位数 | 单位 |
        |---:|---|---|---|---|---|---:|---:|---|
        | 1 | 蒸发计算 | kc | KC | 蒸散折算系数 | [0.50, 2.00] | 1.00 | 3 | - |
        | 2 | 蒸发计算 | wum | WUM | 上层张力水蓄水容量 | [5, 30] | 10 | 2 | mm |
        | 3 | 蒸发计算 | wlm | WLM | 下层/中层张力水蓄水容量 | [40, 90] | 70 | 2 | mm |
        | 4 | 蒸发计算 | wdm | WDM | 深层张力水蓄水容量 | [35, 80] | 50 | 2 | mm |
        | 5 | 蒸发计算 | c | C | 深层蒸散系数 | [0.09, 0.20] | 0.16 | 2 | - |
        | 6 | 产流计算 | im | IMP | 不透水面占全流域面积之比 | [0.005, 0.50] | 0.01 | 3 | - |
        | 7 | 产流计算 | wm | WM | 总张力水蓄水容量，当前实现中为 wum + wlm + wdm | [80, 200] | 120 | 3 | mm |
        | 8 | 产流计算 | b | B | 指数型张力水蓄水容量曲线方次 | [0.1, 0.4] | 0.2 | 1 | - |
        | 9 | 水源划分 | sm | SM | 自由水蓄水容量 | [5, 50] | 10 | 2 | mm |
        | 10 | 水源划分 | ex | EX | 指数型自由水蓄水容量曲线方次 | [1.0, 2.0] | 1.5 | 2 | - |
        | 11 | 汇流计算 | ki | KI | 壤中流/集中流出系数 | [0.10, 0.50] | 0.35 | 2 | - |
        | 12 | 汇流计算 | kg | KG | 地下水出流系数 | [0.10, 0.50] | 0.35 | 2 | - |
        | 13 | 汇流计算 | ci | CI | 壤中流/集中流消减系数 | [0.010, 0.999] | 0.800 | 3 | - |
        | 14 | 汇流计算 | cg | CG | 地下径流消减系数 | [0.050, 0.999] | 0.950 | 3 | - |
        | 15 | 汇流计算 | cr | CS | 计算单元内地表/河网汇流系数 | [0.010, 0.999] | 0.990 | 3 | - |
        | 16 | 汇流计算 | lag | LAG | 计算单元内河网汇流滞时 | [0, 5] | 0 | 1 | 计算时段数 |

        说明：
        - `wm` 不是独立构造参数，而是在 forward 中由 `wum + wlm + wdm` 计算得到。
        - `cr` 对应常用表中的 `CS`；`cr/ci/cg` 均按退水保留系数使用，实际出流比例为 `1 - 系数`。
        - `ki` 和 `kg` 的上界均为 0.5，本时段未由 `ki + kg` 流出的自由水保留在 `sf` 中。
        - `lag_queue` 是内部状态，保存尚未流出滞后单位线的全流域平均水量；多子流域时形状为 [n_subs, 5]。

    === 冰冻圈参数（enable_cold_region=True时启用）===
        - t_snow: 雨雪分界温度 (°C) [-2, 2]
        - t_melt: 融雪温度阈值 (°C) [-1, 3]
        - ddf_snow: 积雪度日因子 (mm/°C/day) [1, 10]
        - k_snow: 积雪消融系数 [0.1, 0.9]
        - ddf_ice: 冰川度日因子 (mm/°C/day) [3, 15]
        - t_ice_melt: 冰川融化温度阈值 (°C) [0, 2]
        - glacier_runoff_frac: 冰川融水地表径流比例 [0.3, 0.9]
        - fi_alpha: 冻结指数衰减系数 [0.5, 0.99]
        - fi_beta: 冻结指数影响系数 [0.01, 0.5]
    """

    def __init__(
        self,
        # 冰冻圈开关
        enable_cold_region=False,
        # 降水校正开关
        enable_precip_correct=False,
        # 冰冻圈参数（仅enable_cold_region=True时使用）
        t_snow=0.0, t_melt=0.0, ddf_snow=3.0, k_snow=0.5,
        glacier_frac=0.0, ddf_ice=6.0, t_ice_melt=0.0,
        glacier_runoff_frac=0.7,
        fi_alpha=0.9, fi_beta=0.1, 
        # 降水校正参数（仅enable_precip_correct=True时使用）
        kp=1.0,
        # 原XAJ参数
        wum=10.0, wlm=70.0, wdm=50.0,
        b=0.2, kc=1.0, c=0.16,
        im=0.01,
        sm=10.0, ex=1.5,
        kg=0.35, ki=0.35,
        cr=0.99, ci=0.8, cg=0.95,
        lag=0.0,
        modules=None
    ):
        super().__init__()
        
        # 冰冻圈开关
        self.enable_cold_region = enable_cold_region
        
        # 降水校正开关
        self.enable_precip_correct = enable_precip_correct
        self.kp = kp
        
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

        # 原XAJ参数
        self.wum = wum
        self.wlm = wlm
        self.wdm = wdm
        self.b = b
        self.kc = kc
        self.c = c

        self.im = im
        self.sm = sm
        self.ex = ex

        self.kg = kg
        self.ki = ki

        self.cr = cr
        self.ci = ci
        self.cg = cg
        self.lag = lag

    # ----------------------------
    # 单时间步计算
    # ----------------------------
    def forward(self, inputs, static_params, dynamic_params, state, return_details=True):

        P = inputs.get("P", 0.0)
        E = inputs.get("E", 0.0)

        # device / dtype helpers
        device = P.device if isinstance(P, torch.Tensor) else None
        dtype = P.dtype if isinstance(P, torch.Tensor) else torch.float32
        def _t(x):
            return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=dtype, device=device)

        P = _t(P)
        E = _t(E)
        P_input = P
        E_input = E
        P_corr = P
        T_input = None
        glacier_frac_input = None
        cryo_result = None
        
        # ==================================================
        # 降水校正（如果启用）
        # ==================================================
        if self.enable_precip_correct:
            kp = _t(static_params.get("kp", self.kp))
            P = apply_precip_correction(P, kp)
        P_corr = P
        
        # ==================================================
        # 冰冻圈过程（如果启用）
        # ==================================================
        snow_melt = _t(0.0)
        glacier_melt = _t(0.0)
        swe = _t(0.0)
        FI = _t(0.0)
        glacier_melt_surface = _t(0.0)
        glacier_melt_ground = _t(0.0)
        frozen_factor = _t(1.0)
        
        if self.enable_cold_region:
            T = _t(inputs.get("T", 10.0))
            T_input = T
            swe = _t(state.get("swe", 0.0))
            FI = _t(state.get("FI", 0.0))
            glacier_frac = _t(inputs.get("glacier_frac", self.glacier_frac))
            glacier_frac_input = glacier_frac
            
            # 构建冰冻圈参数
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
            
            # 调用冰冻圈过程函数
            cryo_result = process_cryosphere(P, T, swe, FI, glacier_frac, cryo_params)
            
            # 提取结果
            P_eff = cryo_result["P_eff"]
            swe = cryo_result["swe"]
            FI = cryo_result["FI"]
            snow_melt = cryo_result["snow_melt"]
            glacier_melt = cryo_result["glacier_melt"]
            glacier_melt_surface = cryo_result["glacier_melt_surface"]
            glacier_melt_ground = cryo_result["glacier_melt_ground"]
            frozen_factor = cryo_result["frozen_factor"]
        else:
            P_eff = P

        # === 参数 ===
        wum = _t(static_params.get("wum", self.wum))
        wlm = _t(static_params.get("wlm", self.wlm))
        wdm = _t(static_params.get("wdm", self.wdm))
        wm = wum + wlm + wdm
        b  = _t(static_params.get("b",  self.b))
        kc = _t(static_params.get("kc", self.kc))
        c = _t(static_params.get("c", self.c))

        im = torch.clamp(_t(static_params.get("im", self.im)), min=0.0, max=1.0)
        pervious_frac = torch.clamp(1 - im, min=0.0)
        sm = _t(static_params.get("sm", self.sm))
        ex = _t(static_params.get("ex", self.ex))

        kg = _t(static_params.get("kg", self.kg))
        ki = _t(static_params.get("ki", self.ki))

        cr = _t(static_params.get("cr", self.cr))
        ci = _t(static_params.get("ci", self.ci))
        cg = _t(static_params.get("cg", self.cg))
        lag = _t(static_params.get("lag", self.lag))

        # === 状态 (ensure tensors) ===
        wu = _t(state.get("wu", wum / 2))
        wl = _t(state.get("wl", wlm / 2))
        wd = _t(state.get("wd", wdm / 2))

        sf = _t(state.get("sf", 0.0))
        si = _t(state.get("si", 0.0))
        sg = _t(state.get("sg", 0.0))

        # ==================================================
        # 1. 三层蒸散发（张量/矢量化实现）
        # ==================================================
        ep = E * kc
        pe = torch.clamp(P_eff - ep, min=0.0)
        precip_impervious = torch.clamp(im * pe, min=0.0)
        precip_pervious = pe

        # 标准新安江三层蒸散：降水先满足 EP，不足部分再按三层张力水规则消耗土壤水。
        precip_evap = torch.minimum(ep, P_eff)
        soil_ep = torch.clamp(ep - P_eff, min=0.0)
        eu_soil = torch.minimum(soil_ep, wu)
        eu = precip_evap + eu_soil
        wu = torch.clamp(wu - eu_soil, min=0.0)

        et_remaining = torch.clamp(soil_ep - eu_soil, min=0.0)
        wl_before = wl
        wl_ratio = torch.clamp(wl_before / torch.clamp(wlm, min=1e-6), min=0.0, max=1.0)
        el_high = torch.minimum(wl_before, et_remaining * wl_ratio)
        el_mid = torch.minimum(wl_before, c * et_remaining)
        el_low = wl_before
        el = torch.where(
            wl_before >= c * wlm,
            el_high,
            torch.where(wl_before >= c * et_remaining, el_mid, el_low),
        )
        ed_demand = torch.where(
            wl_before >= c * wlm,
            torch.zeros_like(et_remaining),
            torch.where(
                wl_before >= c * et_remaining,
                torch.zeros_like(et_remaining),
                torch.clamp(c * et_remaining - el, min=0.0),
            ),
        )
        ed = torch.minimum(wd, ed_demand)
        wl = torch.clamp(wl_before - el, min=0.0)
        wd = torch.clamp(wd - ed, min=0.0)

        w = wu + wl + wd

        # 标准蓄满产流：容量曲线在透水面积上使用 PE 深度，产流量最后再折算为流域平均。
        wmm = wm * (1 + b)
        w_ratio = torch.clamp(w / torch.clamp(wm, min=1e-6), min=0.0, max=1.0)
        A = wmm * (1 - torch.pow(1 - w_ratio, 1 / torch.clamp(1 + b, min=1e-6)))

        ap = A + precip_pervious
        runoff_candidate = precip_pervious - (wm - w) + wm * torch.pow(
            torch.clamp(1 - ap / torch.clamp(wmm, min=1e-6), min=0.0),
            1 + b,
        )
        runoff_alt = precip_pervious - (wm - w)
        runoff_generated = torch.where(ap < wmm, runoff_candidate, runoff_alt)
        runoff_generated = torch.where(precip_pervious <= 0, torch.zeros_like(precip_pervious), runoff_generated)
        runoff_generated = torch.clamp(runoff_generated, min=0.0)
        runoff_generated = torch.minimum(runoff_generated, precip_pervious)

        potential_infil_area = torch.clamp(precip_pervious - runoff_generated, min=0.0)
        actual_infil_area = potential_infil_area
        frozen_excess_area = _t(0.0)
        if self.enable_cold_region:
            actual_infil_area = potential_infil_area * frozen_factor
            frozen_excess_area = potential_infil_area - actual_infil_area
            runoff_generated = runoff_generated + frozen_excess_area

        # 张力水状态 WU/WL/WD 按透水区局地水深维护，因此补给不乘 (1 - im)。
        potential_infil = potential_infil_area
        actual_infil = actual_infil_area
        frozen_excess = frozen_excess_area

        # 标准张力水补给顺序：上层 -> 中层 -> 深层。
        wu_new = wu + actual_infil
        wu_overflow = torch.clamp(wu_new - wum, min=0.0)
        wu = torch.minimum(wu_new, wum)

        wl_new = wl + wu_overflow
        wl_overflow = torch.clamp(wl_new - wlm, min=0.0)
        wl = torch.minimum(wl_new, wlm)

        wd_new = wd + wl_overflow
        wd_overflow = torch.clamp(wd_new - wdm, min=0.0)
        wd = torch.minimum(wd_new, wdm)

        runoff_im = precip_impervious
        runoff_p_local = runoff_generated + wd_overflow
        Pp = precip_pervious

        # 产流面积 FR 与自由水蓄水库 S：
        # - runoff_p_local 是透水区局地产流深，尚未乘 (1 - im)。
        # - fr 是透水区内部的产流面积比例。
        # - sf 按透水区局地自由水深维护，整个自由水与三水源划分过程不乘 pervious_frac。
        fr = torch.where(
            Pp > 1e-6,
            torch.clamp(runoff_p_local / torch.clamp(Pp, min=1e-6), min=0.0, max=1.0),
            torch.zeros_like(Pp),
        )
        free_water_area = torch.where(
            fr > 1e-6,
            torch.clamp(sf / torch.clamp(fr, min=1e-6), min=0.0, max=sm),
            torch.zeros_like(sf),
        )
        free_input_area = torch.where(
            fr > 1e-6,
            runoff_p_local / torch.clamp(fr, min=1e-6),
            torch.zeros_like(runoff_p_local),
        )

        # 自由水容量曲线在产流面积上的局地水深尺度上计算。
        # free_water_area 表示产流面积上的自由水深；free_input_area 表示本时段进入自由水库的局地产流深。
        smm = sm * (1 + ex)
        free_ratio = torch.clamp(free_water_area / torch.clamp(sm, min=1e-6), min=0.0, max=1.0)
        au = smm * (1 - torch.pow(1 - free_ratio, 1 / torch.clamp(1 + ex, min=1e-6)))
        ap_free = au + free_input_area

        # RS 为地面径流分量：
        # 若自由水蓄水曲线尚未达到最大容量，用曲线积分差计算 RS；
        # 若超过最大容量，则超出自由水容量的部分全部作为地面径流。
        rs_area_candidate = free_input_area - (sm - free_water_area) + sm * torch.pow(
            torch.clamp(1 - ap_free / torch.clamp(smm, min=1e-6), min=0.0),
            1 + ex,
        )
        rs_area_alt = free_input_area - (sm - free_water_area)
        rs_area = torch.where(ap_free < smm, rs_area_candidate, rs_area_alt)
        rs_area = torch.clamp(rs_area, min=0.0)
        rs_area = torch.minimum(rs_area, free_input_area + free_water_area)

        # rs_area 是产流面积上的局地地面径流深，rs_local 是透水区局地地面径流深。
        rs_local = fr * rs_area

        # 剩余自由水保存在 S 中，再按 KI/KG 产生壤中流 RI 和地下水 RG。
        # sf、ri_local、rg_local 均为透水区局地水深；若 ki + kg 在数值上超过 1，用 scale 防止抽干超过现有自由水。
        sf = torch.clamp(sf + runoff_p_local - rs_local, min=0.0)
        qi_raw = ki * sf
        qg_raw = kg * sf
        subsurface_out = qi_raw + qg_raw
        subsurface_scale = torch.where(
            subsurface_out > sf,
            sf / torch.clamp(subsurface_out, min=1e-6),
            torch.ones_like(sf),
        )
        ri_local = qi_raw * subsurface_scale  # 壤中流入流 RI（透水区局地）
        rg_local = qg_raw * subsurface_scale  # 地下水入流 RG（透水区局地）
        sf = torch.clamp(sf - ri_local - rg_local, min=0.0)

        # 最终汇总到全流域平均水量：只有这里乘 pervious_frac。
        runoff_p = pervious_frac * runoff_p_local
        runoff_total = runoff_im + runoff_p
        rs = pervious_frac * rs_local
        qi = pervious_frac * ri_local
        qg = pervious_frac * rg_local
        qs = runoff_im + rs  # 地面径流入流 QS = RIM + (1 - IM) * RS

        # ==================================================
        # 4. 各分量汇流（线性水库）
        # ==================================================
        sr = _t(state.get("sr", 0.0))
        sr = sr + qs + glacier_melt_surface  # 冰川地表融水进入地表径流
        sr_release = torch.clamp(1.0 - cr, min=0.0, max=1.0)
        sr_out = torch.minimum(sr_release * sr, sr)
        sr = torch.clamp(sr - sr_out, min=0.0)
        
        # 壤中流水库
        si = si + qi
        si_release = torch.clamp(1.0 - ci, min=0.0, max=1.0)
        si_out = torch.minimum(si_release * si, si)
        si = si - si_out

        # 地下水水库
        sg = sg + qg + glacier_melt_ground  # 冰川地下融水进入地下水
        sg_release = torch.clamp(1.0 - cg, min=0.0, max=1.0)
        sg_out = torch.minimum(sg_release * sg, sg)
        sg = sg - sg_out
        
        # 确保非负
        sr = torch.clamp(sr, min=0.0)
        si = torch.clamp(si, min=0.0)
        sg = torch.clamp(sg, min=0.0)

        q_response = torch.clamp(sr_out + si_out + sg_out, min=0.0)

        # 三水源响应之后的滞后单位线。它只表示子流域内到出口的时间分配，
        # 后续若还有河道 Muskingum，应把这里的输出作为河道侧向入流。
        q_out, lag_queue, lag_queue_storage, lag_first_weight = _route_cdf_unit_hydrograph(
            q_response,
            lag,
            state,
            queue_name="lag_queue",
            max_len=5,
        )
        q_out = torch.clamp(q_out, min=0.0)

        # ==================================================
        # 更新状态
        # ==================================================
        state.update({
            "wu": wu,
            "wl": wl,
            "wd": wd,
            "sf": sf,
            "sr": sr,
            "si": si,
            "sg": sg,
            "lag_queue": lag_queue,
        })
        
        # 冰冻圈状态
        if self.enable_cold_region:
            state.update({
                "swe": swe,
                "FI": FI,
            })

        # 计算总储水量
        if not return_details:
            return {"runoff": q_out}, state

        total_storage = pervious_frac * (wu + wl + wd + sf) + sr + si + sg + lag_queue_storage
        if self.enable_cold_region:
            total_storage = total_storage + swe
        total_et = precip_evap + pervious_frac * (eu_soil + el + ed)

        others = _make_process_outputs(q_out, total_storage, total_et)
        _update_process_outputs(
            others,
            storage_upper_tension=wu,
            storage_lower_tension=wl,
            storage_deep_tension=wd,
            storage_free_water=sf,
            storage_surface_reservoir=sr,
            storage_interflow_reservoir=si,
            storage_groundwater_reservoir=sg,
            storage_lag_queue=lag_queue_storage,
            runoff_impervious=runoff_im,
            runoff_saturation_excess=runoff_p,
            runoff_total_generated=runoff_total,
            runoff_surface_freewater=rs,
            runoff_area_fraction=fr,
            runoff_free_water_out=rs + qi + qg,
            runoff_surface_generated=qs,
            runoff_interflow_generated=qi,
            runoff_groundwater_generated=qg,
            runoff_surface_out=sr_out,
            runoff_interflow_out=si_out,
            runoff_groundwater_out=sg_out,
            runoff_response_before_lag=q_response,
            runoff_lagged=q_out,
            lag_time=lag,
            lag_first_weight=lag_first_weight,
            evap_precip=precip_evap,
            evap_soil_potential=soil_ep,
            evap_upper_soil=eu_soil,
            evap_upper=precip_evap + pervious_frac * eu_soil,
            evap_middle=pervious_frac * el,
            evap_deep=pervious_frac * ed,
            soil_water_total=w,
            pervious_area_fraction=pervious_frac,
            precip_impervious=precip_impervious,
            precip_pervious=Pp,
            net_rain_total=pe,
            net_rain_pervious=Pp,
            free_water_area_depth=free_water_area,
            infiltration_potential=potential_infil,
            infiltration_actual=actual_infil,
            infiltration_frozen_excess=frozen_excess,
        )

        _append_common_inputs(
            others,
            P_input=P_input,
            E_input=E_input,
            P_corr=P_corr,
            P_eff=P_eff,
            T_input=T_input,
            glacier_frac_input=glacier_frac_input,
        )
        _append_cryosphere_outputs(others, cryo_result)

        return {"runoff": q_out, "others": others}, state
    
    def _get_cr_defaults(self):
        """获取冰冻圈参数默认值"""
        return {
            "t_snow": self.t_snow,
            "t_melt": self.t_melt,
            "ddf_snow": self.ddf_snow,
            "k_snow": self.k_snow,
            "glacier_frac": self.glacier_frac,
            "ddf_ice": self.ddf_ice,
            "t_ice_melt": self.t_ice_melt,
            "glacier_runoff_frac": self.glacier_runoff_frac,
            "fi_alpha": self.fi_alpha,
            "fi_beta": self.fi_beta,
        }

    # ----------------------------
    # 参数定义
    # ----------------------------
    def _define_parameters(self):
        # 基础参数
        names = [
            "kc", "wum", "wlm", "wdm", "c",
            "im", "b", "sm", "ex",
            "ki", "kg",
            "ci", "cg", "cr", "lag"
        ]
        bounds = [
            [0.50, 2.00], [5, 30], [40, 90], [35, 80], [0.09, 0.20],
            [0.005, 0.50], [0.1, 0.4], [5, 50], [1.0, 2.0],
            [0.10, 0.50], [0.10, 0.50],
            [0.010, 0.999], [0.050, 0.999], [0.010, 0.999], [0, 0]
        ]
        init = [
            self.kc, self.wum, self.wlm, self.wdm, self.c,
            self.im, self.b, self.sm, self.ex,
            self.ki, self.kg,
            self.ci, self.cg, self.cr, self.lag
        ]
        network_mapping = {
            "wum": "subbasin_runoff", "wlm": "subbasin_runoff", "wdm": "subbasin_runoff",
            "b": "subbasin_runoff", "kc": "subbasin_runoff", "c": "subbasin_runoff",
            "im": "subbasin_runoff", "sm": "subbasin_runoff", "ex": "subbasin_runoff",
            "kg": "subbasin_runoff", "ki": "subbasin_runoff",
            "cr": "hillslope_routing", "ci": "hillslope_routing", "cg": "hillslope_routing",
            "lag": "hillslope_routing"
        }
        # 无延时无调蓄参数：退水保留系数取下界（最快退水），产流参数取init值
        passthrough = [
            self.kc, self.wum, self.wlm, self.wdm, self.c,
            self.im, self.b, self.sm, self.ex,
            self.ki, self.kg,
            0.010, 0.050, 0.010, 0  # ci, cg, cr: 最快退水（下界）；lag: 无额外滞后
        ]
        
        # 如果启用冰冻圈，添加冰冻圈参数
        if self.enable_cold_region:
            cr_names, cr_bounds, cr_init = get_cold_region_params(self._get_cr_defaults())
            names = cr_names + names
            bounds = cr_bounds + bounds
            init = cr_init + init
            passthrough = cr_init + passthrough
            network_mapping.update(get_cold_region_network_mapping())
        
        # 如果启用降水校正，添加降水校正参数
        if self.enable_precip_correct:
            kp_names, kp_bounds, kp_init = get_precip_correct_params(self)
            names = kp_names + names
            bounds = kp_bounds + bounds
            init = kp_init + init
            passthrough = kp_init + passthrough
            network_mapping.update(get_precip_correct_network_mapping())
        
        return {
            "static": {
                "names": names,
                "bounds": bounds,
                "init": init,
                "passthrough": passthrough,
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
            "lag_queue": 0.0,
        }
        if self.enable_cold_region:
            state.update({
                "swe": 0.0,
                "FI": 0.0,
            })
        return state


@register(SUBBASIN_REGISTRY, "gr4j")
class GR4JModel(BaseModel):
    """
    标准 GR4J（日尺度）模型

    蒸发输入为潜在蒸发能力
    x2，流域完全闭合时设置为[0,0]， 后期可设置约束条件，∑(Exch i)≈0
    两个问题：1参数补偿效应，弥补降水观测数据误差；2物理逻辑冲突，相邻子流域无水力联系
    
    参数说明（共4个）：
    
    === 生产水库参数 ===
        - x1: 生产水库容量 (mm) [100-2500]
          物理含义：流域土壤含水量最大值；x1越大，土壤持水能力越强
          模型逻辑：降水分为净雨和下渗，由生产水库容量决定分配比例
          推荐范围：[200-1000] for 湿润地区；[100-500] for 干旱地区
    
    === 地下水交换参数 ===
        - x2: 地下水交换系数 (mm/day) [-10-10]
          物理含义：相邻子流域间的地下水侧向交换量（正向为流入，负向为流出）
          模型逻辑：在路由水库阶段，与该水库的蓄水量成立方关系
          推荐范围：[-2-2] for 独立流域；[-5-5] for 相邻流域间有交换
    
    === 路由水库参数 ===
        - x3: 路由水库容量 (mm) [10-500]
          物理含义：流域地表及地下汇流的蓄水池容量
          模型逻辑：控制基流衰减和汇流延迟的强弱
          推荐范围：[50-300] for 一般流域
    
    === 单位线参数 ===
        - x4: 单位线基础时间 (day) [1-10]
          物理含义：流域汇流时间常数；x4越大，汇流延迟越长
          模型逻辑：生成两条S型单位线（快速和缓慢）：
                   - 快速线：90% 净雨通过x4天内输出，常数为0.9*x4
                   - 缓慢线：10% 净雨通过2*x4天内输出，常数为2*x4
          推荐范围：[2-5] for 中小流域；[5-10] for 大型流域
    
    === 水热耦合参数（可选扩展） ===
        可与HBV或其他冰冻圈模型耦合，添加：
        - t_melt: 融雪临界温度 (°C)
        - ddf: 融雪度日因子 (mm/°C)
    """

    def __init__(
        self,
        # 冰冻圈开关
        enable_cold_region=False,
        # 降水校正开关
        enable_precip_correct=False,
        # 冰冻圈参数（仅enable_cold_region=True时使用）
        t_snow=0.0, t_melt=0.0, ddf_snow=3.0, k_snow=0.5,
        glacier_frac=0.0, ddf_ice=6.0, t_ice_melt=0.0,
        glacier_runoff_frac=0.7,
        fi_alpha=0.9, fi_beta=0.1, 
        # 降水校正参数（仅enable_precip_correct=True时使用）
        kp=1.0,
        # 原GR4J参数
        x1=300.0, x2=0.0, x3=100.0, x4=2.0,
        modules=None
    ):
        super().__init__()
        
        # 冰冻圈开关
        self.enable_cold_region = enable_cold_region
        
        # 降水校正开关
        self.enable_precip_correct = enable_precip_correct
        self.kp = kp
        
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
        
        # 原GR4J参数
        self.x1 = x1  
        self.x2 = x2  
        self.x3 = x3  
        self.x4 = x4  

    def forward(self, inputs, static_params, dynamic_params, state, return_details=True):
        """
        GR4J forward方法，支持向量化（多子流域批量处理）
        """
        P = inputs.get("P", 0.0)
        E = inputs.get("E", 0.0)

        # helpers
        device = P.device if isinstance(P, torch.Tensor) else None
        dtype = P.dtype if isinstance(P, torch.Tensor) else torch.float32
        def _t(x):
            return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=dtype, device=device)

        P = _t(P)
        E = _t(E)
        P_input = P
        E_input = E
        P_corr = P
        T_input = None
        glacier_frac_input = None
        cryo_result = None
        
        # ==================================================
        # 降水校正（如果启用）
        # ==================================================
        if self.enable_precip_correct:
            kp = _t(static_params.get("kp", self.kp))
            P = apply_precip_correction(P, kp)
        P_corr = P
        
        # ==================================================
        # 冰冻圈过程（如果启用）
        # ==================================================
        snow_melt = _t(0.0)
        glacier_melt = _t(0.0)
        swe = _t(0.0)
        FI = _t(0.0)
        glacier_melt_surface = _t(0.0)
        glacier_melt_ground = _t(0.0)
        frozen_factor = _t(1.0)
        
        if self.enable_cold_region:
            T = _t(inputs.get("T", 10.0))
            T_input = T
            swe = _t(state.get("swe", 0.0))
            FI = _t(state.get("FI", 0.0))
            glacier_frac = _t(inputs.get("glacier_frac", self.glacier_frac))
            glacier_frac_input = glacier_frac
            
            # 构建冰冻圈参数
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
            
            # 调用冰冻圈过程函数
            cryo_result = process_cryosphere(P, T, swe, FI, glacier_frac, cryo_params)
            
            # 提取结果
            P_eff = cryo_result["P_eff"]
            swe = cryo_result["swe"]
            FI = cryo_result["FI"]
            snow_melt = cryo_result["snow_melt"]
            glacier_melt = cryo_result["glacier_melt"]
            glacier_melt_surface = cryo_result["glacier_melt_surface"]
            glacier_melt_ground = cryo_result["glacier_melt_ground"]
            frozen_factor = cryo_result["frozen_factor"]
        else:
            P_eff = P
        
        # 检测是否为向量化模式（多个子流域）
        is_vectorized = P_eff.dim() >= 1 and P_eff.numel() > 1
        max_uh_len = 20  # 固定最大单位线长度

        x1 = _t(static_params.get("x1", self.x1))
        x2 = _t(static_params.get("x2", self.x2))
        x2 = torch.zeros_like(x2)
        x3 = _t(static_params.get("x3", self.x3))
        x4 = _t(static_params.get("x4", self.x4))

        S = _t(state.get("S", (x1 / 2)))
        R = _t(state.get("R", (x3 / 2)))
        
        # 初始化UH（支持向量化）
        if is_vectorized:
            n_subs = P.shape[0]
            # 从状态获取UH，如果不存在或形状不对则创建新的
            UH1 = state.get("UH1", None)
            UH2 = state.get("UH2", None)
            # 确保UH有正确的形状 [n_subs, max_uh_len]
            if UH1 is None or not isinstance(UH1, torch.Tensor) or UH1.dim() != 2 or UH1.shape[0] != n_subs or UH1.shape[1] != max_uh_len:
                UH1 = torch.zeros(n_subs, max_uh_len, dtype=dtype, device=device)
            if UH2 is None or not isinstance(UH2, torch.Tensor) or UH2.dim() != 2 or UH2.shape[0] != n_subs or UH2.shape[1] != max_uh_len:
                UH2 = torch.zeros(n_subs, max_uh_len, dtype=dtype, device=device)
        else:
            UH1 = state.get("UH1", None)
            UH2 = state.get("UH2", None)
            if UH1 is None or not isinstance(UH1, torch.Tensor) or UH1.numel() != max_uh_len:
                UH1 = torch.zeros(max_uh_len, dtype=dtype, device=device)
            if UH2 is None or not isinstance(UH2, torch.Tensor) or UH2.numel() != max_uh_len:
                UH2 = torch.zeros(max_uh_len, dtype=dtype, device=device)

        # ========= 1. 净雨 / 净蒸散 (tensorized) =========
        E_adj = E
        Pn = torch.clamp(P_eff - E_adj, min=0.0)
        En = torch.clamp(E_adj - P_eff, min=0.0)

        # ========= 2. 生产水库 (use branch-free selection via where) =========
        mask = Pn > 0

        tmp_pos = torch.tanh(Pn / x1)
        Ps_pos = x1 * (1 - torch.pow(S / x1, 2)) * tmp_pos / (1 + (S / x1) * tmp_pos) 
        S_pos = S + Ps_pos
        Pr_pos = Pn - Ps_pos

        tmp_neg = torch.tanh(En / x1)
        Es_neg = S * (2 - S / x1) * tmp_neg / (1 + (1 - S / x1) * tmp_neg)
        S_neg = S - Es_neg
        Pr_neg = torch.zeros_like(Pn)

        S = torch.where(mask, S_pos, S_neg)
        Pr = torch.where(mask, Pr_pos, Pr_neg)

        # 渗漏（Perc）
        Perc = S * (1 - torch.pow(1 + torch.pow(4 * S / (9 * x1), 4), -0.25))
        S = S - Perc

        # ========= 3. 汇流输入 =========
        Pr = Pr + Perc

        # ========= 4. 单位线（向量化支持）=========
        if is_vectorized:
            UH1 = self._uh_convolution_vec(0.9 * Pr, UH1, x4, max_uh_len, fast=True)
            UH2 = self._uh_convolution_vec(0.1 * Pr, UH2, x4, max_uh_len, fast=False)
            uh1_out = UH1[:, 0]
            uh2_out = UH2[:, 0]
        else:
            UH1 = self._uh_convolution_scalar(0.9 * Pr, UH1, x4, max_uh_len, fast=True)
            UH2 = self._uh_convolution_scalar(0.1 * Pr, UH2, x4, max_uh_len, fast=False)
            uh1_out = UH1[0]
            uh2_out = UH2[0]

        # ========= 5. 地下水交换 =========
        R_ratio = torch.clamp(R / torch.clamp(x3, min=1e-6), min=0.0)
        Exch = x2 * torch.pow(R_ratio, 3.5)
        R = torch.clamp(R + uh1_out + Exch + glacier_melt_ground, min=0.0)

        # ========= 6. 路由水库 =========
        Qr_out = R * (1 - torch.pow(1 + torch.pow(R / x3, 4), -0.25)) 
        R = R - Qr_out

        # ========= 总出流 =========
        Qd_out = torch.clamp(uh2_out + Exch, min=0.0)
        Q = torch.clamp(Qr_out + Qd_out + glacier_melt_surface, min=0.0)

        # shift UH queues
        if is_vectorized:
            UH1 = torch.cat([UH1[:, 1:], torch.zeros(n_subs, 1, dtype=dtype, device=device)], dim=1)
            UH2 = torch.cat([UH2[:, 1:], torch.zeros(n_subs, 1, dtype=dtype, device=device)], dim=1)
        else:
            UH1 = torch.cat([UH1[1:], torch.zeros(1, dtype=dtype, device=device)])
            UH2 = torch.cat([UH2[1:], torch.zeros(1, dtype=dtype, device=device)])

        state.update({
            "S": S,
            "R": R,
            "UH1": UH1,
            "UH2": UH2,
        })
        
        # 冰冻圈状态
        if self.enable_cold_region:
            state.update({
                "swe": swe,
                "FI": FI,
            })

        if not return_details:
            return {"runoff": Q}, state

        # 计算蒸散量和总蓄水量
        # GR4J蒸发逻辑：
        # - 当P>=E时，蒸发需求被降水满足，实际蒸发 = E（降水截留蒸发）
        # - 当P<E时，蒸发需求未被满足，从土壤蒸发 Es_neg = E - P + 从土壤的额外蒸发
        # 正确的ET = min(P, E) + Es_neg（当P<E时Es_neg>0，当P>=E时Es_neg=0）
        # 使用标准蒸发输入 E，冷区 et_factor 不参与 GR4J 蒸发。
        actual_et = torch.where(mask, E_adj, P_eff + Es_neg)  # P>=E时ET=E_adj; P<E时ET=P_eff+Es_neg
        
        # storage 需要包含单位线(UH)中的延迟水量
        if is_vectorized:
            uh_storage = UH1.sum(dim=1) + UH2.sum(dim=1)
            uh1_storage = UH1.sum(dim=1)
            uh2_storage = UH2.sum(dim=1)
        else:
            uh_storage = UH1.sum() + UH2.sum()
            uh1_storage = UH1.sum()
            uh2_storage = UH2.sum()
        total_storage = S + R + uh_storage  # 生产水库 + 路由水库 + 单位线延迟水量
        if self.enable_cold_region:
            total_storage = total_storage + swe  # 加上雪水当量
        
        # GR4J特殊：地下水交换(Exch)是外部水量交换
        # 水量平衡：P + Exch = ET + Q + ΔS
        Ps = torch.where(mask, Ps_pos, torch.zeros_like(Pn))
        Es = torch.where(mask, torch.zeros_like(En), Es_neg)

        others = _make_process_outputs(Q, total_storage, actual_et)
        _update_process_outputs(
            others,
            storage_production=S,
            storage_routing=R,
            storage_uh_total=uh_storage,
            storage_uh1=uh1_storage,
            storage_uh2=uh2_storage,
            exchange_groundwater=Exch,
            percolation=Perc,
            runoff_effective=Pr,
            runoff_routing_out=Qr_out,
            runoff_uh1_out=uh1_out,
            runoff_uh2_out=uh2_out,
            evap_potential=E_adj,
            precip_net=Pn,
            evap_net=En,
            production_store_fill=Ps,
            production_store_evap=Es,
        )

        _append_common_inputs(
            others,
            P_input=P_input,
            E_input=E_input,
            P_corr=P_corr,
            P_eff=P_eff,
            T_input=T_input,
            glacier_frac_input=glacier_frac_input,
        )
        _append_cryosphere_outputs(others, cryo_result)

        return {"runoff": Q, "others": others}, state

    # ==========================================================
    # 向量化单位线
    # ==========================================================
    def _uh_convolution_vec(self, inflow, UH, x4, max_len, fast=True):
        """向量化单位线卷积: inflow[n_subs], UH[n_subs,max_len], x4可为标量或向量"""
        n_subs = inflow.shape[0]
        device, dtype = inflow.device, inflow.dtype
        
        t = torch.arange(1, max_len + 1, device=device, dtype=dtype).unsqueeze(0)
        
        # 处理x4可能是标量的情况
        if x4.dim() == 0:
            x4_exp = torch.clamp(x4.unsqueeze(0).unsqueeze(1).expand(n_subs, max_len), min=1.0)
        else:
            x4_exp = torch.clamp(x4.unsqueeze(1).expand(-1, max_len), min=1.0)
        
        # 使用clamp确保指数底数为正，避免负数的非整数次幂产生NaN
        ratio = torch.clamp(t / x4_exp, min=0.0, max=2.0)
        
        if fast:
            S = torch.where(t <= x4_exp, torch.pow(ratio, 2.5), torch.ones(n_subs, max_len, device=device, dtype=dtype))
        else:
            ratio2 = torch.clamp(2.0 - ratio, min=0.0)  # 确保非负
            S = torch.where(t <= x4_exp, 0.5 * torch.pow(ratio, 2.5),
                    torch.where(t <= 2 * x4_exp, 1 - 0.5 * torch.pow(ratio2, 2.5),
                                torch.ones(n_subs, max_len, device=device, dtype=dtype)))
        
        diffs = torch.cat([S[:, :1], S[:, 1:] - S[:, :-1]], dim=1)
        return UH + inflow.unsqueeze(1) * diffs

    def _uh_convolution_scalar(self, inflow, UH, x4, max_len, fast=True):
        """标量单位线卷积"""
        device, dtype = inflow.device, inflow.dtype
        t = torch.arange(1, max_len + 1, device=device, dtype=dtype)
        x4 = torch.clamp(x4, min=1.0)  # 确保x4 >= 1
        
        ratio = torch.clamp(t / x4, min=0.0, max=2.0)
        
        if fast:
            S = torch.where(t <= x4, torch.pow(ratio, 2.5), torch.ones_like(t))
        else:
            ratio2 = torch.clamp(2.0 - ratio, min=0.0)
            S = torch.where(t <= x4, 0.5 * torch.pow(ratio, 2.5),
                    torch.where(t <= 2 * x4, 1 - 0.5 * torch.pow(ratio2, 2.5), torch.ones_like(t)))
        
        diffs = torch.cat([S[:1], S[1:] - S[:-1]])
        return UH + inflow * diffs

    def _get_cr_defaults(self):
        """获取冰冻圈参数默认值"""
        return {
            "t_snow": self.t_snow,
            "t_melt": self.t_melt,
            "ddf_snow": self.ddf_snow,
            "k_snow": self.k_snow,
            "glacier_frac": self.glacier_frac,
            "ddf_ice": self.ddf_ice,
            "t_ice_melt": self.t_ice_melt,
            "glacier_runoff_frac": self.glacier_runoff_frac,
            "fi_alpha": self.fi_alpha,
            "fi_beta": self.fi_beta,
        }

    def init_state(self):
        state = {
            "S": self.x1 / 2,
            "R": self.x3 / 2,
            "UH1": torch.zeros(0, dtype=torch.float32),
            "UH2": torch.zeros(0, dtype=torch.float32),
        }
        if self.enable_cold_region:
            state.update({
                "swe": 0.0,
                "FI": 0.0,
            })
        return state

    def _define_parameters(self):
        # 基础参数
        names = ["x1", "x2", "x3", "x4"]
        bounds = [
            [100, 2500],
            [0, 0],
            [10, 500],
            [1, 10],
        ]
        init = [self.x1, self.x2, self.x3, self.x4]
        network_mapping = {
            # 产流参数：生产水库
            "x1": "subbasin_runoff", "x2": "subbasin_runoff",
            # 汇流参数：汇流水库容量和单位线时间
            "x3": "hillslope_routing", "x4": "hillslope_routing"
        }
        # 无延时无调蓄参数：x3取下界（最小蓄水），x4取下界（最短延迟）
        passthrough = [self.x1, self.x2, 10, 1]  # x3=10, x4=1
        
        # 如果启用冰冻圈，添加冰冻圈参数
        if self.enable_cold_region:
            cr_names, cr_bounds, cr_init = get_cold_region_params(self._get_cr_defaults())
            names = cr_names + names
            bounds = cr_bounds + bounds
            init = cr_init + init
            passthrough = cr_init + passthrough
            network_mapping.update(get_cold_region_network_mapping())
        
        # 如果启用降水校正，添加降水校正参数
        if self.enable_precip_correct:
            kp_names, kp_bounds, kp_init = get_precip_correct_params(self)
            names = kp_names + names
            bounds = kp_bounds + bounds
            init = kp_init + init
            passthrough = kp_init + passthrough
            network_mapping.update(get_precip_correct_network_mapping())
        
        return {
            "static": {
                "names": names,
                "bounds": bounds,
                "init": init,
                "passthrough": passthrough,
                "network_mapping": network_mapping
            },
            "dynamic": {"names": [], "bounds": []},
        }


@register(SUBBASIN_REGISTRY, "sacsma")
class SACSMAModel(BaseModel):
    """
    SAC-SMA (Sacramento Soil Moisture Accounting) - 标准 16 参数版

    核心逻辑：两层土壤、张力水/自由水分离、非线性渗透、多出流路径、
    固定不透水面积和附加不透水面积处理。当前实现保留项目统一的冰冻圈和
    降水校正接口；蒸散不使用冷区 et_factor。

    参数说明（保留当前代码参数名，对应常用 SAC-SMA 参数表）：

        | 序号 | 模型结构 | 当前参数 | 标准名称 | 参数名称 | 取值范围 | 默认值 | 单位 |
        |---:|---|---|---|---|---|---:|---|
        | 1 | 上层蓄水 | uztwm | UZTWM | 上层张力水容量 | [5, 100] | 30 | mm |
        | 2 | 上层蓄水 | uzfwm | UZFWM | 上层自由水容量 | [40, 300] | 80 | mm |
        | 3 | 下层蓄水 | lztwm | LZTWM | 下层张力水容量 | [50, 500] | 150 | mm |
        | 4 | 下层蓄水 | lzfpm | LZFPM | 下层主自由水容量 | [50, 1000] | 200 | mm |
        | 5 | 下层蓄水 | lzfsm | LZFSM | 下层补充自由水容量 | [5, 200] | 40 | mm |
        | 6 | 不透水面积 | pctim | PCTIM | 固定不透水面积比例 | [0.0, 0.4] | 0.0 | - |
        | 7 | 不透水面积 | adimp | ADIMP | 附加不透水面积比例 | [0.0, 0.3] | 0.0 | - |
        | 8 | 渗透计算 | zperc | ZPERC | 下层缺水对渗透的放大系数 | [1, 250] | 1.5 | - |
        | 9 | 渗透计算 | rexp | REXP | 渗透非线性指数 | [0, 5] | 1.5 | - |
        | 10 | 汇流计算 | uzk | UZK | 上层自由水退水系数 | [0.10, 0.50] | 0.35 | 1/day |
        | 11 | 汇流计算 | lzsk | LZSK | 下层补充自由水退水系数 | [0.01, 0.10] | 0.05 | 1/day |
        | 12 | 汇流计算 | lzpk | LZPK | 下层主自由水退水系数 | [0.002, 0.05] | 0.02 | 1/day |
        | 13 | 渗透分配 | pfree | PFREE | 渗漏水直接进入下层自由水的比例 | [0.0, 0.6] | 0.2 | - |
        | 14 | 下层出流 | rserv | RSERV | 下层主自由水不可排泄保留比例 | [0.0, 0.5] | 0.3 | - |
        | 15 | 下层出流 | side | SIDE | 深层侧向损失/增益比例 | [0.0, 0.2] | 0.0 | - |
        | 16 | 蒸散计算 | iv | RIVA | 河岸/开阔水面面积比例 | [0.0, 0.1] | 0.0 | - |

        说明：
        - `iv` 对应 SAC-SMA 常用参数 `RIVA`。
        - 标准 SAC-SMA 中没有 `pxv`，本实现已去除；下层蒸散由 `lztw / (uztwm + lztwm)` 控制。
        - `side=0` 时不产生深层侧向水量调整；非零时侧向出流计入总出流并保留 `runoff_sideflow` 诊断项。
        - 冷区扩展中，冰川地下融水进入 `lzfp`；若超过 `lzfpm`，超出部分作为下层溢流出流。
    """
    def __init__(
        self,
        # 冰冻圈开关
        enable_cold_region=False,
        # 降水校正开关
        enable_precip_correct=False,
        # 冰冻圈参数（仅enable_cold_region=True时使用）
        t_snow=0.0, t_melt=0.0, ddf_snow=3.0, k_snow=0.5,
        glacier_frac=0.0, ddf_ice=6.0, t_ice_melt=0.0,
        glacier_runoff_frac=0.7,
        fi_alpha=0.9, fi_beta=0.1, 
        # 降水校正参数（仅enable_precip_correct=True时使用）
        kp=1.0,
        # SAC-SMA参数
        uztwm=30.0, uzfwm=80.0, lztwm=150.0, lzfpm=200.0, lzfsm=40.0,
        pctim=0.0, adimp=0.0,
        rexp=1.5, zperc=1.5,
        lzpk=0.02, lzsk=0.05, uzk=0.35,
        pfree=0.2, rserv=0.3, side=0.0,
        iv=0.0,
        modules=None
    ):
        super().__init__()
        
        # 冰冻圈开关
        self.enable_cold_region = enable_cold_region
        
        # 降水校正开关
        self.enable_precip_correct = enable_precip_correct
        self.kp = kp
        
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
        
        # SAC-SMA参数
        self.uztwm = uztwm
        self.uzfwm = uzfwm
        self.lztwm = lztwm
        self.lzfpm = lzfpm
        self.lzfsm = lzfsm
        self.pctim = pctim
        self.adimp = adimp
        self.rexp = rexp
        self.zperc = zperc
        self.lzpk = lzpk
        self.lzsk = lzsk
        self.uzk = uzk
        self.pfree = pfree
        self.rserv = rserv
        self.side = side
        self.iv = iv

    def _get_cr_defaults(self):
        """获取冰冻圈参数默认值"""
        return {
            "t_snow": self.t_snow,
            "t_melt": self.t_melt,
            "ddf_snow": self.ddf_snow,
            "k_snow": self.k_snow,
            "glacier_frac": self.glacier_frac,
            "ddf_ice": self.ddf_ice,
            "t_ice_melt": self.t_ice_melt,
            "glacier_runoff_frac": self.glacier_runoff_frac,
            "fi_alpha": self.fi_alpha,
            "fi_beta": self.fi_beta,
        }

    def _define_parameters(self):
        # 基础参数
        names = [
            "uztwm", "uzfwm", "lztwm", "lzfpm", "lzfsm",
            "pctim", "adimp", "zperc", "rexp",
            "uzk", "lzsk", "lzpk", "pfree", "rserv", "side", "iv"
        ]
        bounds = [
            [5, 100], [40, 300], [50, 500], [50, 1000], [5, 200],
            [0.0, 0.4], [0.0, 0.3], [1.0, 250.0], [0.0, 5.0],
            [0.1, 0.5], [0.01, 0.1], [0.002, 0.05],
            [0.0, 0.6], [0.0, 0.5], [0.0, 0.2], [0.0, 0.1]
        ]
        init = [
            self.uztwm, self.uzfwm, self.lztwm, self.lzfpm, self.lzfsm,
            self.pctim, self.adimp, self.zperc, self.rexp,
            self.uzk, self.lzsk, self.lzpk,
            self.pfree, self.rserv, self.side, self.iv
        ]
        # 四类参数分类：产流参数(subbasin_runoff) vs 汇流参数(hillslope_routing)
        network_mapping = {
            # 产流参数：蓄水容量、不透水面积、渗透参数
            "uztwm": "subbasin_runoff", "uzfwm": "subbasin_runoff",
            "lztwm": "subbasin_runoff", "lzfpm": "subbasin_runoff", "lzfsm": "subbasin_runoff",
            "pctim": "subbasin_runoff", "adimp": "subbasin_runoff",
            "rexp": "subbasin_runoff", "zperc": "subbasin_runoff",
            "pfree": "subbasin_runoff", "rserv": "subbasin_runoff", "side": "subbasin_runoff",
            "iv": "subbasin_runoff",
            # 汇流参数：出流衰减系数
            "lzpk": "hillslope_routing", "lzsk": "hillslope_routing", "uzk": "hillslope_routing"
        }
        # 无延时无调蓄参数：衰减系数取上界（最快排空）
        passthrough = [
            self.uztwm, self.uzfwm, self.lztwm, self.lzfpm, self.lzfsm,
            self.pctim, self.adimp, self.zperc, self.rexp,
            0.5, 0.1, 0.05,  # uzk, lzsk, lzpk: 最快衰减（上界）
            self.pfree, self.rserv, self.side, self.iv
        ]
        
        # 如果启用冰冻圈，添加冰冻圈参数
        if self.enable_cold_region:
            cr_names, cr_bounds, cr_init = get_cold_region_params(self._get_cr_defaults())
            names = cr_names + names
            bounds = cr_bounds + bounds
            init = cr_init + init
            passthrough = cr_init + passthrough
            network_mapping.update(get_cold_region_network_mapping())
        
        # 如果启用降水校正，添加降水校正参数
        if self.enable_precip_correct:
            kp_names, kp_bounds, kp_init = get_precip_correct_params(self)
            names = kp_names + names
            bounds = kp_bounds + bounds
            init = kp_init + init
            passthrough = kp_init + passthrough
            network_mapping.update(get_precip_correct_network_mapping())
        
        return {
            "static": {
                "names": names,
                "bounds": bounds,
                "init": init,
                "passthrough": passthrough,
                "network_mapping": network_mapping
            },
            "dynamic": {"names": [], "bounds": []},
        }

    def init_state(self):
        state = {
            "uztw": self.uztwm / 2,
            "uzfw": self.uzfwm / 2,
            "lztw": self.lztwm / 2,
            "lzfp": self.lzfpm / 2,
            "lzfs": self.lzfsm / 2,
            "adimc": self.uztwm / 2 + self.lztwm / 2,
        }
        if self.enable_cold_region:
            state.update({
                "swe": 0.0,
                "FI": 0.0,
            })
        return state

    def forward(self, inputs, static_params, dynamic_params, state, return_details=True):
        """Standard 16-parameter SAC-SMA single-step calculation."""
        P = inputs.get("P", 0.0)
        E = inputs.get("E", 0.0)

        device = P.device if isinstance(P, torch.Tensor) else None
        dtype = P.dtype if isinstance(P, torch.Tensor) else torch.float32

        def _t(x):
            return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=dtype, device=device)

        P, E = _t(P), _t(E)
        P_input = P
        E_input = E
        P_corr = P
        T_input = None
        glacier_frac_input = None
        cryo_result = None

        if self.enable_precip_correct:
            kp = _t(static_params.get("kp", self.kp))
            P = apply_precip_correction(P, kp)
        P_corr = P

        snow_melt = _t(0.0)
        glacier_melt = _t(0.0)
        swe = _t(0.0)
        FI = _t(0.0)
        glacier_melt_surface = _t(0.0)
        glacier_melt_ground = _t(0.0)
        frozen_factor = _t(1.0)

        if self.enable_cold_region:
            T = _t(inputs.get("T", 10.0))
            T_input = T
            swe = _t(state.get("swe", 0.0))
            FI = _t(state.get("FI", 0.0))
            glacier_frac = _t(inputs.get("glacier_frac", self.glacier_frac))
            glacier_frac_input = glacier_frac

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
        else:
            P_eff = P
        P_eff_input = P_eff

        uztwm = _t(static_params.get("uztwm", self.uztwm))
        uzfwm = _t(static_params.get("uzfwm", self.uzfwm))
        lztwm = _t(static_params.get("lztwm", self.lztwm))
        lzfpm = _t(static_params.get("lzfpm", self.lzfpm))
        lzfsm = _t(static_params.get("lzfsm", self.lzfsm))
        pctim = _t(static_params.get("pctim", self.pctim))
        adimp = _t(static_params.get("adimp", self.adimp))
        rexp = _t(static_params.get("rexp", self.rexp))
        zperc = _t(static_params.get("zperc", self.zperc))
        lzpk = _t(static_params.get("lzpk", self.lzpk))
        lzsk = _t(static_params.get("lzsk", self.lzsk))
        uzk = _t(static_params.get("uzk", self.uzk))
        pfree = _t(static_params.get("pfree", self.pfree))
        rserv = _t(static_params.get("rserv", self.rserv))
        side = _t(static_params.get("side", self.side))
        iv = _t(static_params.get("iv", self.iv))

        uztw = _t(state.get("uztw", uztwm / 2))
        uzfw = _t(state.get("uzfw", uzfwm / 2))
        lztw = _t(state.get("lztw", lztwm / 2))
        lzfp = _t(state.get("lzfp", lzfpm / 2))
        lzfs = _t(state.get("lzfs", lzfsm / 2))
        adimc = _t(state.get("adimc", uztw + lztw))

        uztwm_safe = torch.clamp(uztwm, min=1e-6)
        uzfwm_safe = torch.clamp(uzfwm, min=1e-6)
        lztwm_safe = torch.clamp(lztwm, min=1e-6)
        lzfpm_safe = torch.clamp(lzfpm, min=1e-6)
        lzfsm_safe = torch.clamp(lzfsm, min=1e-6)

        pctim = torch.clamp(pctim, min=0.0, max=1.0)
        adimp = torch.clamp(adimp, min=0.0, max=1.0)
        adimp_eff = adimp
        parea = torch.clamp(1.0 - pctim - adimp, min=0.0)
        pfree = torch.clamp(pfree, min=0.0, max=1.0)
        rserv = torch.clamp(rserv, min=0.0, max=1.0)
        iv = torch.clamp(iv, min=0.0, max=1.0)
        side = torch.clamp(side, min=0.0)
        zperc = torch.clamp(zperc, min=0.0)
        rexp = torch.clamp(rexp, min=0.0)
        uzk = torch.clamp(uzk, min=0.0, max=1.0)
        lzsk = torch.clamp(lzsk, min=0.0, max=1.0)
        lzpk = torch.clamp(lzpk, min=0.0, max=1.0)

        uztw = torch.clamp(uztw, min=0.0, max=uztwm)
        uzfw = torch.clamp(uzfw, min=0.0, max=uzfwm)
        lztw = torch.clamp(lztw, min=0.0, max=lztwm)
        lzfp = torch.clamp(lzfp, min=0.0, max=lzfpm)
        lzfs = torch.clamp(lzfs, min=0.0, max=lzfsm)
        adimc = torch.clamp(adimc, min=uztw, max=uztwm + lztwm)

        pr = torch.clamp(P_eff_input, min=0.0)
        edmnd = torch.clamp(E, min=0.0)
        E_eff = edmnd

        # Upper-zone and lower-zone evapotranspiration follows the classic
        # SAC-SMA tension-water accounting; no extra deep ET coefficient is used.
        et_uztw_potential = edmnd * torch.clamp(uztw / uztwm_safe, min=0.0, max=1.0)
        et_uztw = torch.minimum(et_uztw_potential, uztw)
        uztw = torch.clamp(uztw - et_uztw, min=0.0)
        et_remaining = torch.clamp(edmnd - et_uztw, min=0.0)

        upper_tension_empty = et_uztw_potential > et_uztw
        et_uzfw = torch.where(upper_tension_empty, torch.minimum(uzfw, et_remaining), torch.zeros_like(uzfw))
        uzfw = torch.clamp(uzfw - et_uzfw, min=0.0)
        et_remaining = torch.where(upper_tension_empty, torch.clamp(et_remaining - et_uzfw, min=0.0), et_remaining)

        uz_tension_ratio = torch.clamp(uztw / uztwm_safe, min=0.0, max=1.0)
        uz_free_ratio = torch.clamp(uzfw / uzfwm_safe, min=0.0, max=1.0)
        uz_balance = torch.clamp((uztw + uzfw) / torch.clamp(uztwm + uzfwm, min=1e-6), min=0.0, max=1.0)
        rebalance_upper = (~upper_tension_empty) & (uz_tension_ratio < uz_free_ratio)
        uztw = torch.where(rebalance_upper, uztwm * uz_balance, uztw)
        uzfw = torch.where(rebalance_upper, uzfwm * uz_balance, uzfw)

        lower_et_fraction = torch.clamp(lztw / torch.clamp(uztwm + lztwm, min=1e-6), min=0.0, max=1.0)
        et_lztw = torch.minimum(lztw, et_remaining * lower_et_fraction)
        lztw = torch.clamp(lztw - et_lztw, min=0.0)
        et_remaining = torch.clamp(et_remaining - et_lztw, min=0.0)

        saved_lower_free = rserv * (lzfpm + lzfsm)
        lower_ratio_den = torch.clamp(lztwm + lzfpm + lzfsm - saved_lower_free, min=1e-6)
        rat_lzt = torch.clamp(lztw / lztwm_safe, min=0.0, max=1.0)
        rat_lz = torch.clamp((lztw + lzfp + lzfs - saved_lower_free) / lower_ratio_den, min=0.0, max=1.0)
        transfer_need = torch.where(rat_lzt < rat_lz, (rat_lz - rat_lzt) * lztwm, torch.zeros_like(lztw))
        transfer = torch.minimum(transfer_need, lzfp + lzfs)
        transfer_from_lzfs = torch.minimum(lzfs, transfer)
        lzfs = torch.clamp(lzfs - transfer_from_lzfs, min=0.0)
        transfer_from_lzfp = torch.minimum(lzfp, torch.clamp(transfer - transfer_from_lzfs, min=0.0))
        lzfp = torch.clamp(lzfp - transfer_from_lzfp, min=0.0)
        lztw = torch.minimum(lztw + transfer_from_lzfs + transfer_from_lzfp, lztwm)

        et_adimp_potential = et_uztw + (et_remaining + et_uzfw) * torch.clamp(
            (adimc - et_uztw - uztw) / torch.clamp(uztwm + lztwm, min=1e-6),
            min=0.0,
            max=1.0,
        )
        et_adimp = torch.minimum(adimc, torch.clamp(et_adimp_potential, min=0.0))
        adimc = torch.clamp(adimc - et_adimp, min=0.0)

        et_uz = et_uztw + et_uzfw
        et_lz = et_lztw
        et_lzfs = torch.zeros_like(lzfs)
        et_lzfp = torch.zeros_like(lzfp)

        uztw_before_pr = uztw
        twx = pr + uztw - uztwm
        no_tension_excess = twx <= 0.0
        uztw = torch.where(no_tension_excess, uztw + pr, uztwm)
        twx = torch.where(no_tension_excess, torch.zeros_like(twx), twx)
        dt_uztw = torch.clamp(uztw - uztw_before_pr, min=0.0)
        adimc = torch.clamp(adimc + pr - twx, min=0.0)

        ninc = torch.floor(1.0 + 0.2 * (uzfw + twx)).clamp(min=1.0)
        max_ninc = int(ninc.detach().max().item()) if isinstance(ninc, torch.Tensor) else int(ninc)
        dinc = 1.0 / ninc
        pinc = twx / ninc
        duz = 1.0 - torch.pow(torch.clamp(1.0 - uzk, min=0.0, max=1.0), dinc)
        dlzp = 1.0 - torch.pow(torch.clamp(1.0 - lzpk, min=0.0, max=1.0), dinc)
        dlzs = 1.0 - torch.pow(torch.clamp(1.0 - lzsk, min=0.0, max=1.0), dinc)

        roimp = pr * pctim
        sdro = torch.zeros_like(pr)
        ssur = torch.zeros_like(pr)
        sif = torch.zeros_like(pr)
        base_primary = torch.zeros_like(pr)
        base_supplemental = torch.zeros_like(pr)
        sideflow = torch.zeros_like(pr)
        sperc = torch.zeros_like(pr)
        lz_overflow = torch.zeros_like(pr)
        dt_lztw = torch.zeros_like(pr)
        dt_lzfp = torch.zeros_like(pr)
        dt_lzfs = torch.zeros_like(pr)
        dt_uzfw = torch.zeros_like(pr)

        for inc in range(max_ninc):
            active = inc < ninc

            adimp_ratio = torch.clamp((adimc - uztw) / lztwm_safe, min=0.0, max=1.0)
            addro = torch.where(active, pinc * torch.pow(adimp_ratio, 2.0), torch.zeros_like(pr))

            qlzfp_withdrawal = torch.where(active, lzfp * dlzp, torch.zeros_like(pr))
            qlzfp_local = qlzfp_withdrawal / torch.clamp(1.0 + side, min=1e-6)
            sideflow_local = qlzfp_withdrawal - qlzfp_local
            lzfp = torch.where(active, torch.clamp(lzfp - qlzfp_withdrawal, min=0.0), lzfp)
            qlzfs_local = torch.where(active, lzfs * dlzs, torch.zeros_like(pr))
            lzfs = torch.where(active, torch.clamp(lzfs - qlzfs_local, min=0.0), lzfs)
            base_primary = base_primary + qlzfp_local * parea
            base_supplemental = base_supplemental + qlzfs_local * parea
            sideflow = sideflow + sideflow_local * parea

            small_upper = active & ((pinc + uzfw) <= 0.01)
            uzfw = torch.where(small_upper, uzfw + pinc, uzfw)
            dt_uzfw = dt_uzfw + torch.where(small_upper, pinc * parea, torch.zeros_like(pr))

            process = active & (~small_upper)
            percm = lzfpm * dlzp + lzfsm * dlzs
            perc = percm * torch.clamp(uzfw / uzfwm_safe, min=0.0, max=1.0)
            lower_deficit_ratio = torch.clamp(
                1.0 - (lztw + lzfp + lzfs) / torch.clamp(lztwm + lzfpm + lzfsm, min=1e-6),
                min=0.0,
                max=1.0,
            )
            perc = perc * (1.0 + zperc * torch.pow(lower_deficit_ratio, rexp))
            perc = torch.minimum(perc, uzfw)

            lower_excess = torch.clamp(lztw + lzfp + lzfs + perc - (lztwm + lzfpm + lzfsm), min=0.0)
            perc = torch.clamp(perc - lower_excess, min=0.0)
            uzfw = torch.where(process, uzfw + lower_excess, uzfw)

            sperc = sperc + torch.where(process, perc * parea, torch.zeros_like(pr))
            uzfw = torch.where(process, torch.clamp(uzfw - perc, min=0.0), uzfw)

            interflow_local = uzfw * duz
            sif = sif + torch.where(process, interflow_local * parea, torch.zeros_like(pr))
            uzfw = torch.where(process, torch.clamp(uzfw - interflow_local, min=0.0), uzfw)

            perct = perc * (1.0 - pfree)
            lztw_add = torch.minimum(perct, torch.clamp(lztwm - lztw, min=0.0))
            lztw = torch.where(process, lztw + lztw_add, lztw)
            dt_lztw = dt_lztw + torch.where(process, lztw_add * parea, torch.zeros_like(pr))
            percf = perc * pfree + torch.clamp(perct - lztw_add, min=0.0)

            hpl = torch.clamp(lzfpm / torch.clamp(lzfpm + lzfsm, min=1e-6), min=0.0, max=1.0)
            ratlp = torch.clamp(lzfp / lzfpm_safe, min=0.0, max=1.0)
            ratls = torch.clamp(lzfs / lzfsm_safe, min=0.0, max=1.0)
            frac_primary = hpl * 2.0 * (1.0 - ratlp) / torch.clamp(2.0 - ratlp - ratls, min=1e-6)
            frac_primary = torch.clamp(frac_primary, min=0.0, max=1.0)

            percf_primary = percf * frac_primary
            percf_supplemental = percf - percf_primary
            lzfs_add = torch.minimum(percf_supplemental, torch.clamp(lzfsm - lzfs, min=0.0))
            lzfs = torch.where(process, lzfs + lzfs_add, lzfs)
            dt_lzfs = dt_lzfs + torch.where(process, lzfs_add * parea, torch.zeros_like(pr))
            percf_primary = percf_primary + torch.clamp(percf_supplemental - lzfs_add, min=0.0)

            lzfp_add = torch.minimum(percf_primary, torch.clamp(lzfpm - lzfp, min=0.0))
            lzfp = torch.where(process, lzfp + lzfp_add, lzfp)
            dt_lzfp = dt_lzfp + torch.where(process, lzfp_add * parea, torch.zeros_like(pr))
            free_excess = torch.clamp(percf_primary - lzfp_add, min=0.0)
            lztw_add_excess = torch.minimum(free_excess, torch.clamp(lztwm - lztw, min=0.0))
            lztw = torch.where(process, lztw + lztw_add_excess, lztw)
            dt_lztw = dt_lztw + torch.where(process, lztw_add_excess * parea, torch.zeros_like(pr))
            remaining_excess = torch.clamp(free_excess - lztw_add_excess, min=0.0)
            lz_overflow = lz_overflow + torch.where(process, remaining_excess * parea, torch.zeros_like(pr))

            surface_excess = torch.clamp(pinc + uzfw - uzfwm, min=0.0)
            uzfw_before_surface = uzfw
            uzfw_after_surface = torch.where(surface_excess > 0.0, uzfwm, uzfw + pinc)
            ssur = ssur + torch.where(process, surface_excess * parea, torch.zeros_like(pr))
            adsur = torch.where(
                (pinc > 0.0) & process,
                surface_excess * torch.clamp(1.0 - addro / torch.clamp(pinc, min=1e-6), min=0.0, max=1.0),
                torch.zeros_like(pr),
            )
            ssur = ssur + adsur * adimp
            uzfw = torch.where(process, uzfw_after_surface, uzfw)
            dt_uzfw = dt_uzfw + torch.where(
                process,
                torch.clamp(uzfw_after_surface - uzfw_before_surface, min=0.0) * parea,
                torch.zeros_like(pr),
            )

            adimc = torch.where(active, adimc + pinc - addro - adsur, adimc)
            adimc_excess = torch.clamp(adimc - (uztwm + lztwm), min=0.0)
            adimc = torch.minimum(adimc, uztwm + lztwm)
            addro = addro + adimc_excess
            sdro = sdro + addro * adimp
            adimc = torch.maximum(adimc, uztw)

        runoff_pctim = roimp
        runoff_adimp = sdro
        runoff_surface = ssur
        quz = sif
        qlzfp = base_primary
        qlzfs = base_supplemental
        lz_remain = lz_overflow

        runoff = runoff_pctim + runoff_adimp + runoff_surface + quz + qlzfp + qlzfs + sideflow + lz_remain
        runoff = runoff + glacier_melt_surface
        lzfp_with_glacier = lzfp + glacier_melt_ground
        glacier_ground_overflow = torch.clamp(lzfp_with_glacier - lzfpm, min=0.0)
        lzfp = torch.minimum(lzfp_with_glacier, lzfpm)
        runoff = runoff + glacier_ground_overflow

        channel_evap_potential = torch.clamp(edmnd - (et_uz + et_lz), min=0.0) * iv
        channel_evap = torch.minimum(runoff, channel_evap_potential)
        runoff = torch.clamp(runoff - channel_evap, min=0.0)

        P_after_pctim = pr * (1.0 - pctim)
        P_pervious = pr * parea
        infiltration_input = pr * parea
        frozen_excess = torch.zeros_like(pr)
        infiltration_overflow = runoff_surface + runoff_adimp + runoff_pctim

        state.update({
            "uztw": uztw,
            "uzfw": uzfw,
            "lztw": lztw,
            "lzfp": lzfp,
            "lzfs": lzfs,
            "adimc": adimc,
        })

        if self.enable_cold_region:
            state.update({
                "swe": swe,
                "FI": FI,
            })

        if not return_details:
            return {"runoff": runoff}, state

        total_storage = parea * (uztw + uzfw + lztw + lzfp + lzfs) + adimp * adimc
        if self.enable_cold_region:
            total_storage = total_storage + swe

        total_et = parea * (et_uz + et_lz) + adimp * et_adimp + channel_evap
        others = _make_process_outputs(runoff, total_storage, total_et)
        _update_process_outputs(
            others,
            storage_upper_tension=uztw,
            storage_upper_free=uzfw,
            storage_lower_tension=lztw,
            storage_lower_primary=lzfp,
            storage_lower_supplemental=lzfs,
            storage_adimp=adimc,
            evap_upper_total=et_uz,
            evap_lower_total=et_lz,
            evap_adimp=adimp * et_adimp,
            evap_open_water=channel_evap,
            runoff_impervious_pctim=runoff_pctim,
            runoff_impervious_adimp=runoff_adimp,
            runoff_surface=runoff_surface,
            runoff_upper_zone=quz,
            runoff_lower_primary=qlzfp,
            runoff_lower_supplemental=qlzfs,
            runoff_sideflow=sideflow,
            percolation=sperc,
            adimp_fraction_effective=adimp_eff,
            precip_after_pctim=P_after_pctim,
            precip_pervious=P_pervious,
            infiltration_actual=infiltration_input,
            infiltration_frozen_excess=frozen_excess,
            infiltration_overflow=infiltration_overflow,
            percolation_storage_excess=lz_remain,
            runoff_glacier_ground_overflow=glacier_ground_overflow,
            evap_upper_tension=et_uztw,
            evap_upper_free=et_uzfw,
            evap_lower_tension=et_lztw,
            evap_lower_supplemental=et_lzfs,
            evap_lower_primary=et_lzfp,
            recharge_upper_tension=dt_uztw,
            recharge_upper_free=dt_uzfw,
            recharge_lower_tension=dt_lztw,
            recharge_lower_primary=dt_lzfp,
            recharge_lower_supplemental=dt_lzfs,
            evap_potential=E_eff,
        )

        _append_common_inputs(
            others,
            P_input=P_input,
            E_input=E_input,
            P_corr=P_corr,
            P_eff=P_eff_input,
            T_input=T_input,
            glacier_frac_input=glacier_frac_input,
        )
        _append_cryosphere_outputs(others, cryo_result)

        return {"runoff": runoff, "others": others}, state


@register(SUBBASIN_REGISTRY, "tank")
class TankModel(BaseModel):
    """
    Tank Model - 日本水文模型，多层水库级联
    完全可微实现（PyTorch）
    
    日本气象和水文学院（JMA）开发的经典模型，特点为四层水库级联结构
    核心逻辑：从顶至底（表层→中层→深层→地下水），多出流口设计
    
    参数说明（共15个）：
    
    === 第1层（表层/快速流） ===
        - S1: 第1层蓄水容量 (mm) [50-500]
          物理含义：表层土壤蓄水库；响应快速径流
          推荐范围：[80-200] for 一般流域
        
        - K1: 第1层出流系数 [0.1-0.5]
          物理含义：第1层日衰减系数；控制表流速度
          推荐范围：[0.25-0.45] for 快速响应
        
        - H1: 第1层出流水位 (mm) [10-100]
          物理含义：第1层开始产流的水位阈值
          推荐范围：[20-60] for 一般流域
    
    === 第2层（中层/中间流） ===
        - S2: 第2层蓄水容量 (mm) [50-400]
          物理含义：中层土壤蓄水库；响应中间流
          推荐范围：[60-150] for 一般流域
        
        - K2: 第2层出流系数 [0.05-0.3]
          物理含义：第2层日衰减系数；中间流延迟控制
          推荐范围：[0.1-0.2] for 中等延迟
        
        - H2: 第2层出流水位 (mm) [5-80]
          物理含义：第2层开始产流的水位阈值
          推荐范围：[15-45] for 一般流域
        
        - Perc1: 第1层下渗系数 [0.01-0.3]
          物理含义：第1层向第2层的下渗比例
          推荐范围：[0.05-0.15] for 一般流域
    
    === 第3层（深层/缓流） ===
        - S3: 第3层蓄水容量 (mm) [50-400]
          物理含义：深层土壤蓄水库；缓慢释放
          推荐范围：[100-250] for 一般流域
        
        - K3: 第3层出流系数 [0.01-0.2]
          物理含义：第3层日衰减系数；控制基流延迟
          推荐范围：[0.03-0.1] for 缓流区
        
        - H3: 第3层出流水位 (mm) [5-60]
          物理含义：第3层开始产流的水位阈值
          推荐范围：[10-30] for 一般流域
        
        - Perc2: 第2层下渗系数 [0.01-0.2]
          物理含义：第2层向第3层的下渗比例
          推荐范围：[0.03-0.1] for 一般流域
    
    === 第4层（地下水/极缓流） ===
        - S4: 第4层蓄水容量 (mm) [50-300]
          物理含义：地下水蓄水库；极缓释放
          推荐范围：[80-200] for 一般流域
        
        - K4: 第4层出流系数 [0.001-0.05]
          物理含义：第4层日衰减系数；基流衰减最慢
          推荐范围：[0.01-0.03] for 地下水流
        
        - H4: 第4层出流水位 (mm) [1-40]
          物理含义：第4层开始产流的水位阈值
          推荐范围：[5-20] for 一般流域
        
        - Perc3: 第3层下渗系数 [0.001-0.1]
          物理含义：第3层向第4层的下渗比例
          推荐范围：[0.01-0.05] for 一般流域
    
    === 关键特点 ===
    级联结构：降水 → 水库1(快) → 水库2(中) → 水库3(缓) → 水库4(极缓)，形成分阶段释放
    多出流口：每层有独立的K和H，可产生多峰流量（接近真实流量变化）
    下渗控制：通过Perc系数控制相邻层之间的水分交换
    """
    def __init__(
        self,
        # 冰冻圈开关
        enable_cold_region=False,
        # 降水校正开关
        enable_precip_correct=False,
        # 冰冻圈参数（仅enable_cold_region=True时使用）
        t_snow=0.0, t_melt=0.0, ddf_snow=3.0, k_snow=0.5,
        glacier_frac=0.0, ddf_ice=6.0, t_ice_melt=0.0,
        glacier_runoff_frac=0.7,
        fi_alpha=0.9, fi_beta=0.1, 
        # 降水校正参数（仅enable_precip_correct=True时使用）
        kp=1.0,
        modules=None
    ):
        super().__init__()
        
        # 冰冻圈开关
        self.enable_cold_region = enable_cold_region
        
        # 降水校正开关
        self.enable_precip_correct = enable_precip_correct
        self.kp = kp
        
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
        
        # Tank模型参数
        self.S = [200.0, 150.0, 100.0, 50.0]  # 各层容量
        self.K = [0.3, 0.2, 0.1, 0.05]        # 出流系数
        self.H = [40.0, 30.0, 20.0, 10.0]     # 出流水位
        self.Perc = [0.1, 0.1, 0.1]           # 下渗系数
        
        # 添加单独的属性用于参数获取
        self.S1, self.S2, self.S3, self.S4 = self.S
        self.K1, self.K2, self.K3, self.K4 = self.K
        self.H1, self.H2, self.H3, self.H4 = self.H
        self.Perc1, self.Perc2, self.Perc3 = self.Perc

    def forward(self, inputs, static_params, dynamic_params, state, return_details=True):
        P = inputs.get("P", 0.0)
        E = inputs.get("E", 0.0)

        device = P.device if isinstance(P, torch.Tensor) else None
        dtype = P.dtype if isinstance(P, torch.Tensor) else torch.float32
        def _t(x):
            return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=dtype, device=device)
        P, E = _t(P), _t(E)
        P_input = P
        E_input = E
        P_corr = P
        T_input = None
        glacier_frac_input = None
        cryo_result = None

        # -----------------
        # 降水校正
        # -----------------
        if self.enable_precip_correct:
            kp = _t(static_params.get("kp", self.kp))
            P = apply_precip_correction(P, kp)
        P_corr = P

        # -----------------
        # 冰冻圈过程
        # -----------------
        snow_melt = _t(0.0)
        glacier_melt = _t(0.0)
        swe = _t(0.0)
        FI = _t(0.0)
        glacier_melt_surface = _t(0.0)
        glacier_melt_ground = _t(0.0)
        frozen_factor = _t(1.0)
        
        if self.enable_cold_region:
            T = _t(inputs.get("T", 10.0))
            T_input = T
            swe = _t(state.get("swe", 0.0))
            FI = _t(state.get("FI", 0.0))
            glacier_frac = _t(inputs.get("glacier_frac", self.glacier_frac))
            glacier_frac_input = glacier_frac
            
            # 构建冰冻圈参数
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
            
            # 调用冰冻圈过程函数
            cryo_result = process_cryosphere(P, T, swe, FI, glacier_frac, cryo_params)
            
            # 提取结果
            P_eff = cryo_result["P_eff"]
            swe = cryo_result["swe"]
            FI = cryo_result["FI"]
            snow_melt = cryo_result["snow_melt"]
            glacier_melt = cryo_result["glacier_melt"]
            glacier_melt_surface = cryo_result["glacier_melt_surface"]
            glacier_melt_ground = cryo_result["glacier_melt_ground"]
            frozen_factor = cryo_result["frozen_factor"]
        else:
            P_eff = P

        # -----------------
        # 获取各层状态
        # -----------------
        h = [_t(state.get(f"h{i+1}", self.S[i] / 2.0)) for i in range(4)]

        # -----------------
        # 获取参数
        # -----------------
        S = [_t(static_params.get(f"S{i+1}", self.S[i])) for i in range(4)]
        K = [_t(static_params.get(f"K{i+1}", self.K[i])) for i in range(4)]
        H = [_t(static_params.get(f"H{i+1}", self.H[i])) for i in range(4)]
        Perc = [_t(static_params.get(f"Perc{i+1}", self.Perc[i])) for i in range(3)]

        # -----------------
        # 冻土影响入渗：frozen_factor 减少入渗，多余部分变为地表径流
        # -----------------
        infil_capacity = P_eff * frozen_factor if self.enable_cold_region else P_eff
        frozen_excess = P_eff - infil_capacity if self.enable_cold_region else _t(0.0)
        runoff = frozen_excess  # 冻土超渗产流

        # -----------------
        # 降水进入第一层（修正：超出容量的部分直接产流）
        # -----------------
        h0_new = h[0] + infil_capacity
        overflow = torch.clamp(h0_new - S[0], min=0.0)  # 超出容量的部分
        h[0] = torch.minimum(h0_new, S[0])  # h[0]不超过容量
        runoff = runoff + overflow

        # -----------------
        # 蒸散从第一层，冷区 et_factor 不参与 Tank 蒸发。
        # -----------------
        E_adj = E
        et_actual = torch.minimum(h[0], E_adj)
        h[0] = h[0] - et_actual 

        # -----------------
        # 多层产流和下渗
        # -----------------
        layer_outflows = []
        layer_percolations = []
        layer_overflows = []
        for i in range(4):
            # 溢流
            q = K[i] * torch.maximum(h[i] - H[i], torch.tensor(0.0, dtype=dtype, device=device))
            layer_outflows.append(q)
            runoff = runoff + q
            h[i] = h[i] - q

            # 下渗到下一层（最后一层没有下渗）
            if i < 3:
                percolation = Perc[i] * h[i]
                layer_percolations.append(percolation)
                h[i] = h[i] - percolation
                h_next_new = h[i+1] + percolation
                perc_overflow = torch.clamp(h_next_new - S[i+1], min=0.0)  # 计算溢流
                layer_overflows.append(perc_overflow)
                h[i+1] = torch.minimum(h_next_new, S[i+1])  # 下层不超过容量
                runoff = runoff + perc_overflow  # 溢流加入径流

        # -----------------
        # 冰川融水贡献
        # -----------------
        runoff = runoff + glacier_melt_surface  # 冰川地表融水进入地表径流
        h[3] = h[3] + glacier_melt_ground  # 冰川地下融水进入最底层水箱（第4层）

        # -----------------
        # 更新状态
        # -----------------
        state.update({f"h{i+1}": h[i] for i in range(4)})
        
        # 冰冻圈状态
        if self.enable_cold_region:
            state.update({
                "swe": swe,
                "FI": FI,
            })
        
        # 计算总储水量（mm）：四层水库之和
        # 支持向量化（多子流域）
        if not return_details:
            return {"runoff": runoff}, state

        total_storage = h[0] + h[1] + h[2] + h[3]
        if self.enable_cold_region:
            total_storage = total_storage + swe

        others = _make_process_outputs(runoff, total_storage, et_actual)
        _update_process_outputs(
            others,
            storage_tank1=h[0],
            storage_tank2=h[1],
            storage_tank3=h[2],
            storage_tank4=h[3],
            evap_tank1=et_actual,
            infiltration_actual=infil_capacity,
            infiltration_frozen_excess=frozen_excess,
            runoff_surface_overflow=overflow,
            runoff_tank1=layer_outflows[0],
            runoff_tank2=layer_outflows[1],
            runoff_tank3=layer_outflows[2],
            runoff_tank4=layer_outflows[3],
            percolation_12=layer_percolations[0],
            percolation_23=layer_percolations[1],
            percolation_34=layer_percolations[2],
            overflow_12=layer_overflows[0],
            overflow_23=layer_overflows[1],
            overflow_34=layer_overflows[2],
            evap_potential=E_adj,
        )

        _append_common_inputs(
            others,
            P_input=P_input,
            E_input=E_input,
            P_corr=P_corr,
            P_eff=P_eff,
            T_input=T_input,
            glacier_frac_input=glacier_frac_input,
        )
        _append_cryosphere_outputs(others, cryo_result)

        return {"runoff": runoff, "others": others}, state

    def _get_cr_defaults(self):
        """获取冰冻圈参数默认值"""
        return {
            "t_snow": self.t_snow,
            "t_melt": self.t_melt,
            "ddf_snow": self.ddf_snow,
            "k_snow": self.k_snow,
            "glacier_frac": self.glacier_frac,
            "ddf_ice": self.ddf_ice,
            "t_ice_melt": self.t_ice_melt,
            "glacier_runoff_frac": self.glacier_runoff_frac,
            "fi_alpha": self.fi_alpha,
            "fi_beta": self.fi_beta,
        }

    def _define_parameters(self):
        # 基础参数
        names = ["S1","S2","S3","S4","K1","K2","K3","K4",
                 "H1","H2","H3","H4","Perc1","Perc2","Perc3"]
        bounds = [[10,500]]*4 + [[0.01,0.5]]*4 + [[0,100]]*4 + [[0.01,0.5]]*3
        init = self.S + self.K + self.H + self.Perc
        # 四类参数分类：产流参数(subbasin_runoff) vs 汇流参数(hillslope_routing)
        network_mapping = {
            # 产流参数：水库容量、出流孔高度、渗透系数
            "S1": "subbasin_runoff", "S2": "subbasin_runoff", "S3": "subbasin_runoff", "S4": "subbasin_runoff",
            "H1": "subbasin_runoff", "H2": "subbasin_runoff", "H3": "subbasin_runoff", "H4": "subbasin_runoff",
            "Perc1": "subbasin_runoff", "Perc2": "subbasin_runoff", "Perc3": "subbasin_runoff",
            # 汇流参数：出流系数
            "K1": "hillslope_routing", "K2": "hillslope_routing", "K3": "hillslope_routing", "K4": "hillslope_routing"
        }
        # 无延时无调蓄参数：出流系数取上界（最快出流）
        passthrough = (list(self.S) + [0.5, 0.5, 0.5, 0.5] +  # K1-K4: 最快出流（上界）
                       list(self.H) + list(self.Perc))
        
        # 如果启用冰冻圈，添加冰冻圈参数
        if self.enable_cold_region:
            cr_names, cr_bounds, cr_init = get_cold_region_params(self._get_cr_defaults())
            names = cr_names + names
            bounds = cr_bounds + bounds
            init = cr_init + init
            passthrough = cr_init + passthrough
            network_mapping.update(get_cold_region_network_mapping())
        
        # 如果启用降水校正，添加降水校正参数
        if self.enable_precip_correct:
            kp_names, kp_bounds, kp_init = get_precip_correct_params(self)
            names = kp_names + names
            bounds = kp_bounds + bounds
            init = kp_init + init
            passthrough = kp_init + passthrough
            network_mapping.update(get_precip_correct_network_mapping())
        
        return {
            "static": {
                "names": names,
                "bounds": bounds,
                "init": init,
                "passthrough": passthrough,
                "network_mapping": network_mapping
            },
            "dynamic": {"names": [], "bounds": []},
        }

    def init_state(self):
        state = {f"h{i+1}": torch.tensor(self.S[i]/2.0) for i in range(4)}
        if self.enable_cold_region:
            state.update({
                "swe": 0.0,
                "FI": 0.0,
            })
        return state


@register(SUBBASIN_REGISTRY, "hbv")
class HBVModel(BaseModel):
    """
    HBV (Hydrologiska Byrans Vattenbalansmodell) 子流域产流模型。

    当前实现保持项目原有接口和命名，不引入示例函数中的大写变量名：
    - 输入：`P`、`E`，非冷区融雪使用 `T`；冷区模式额外使用统一冷区模块输入。
    - 状态：`snow`、`sm`、`uz`、`lz`、`qh_queue`。
    - 参数：`fc`、`lp`、`beta`、`perc`、`uzl`、`k0`、`k1`、`k2`、`maxbas`、`t_melt`、`ddf`。

    参数说明：

    === 土壤水分模块 ===
        - fc: 土壤最大含水量/田间持水量 (mm) [50, 400]
          控制土壤蓄水上限，也是非线性产流和蒸散限制的核心尺度。

        - lp: 蒸散临界含水量比例 [-] [0.3, 0.9]
          当 `sm < lp * fc` 时实际蒸散按土壤湿度比例降低；当 `sm >= lp * fc` 时取潜在蒸散。

        - beta: 土壤非线性产流指数 [-] [0.5, 2.0]
          控制 `Peff * (sm / fc) ** beta` 的产流响应。值越大，干土条件下产流越弱。

    === 响应水库与出流模块 ===
        - perc: 上层水库向下层水库的最大渗漏量 (mm/day) [0, 5]
          每个时间步从 `uz` 转入 `lz`，并与 `q0/q1` 一起受上层水库可用水量约束。

        - uzl: 上层快流阈值 (mm) [0, 100]
          只有 `uz` 超过该阈值的部分参与快流 `q0 = k0 * max(uz - uzl, 0)`。

        - k0: 上层快流退水系数 [-] [0.1, 0.5]
          控制超过 `uzl` 的快速径流释放。

        - k1: 上层中间流退水系数 [-] [0.01, 0.1]
          控制 `uz` 的线性中间流释放。

        - k2: 下层基流退水系数 [-] [0.001, 0.02]
          控制 `lz` 的线性基流释放。

    === 单位线路由模块 ===
        - maxbas: HBV 三角单位线宽度/汇流平滑时间 (day) [1, 10]
          `maxbas <= 1` 时径流直接输出；否则使用 CDF 差分权重队列对 `q0 + q1 + q2` 进行延迟和平滑。
          多子流域向量化时，标量表示各子流域共用同一单位线；长度为 `n_subs` 的张量表示各子流域使用各自的单位线。

    === 非冷区融雪模块 ===
        - t_melt: 雨雪/融雪温度阈值 (deg C) [-5, 5]
          非冷区模式下，`T > t_melt` 时降水作为雨，反之作为雪。

        - ddf: 度日融雪因子 (mm/deg C/day) [0.5, 10]
          非冷区模式下融雪量为 `ddf * max(T - t_melt, 0)`，且不超过当前 `snow`。

    === 降水校正扩展 ===
        - kp: 降水校正系数 [-]
          仅当 `enable_precip_correct=True` 时启用，用于在进入 HBV 主体前修正降水。

    === 冷区过程扩展 ===
        - t_snow: 雨雪分离温度阈值 (deg C)
        - t_melt_cr: 统一冷区模块的融雪温度阈值 (deg C)
        - ddf_snow: 积雪度日融雪因子 (mm/deg C/day)
        - k_snow: 积雪融水释放系数 [-]
        - glacier_frac: 冰川覆盖比例 [-]
        - ddf_ice: 冰川度日融化因子 (mm/deg C/day)
        - t_ice_melt: 冰川开始融化温度阈值 (deg C)
        - glacier_runoff_frac: 冰川融水进入地表快速径流的比例 [-]
        - fi_alpha: 冻结指数记忆/衰减系数 [-]
        - fi_beta: 冻结指数对入渗和蒸散抑制的强度系数 [-]

        冷区模式下，HBV 自带的 `t_melt/ddf` 融雪参数不进入参数表；
        有效降水、积雪和冰川融水由统一冷区模块计算后再进入 HBV 主体。

    主体内核按标准 HBV 单步过程实现：
    1. 非冷区模式下按温度阈值 `t_melt` 分雨雪，并用 `ddf` 计算融雪；冷区模式下由统一冷区模块给出有效降水。
    2. 土壤蒸散按 `sm < lp * fc` 时受限、否则取潜在蒸散，并限制不超过现有土壤水。
    3. 土壤产流为 `Peff * (sm / fc) ** beta`，并受可用水量和 `fc` 容量约束。
    4. 上层水库 `uz` 同时产生超过 `uzl` 的快流 `q0`、线性中间流 `q1`，并通过 `perc` 向下层水库 `lz` 渗漏；若 `q0 + q1 + perc` 超过 `uz`，按比例缩放。
    5. 下层水库 `lz` 产生线性基流 `q2`。
    6. `q0 + q1 + q2` 统一进入 `maxbas` 单位线路由；`maxbas <= 1` 时直接输出，否则使用 HBV 标准 CDF 差分权重队列。

    向量化说明：
    - 标量 `maxbas` 表示所有子流域共用同一单位线。
    - 长度为 `n_subs` 的 `maxbas` 张量表示每个子流域使用自己的单位线权重。
    - `qh_queue` 形状为 `[n_subs, max_uh_len]`，每个子流域独立维护延迟队列。

    项目扩展：
    - `enable_precip_correct=True` 时先对降水应用 `kp` 校正。
    - `enable_cold_region=True` 时使用统一冷区模块处理雨雪、积雪、冰川融水、冻土入渗影响。
    """

    def __init__(
        self,
        # 冰冻圈开关
        enable_cold_region=False,
        # 降水校正开关
        enable_precip_correct=False,
        # 冰冻圈参数（仅enable_cold_region=True时使用）
        t_snow=0.0, t_melt_cr=0.0, ddf_snow=3.0, k_snow=0.5,
        glacier_frac=0.0, ddf_ice=6.0, t_ice_melt=0.0,
        glacier_runoff_frac=0.7,
        fi_alpha=0.9, fi_beta=0.1, 
        # 降水校正参数（仅enable_precip_correct=True时使用）
        kp=1.0,
        # HBV原有参数
        fc=200.0, lp=0.5, beta=1.5, perc=1.0, uzl=20.0,
        k0=0.3, k1=0.05, k2=0.01, maxbas=3,
        t_melt=0.0, ddf=3.0, modules=None
    ):
        super().__init__()
        
        # 冰冻圈开关
        self.enable_cold_region = enable_cold_region
        
        # 降水校正开关
        self.enable_precip_correct = enable_precip_correct
        self.kp = kp
        
        # 冰冻圈参数
        self.t_snow = t_snow
        self.t_melt_cr = t_melt_cr
        self.ddf_snow = ddf_snow
        self.k_snow = k_snow
        self.glacier_frac = glacier_frac
        self.ddf_ice = ddf_ice
        self.t_ice_melt = t_ice_melt
        self.glacier_runoff_frac = glacier_runoff_frac
        self.fi_alpha = fi_alpha
        self.fi_beta = fi_beta
        
        # HBV原有参数
        self.fc = fc
        self.lp = lp
        self.beta = beta
        self.perc = perc
        self.uzl = uzl
        self.k0 = k0
        self.k1 = k1
        self.k2 = k2
        self.maxbas = maxbas
        self.t_melt = t_melt
        self.ddf = ddf

    def forward(self, inputs, static_params, dynamic_params, state, return_details=True):
        P = inputs.get("P", 0.0)
        E = inputs.get("E", 0.0)  # 蒸发能力（统一使用E）

        # 支持 PyTorch 自动微分
        device = P.device if isinstance(P, torch.Tensor) else None
        dtype = P.dtype if isinstance(P, torch.Tensor) else torch.float32
        def _t(x):
            return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=dtype, device=device)
        P, E = _t(P), _t(E)
        P_input = P
        E_input = E
        P_corr = P
        T_input = None
        glacier_frac_input = None
        cryo_result = None

        # -----------------
        # 降水校正
        # -----------------
        if self.enable_precip_correct:
            kp = _t(static_params.get("kp", self.kp))
            P = apply_precip_correction(P, kp)
        P_corr = P

        # -----------------
        # 冰冻圈过程
        # -----------------
        snow_melt = _t(0.0)
        glacier_melt = _t(0.0)
        swe = _t(0.0)
        FI = _t(0.0)
        glacier_melt_surface = _t(0.0)
        glacier_melt_ground = _t(0.0)
        frozen_factor = _t(1.0)
        if self.enable_cold_region:
            T = _t(inputs.get("T", 10.0))
            T_input = T
            swe = _t(state.get("swe", 0.0))
            FI = _t(state.get("FI", 0.0))
            glacier_frac = _t(inputs.get("glacier_frac", self.glacier_frac))
            glacier_frac_input = glacier_frac
            
            # 构建冰冻圈参数
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
            
            # 调用冰冻圈过程函数
            cryo_result = process_cryosphere(P, T, swe, FI, glacier_frac, cryo_params)
            
            # 提取结果
            P_eff = cryo_result["P_eff"]
            swe = cryo_result["swe"]
            FI = cryo_result["FI"]
            snow_melt = cryo_result["snow_melt"]
            glacier_melt = cryo_result["glacier_melt"]
            glacier_melt_surface = cryo_result["glacier_melt_surface"]
            glacier_melt_ground = cryo_result["glacier_melt_ground"]
            frozen_factor = cryo_result["frozen_factor"]
        else:
            T = _t(inputs.get("T", 0.0))
            T_input = T
            P_eff = P

        fc = _t(static_params.get("fc", self.fc))
        lp = _t(static_params.get("lp", self.lp))
        beta = _t(static_params.get("beta", self.beta))
        perc_param = _t(static_params.get("perc", self.perc))
        uzl = _t(static_params.get("uzl", self.uzl))
        k0 = _t(static_params.get("k0", self.k0))
        k1 = _t(static_params.get("k1", self.k1))
        k2 = _t(static_params.get("k2", self.k2))
        maxbas = static_params.get("maxbas", self.maxbas)
        t_melt = _t(static_params.get("t_melt", self.t_melt))
        ddf = _t(static_params.get("ddf", self.ddf))

        # 状态量
        sm = _t(state.get("sm", fc / 2.0))
        uz = _t(state.get("uz", 10.0))
        lz = _t(state.get("lz", 10.0))
        snow = _t(state.get("snow", 0.0))

        fc_safe = fc + 1e-12
        if not self.enable_cold_region:
            is_rain = T > t_melt
            Prain = torch.where(is_rain, P_eff, torch.zeros_like(P_eff))
            Psnow = torch.where(is_rain, torch.zeros_like(P_eff), P_eff)
            melt = torch.where(is_rain, ddf * (T - t_melt), torch.zeros_like(P_eff))
            melt = torch.minimum(melt, snow)
            snow = snow + Psnow - melt
            Peff = Prain + melt
        else:
            Prain = cryo_result["P_rain"] if cryo_result is not None else P_eff
            Psnow = cryo_result["P_snow"] if cryo_result is not None else torch.zeros_like(P_eff)
            melt = snow_melt
            snow = swe
            Peff = P_eff
        P_eff_model = Peff

        # Soil evaporation and recharge follow the supplied hbv_step order.
        E_adj = E
        lp_fc = lp * fc
        et = torch.where(sm < lp_fc, E_adj * (sm / (lp_fc + 1e-12)), E_adj)
        et = torch.minimum(torch.clamp(et, min=0.0), sm)

        sm_ratio = torch.clamp(sm / fc_safe, min=0.0)
        recharge = Peff * torch.pow(sm_ratio, beta)
        recharge = torch.minimum(recharge, torch.clamp(sm + Peff - et, min=0.0))
        sm = sm + Peff - recharge - et
        sm_overflow = torch.clamp(sm - fc, min=0.0)
        recharge = recharge + sm_overflow
        sm = torch.where(sm > fc, fc, sm)
        sm = torch.clamp(sm, min=0.0)

        # Upper and lower response reservoirs: q0/q1 leave uz, perc moves to lz, q2 leaves lz.
        uz = uz + recharge
        q0 = k0 * torch.clamp(uz - uzl, min=0.0)
        q1 = k1 * uz
        perc_to_lz = torch.minimum(perc_param, uz)
        upper_out = q0 + q1 + perc_to_lz
        scale = torch.where((upper_out > uz) & (upper_out > 0.0), uz / upper_out, torch.ones_like(upper_out))
        q0 = q0 * scale
        q1 = q1 * scale
        perc_to_lz = perc_to_lz * scale
        uz = torch.clamp(uz - q0 - q1 - perc_to_lz, min=0.0)

        lz = lz + perc_to_lz + glacier_melt_ground
        q2 = torch.minimum(k2 * lz, lz)
        lz = torch.clamp(lz - q2, min=0.0)

        q_mm = torch.clamp(q0 + q1 + q2 + glacier_melt_surface, min=0.0)

        # MAXBAS routing uses the CDF-difference weights from the supplied function.
        max_uh_len = 10
        is_vectorized = q_mm.dim() >= 1 and q_mm.numel() > 1
        maxbas_t = maxbas if isinstance(maxbas, torch.Tensor) else _t(maxbas)
        if isinstance(maxbas_t, torch.Tensor):
            maxbas_t = maxbas_t.to(dtype=dtype)
            if device is not None:
                maxbas_t = maxbas_t.to(device=device)
        maxbas_t = torch.clamp(maxbas_t, min=1.0, max=float(max_uh_len))

        qh_queue = state.get("qh_queue", None)

        if is_vectorized:
            n_subs = q_mm.shape[0]
            if qh_queue is None or not isinstance(qh_queue, torch.Tensor) or qh_queue.dim() != 2 or qh_queue.shape != (n_subs, max_uh_len):
                qh_queue = torch.zeros(n_subs, max_uh_len, dtype=dtype, device=device)
        else:
            if qh_queue is None or not isinstance(qh_queue, torch.Tensor) or qh_queue.dim() != 1 or qh_queue.numel() != max_uh_len:
                qh_queue = torch.zeros(max_uh_len, dtype=dtype, device=device)

        x0 = torch.arange(0, max_uh_len, dtype=dtype, device=device)
        x1 = x0 + 1.0
        if is_vectorized:
            mb = maxbas_t.reshape(1, 1).expand(n_subs, 1) if maxbas_t.numel() == 1 else maxbas_t.reshape(n_subs, 1)
            xx0 = x0.unsqueeze(0)
            xx1 = x1.unsqueeze(0)
        else:
            mb = maxbas_t.reshape(())
            xx0 = x0
            xx1 = x1

        def _maxbas_cdf(x, mb_value):
            x_clip = torch.minimum(torch.clamp(x, min=0.0), mb_value)
            half = 0.5 * mb_value
            lower = 2.0 * torch.pow(x_clip / mb_value, 2.0)
            upper = 1.0 - 2.0 * torch.pow((mb_value - x_clip) / mb_value, 2.0)
            return torch.where(x_clip <= half, lower, upper)

        weights = _maxbas_cdf(xx1, mb) - _maxbas_cdf(xx0, mb)
        weight_sum = weights.sum(dim=1, keepdim=True) if is_vectorized else weights.sum()
        weights = weights / torch.clamp(weight_sum, min=1e-12)

        if is_vectorized:
            qh_queue = qh_queue + q_mm.unsqueeze(1) * weights
            routed = qh_queue[:, 0]
            shifted_queue = torch.cat([qh_queue[:, 1:], torch.zeros(n_subs, 1, dtype=dtype, device=device)], dim=1)
            direct_mask = (maxbas_t.reshape(-1) <= 1.0) if maxbas_t.numel() > 1 else (maxbas_t.reshape(1) <= 1.0).expand(n_subs)
            runoff = torch.where(direct_mask, q_mm, routed)
            qh_queue = torch.where(direct_mask.unsqueeze(1), torch.zeros_like(shifted_queue), shifted_queue)
            first_weight = torch.where(direct_mask, torch.ones_like(routed), weights[:, 0])
        else:
            if bool((maxbas_t <= 1.0).detach().item()):
                runoff = q_mm
                qh_queue = torch.zeros_like(qh_queue)
                first_weight = torch.ones_like(q_mm)
            else:
                qh_queue = qh_queue + q_mm * weights
                runoff = qh_queue[0]
                qh_queue = torch.cat([qh_queue[1:], torch.zeros(1, dtype=dtype, device=device)])
                first_weight = weights[0]

        runoff = torch.clamp(runoff, min=0.0)
        qh_idx = _t(state.get("qh_idx", 0.0)) + 1

        qf, qs, qg = q0, q1, q2
        quick_runoff = q_mm
        runoff_delayed = runoff
        infil_capacity = Peff
        frozen_excess = _t(0.0)
        uz_overflow = torch.zeros_like(q_mm)
        lz_overflow = torch.zeros_like(q_mm)
        lz_overflow2 = torch.zeros_like(q_mm)
        state.update({
            "sm": sm,
            "uz": uz,
            "lz": lz,
            "snow": snow,
            "qh_queue": qh_queue,
            "qh_idx": qh_idx,
        })
        
        # 冰冻圈状态
        if self.enable_cold_region:
            state.update({
                "swe": swe,
                "FI": FI,
            })

        # 计算总储水量（mm）：土壤含水 + 快流水库 + 慢流水库 + 雪储存 + 单位线队列
        # 支持向量化（多子流域）
        # 注意：非冰冻圈模式用snow，冰冻圈模式用swe（两者不同时使用）
        if not return_details:
            return {"runoff": runoff}, state

        qh_storage = qh_queue.sum(dim=1) if qh_queue.dim() == 2 else qh_queue.sum()
        if self.enable_cold_region:
            total_storage = sm + uz + lz + swe + qh_storage
        else:
            total_storage = sm + uz + lz + snow + qh_storage

        others = _make_process_outputs(runoff, total_storage, et)
        _update_process_outputs(
            others,
            recharge=recharge,
            storage_soil=sm,
            storage_upper_zone=uz,
            storage_lower_zone=lz,
            storage_snowpack=(swe if self.enable_cold_region else snow),
            storage_delay_queue=qh_storage,
            runoff_quick_component=qf,
            runoff_intermediate_component=qs,
            runoff_baseflow_component=qg,
            runoff_quick_generated=quick_runoff,
            runoff_delayed=runoff_delayed,
            runoff_total_generated=q_mm,
            precip_rain=Prain,
            precip_snow=Psnow,
            infiltration_actual=infil_capacity,
            infiltration_frozen_excess=frozen_excess,
            snowpack_melt=melt,
            evap_actual=et,
            evap_potential=E_adj,
            storage_ratio_soil=sm_ratio,
            overflow_soil=sm_overflow,
            overflow_upper_zone=uz_overflow,
            overflow_lower_zone_1=lz_overflow,
            overflow_lower_zone_2=lz_overflow2,
            routing_weight_first=first_weight,
            routing_weights=weights,
            routing_decay=first_weight,
        )

        _append_common_inputs(
            others,
            P_input=P_input,
            E_input=E_input,
            P_corr=P_corr,
            P_eff=P_eff_model,
            T_input=T_input,
            glacier_frac_input=glacier_frac_input,
        )
        _append_cryosphere_outputs(others, cryo_result)

        return {"runoff": runoff, "others": others}, state

    def _get_cr_defaults(self):
        """获取冰冻圈参数默认值"""
        return {
            "t_snow": self.t_snow,
            "t_melt": self.t_melt_cr,
            "ddf_snow": self.ddf_snow,
            "k_snow": self.k_snow,
            "glacier_frac": self.glacier_frac,
            "ddf_ice": self.ddf_ice,
            "t_ice_melt": self.t_ice_melt,
            "glacier_runoff_frac": self.glacier_runoff_frac,
            "fi_alpha": self.fi_alpha,
            "fi_beta": self.fi_beta,
        }

    def _define_parameters(self):
        # 基础参数（不含冰冻圈时包含t_melt和ddf）
        if self.enable_cold_region:
            # 冰冻圈模式：移除t_melt和ddf，由冰冻圈模块的t_melt_cr和ddf_snow替代
            names = ["fc", "lp", "beta", "perc", "uzl", "k0", "k1", "k2", "maxbas"]
            bounds = [
                [50, 400], [0.3, 0.9], [0.5, 2.0], [0.0, 5.0], [0.0, 100.0],
                [0.1, 0.5], [0.01, 0.1], [0.001, 0.02],
                [1, 10]
            ]
            init = [self.fc, self.lp, self.beta, self.perc, self.uzl, self.k0, self.k1, self.k2, self.maxbas]
            network_mapping = {
                "fc": "subbasin_runoff", "lp": "subbasin_runoff", "beta": "subbasin_runoff",
                "perc": "subbasin_runoff", "uzl": "subbasin_runoff",
                "k0": "hillslope_routing", "k1": "hillslope_routing", "k2": "hillslope_routing",
                "maxbas": "hillslope_routing"
            }
            # 添加冰冻圈参数
            cr_names, cr_bounds, cr_init = get_cold_region_params(self._get_cr_defaults())
            names = cr_names + names
            bounds = cr_bounds + bounds
            init = cr_init + init
            network_mapping.update(get_cold_region_network_mapping())
        else:
            # 非冰冻圈模式：保留HBV原有的t_melt和ddf融雪参数
            names = ["fc", "lp", "beta", "perc", "uzl", "k0", "k1", "k2", "maxbas", "t_melt", "ddf"]
            bounds = [
                [50, 400], [0.3, 0.9], [0.5, 2.0], [0.0, 5.0], [0.0, 100.0],
                [0.1, 0.5], [0.01, 0.1], [0.001, 0.02],
                [1, 10], [-5, 5], [0.5, 10.0]
            ]
            init = [self.fc, self.lp, self.beta, self.perc, self.uzl, self.k0, self.k1, self.k2, self.maxbas, self.t_melt, self.ddf]
            network_mapping = {
                "fc": "subbasin_runoff", "lp": "subbasin_runoff", "beta": "subbasin_runoff",
                "perc": "subbasin_runoff", "uzl": "subbasin_runoff",
                "t_melt": "subbasin_runoff", "ddf": "subbasin_runoff",
                "k0": "hillslope_routing", "k1": "hillslope_routing", "k2": "hillslope_routing",
                "maxbas": "hillslope_routing"
            }
        
        # 无延时无调蓄参数：汇流系数取上界/时间取下界（最快汇流）
        passthrough_mapping = {
            "k0": 0.5, "k1": 0.1, "k2": 0.02, "maxbas": 1  # 最快衰减（上界），最短延迟（下界）
        }
        passthrough = [passthrough_mapping.get(n, init[i]) for i, n in enumerate(names)]
        
        # 如果启用降水校正，添加降水校正参数
        if self.enable_precip_correct:
            kp_names, kp_bounds, kp_init = get_precip_correct_params(self)
            names = kp_names + names
            bounds = kp_bounds + bounds
            init = kp_init + init
            passthrough = kp_init + passthrough
            network_mapping.update(get_precip_correct_network_mapping())
        
        return {
            "static": {
                "names": names,
                "bounds": bounds,
                "init": init,
                "passthrough": passthrough,
                "network_mapping": network_mapping
            },
            "dynamic": {"names": [], "bounds": []},
        }

    def init_state(self):
        # 使用固定大小的Tensor队列代替list，支持向量化计算
        # maxbas最大为10，所以队列长度固定为10
        state = {
            "sm": self.fc / 2.0,
            "uz": 10.0,
            "lz": 10.0,
            "snow": 0.0,
            "qh_queue": 0.0,      # 队列存储（展平为单值，实际在forward中处理）
            "qh_idx": 0.0,        # 队列当前索引
        }
        if self.enable_cold_region:
            state.update({
                "swe": 0.0,
                "FI": 0.0,
            })
        return state

