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


# ==============================================================================
# 水文模型定义
# ==============================================================================

@register(SUBBASIN_REGISTRY, "xaj")

class XAJModel(BaseModel):
    """
    新安江模型（日尺度）- 支持冰冻圈开关

    参数说明：
    
    === 土壤蓄水参数 ===
        - wum: 上层蓄水容量 (mm) [10-100]
        - wlm: 中层蓄水容量 (mm) [20-200]
        - wdm: 下层蓄水容量 (mm) [50-300]
        - b: 产流形状系数 [0.0-1.5]
        - kc: 蒸散能力系数 [0.1-1.0]
    
    === 产流划分参数 ===
        - im: 不透水面积系数 [0-0.4]
        - sm: 自由水蓄水容量 (mm) [20-500]
        - ex: 自由水蒸发指数 [0.5-2.0]
    
    === 三水源分配系数（kg + ki + ks 自动归一化为1.0）===
        - kg: 基流入渗系数 [0.05-0.35]
        - ki: 壤中流入渗系数 [0.15-0.55]
        - ks: 地表径流系数 [0.30-0.70]
    
    === 线性水库衰减系数 ===
        - cr: 快速流衰减系数 [0.1-0.9]
        - ci: 中间流衰减系数 [0.1-0.9]
        - cg: 基流衰减系数 [0.01-0.5]
    
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
        wum=30.0, wlm=60.0, wdm=80.0,
        b=0.5, kc=1.0,
        im=0.1,
        sm=100.0, ex=1.5,
        kg=0.15, ki=0.35, ks=0.50,
        cr=0.5, ci=0.5, cg=0.1,
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

        self.im = im
        self.sm = sm
        self.ex = ex

        self.kg = kg
        self.ki = ki
        self.ks = ks

        self.cr = cr
        self.ci = ci
        self.cg = cg

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
        et_factor = _t(1.0)
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
            et_factor = cryo_result["et_factor"]
        else:
            P_eff = P

        # === 参数 ===
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
        
        # 归一化：确保 kg + ki + ks = 1
        total_k = kg + ki + ks
        kg = kg / total_k
        ki = ki / total_k
        ks = ks / total_k

        cr = _t(static_params.get("cr", self.cr))
        ci = _t(static_params.get("ci", self.ci))
        cg = _t(static_params.get("cg", self.cg))

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
        et = E * kc * et_factor  # 冰冻圈：低温时蒸发减少

        eu = torch.minimum(wu, et)
        wu = wu - eu
        et = et - eu

        el = torch.minimum(wl, et)
        wl = wl - el
        et = et - el

        ed = torch.minimum(wd, et)
        wd = wd - ed

        w = wu + wl + wd

        # ==================================================
        # 2. 产流计算（蓄满产流）
        # ==================================================
        runoff_im = im * P_eff
        Pp = (1 - im) * P_eff

        # 数值稳定性：确保 w/wm 在 [0,1] 范围内
        w_ratio = torch.clamp(w / (wm + 1e-6), min=0.0, max=1.0)
        A = wm * (1 - torch.pow(1 - w_ratio + 1e-6, 1 + b))

        # 两个 case 的候选值（数值稳定性处理）
        A_Pp_ratio = torch.clamp((A + Pp) / (wm + 1e-6), min=1e-6, max=1.0 - 1e-6)
        tmp = wm * (1 - torch.pow(A_Pp_ratio, 1 / (1 + b + 1e-6))) - w
        runoff_p_candidate = Pp - tmp
        runoff_p_alt = Pp - (wm - w)

        runoff_p_pos = torch.where((A + Pp) < wm, runoff_p_candidate, runoff_p_alt)
        runoff_p = torch.where(Pp <= 0, torch.zeros_like(Pp), runoff_p_pos)
        runoff_p = torch.clamp(runoff_p, min=0.0)  # 产流不能为负
        runoff_p = torch.minimum(runoff_p, Pp)      # 产流不能超过有效降水
        
        # 冻土超渗产流（仅冰冻圈启用时）
        # 冻土会减少入渗能力，被拒绝的入渗变成地表产流
        potential_infil = Pp - runoff_p
        actual_infil = potential_infil
        frozen_excess = _t(0.0)
        if self.enable_cold_region:
            actual_infil = potential_infil * frozen_factor  # 实际入渗 = 潜在入渗 × 冻土因子
            frozen_excess = potential_infil - actual_infil  # 被冻土拒绝的部分变成产流
            runoff_p = runoff_p + frozen_excess
            infil = actual_infil
        else:
            infil = actual_infil
        
        # 上层入渗
        wu_space = wum - wu
        wu_add = torch.minimum(wu_space, infil * (wum / wm))
        wu = wu + wu_add
        infil_remain = infil - wu_add
        
        # 中层入渗
        wl_space = wlm - wl
        wl_add = torch.minimum(wl_space, infil_remain * (wlm / (wlm + wdm)))
        wl = wl + wl_add
        infil_remain = infil_remain - wl_add
        
        # 下层入渗（剩余全部）
        wd_space = wdm - wd
        wd_add = torch.minimum(wd_space, infil_remain)
        wd = wd + wd_add
        infil_remain = infil_remain - wd_add
        
        # 溢出的入渗加入产流
        runoff_p = runoff_p + infil_remain

        # ==================================================
        # 3. 自由水蓄水与三分流
        # ==================================================
        runoff_total = runoff_im + runoff_p
        sf = sf + runoff_total

        # 自由水出流量计算（确保不超过蓄水量，添加数值稳定性）
        sf_ratio = torch.clamp(sf / (sm + 1e-6), min=0.0, max=1.0)
        qf_calc = torch.where(
            sf <= sm,
            sf * (1 - torch.pow(1 - sf_ratio + 1e-6, ex)),
            sf
        )
        qf = torch.minimum(qf_calc, sf)
        sf = torch.clamp(sf - qf, min=0.0)

        qs = ks * qf  # 地表径流入流
        qi = ki * qf  # 壤中流入流
        qg = kg * qf  # 地下水入流

        # ==================================================
        # 4. 各分量汇流（线性水库）
        # ==================================================
        sr = _t(state.get("sr", 0.0))
        sr = sr + qs + glacier_melt_surface  # 冰川地表融水进入地表径流
        sr_out = torch.minimum(cr * sr, sr)
        sr = torch.clamp(sr - sr_out, min=0.0)
        
        # 壤中流水库
        si = si + qi
        si_out = torch.minimum(ci * si, si)
        si = si - si_out

        # 地下水水库
        sg = sg + qg + glacier_melt_ground  # 冰川地下融水进入地下水
        sg_out = torch.minimum(cg * sg, sg)
        sg = sg - sg_out
        
        # 确保非负
        sr = torch.clamp(sr, min=0.0)
        si = torch.clamp(si, min=0.0)
        sg = torch.clamp(sg, min=0.0)

        q_out = sr_out + si_out + sg_out
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

        total_storage = wu + wl + wd + sf + sr + si + sg
        if self.enable_cold_region:
            total_storage = total_storage + swe
        total_et = eu + el + ed

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
            runoff_impervious=runoff_im,
            runoff_saturation_excess=runoff_p,
            runoff_total_generated=runoff_total,
            runoff_free_water_out=qf,
            runoff_surface_generated=qs,
            runoff_interflow_generated=qi,
            runoff_groundwater_generated=qg,
            runoff_surface_out=sr_out,
            runoff_interflow_out=si_out,
            runoff_groundwater_out=sg_out,
            evap_upper=eu,
            evap_middle=el,
            evap_deep=ed,
            soil_water_total=w,
            precip_pervious=Pp,
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
            "wum": "subbasin_runoff", "wlm": "subbasin_runoff", "wdm": "subbasin_runoff",
            "b": "subbasin_runoff", "kc": "subbasin_runoff",
            "im": "subbasin_runoff", "sm": "subbasin_runoff", "ex": "subbasin_runoff",
            "kg": "subbasin_runoff", "ki": "subbasin_runoff", "ks": "subbasin_runoff",
            "cr": "hillslope_routing", "ci": "hillslope_routing", "cg": "hillslope_routing"
        }
        # 无延时无调蓄参数：汇流参数取上界（最快退水），产流参数取init值
        passthrough = [
            self.wum, self.wlm, self.wdm, self.b, self.kc,
            self.im, self.sm, self.ex,
            self.kg, self.ki, self.ks,
            0.9, 0.9, 0.5  # cr, ci, cg: 最快退水（上界）
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
        et_factor = _t(1.0)
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
            et_factor = cryo_result["et_factor"]
        else:
            P_eff = P
        
        # 检测是否为向量化模式（多个子流域）
        is_vectorized = P_eff.dim() >= 1 and P_eff.numel() > 1
        max_uh_len = 20  # 固定最大单位线长度

        x1 = _t(static_params.get("x1", self.x1))
        x2 = _t(static_params.get("x2", self.x2))
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
        E_adj = E * et_factor  # 冰冻圈：低温时蒸发减少
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
        R_ratio = torch.clamp(R / x3, min=0.0, max=1.0)
        Exch = x2 * torch.pow(R_ratio, 3)
        Exch = torch.maximum(Exch, -R*0.1)  # 下限
        Exch = torch.minimum(Exch, R*0.1)   # 上限
        R = R + uh1_out + Exch + glacier_melt_ground  # 冰川地下融水进入路由水库

        # ========= 6. 路由水库 =========
        Qr_out = R * (1 - torch.pow(1 + torch.pow(R / x3, 4), -0.25)) 
        R = R - Qr_out

        # ========= 总出流 =========
        Q = torch.clamp(Qr_out + uh2_out + glacier_melt_surface, min=0.0)  # 冰川地表融水直接汇入出流

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
        # 注意：这里使用调整后的蒸发 E_adj（考虑冰冻圈低温蒸发减少）
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
            evap_adjusted=E_adj,
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
    完整 SAC-SMA (Sacramento Soil Moisture Accounting) - USGS 标准版
    
    美国地质调查所（USGS）开发的完整非线性蓄水函数模型，应用于美国西部水文预报
    核心逻辑：两层土壤、张力水自由水分离、复杂非线性渗透、多出流路径、不透水面处理
    
    参数说明（共14个）：
    
    === 上层蓄水参数 ===
        - uztwm: 上层张力水最大容量 (mm) [5-100]
          物理含义：表层根系层张力水容量；代表根系可吸收的毛管水
          推荐范围：[20-50] for 中等土壤；[5-20] for 砂土；[50-100] for 粘土
        
        - uzfwm: 上层自由水最大容量 (mm) [40-300]
          物理含义：表层土壤自由水容量；降雨后立即产流的蓄水
          推荐范围：[60-120] for 一般土壤；[40-80] for 砂土；[150-300] for 粘土
    
    === 下层蓄水参数 ===
        - lztwm: 下层张力水最大容量 (mm) [50-500]
          物理含义：深层（毛管）张力水容量；影响蒸散模式改变点
          推荐范围：[100-300] for 一般流域
        
        - lzfpm: 下层主蓄水最大容量 (mm) [50-1000]
          物理含义：地下水主蓄水库；缓慢释放（地下水、孔隙度低含水层）
          推荐范围：[150-400] for 一般流域；[50-150] for 干旱；[500-1000] for 湿润
        
        - lzfsm: 下层补蓄水最大容量 (mm) [5-200]
          物理含义：地下水补蓄水库；快速补充主蓄水（裂隙网络、高孔隙度含水层）
          推荐范围：[20-80] for 一般流域；[5-30] for 基岩；[100-200] for 冲积
    
    === 不透水面积参数 ===
        - pctim: 不透水面积比例 [0.0-0.4]
          物理含义：城市建筑、道路等完全不透水面
          推荐范围：[0.0] for 自然流域；[0.05-0.15] for 城市化；[0.25-0.4] for 高度城市化
        
        - adimp: 附加不透水面积比例 [0.0-0.3]
          物理含义：部分冻融、薄层基岩等有限渗透的面积（产流但不入渗）
          推荐范围：[0.0] for 无冻融；[0.05-0.2] for 有限冻融；[0.15-0.3] for 重度冻融
    
    === 渗透参数（非线性过程控制） ===
        - rexp: 渗透指数 [1.0-3.0]
          物理含义：下渗速率随含水量变化的非线性度
          推荐范围：[1.5-2.5] for 一般土壤
          模型逻辑：Perc = PERC_MAX × (upper_storage/upper_capacity)^REXP
                   值越大，干旱时渗透越慢；越接近1渗透越线性
        
        - zperc: 渗透形状系数 [0.5-3.0]
          物理含义：上下层水分交换的强度调节
          推荐范围：[1.0-2.0] for 中等可渗透性
          模型逻辑：控制上层向下层的水分入流速率与其他参数的耦合
    
    === 出流系数参数 ===
        - lzpk: 下层主蓄水衰减系数 (1/day) [0.002-0.05]
          物理含义：地下水主库的日衰减系数
          推荐范围：[0.01-0.03] for 一般流域；[0.002-0.01] for 缓流；[0.03-0.05] for 快流
        
        - lzsk: 下层补蓄水衰减系数 (1/day) [0.01-0.1]
          物理含义：地下水补库的日衰减系数，通常>LZPK
          推荐范围：[0.03-0.08] for 一般流域
        
        - uzk: 上层自由水衰减系数 (1/day) [0.1-0.5]
          物理含义：表层自由水的日衰减系数，最快
          推荐范围：[0.25-0.45] for 一般流域
    
    === 蒸散参数 ===
        - pxv: 深层蒸散参数 [0.5-1.0]
          物理含义：从下层张力水蒸散的速率参数（相对于从上层的速率）
          推荐范围：[0.8-1.0] for 一般；低于0.5时表示深层蒸散有限
        
        - iv: 河流面积参数 [0.0-0.1]
          物理含义：河流水面面积占流域的比例（直接蒸散，不参与汇流）
          推荐范围：[0.0] for 一般；[0.01-0.05] for 湖泊湿地较多
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
        # 原SAC-SMA参数
        uztwm=30.0, uzfwm=80.0, lztwm=150.0, lzfpm=200.0, lzfsm=40.0,
        pctim=0.0, adimp=0.0,
        rexp=1.5, zperc=1.5,
        lzpk=0.02, lzsk=0.05, uzk=0.35,
        pxv=0.9, iv=0.0,
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
        
        # 原SAC-SMA参数
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
        self.pxv = pxv
        self.iv = iv

    def forward(self, inputs, static_params, dynamic_params, state, return_details=True):
        """完整 SAC-SMA 单时间步计算"""
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
        et_factor = _t(1.0)
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
            et_factor = cryo_result["et_factor"]
        else:
            P_eff = P
        P_eff_input = P_eff

        # 获取参数
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
        pxv = _t(static_params.get("pxv", self.pxv))
        iv = _t(static_params.get("iv", self.iv))

        # 获取状态
        uztw = _t(state.get("uztw", uztwm / 2))
        uzfw = _t(state.get("uzfw", uzfwm / 2))
        lztw = _t(state.get("lztw", lztwm / 2))
        lzfp = _t(state.get("lzfp", lzfpm / 2))
        lzfs = _t(state.get("lzfs", lzfsm / 2))

        # 不透水面降水直接产流（使用有效降水）
        runoff_pctim = P_eff_input * pctim
        runoff = runoff_pctim
        P_after_pctim = P_eff_input * (1 - pctim)

        # 附加不透水面产流（可能产流）
        # 冻土影响：附加不透水面积受冻土影响增加
        adimp_eff = adimp + (1 - frozen_factor) * (1 - adimp) * 0.3 if self.enable_cold_region else adimp
        runoff_adimp = P_after_pctim * adimp_eff
        runoff = runoff + runoff_adimp
        P_pervious = P_after_pctim * (1 - adimp_eff)

        # ===== 1. 蒸散模块（分层） =====
        # 冰冻圈：低温时蒸发减少
        E_eff = E * et_factor
        
        uz_total = uztw + uzfw
        lz_total = lztw + lzfp + lzfs

        # 上层蒸散优先
        et_uz = torch.minimum(uz_total, E_eff) # 上层实际蒸散量
        et_remain = E_eff - et_uz
        
        # 张力水优先被蒸发
        et_uztw = torch.minimum(uztw, et_uz)
        uztw = uztw - et_uztw
        et_uz_remain = et_uz - et_uztw
        uzfw = torch.clamp(uzfw - et_uz_remain, min=0.0)

        # 下层蒸散（自由水和张力水都参与）
        et_lz = et_remain * pxv * torch.clamp(lz_total / (lztwm + lzfpm + lzfsm), min=0.0, max=1.0)
        et_lz = torch.minimum(et_lz, lz_total)  # 蒸发不超过下层储水总量
        
        # 张力水优先蒸发
        et_lztw = torch.minimum(lztw, et_lz)
        lztw = lztw - et_lztw
        et_lz_remain = et_lz - et_lztw
        
        # 补蓄水次优先
        et_lzfs = torch.minimum(lzfs, et_lz_remain)
        lzfs = lzfs - et_lzfs
        et_lz_remain = et_lz_remain - et_lzfs
        
        # 主蓄水最后
        et_lzfp = torch.minimum(lzfp, et_lz_remain)
        lzfp = torch.clamp(lzfp - et_lzfp, min=0.0)

        # ===== 2. 上层蓄水模块 =====
        # 冻土影响入渗：frozen_factor 减少入渗，多余部分变为地表径流
        infiltration_input = P_pervious * frozen_factor if self.enable_cold_region else P_pervious
        frozen_excess = P_pervious - infiltration_input if self.enable_cold_region else _t(0.0)
        runoff = runoff + frozen_excess  # 冻土超渗产流
        
        # 张力水补充（优先）
        infil_remaining = infiltration_input
        dt_uztw = torch.minimum(infil_remaining, uztwm - uztw)
        uztw = uztw + dt_uztw
        infil_remaining = infil_remaining - dt_uztw

        # 自由水补充（次优先）
        dt_uzfw = torch.minimum(infil_remaining, uzfwm - uzfw)
        uzfw = uzfw + dt_uzfw
        # 超量溢出为地表径流
        infiltration_overflow = infil_remaining - dt_uzfw
        runoff = runoff + infiltration_overflow

        # ===== 3. 渗透模块（非线性，关键） =====
        uz_cap = uztwm + uzfwm
        uz_current = uztw + uzfw
        
        # 非线性渗透速率
        perc_ratio = torch.clamp(uz_current / uz_cap, min=0.0, max=1.0)
        perc_max = 10.0 * zperc  # 基准渗透速率
        perc = perc_max * torch.pow(perc_ratio, rexp)  # 非线性关系
        perc = torch.minimum(perc, uzfw)  # 不超过现有水量

        # 从自由水中渗透
        uzfw = torch.clamp(uzfw - perc, min=0.0)

        # ===== 4. 上层出流 =====
        # 自由水直接出流
        quz = uzk * uzfw
        runoff = runoff + quz
        uzfw = uzfw - quz

        # ===== 5. 下层蓄水与出流 =====
        # 渗透水分配到下层（复杂的容量分配）
        lz_remain = perc
        
        # 先补充下层张力水
        dt_lztw = torch.minimum(lz_remain, lztwm - lztw)
        lztw = lztw + dt_lztw
        lz_remain = lz_remain - dt_lztw

        # 再补充主蓄水
        dt_lzfp = torch.minimum(lz_remain, lzfpm - lzfp)
        lzfp = lzfp + dt_lzfp
        lz_remain = lz_remain - dt_lzfp

        # 补充补蓄水
        dt_lzfs = torch.minimum(lz_remain, lzfsm - lzfs)
        lzfs = lzfs + dt_lzfs
        lz_remain = lz_remain - dt_lzfs

        # 超量溢出
        runoff = runoff + lz_remain

        # 下层线性出流
        qlzfp = lzpk * lzfp
        runoff = runoff + qlzfp
        lzfp = lzfp - qlzfp

        qlzfs = lzsk * lzfs
        runoff = runoff + qlzfs
        lzfs = lzfs - qlzfs
        
        # 冰川融水贡献
        runoff = runoff + glacier_melt_surface  # 冰川地表融水进入地表径流
        lzfp = lzfp + glacier_melt_ground  # 冰川地下融水进入主蓄水库

        # ===== 6. 河流直接蒸散 =====
        runoff = runoff * (1 - iv)

        # ===== 7. 更新状态 =====
        state.update({
            "uztw": uztw,
            "uzfw": uzfw,
            "lztw": lztw,
            "lzfp": lzfp,
            "lzfs": lzfs,
        })
        
        # 冰冻圈状态
        if self.enable_cold_region:
            state.update({
                "swe": swe,
                "FI": FI,
            })

        # 计算总储水量（mm）：上层张力水+自由水 + 下层张力水+两级自由水
        # 支持向量化（多子流域）
        if not return_details:
            return {"runoff": runoff}, state

        total_storage = uztw + uzfw + lztw + lzfp + lzfs
        if self.enable_cold_region:
            total_storage = total_storage + swe

        total_et = et_uz + et_lz
        others = _make_process_outputs(runoff, total_storage, total_et)
        _update_process_outputs(
            others,
            storage_upper_tension=uztw,
            storage_upper_free=uzfw,
            storage_lower_tension=lztw,
            storage_lower_primary=lzfp,
            storage_lower_supplemental=lzfs,
            evap_upper_total=et_uz,
            evap_lower_total=et_lz,
            runoff_impervious_pctim=runoff_pctim,
            runoff_impervious_adimp=runoff_adimp,
            runoff_upper_zone=quz,
            runoff_lower_primary=qlzfp,
            runoff_lower_supplemental=qlzfs,
            percolation=perc,
            adimp_fraction_effective=adimp_eff,
            precip_after_pctim=P_after_pctim,
            precip_pervious=P_pervious,
            infiltration_actual=infiltration_input,
            infiltration_frozen_excess=frozen_excess,
            infiltration_overflow=infiltration_overflow,
            percolation_storage_excess=lz_remain,
            evap_upper_tension=et_uztw,
            evap_lower_tension=et_lztw,
            evap_lower_supplemental=et_lzfs,
            evap_lower_primary=et_lzfp,
            recharge_upper_tension=dt_uztw,
            recharge_upper_free=dt_uzfw,
            recharge_lower_tension=dt_lztw,
            recharge_lower_primary=dt_lzfp,
            recharge_lower_supplemental=dt_lzfs,
            evap_potential_adjusted=E_eff,
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
            "pctim", "adimp", "rexp", "zperc",
            "lzpk", "lzsk", "uzk", "pxv", "iv"
        ]
        bounds = [
            [5, 100], [40, 300], [50, 500], [50, 1000], [5, 200],
            [0.0, 0.4], [0.0, 0.3], [1.0, 3.0], [0.5, 3.0],
            [0.002, 0.05], [0.01, 0.1], [0.1, 0.5], [0.5, 1.0], [0.0, 0.1]
        ]
        init = [
            self.uztwm, self.uzfwm, self.lztwm, self.lzfpm, self.lzfsm,
            self.pctim, self.adimp, self.rexp, self.zperc,
            self.lzpk, self.lzsk, self.uzk, self.pxv, self.iv
        ]
        # 四类参数分类：产流参数(subbasin_runoff) vs 汇流参数(hillslope_routing)
        network_mapping = {
            # 产流参数：蓄水容量、不透水面积、渗透参数
            "uztwm": "subbasin_runoff", "uzfwm": "subbasin_runoff",
            "lztwm": "subbasin_runoff", "lzfpm": "subbasin_runoff", "lzfsm": "subbasin_runoff",
            "pctim": "subbasin_runoff", "adimp": "subbasin_runoff",
            "rexp": "subbasin_runoff", "zperc": "subbasin_runoff",
            "pxv": "subbasin_runoff", "iv": "subbasin_runoff",
            # 汇流参数：出流衰减系数
            "lzpk": "hillslope_routing", "lzsk": "hillslope_routing", "uzk": "hillslope_routing"
        }
        # 无延时无调蓄参数：衰减系数取上界（最快排空）
        passthrough = [
            self.uztwm, self.uzfwm, self.lztwm, self.lzfpm, self.lzfsm,
            self.pctim, self.adimp, self.rexp, self.zperc,
            0.05, 0.1, 0.5,  # lzpk, lzsk, uzk: 最快衰减（上界）
            self.pxv, self.iv
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
        }
        if self.enable_cold_region:
            state.update({
                "swe": 0.0,
                "FI": 0.0,
            })
        return state


    """
    Near-standard SAC-SMA implementation.

    Compared with the legacy version, this variant keeps the same parameter set
    but makes two targeted changes to better match the classic SAC-SMA logic:
    1. lower-zone evaporation is tension-water dominated instead of drawing
       directly from all lower-zone storages;
    2. riparian/open-water evaporation is represented as a separate loss term
       rather than applying a blanket multiplicative reduction to total runoff.
    """

    def forward(self, inputs, static_params, dynamic_params, state, return_details=True):
        """Near-standard SAC-SMA single-step calculation."""
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
        et_factor = _t(1.0)
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
            et_factor = cryo_result["et_factor"]
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
        pxv = _t(static_params.get("pxv", self.pxv))
        iv = _t(static_params.get("iv", self.iv))

        uztw = _t(state.get("uztw", uztwm / 2))
        uzfw = _t(state.get("uzfw", uzfwm / 2))
        lztw = _t(state.get("lztw", lztwm / 2))
        lzfp = _t(state.get("lzfp", lzfpm / 2))
        lzfs = _t(state.get("lzfs", lzfsm / 2))

        runoff_pctim = P_eff_input * pctim
        runoff = runoff_pctim
        P_after_pctim = P_eff_input * (1 - pctim)

        adimp_eff = adimp + (1 - frozen_factor) * (1 - adimp) * 0.3 if self.enable_cold_region else adimp
        runoff_adimp = P_after_pctim * adimp_eff
        runoff = runoff + runoff_adimp
        P_pervious = P_after_pctim * (1 - adimp_eff)

        E_eff = E * et_factor

        et_uztw = torch.minimum(uztw, E_eff)
        uztw = uztw - et_uztw
        et_remaining = E_eff - et_uztw

        uzfw_fraction = torch.clamp(uzfw / torch.clamp(uzfwm, min=1e-6), min=0.0, max=1.0)
        et_uzfw = torch.minimum(uzfw, et_remaining * uzfw_fraction)
        uzfw = torch.clamp(uzfw - et_uzfw, min=0.0)
        et_remaining = et_remaining - et_uzfw

        lztw_fraction = torch.clamp(lztw / torch.clamp(lztwm, min=1e-6), min=0.0, max=1.0)
        et_lztw = torch.minimum(lztw, et_remaining * pxv * lztw_fraction)
        lztw = lztw - et_lztw
        et_remaining = et_remaining - et_lztw

        et_uz = et_uztw + et_uzfw
        et_lz = et_lztw
        et_lzfs = _t(0.0)
        et_lzfp = _t(0.0)

        infiltration_input = P_pervious * frozen_factor if self.enable_cold_region else P_pervious
        frozen_excess = P_pervious - infiltration_input if self.enable_cold_region else _t(0.0)
        runoff = runoff + frozen_excess

        infil_remaining = infiltration_input
        dt_uztw = torch.minimum(infil_remaining, uztwm - uztw)
        uztw = uztw + dt_uztw
        infil_remaining = infil_remaining - dt_uztw

        dt_uzfw = torch.minimum(infil_remaining, uzfwm - uzfw)
        uzfw = uzfw + dt_uzfw
        infiltration_overflow = infil_remaining - dt_uzfw
        runoff = runoff + infiltration_overflow

        uz_cap = uztwm + uzfwm
        uz_current = uztw + uzfw
        perc_ratio = torch.clamp(uz_current / torch.clamp(uz_cap, min=1e-6), min=0.0, max=1.0)
        perc_max = 10.0 * zperc
        perc = perc_max * torch.pow(perc_ratio, rexp)
        perc = torch.minimum(perc, uzfw)
        uzfw = torch.clamp(uzfw - perc, min=0.0)

        quz = uzk * uzfw
        runoff = runoff + quz
        uzfw = uzfw - quz

        lz_remain = perc
        dt_lztw = torch.minimum(lz_remain, lztwm - lztw)
        lztw = lztw + dt_lztw
        lz_remain = lz_remain - dt_lztw

        dt_lzfp = torch.minimum(lz_remain, lzfpm - lzfp)
        lzfp = lzfp + dt_lzfp
        lz_remain = lz_remain - dt_lzfp

        dt_lzfs = torch.minimum(lz_remain, lzfsm - lzfs)
        lzfs = lzfs + dt_lzfs
        lz_remain = lz_remain - dt_lzfs

        runoff = runoff + lz_remain

        qlzfp = lzpk * lzfp
        runoff = runoff + qlzfp
        lzfp = lzfp - qlzfp

        qlzfs = lzsk * lzfs
        runoff = runoff + qlzfs
        lzfs = lzfs - qlzfs

        runoff = runoff + glacier_melt_surface
        lzfp = lzfp + glacier_melt_ground

        channel_evap = torch.minimum(runoff, E_eff * iv)
        runoff = torch.clamp(runoff - channel_evap, min=0.0)

        state.update({
            "uztw": uztw,
            "uzfw": uzfw,
            "lztw": lztw,
            "lzfp": lzfp,
            "lzfs": lzfs,
        })

        if self.enable_cold_region:
            state.update({
                "swe": swe,
                "FI": FI,
            })

        if not return_details:
            return {"runoff": runoff}, state

        total_storage = uztw + uzfw + lztw + lzfp + lzfs
        if self.enable_cold_region:
            total_storage = total_storage + swe

        total_et = et_uz + et_lz + channel_evap
        others = _make_process_outputs(runoff, total_storage, total_et)
        _update_process_outputs(
            others,
            storage_upper_tension=uztw,
            storage_upper_free=uzfw,
            storage_lower_tension=lztw,
            storage_lower_primary=lzfp,
            storage_lower_supplemental=lzfs,
            evap_upper_total=et_uz,
            evap_lower_total=et_lz,
            evap_open_water=channel_evap,
            runoff_impervious_pctim=runoff_pctim,
            runoff_impervious_adimp=runoff_adimp,
            runoff_upper_zone=quz,
            runoff_lower_primary=qlzfp,
            runoff_lower_supplemental=qlzfs,
            percolation=perc,
            adimp_fraction_effective=adimp_eff,
            precip_after_pctim=P_after_pctim,
            precip_pervious=P_pervious,
            infiltration_actual=infiltration_input,
            infiltration_frozen_excess=frozen_excess,
            infiltration_overflow=infiltration_overflow,
            percolation_storage_excess=lz_remain,
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
            evap_potential_adjusted=E_eff,
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
        et_factor = _t(1.0)
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
            et_factor = cryo_result["et_factor"]
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
        # 蒸散从第一层（蒸发乘以 et_factor）
        # -----------------
        E_adj = E * et_factor
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
            evap_potential_adjusted=E_adj,
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
    HBV (Hydrologiska Byr?ns Vattenbalansmodell) - 瑞典模型
    瑞典气象和水文学院（SMHI）开发，结合了冰冻圈和水热耦合的水文模型
    
    核心逻辑：雪融 → 土壤储水 → 非线性产流 → 三分流汇流 → 单位线延迟
    特点：简洁高效、参数物理意义明确、对冷地区适应性强
    
    参数说明（共9个）：
    
    === 土壤含水模块 ===
        - FC: 田间含水量 (mm) [50-400]
          物理含义：土壤含水量最大值（田间持水量）；控制产流和蒸散
          推荐范围：[100-250] for 中等土壤；[50-150] for 砂土；[200-400] for 粘土
        
        - LP: 蒸散临界含水量比 [0.3-0.9]
          物理含义：土壤含水量 / FC 的比值，当低于此值时蒸散受限
          推荐范围：[0.4-0.6] for 一般流域（0.5表示50%含水时蒸散开始降低）
          模型逻辑：实际蒸散 = PET × min(SM/FC, 1) × min(SM/(LP×FC), 1)
    
    === 产流模块 ===
        - BETA: 非线性产流指数 [0.5-2.0]
          物理含义：控制土壤储水与产流的非线性关系强度
          推荐范围：[1.2-1.8] for 一般流域
          模型逻辑：Runoff_contrib = P × (SM/FC)^BETA；BETA越大，则越干时产流越少
          物理解释：高BETA值表示干燥土壤持水能力强，降水易入渗；BETA=1时为线性关系
    
    === 快速汇流模块 ===
        - K0: 快速流（地表径流）系数 [0.1-0.5]
          物理含义：表层产流的日衰减系数；直接流出
          推荐范围：[0.25-0.45] for 快速响应流域
        
        - MAXBAS: 单位线响应时间 (day) [1-10]
          物理含义：流域汇流时间常数（平均流动时间）
          推荐范围：[2-5] for 中小流域；[5-10] for 大型流域
          模型逻辑：快速流通过长度为MAXBAS的单位线延迟输出
    
    === 缓流/基流模块 ===
        - K1: 慢速流（间接流）系数 [0.01-0.1]
          物理含义：土壤间隙水日衰减系数；较缓慢汇流
          推荐范围：[0.02-0.08] for 一般流域
        
        - K2: 地下水流系数 [0.001-0.02]
          物理含义：地下水库日衰减系数；基流输出
          推荐范围：[0.01-0.02] for 基流占比高的流域；[0.001-0.01] for 基流占比低的流域
    
    === 冰冻圈模块 ===
        - T_MELT: 融雪临界温度 (°C) [-5-3]
          物理含义：温度阈值；高于此温度时积雪开始融化
          推荐范围：[0.0-2.0] for 一般气候；[-2.0-0.0] for 高寒区（温度梯度不清晰）
          模型逻辑：降雪条件为 T < T_MELT；融雪量 = DDF × max(T - T_MELT, 0)
        
        - DDF: 融雪度日因子 (mm/°C) [2.0-10.0]
          物理含义：单位温度升高导致的日融雪量
          推荐范围：[3.0-5.0] for 冷季融雪（阴坡、森林覆盖）
          推荐范围：[5.0-8.0] for 暖季融雪（阳坡、裸露）
          推荐范围：[8.0-10.0] for 冰川融化（低反照率）
    
    === 关键约束 ===
    土壤水热平衡：蒸散受限于 PET、土壤含水 和 LP 参数
    级联延迟：K0(快) << K1(中) << K2(缓)，形成分阶段流量释放
    冰冻圈耦合：融雪作为降雨进入土壤模块，与其他降水同等对待
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
        fc=200.0, lp=0.5, beta=1.5,
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
        et_factor = _t(1.0)
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
            et_factor = cryo_result["et_factor"]
        else:
            T = _t(inputs.get("T", 0.0))
            T_input = T
            P_eff = P

        fc = _t(static_params.get("fc", self.fc))
        lp = _t(static_params.get("lp", self.lp))
        beta = _t(static_params.get("beta", self.beta))
        k0 = _t(static_params.get("k0", self.k0))
        k1 = _t(static_params.get("k1", self.k1))
        k2 = _t(static_params.get("k2", self.k2))
        maxbas = static_params.get("maxbas", self.maxbas)  # 保持为标量/整数
        t_melt = _t(static_params.get("t_melt", self.t_melt))
        ddf = _t(static_params.get("ddf", self.ddf))

        # 状态量
        sm = _t(state.get("sm", fc / 2.0))      # 土壤水分
        uz = _t(state.get("uz", 10.0))          # 快速流
        lz = _t(state.get("lz", 10.0))          # 慢速流
        snow = _t(state.get("snow", 0.0))       # 雪储存（HBV原有雪模块，非冰冻圈模式使用）
        
        # 单位线延迟：使用简化的累积延迟方法（避免使用list队列）
        qh_queue = _t(state.get("qh_queue", 0.0))  # 累积快速径流
        qh_idx = _t(state.get("qh_idx", 0.0))      # 时间步计数器

        # --- 1. 雪/融雪模块（仅在非冰冻圈模式下使用HBV原有雪模块） ---
        if not self.enable_cold_region:
            snow = torch.where(T < t_melt, snow + P, snow)
            melt = ddf * torch.clamp(T - t_melt, min=0.0)
            melt = torch.minimum(melt, snow)
            snow = snow - melt
            P_eff = torch.where(T < t_melt, melt, P + melt)
        else:
            # 冰冻圈模式下，P_eff已经在上面计算过了
            melt = snow_melt
        P_eff_model = P_eff

        # --- 2. 土壤蒸散（向量化，乘以et_factor） ---
        E_adj = E * et_factor
        et = E_adj * torch.clamp(sm / (lp * fc), max=1.0)
        et = torch.minimum(et, sm)
        sm = sm - et

        # --- 3. 产流（BETA 函数，向量化） ---
        # 冻土影响入渗：frozen_factor 减少入渗，多余部分变为地表径流
        infil_capacity = P_eff * frozen_factor if self.enable_cold_region else P_eff
        frozen_excess = P_eff - infil_capacity if self.enable_cold_region else _t(0.0)
        
        # 确保fc为正，避免除零
        fc_safe = torch.clamp(fc, min=1.0)
        beta_safe = torch.clamp(beta, min=0.1, max=5.0)
        
        # 计算产流（添加数值保护）
        sm_ratio = torch.clamp(sm / fc_safe, min=0.0, max=1.0)
        recharge = infil_capacity * torch.pow(sm_ratio + 1e-8, beta_safe)
        recharge = torch.clamp(recharge, min=0.0)
        recharge = torch.minimum(recharge, infil_capacity)
        sm_new = sm + (infil_capacity - recharge)
        # 土壤容量溢流直接产流
        sm_overflow = torch.clamp(sm_new - fc_safe, min=0.0)
        sm = torch.clamp(torch.minimum(sm_new, fc_safe), min=0.0)

        # --- 4. 快慢流和地下水流（向量化） ---
        to_uz = recharge * 0.6
        to_lz = recharge * 0.4
        
        uz_new = uz + to_uz
        uz_overflow = torch.clamp(uz_new - fc_safe, min=0.0)
        uz = torch.clamp(torch.minimum(uz_new, fc_safe), min=0.0)
        qf = k0 * uz
        uz = torch.clamp(uz - qf, min=0.0)

        lz_new = lz + to_lz
        lz_overflow = torch.clamp(lz_new - fc_safe * 2, min=0.0)
        lz = torch.clamp(torch.minimum(lz_new, fc_safe * 2), min=0.0)
        qs = k1 * lz
        lz = torch.clamp(lz - qs, min=0.0)
        
        qg = k2 * lz
        lz = torch.clamp(lz - qg, min=0.0)
        lz = torch.minimum(lz, fc_safe * 2)
        lz_overflow2 = torch.clamp(lz - fc * 2, min=0.0)

        # --- 5. 单位线延迟（maxbas） ---
        # 快速径流组分进入延迟队列
        quick_runoff = qf + qs + sm_overflow + uz_overflow + lz_overflow + lz_overflow2 + frozen_excess
        quick_runoff = quick_runoff + glacier_melt_surface  # 冰川地表融水进入快速径流
        quick_runoff = torch.clamp(quick_runoff, min=0.0)  # 确保非负
        
        # 简化的延迟机制：使用指数衰减代替精确的单位线
        # 这种方法支持向量化计算，同时保持延迟效果的物理意义
        maxbas_t = _t(maxbas) if not isinstance(maxbas, torch.Tensor) else maxbas
        maxbas_t = torch.clamp(maxbas_t, min=1.0, max=10.0)
        
        # 衰减系数：1/maxbas，表示每个时间步释放的比例
        decay = 1.0 / maxbas_t
        decay = torch.clamp(decay, min=0.1, max=1.0)  # 确保衰减系数在合理范围
        
        # 将当前快速径流加入累积队列
        qh_queue = qh_queue + quick_runoff
        qh_queue = torch.clamp(qh_queue, min=0.0)  # 确保非负
        
        # 从累积队列中释放一部分作为延迟后的径流
        runoff_delayed = qh_queue * decay
        qh_queue = qh_queue * (1.0 - decay)  # 保持数值稳定
        qh_queue = torch.clamp(qh_queue, min=0.0)  # 确保非负
        
        # 更新时间步计数
        qh_idx = qh_idx + 1
        
        # 总径流 = 延迟后的快速径流 + 基流（基流不延迟）
        runoff = runoff_delayed + qg
        runoff = torch.clamp(runoff, min=0.0)  # 确保径流非负
        
        # 冰川地下融水进入慢流水库
        lz = lz + glacier_melt_ground

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

        if self.enable_cold_region:
            total_storage = sm + uz + lz + swe + qh_queue
        else:
            total_storage = sm + uz + lz + snow + qh_queue

        others = _make_process_outputs(runoff, total_storage, et)
        _update_process_outputs(
            others,
            recharge=recharge,
            storage_soil=sm,
            storage_upper_zone=uz,
            storage_lower_zone=lz,
            storage_snowpack=(swe if self.enable_cold_region else snow),
            storage_delay_queue=qh_queue,
            runoff_quick_component=qf,
            runoff_intermediate_component=qs,
            runoff_baseflow_component=qg,
            runoff_quick_generated=quick_runoff,
            runoff_delayed=runoff_delayed,
            infiltration_actual=infil_capacity,
            infiltration_frozen_excess=frozen_excess,
            snowpack_melt=melt,
            evap_potential_adjusted=E_adj,
            storage_ratio_soil=sm_ratio,
            overflow_soil=sm_overflow,
            overflow_upper_zone=uz_overflow,
            overflow_lower_zone_1=lz_overflow,
            overflow_lower_zone_2=lz_overflow2,
            routing_decay=decay,
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
            names = ["fc", "lp", "beta", "k0", "k1", "k2", "maxbas"]
            bounds = [
                [50, 400], [0.3, 0.9], [0.5, 2.0],
                [0.1, 0.5], [0.01, 0.1], [0.001, 0.02],
                [1, 10]
            ]
            init = [self.fc, self.lp, self.beta, self.k0, self.k1, self.k2, self.maxbas]
            network_mapping = {
                "fc": "subbasin_runoff", "lp": "subbasin_runoff", "beta": "subbasin_runoff",
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
            names = ["fc", "lp", "beta", "k0", "k1", "k2", "maxbas", "t_melt", "ddf"]
            bounds = [
                [50, 400], [0.3, 0.9], [0.5, 2.0],
                [0.1, 0.5], [0.01, 0.1], [0.001, 0.02],
                [1, 10], [-5, 5], [0.5, 10.0]
            ]
            init = [self.fc, self.lp, self.beta, self.k0, self.k1, self.k2, self.maxbas, self.t_melt, self.ddf]
            network_mapping = {
                "fc": "subbasin_runoff", "lp": "subbasin_runoff", "beta": "subbasin_runoff",
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

