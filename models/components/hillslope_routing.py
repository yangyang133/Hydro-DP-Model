"""
坡面汇流模块（Hillslope Routing）

本模块实现网格产流到河网节点的汇流过程，位于模型调用链的中间环节：
    grid_runoff（网格产流） => hillslope_routing（坡面汇流） => river_routing（河道汇流）

==============================================================================
模块功能
==============================================================================
将网格产流（来自grid_runoff模块的runoff输出）汇集并演算到河网节点，
生成各节点的侧向入流序列（Q_lateral），作为river_routing模块的输入。

==============================================================================
输入数据说明
==============================================================================
inputs字典必须包含：
    - runoff: tensor (n_subbasins, T) 或 (T, H, W)
        网格/子流域产流序列（来自grid_runoff或subbasin_runoff模块）
    
    对于基于掩膜的方法，还需要：
    - upstream_mask: tensor (n_nodes, H, W) 或 (n_nodes, n_grids), bool
        每个河网节点的上游网格/子流域掩膜矩阵
    
    对于基于D8流向的方法，还需要：
    - flow_dir: tensor (H, W)
        D8流向编码矩阵
    - river_mask: tensor (H, W), bool
        河网节点位置掩膜

可选输入：
    - cell_area: tensor (H, W) 或 float
        网格面积（m²），用于产流深度到体积的转换
    - dt: float
        时间步长（天），默认1.0

==============================================================================
输出说明
==============================================================================
outputs字典包含：
    - Q_lateral: tensor (n_nodes, T)
        各河网节点的侧向入流序列，可直接传递给river_routing作为Q_in
    - Q_total: tensor (T,)
        所有节点入流的汇总

==============================================================================
可用汇流方法
==============================================================================
基于掩膜的子流域汇流（推荐）：
    - mask_direct          : 直接汇集法（无演算）
    - mask_linear_reservoir: 线性水库法
    - mask_nash_cascade    : Nash串联水库法
    - mask_unit_hydrograph : 单位线法（SCS/Nash/三角形/Snyder/Clark）
    - mask_time_area       : 时间面积法
    - mask_gamma_uh        : Gamma分布单位线
    - mask_lag_route       : 滞后演算法

基于D8流向的分布式汇流：
    - d8_travel_time       : D8流向汇流时间法

模块导出：通过注册器将模型注册到 `HILLSLOPE_REGISTRY`，以便整体框架按名称使用。
"""

import torch
import numpy as np
import math
from models.base import BaseModel
from models.registry import HILLSLOPE_REGISTRY, register


# ==============================================================================
# 工具函数
# ==============================================================================

# D8 流向编码：1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE
D8_CODES = [1, 2, 4, 8, 16, 32, 64, 128]
D8_OFFSETS = {
    1:   (0, 1),    # E
    2:   (1, 1),    # SE
    4:   (1, 0),    # S
    8:   (1, -1),   # SW
    16:  (0, -1),   # W
    32:  (-1, -1),  # NW
    64:  (-1, 0),   # N
    128: (-1, 1),   # NE
}
D8_DISTANCES = {
    1: 1.0, 2: 1.414, 4: 1.0, 8: 1.414,
    16: 1.0, 32: 1.414, 64: 1.0, 128: 1.414
}


def trace_d8_to_boundary(flow_dir, basin_mask, cellsize=1.0):
    """
    追踪每个网格沿D8流向到达其所属子流域下游边界的路径距离
    
    Parameters
    ----------
    flow_dir : ndarray (H, W)
        D8流向编码矩阵
    basin_mask : ndarray (H, W), int
        子流域编号掩膜，每个网格标注所属子流域（0, 1, 2, ...）
        -1 或 NaN 表示不属于任何子流域
    cellsize : float
        网格边长（米）
        
    Returns
    -------
    basin_id : ndarray (H, W), int
        每个网格所属的子流域编号（-1表示无效）
    travel_dist : ndarray (H, W), float
        每个网格到其子流域下游边界的路径距离（米）
    n_basins : int
        子流域总数
    """
    H, W = flow_dir.shape
    
    # 处理掩膜，确保为整数类型
    basin_mask = np.asarray(basin_mask)
    if np.issubdtype(basin_mask.dtype, np.floating):
        basin_mask = np.where(np.isnan(basin_mask), -1, basin_mask).astype(np.int32)
    else:
        basin_mask = basin_mask.astype(np.int32)
    
    # 获取子流域数量
    valid_basins = basin_mask[basin_mask >= 0]
    n_basins = int(valid_basins.max()) + 1 if len(valid_basins) > 0 else 0
    
    # 初始化输出
    basin_id = basin_mask.copy()
    travel_dist = np.full((H, W), np.inf, dtype=np.float32)
    
    max_steps = H * W  # 防止无限循环
    
    for i in range(H):
        for j in range(W):
            current_basin = basin_mask[i, j]
            if current_basin < 0:
                continue  # 无效网格
            
            # 沿流向追踪，直到离开当前子流域
            curr_r, curr_c = i, j
            dist = 0.0
            visited = set()
            
            for _ in range(max_steps):
                if (curr_r, curr_c) in visited:
                    break  # 检测到循环
                visited.add((curr_r, curr_c))
                
                # 获取下游位置
                fd = flow_dir[curr_r, curr_c]
                if fd not in D8_OFFSETS:
                    # 无效流向或出口，当前位置即为边界
                    travel_dist[i, j] = dist
                    break
                
                dr, dc = D8_OFFSETS[fd]
                next_r, next_c = curr_r + dr, curr_c + dc
                
                # 边界检查
                if not (0 <= next_r < H and 0 <= next_c < W):
                    # 流出边界，当前位置即为边界
                    travel_dist[i, j] = dist
                    break
                
                # 检查下游是否离开当前子流域
                next_basin = basin_mask[next_r, next_c]
                if next_basin != current_basin:
                    # 到达子流域边界
                    travel_dist[i, j] = dist
                    break
                
                # 累加距离，继续追踪
                dist += D8_DISTANCES[fd] * cellsize
                curr_r, curr_c = next_r, next_c
    
    return basin_id, travel_dist, n_basins


def aggregate_by_mask(runoff, upstream_mask, cell_area=None):
    """
    根据上游掩膜汇集产流到各河网节点
    
    支持两种输入格式：
    1. 栅格格式: runoff (T, H, W), upstream_mask (n_nodes, H, W)
    2. 子流域格式: runoff (n_subbasins, T), upstream_mask (n_nodes, n_subbasins)
    
    Parameters
    ----------
    runoff : tensor
        产流时间序列
    upstream_mask : tensor, bool
        每个河网节点的上游掩膜
    cell_area : tensor or float, optional
        网格面积（m²），用于将产流深度转换为体积
        
    Returns
    -------
    Q_lateral : tensor (n_nodes, T)
        各节点汇集的入流序列
    """
    device = runoff.device
    dtype = runoff.dtype
    
    # 判断输入格式
    if runoff.ndim == 3:
        # 栅格格式: (T, H, W)
        T, H, W = runoff.shape
        n_nodes = upstream_mask.shape[0]
        
        # 处理面积
        if cell_area is not None:
            if isinstance(cell_area, (int, float)):
                area = torch.full((H, W), cell_area, device=device, dtype=dtype)
            elif not isinstance(cell_area, torch.Tensor):
                area = torch.tensor(cell_area, device=device, dtype=dtype)
            else:
                area = cell_area
        else:
            area = torch.ones(H, W, device=device, dtype=dtype)
        
        # 确保掩膜是float类型
        mask_float = upstream_mask.float()  # (n_nodes, H, W)
        
        # 向量化计算
        # runoff * area: (T, H, W)
        # mask_float: (n_nodes, H, W)
        runoff_vol = runoff * area  # (T, H, W)
        
        # 使用einsum进行高效计算
        # Q_lateral[n, t] = sum over (h,w) of mask[n,h,w] * runoff_vol[t,h,w]
        Q_lateral = torch.einsum('nhw,thw->nt', mask_float, runoff_vol)
        
    else:
        # 子流域格式: runoff (n_subbasins, T), upstream_mask (n_nodes, n_subbasins)
        n_subbasins, T = runoff.shape
        n_nodes = upstream_mask.shape[0]
        
        mask_float = upstream_mask.float()  # (n_nodes, n_subbasins)
        
        # 处理面积（子流域面积）
        if cell_area is not None:
            if isinstance(cell_area, (int, float)):
                area = torch.full((n_subbasins,), cell_area, device=device, dtype=dtype)
            elif not isinstance(cell_area, torch.Tensor):
                area = torch.tensor(cell_area, device=device, dtype=dtype)
            else:
                area = cell_area
            runoff_vol = runoff * area.unsqueeze(1)  # (n_subbasins, T)
        else:
            runoff_vol = runoff
        
        # Q_lateral[n, t] = sum over s of mask[n,s] * runoff_vol[s,t]
        Q_lateral = torch.einsum('ns,st->nt', mask_float, runoff_vol)
    
    return Q_lateral


# ==============================================================================
# 基于掩膜的子流域汇流模型
# ==============================================================================

@register(HILLSLOPE_REGISTRY, "mask_direct")
class MaskDirectRouting(BaseModel):
    """
    直接汇集法（无汇流演算）
    
    将掩膜区域内的产流直接汇集到河网节点，不做任何汇流延迟处理。
    适用于小流域或作为基准对比。
    
    输入：
        - runoff: (n_subbasins, T) 或 (T, H, W) 产流序列
        - upstream_mask: (n_nodes, n_subbasins) 或 (n_nodes, H, W) 上游掩膜
        - cell_area: 可选，网格/子流域面积
    
    输出：
        - Q_lateral: (n_nodes, T) 各节点侧向入流
    """
    
    def __init__(self, modules=None):
        super().__init__()
    
    def forward(self, inputs, static_params, dynamic_params, state):
        runoff = inputs["runoff"]
        upstream_mask = inputs["upstream_mask"]
        cell_area = inputs.get("cell_area", None)
        
        Q_lateral = aggregate_by_mask(runoff, upstream_mask, cell_area)
        
        return {
            "Q_lateral": Q_lateral,
            "Q_total": Q_lateral.sum(dim=0)
        }, state
    
    def _define_parameters(self):
        return {
            "static": {"names": [], "bounds": [], "init": []},
            "dynamic": {"names": [], "bounds": []},
        }


@register(HILLSLOPE_REGISTRY, "mask_linear_reservoir")
class MaskLinearReservoir(BaseModel):
    """
    掩膜汇集 + 线性水库法
    
    先将掩膜区域内产流汇集，再通过线性水库进行汇流演算。
    
    线性水库方程：
        dS/dt = Q_in - Q_out
        S = K * Q_out
        
    离散形式：
        Q_out[t] = α * Q_in[t] + β * Q_out[t-1]
        其中 α = dt/(K+dt), β = K/(K+dt)
    
    参数说明：
        K : float
            水库常数/滞后时间（天）[0.1-10.0]
    """
    
    def __init__(self, K=1.0, modules=None):
        super().__init__()
        self.K = K
    
    def forward(self, inputs, static_params, dynamic_params, state):
        runoff = inputs["runoff"]
        upstream_mask = inputs["upstream_mask"]
        cell_area = inputs.get("cell_area", None)
        dt = inputs.get("dt", 1.0)
        
        device = runoff.device
        dtype = runoff.dtype
        
        # 获取参数
        K = static_params.get("K", self.K)
        K_t = K if isinstance(K, torch.Tensor) else torch.tensor(K, device=device, dtype=dtype)
        
        # 汇集产流
        Q_in = aggregate_by_mask(runoff, upstream_mask, cell_area)  # (n_nodes, T)
        n_nodes, T = Q_in.shape
        
        # 支持每个节点不同的K值
        if K_t.ndim == 0:
            K_t = K_t.unsqueeze(0).expand(n_nodes).unsqueeze(1)
        elif K_t.ndim == 1:
            K_t = K_t.unsqueeze(1)
        
        alpha = dt / (K_t + dt)
        beta = K_t / (K_t + dt)
        
        Q_prev = state.get("Q_prev", torch.zeros(n_nodes, 1, device=device, dtype=dtype))
        
        outputs = []
        for t in range(T):
            q_in_t = Q_in[:, t:t+1]
            q_out_t = alpha * q_in_t + beta * Q_prev
            outputs.append(q_out_t)
            Q_prev = q_out_t
        
        Q_lateral = torch.cat(outputs, dim=1)
        state["Q_prev"] = Q_prev
        
        return {
            "Q_lateral": Q_lateral,
            "Q_total": Q_lateral.sum(dim=0),
            "Q_in": Q_in,
        }, state
    
    def _define_parameters(self):
        return {
            "static": {
                "names": ["K"],
                "bounds": [[0.1, 10.0]],
                "init": [self.K],
                # 无延时无调蓄（K=0: Q_out=Q_in）
                "passthrough": [0.0],
                "network_mapping": {"K": "hillslope_routing"}
            },
            "dynamic": {"names": [], "bounds": []},
        }


@register(HILLSLOPE_REGISTRY, "mask_nash_cascade")
class MaskNashCascade(BaseModel):
    """
    掩膜汇集 + Nash串联水库法
    
    将n个相同时间常数K的线性水库串联，模拟汇流过程。
    该方法产生的出流过程符合Gamma分布形状。
    
    Nash级联方程（连续形式）：
        u(t) = (1/K) * (t/K)^(n-1) * exp(-t/K) / Γ(n)
    
    离散递推形式（对每个水库）：
        Q_i[t] = α * Q_{i-1}[t] + β * Q_i[t-1]
        其中 α = dt/(K+dt), β = K/(K+dt)
    
    参数说明：
        n : int
            水库个数（级数）[1-10]
        K : float
            单个水库的时间常数（天）[0.1-10.0]
    """
    
    def __init__(self, n=3, K=1.0, modules=None):
        super().__init__()
        self.n = int(n)
        self.K = K
    
    def forward(self, inputs, static_params, dynamic_params, state):
        runoff = inputs["runoff"]
        upstream_mask = inputs["upstream_mask"]
        cell_area = inputs.get("cell_area", None)
        dt = inputs.get("dt", 1.0)
        
        device = runoff.device
        dtype = runoff.dtype
        
        # 获取参数
        n = int(static_params.get("n", self.n))
        K = static_params.get("K", self.K)
        K_t = K if isinstance(K, torch.Tensor) else torch.tensor(K, device=device, dtype=dtype)
        
        # 汇集产流
        Q_in = aggregate_by_mask(runoff, upstream_mask, cell_area)
        n_nodes, T = Q_in.shape
        
        # 参数处理
        if K_t.ndim == 0:
            K_t = K_t.unsqueeze(0).expand(n_nodes).unsqueeze(1)
        elif K_t.ndim == 1:
            K_t = K_t.unsqueeze(1)
        
        alpha = dt / (K_t + dt)
        beta = K_t / (K_t + dt)
        
        # 初始化各级水库状态
        reservoir_states = []
        for i in range(n):
            key = f"Q_reservoir_{i}"
            S_i = state.get(key, torch.zeros(n_nodes, 1, device=device, dtype=dtype))
            reservoir_states.append(S_i)
        
        # Nash串联演算
        outputs = []
        for t in range(T):
            q_current = Q_in[:, t:t+1]
            
            for i in range(n):
                q_out = alpha * q_current + beta * reservoir_states[i]
                reservoir_states[i] = q_out
                q_current = q_out
            
            outputs.append(q_current)
        
        Q_lateral = torch.cat(outputs, dim=1)
        
        # 更新状态
        for i in range(n):
            state[f"Q_reservoir_{i}"] = reservoir_states[i]
        
        return {
            "Q_lateral": Q_lateral,
            "Q_total": Q_lateral.sum(dim=0),
            "Q_in": Q_in,
        }, state
    
    def _define_parameters(self):
        return {
            "static": {
                "names": ["n", "K"],
                "bounds": [[1, 10], [0.1, 10.0]],
                "init": [self.n, self.K],
                # 无延时无调蓄（单个水库K=0: 直通）
                "passthrough": [1, 0.0],
                "network_mapping": {
                    "n": "hillslope_routing",
                    "K": "hillslope_routing"
                }
            },
            "dynamic": {"names": [], "bounds": []},
        }


@register(HILLSLOPE_REGISTRY, "mask_unit_hydrograph")
class MaskUnitHydrograph(BaseModel):
    """
    掩膜汇集 + 单位线法
    
    先将掩膜区域内产流汇集，再通过单位线卷积进行汇流演算。
    
    支持多种单位线类型：
        - scs      : SCS三角形单位线（NRCS/USDA）
        - nash     : Nash瞬时单位线（Gamma分布）
        - triangle : 通用三角形单位线
        - snyder   : Snyder合成单位线
        - clark    : Clark时间面积单位线（简化形式）
    
    参数说明：
        Tp : float
            峰值时间（天）[0.5-10.0]
        n : int
            Nash级数（仅nash类型使用）[1-10]
        k : float
            时间常数（天，仅nash类型使用）[0.1-10.0]
        uh_type : str
            单位线类型
        uh_length : int
            单位线长度（时间步数）
    """
    
    def __init__(self, Tp=1.0, n=3, k=1.0, uh_type="scs", uh_length=20, modules=None):
        super().__init__()
        self.Tp = Tp
        self.n = int(n)
        self.k = k
        self.uh_type = uh_type
        self.uh_length = uh_length
    
    def _generate_unit_hydrograph(self, uh_type, Tp, n, k, length, device, dtype):
        """生成归一化单位线"""
        t = torch.arange(1, length + 1, device=device, dtype=dtype)
        
        if uh_type == "nash":
            # Nash瞬时单位线 (Gamma分布)
            gamma_n = math.gamma(n)
            uh = (1.0 / k) * (t / k) ** (n - 1) * torch.exp(-t / k) / gamma_n
            
        elif uh_type == "triangle":
            # 通用三角形单位线
            Tp_t = max(1.0, Tp)
            Tb = 2.67 * Tp_t
            
            uh = torch.zeros_like(t)
            rising = t <= Tp_t
            falling = (t > Tp_t) & (t <= Tb)
            
            uh[rising] = t[rising] / (Tp_t ** 2)
            uh[falling] = (Tb - t[falling]) / (Tp_t * (Tb - Tp_t))
            uh = torch.clamp(uh, min=0.0)
            
        elif uh_type == "snyder":
            # Snyder合成单位线（对数正态分布近似）
            Tp_t = max(1.0, Tp)
            sigma = 0.4
            mu = math.log(Tp_t)
            uh = torch.exp(-0.5 * ((torch.log(t) - mu) / sigma) ** 2) / (t * sigma * math.sqrt(2 * math.pi))
            
        elif uh_type == "clark":
            # Clark时间面积法（简化形式）
            Tc = max(1.0, Tp)
            uh = torch.zeros_like(t)
            mask = t <= Tc
            uh[mask] = t[mask] / Tc
            uh = torch.diff(uh, prepend=torch.zeros(1, device=device, dtype=dtype))
            uh = torch.clamp(uh, min=0.0)
            
        else:  # scs (默认)
            # SCS三角形单位线
            Tp_t = max(1.0, Tp)
            Tb = 2.67 * Tp_t
            
            uh = torch.zeros_like(t)
            rising = t <= Tp_t
            falling = (t > Tp_t) & (t <= Tb)
            
            uh[rising] = t[rising] / (Tp_t ** 2)
            uh[falling] = (Tb - t[falling]) / (Tp_t * (Tb - Tp_t))
            uh = torch.clamp(uh, min=0.0)
        
        # 归一化
        uh_sum = uh.sum()
        if uh_sum > 0:
            uh = uh / uh_sum
        
        return uh
    
    def forward(self, inputs, static_params, dynamic_params, state):
        runoff = inputs["runoff"]
        upstream_mask = inputs["upstream_mask"]
        cell_area = inputs.get("cell_area", None)
        
        device = runoff.device
        dtype = runoff.dtype
        
        # 获取参数
        Tp = float(static_params.get("Tp", self.Tp))
        n = int(static_params.get("n", self.n))
        k = float(static_params.get("k", self.k))
        uh_type = static_params.get("uh_type", self.uh_type)
        uh_length = int(static_params.get("uh_length", self.uh_length))
        
        # 汇集产流
        Q_in = aggregate_by_mask(runoff, upstream_mask, cell_area)
        n_nodes, T = Q_in.shape
        
        # 生成单位线
        uh = self._generate_unit_hydrograph(uh_type, Tp, n, k, uh_length, device, dtype)
        
        # 初始化延迟队列
        delay_queue = state.get(
            "delay_queue",
            torch.zeros(n_nodes, uh_length, device=device, dtype=dtype)
        )
        
        outputs = []
        for t in range(T):
            q_in_t = Q_in[:, t:t+1]
            
            # 入流对未来各时刻的贡献
            delay_queue[:, :len(uh)] += q_in_t * uh.unsqueeze(0)
            
            # 当前时刻出流
            q_out_t = delay_queue[:, 0:1]
            outputs.append(q_out_t)
            
            # 延迟队列前移
            delay_queue = torch.roll(delay_queue, -1, dims=1)
            delay_queue[:, -1] = 0.0
        
        Q_lateral = torch.cat(outputs, dim=1)
        state["delay_queue"] = delay_queue
        
        return {
            "Q_lateral": Q_lateral,
            "Q_total": Q_lateral.sum(dim=0),
            "Q_in": Q_in,
        }, state
    
    def _define_parameters(self):
        return {
            "static": {
                "names": ["Tp", "n", "k"],
                "bounds": [[0.5, 10.0], [1, 10], [0.1, 10.0]],
                "init": [self.Tp, self.n, self.k],
                # 无延时无调蓄（单位线近似delta函数）
                "passthrough": [0.01, 1, 0.01],
                "network_mapping": {
                    "Tp": "hillslope_routing",
                    "n": "hillslope_routing",
                    "k": "hillslope_routing"
                }
            },
            "dynamic": {"names": [], "bounds": []},
        }


@register(HILLSLOPE_REGISTRY, "mask_time_area")
class MaskTimeAreaRouting(BaseModel):
    """
    掩膜汇集 + 时间面积法
    
    假设子流域内的汇流时间按面积线性分布，不同距离的产流按其汇流时间到达出口。
    
    时间面积法假设：
    - 汇流时间Tc内，产流按面积分布逐步到达出口
    - 每个时间步到达的比例由面积分布决定
    
    简化形式：假设面积-时间关系为线性
        A(t) = t / Tc, 0 <= t <= Tc
    
    参数说明：
        Tc : float
            汇流时间（天）[0.5-20.0]
        K : float
            线性水库调蓄常数（天）[0.1-10.0]
    """
    
    def __init__(self, Tc=2.0, K=1.0, modules=None):
        super().__init__()
        self.Tc = Tc
        self.K = K
    
    def forward(self, inputs, static_params, dynamic_params, state):
        runoff = inputs["runoff"]
        upstream_mask = inputs["upstream_mask"]
        cell_area = inputs.get("cell_area", None)
        dt = inputs.get("dt", 1.0)
        
        device = runoff.device
        dtype = runoff.dtype
        
        # 获取参数
        Tc = float(static_params.get("Tc", self.Tc))
        K = static_params.get("K", self.K)
        K_t = K if isinstance(K, torch.Tensor) else torch.tensor(K, device=device, dtype=dtype)
        
        # 汇集产流
        Q_in = aggregate_by_mask(runoff, upstream_mask, cell_area)
        n_nodes, T = Q_in.shape
        
        # 构建时间面积曲线（离散化）
        n_steps = max(1, int(Tc / dt))
        ta_weights = torch.ones(n_steps, device=device, dtype=dtype) / n_steps
        
        # 初始化延迟队列
        delay_queue = state.get(
            "delay_queue",
            torch.zeros(n_nodes, n_steps, device=device, dtype=dtype)
        )
        
        # 线性水库参数
        if K_t.ndim == 0:
            K_t = K_t.unsqueeze(0).expand(n_nodes).unsqueeze(1)
        elif K_t.ndim == 1:
            K_t = K_t.unsqueeze(1)
        
        alpha = dt / (K_t + dt)
        beta = K_t / (K_t + dt)
        
        Q_prev = state.get("Q_prev", torch.zeros(n_nodes, 1, device=device, dtype=dtype))
        
        outputs = []
        for t in range(T):
            q_in_t = Q_in[:, t:t+1]
            
            # 产流按时间面积曲线分配到延迟队列
            delay_queue += q_in_t * ta_weights.unsqueeze(0)
            
            # 从延迟队列取出当前时刻到达的产流
            q_arrive_t = delay_queue[:, 0:1]
            
            # 延迟队列前移
            delay_queue = torch.roll(delay_queue, -1, dims=1)
            delay_queue[:, -1] = 0.0
            
            # 线性水库调蓄
            q_out_t = alpha * q_arrive_t + beta * Q_prev
            outputs.append(q_out_t)
            Q_prev = q_out_t
        
        Q_lateral = torch.cat(outputs, dim=1)
        
        state["delay_queue"] = delay_queue
        state["Q_prev"] = Q_prev
        
        return {
            "Q_lateral": Q_lateral,
            "Q_total": Q_lateral.sum(dim=0),
            "Q_in": Q_in,
        }, state
    
    def _define_parameters(self):
        return {
            "static": {
                "names": ["Tc", "K"],
                "bounds": [[0.5, 20.0], [0.1, 10.0]],
                "init": [self.Tc, self.K],
                # 无延时无调蓄（Tc→0: 瞬时到达，K=0: 无调蓄）
                "passthrough": [0.01, 0.0],
                "network_mapping": {
                    "Tc": "hillslope_routing",
                    "K": "hillslope_routing"
                }
            },
            "dynamic": {"names": [], "bounds": []},
        }


@register(HILLSLOPE_REGISTRY, "mask_gamma_uh")
class MaskGammaUnitHydrograph(BaseModel):
    """
    掩膜汇集 + Gamma分布单位线
    
    使用Gamma分布作为瞬时单位线(IUH)，这是Nash级联法的连续形式。
    
    Gamma分布单位线：
        u(t) = (1 / (k * Γ(α))) * (t/k)^(α-1) * exp(-t/k)
    
    其中：
        α (shape) : 形状参数，控制单位线的形状（相当于Nash级数n）
        k (scale) : 尺度参数，控制时间常数
        
    峰值时间 Tp = (α-1) * k （当α>1时）
    
    参数说明：
        alpha : float
            形状参数 [1.0-10.0]
        k : float
            尺度参数（天）[0.1-10.0]
        uh_length : int
            单位线长度
    """
    
    def __init__(self, alpha=3.0, k=1.0, uh_length=20, modules=None):
        super().__init__()
        self.alpha = alpha
        self.k = k
        self.uh_length = uh_length
    
    def forward(self, inputs, static_params, dynamic_params, state):
        runoff = inputs["runoff"]
        upstream_mask = inputs["upstream_mask"]
        cell_area = inputs.get("cell_area", None)
        
        device = runoff.device
        dtype = runoff.dtype
        
        # 获取参数
        alpha = float(static_params.get("alpha", self.alpha))
        k = float(static_params.get("k", self.k))
        uh_length = int(static_params.get("uh_length", self.uh_length))
        
        # 汇集产流
        Q_in = aggregate_by_mask(runoff, upstream_mask, cell_area)
        n_nodes, T = Q_in.shape
        
        # 生成Gamma分布单位线
        t = torch.arange(1, uh_length + 1, device=device, dtype=dtype)
        gamma_alpha = math.gamma(alpha)
        uh = (1.0 / (k * gamma_alpha)) * (t / k) ** (alpha - 1) * torch.exp(-t / k)
        uh = uh / uh.sum()  # 归一化
        
        # 初始化延迟队列
        delay_queue = state.get(
            "delay_queue",
            torch.zeros(n_nodes, uh_length, device=device, dtype=dtype)
        )
        
        outputs = []
        for t_idx in range(T):
            q_in_t = Q_in[:, t_idx:t_idx+1]
            
            delay_queue[:, :len(uh)] += q_in_t * uh.unsqueeze(0)
            q_out_t = delay_queue[:, 0:1]
            outputs.append(q_out_t)
            
            delay_queue = torch.roll(delay_queue, -1, dims=1)
            delay_queue[:, -1] = 0.0
        
        Q_lateral = torch.cat(outputs, dim=1)
        state["delay_queue"] = delay_queue
        
        return {
            "Q_lateral": Q_lateral,
            "Q_total": Q_lateral.sum(dim=0),
            "Q_in": Q_in,
        }, state
    
    def _define_parameters(self):
        return {
            "static": {
                "names": ["alpha", "k"],
                "bounds": [[1.0, 10.0], [0.1, 10.0]],
                "init": [self.alpha, self.k],
                # 无延时无调蓄（Gamma单位线近似delta函数）
                "passthrough": [1.0, 0.01],
                "network_mapping": {
                    "alpha": "hillslope_routing",
                    "k": "hillslope_routing"
                }
            },
            "dynamic": {"names": [], "bounds": []},
        }


@register(HILLSLOPE_REGISTRY, "mask_lag_route")
class MaskLagAndRoute(BaseModel):
    """
    掩膜汇集 + 滞后演算法（Lag and Route）
    
    两步汇流方法：
    1. 滞后（Lag）：将入流整体延迟一个滞时时间
    2. 演算（Route）：通过线性水库进行削峰调蓄
    
    适用于汇流过程主要由平移和调蓄两部分组成的情况。
    
    参数说明：
        lag : float
            滞时时间（天）[0.1-10.0]
        K : float
            线性水库时间常数（天）[0.1-10.0]
    """
    
    def __init__(self, lag=1.0, K=1.0, modules=None):
        super().__init__()
        self.lag = lag
        self.K = K
    
    def forward(self, inputs, static_params, dynamic_params, state):
        runoff = inputs["runoff"]
        upstream_mask = inputs["upstream_mask"]
        cell_area = inputs.get("cell_area", None)
        dt = inputs.get("dt", 1.0)
        
        device = runoff.device
        dtype = runoff.dtype
        
        # 获取参数
        lag = float(static_params.get("lag", self.lag))
        K = static_params.get("K", self.K)
        K_t = K if isinstance(K, torch.Tensor) else torch.tensor(K, device=device, dtype=dtype)
        
        # 汇集产流
        Q_in = aggregate_by_mask(runoff, upstream_mask, cell_area)
        n_nodes, T = Q_in.shape
        
        # 滞时处理（整数时间步延迟）
        lag_steps = max(1, int(lag / dt))
        
        # 初始化滞时队列
        lag_queue = state.get(
            "lag_queue",
            torch.zeros(n_nodes, lag_steps, device=device, dtype=dtype)
        )
        
        # 线性水库参数
        if K_t.ndim == 0:
            K_t = K_t.unsqueeze(0).expand(n_nodes).unsqueeze(1)
        elif K_t.ndim == 1:
            K_t = K_t.unsqueeze(1)
        
        alpha = dt / (K_t + dt)
        beta = K_t / (K_t + dt)
        
        Q_prev = state.get("Q_prev", torch.zeros(n_nodes, 1, device=device, dtype=dtype))
        
        outputs = []
        for t in range(T):
            q_in_t = Q_in[:, t:t+1]
            
            # 从滞时队列取出延迟后的入流
            q_lagged = lag_queue[:, 0:1]
            
            # 更新滞时队列
            lag_queue = torch.roll(lag_queue, -1, dims=1)
            lag_queue[:, -1:] = q_in_t
            
            # 线性水库演算
            q_out_t = alpha * q_lagged + beta * Q_prev
            outputs.append(q_out_t)
            Q_prev = q_out_t
        
        Q_lateral = torch.cat(outputs, dim=1)
        
        state["lag_queue"] = lag_queue
        state["Q_prev"] = Q_prev
        
        return {
            "Q_lateral": Q_lateral,
            "Q_total": Q_lateral.sum(dim=0),
            "Q_in": Q_in,
        }, state
    
    def _define_parameters(self):
        return {
            "static": {
                "names": ["lag", "K"],
                "bounds": [[0.1, 10.0], [0.1, 10.0]],
                "init": [self.lag, self.K],
                # 无延时无调蓄（lag→0: 最小滞后，K=0: 无调蓄）
                "passthrough": [0.001, 0.0],
                "network_mapping": {
                    "lag": "hillslope_routing",
                    "K": "hillslope_routing"
                }
            },
            "dynamic": {"names": [], "bounds": []},
        }


# ==============================================================================
# 基于D8流向的分布式汇流模型
# ==============================================================================

@register(HILLSLOPE_REGISTRY, "d8_travel_time")
class D8TravelTimeRouting(BaseModel):
    """
    D8流向汇流时间法
    
    利用D8流向追踪每个网格到其所属子流域下游边界的路径，结合网格流速计算汇流时间，
    将网格产流按到达时间比例分配到相邻两个时间步，最后通过线性水库削峰。
    
    本方法连接 grid_runoff（网格产流）和 river_routing（河网汇流），作为中间环节。
    
    输入：
        - runoff: (T, H, W) 网格产流序列
        - flow_dir: (H, W) D8流向编码矩阵
        - basin_mask: (H, W) 子流域编号掩膜（0, 1, 2, ...表示所属子流域）
        - velocity: (H, W) 各网格流速（米/天），静态参数
        - cellsize: 网格边长（米），默认1000.0
        - dt: 时间步长（天），默认1.0
    
    参数说明：
        velocity : tensor (H, W)
            各网格流速（米/天）[100.0-10000.0]，静态参数，仅初始化时计算一次
        K : float
            坡面汇流线性水库时间常数（天）[0.1-10.0]
    
    注意：
        流速为静态参数，汇流时间矩阵在首次调用时计算并缓存，后续调用直接复用。
    """
    
    def __init__(self, velocity=1000.0, K=1.0, modules=None):
        super().__init__()
        self.velocity = velocity  # 默认值，可被(H,W)张量覆盖
        self.K = K
        
        # 缓存追踪结果（静态参数变化时重新计算）
        self._cached_basin_id = None
        self._cached_travel_time = None
        self._cached_n_basins = None
        self._cached_velocity = None  # 缓存上次使用的velocity，用于检测参数变化
        self._cached_travel_dist = None  # 缓存距离矩阵（与velocity无关）
    
    def _velocity_changed(self, velocity, device, dtype):
        """检测velocity参数是否发生变化"""
        if self._cached_velocity is None:
            return True
        
        # 转换为tensor进行比较
        if isinstance(velocity, (int, float)):
            vel_new = torch.tensor(velocity, device=device, dtype=dtype)
        elif not isinstance(velocity, torch.Tensor):
            vel_new = torch.tensor(velocity, device=device, dtype=dtype)
        else:
            vel_new = velocity.to(device, dtype=dtype)
        
        cached_vel = self._cached_velocity
        if cached_vel.shape != vel_new.shape:
            return True
        
        return not torch.allclose(cached_vel, vel_new, rtol=1e-5, atol=1e-8)
    
    def _compute_travel_dist(self, flow_dir, basin_mask, cellsize, device, dtype):
        """
        计算距离矩阵（仅需计算一次，与velocity无关）
        """
        fd_np = flow_dir.cpu().numpy() if isinstance(flow_dir, torch.Tensor) else flow_dir
        bm_np = basin_mask.cpu().numpy() if isinstance(basin_mask, torch.Tensor) else basin_mask
        
        basin_id_np, travel_dist_np, n_basins = trace_d8_to_boundary(fd_np, bm_np, cellsize)
        
        basin_id = torch.from_numpy(basin_id_np).to(device)
        travel_dist = torch.from_numpy(travel_dist_np).to(device, dtype=dtype)
        
        return basin_id, travel_dist, n_basins
    
    def _compute_travel_time(self, travel_dist, velocity, dt, device, dtype):
        """
        根据距离和流速计算汇流时间（velocity变化时重新计算）
        """
        # 处理velocity
        if isinstance(velocity, (int, float)):
            vel = torch.full_like(travel_dist, velocity)
        elif not isinstance(velocity, torch.Tensor):
            vel = torch.tensor(velocity, device=device, dtype=dtype)
        else:
            vel = velocity.to(device, dtype=dtype)
        
        if vel.ndim == 0:
            vel = vel.expand_as(travel_dist)
        
        # 缓存当前velocity
        self._cached_velocity = vel.clone()
        
        # 计算汇流时间（连续时间步数）
        vel = torch.clamp(vel, min=1.0)  # 避免除零
        travel_time = travel_dist / vel / dt
        
        # 处理无穷值（无法到达边界的网格）
        travel_time = torch.where(torch.isinf(travel_time), torch.zeros_like(travel_time), travel_time)
        
        return travel_time
    
    def forward(self, inputs, static_params, dynamic_params, state):
        runoff = inputs["runoff"]  # (T, H, W)
        flow_dir = inputs["flow_dir"]  # (H, W)
        basin_mask = inputs["basin_mask"]  # (H, W) 子流域编号
        cellsize = inputs.get("cellsize", 1000.0)
        dt = inputs.get("dt", 1.0)
        
        device = runoff.device
        dtype = runoff.dtype
        T, H, W = runoff.shape
        
        # 获取参数
        velocity = static_params.get("velocity", self.velocity)
        K = static_params.get("K", self.K)
        
        # 1. 计算距离矩阵（仅首次计算，与velocity无关）
        if self._cached_travel_dist is None:
            basin_id, travel_dist, n_basins = self._compute_travel_dist(
                flow_dir, basin_mask, cellsize, device, dtype
            )
            self._cached_basin_id = basin_id
            self._cached_travel_dist = travel_dist
            self._cached_n_basins = n_basins
        
        basin_id = self._cached_basin_id
        travel_dist = self._cached_travel_dist
        n_basins = self._cached_n_basins
        
        # 2. 计算汇流时间（velocity变化时重新计算）
        if self._velocity_changed(velocity, device, dtype):
            travel_time = self._compute_travel_time(travel_dist, velocity, dt, device, dtype)
            self._cached_travel_time = travel_time
        
        travel_time = self._cached_travel_time
        
        # 分解汇流时间为整数部分和小数部分
        travel_floor = travel_time.floor().long()
        travel_frac = travel_time - travel_floor.float()
        
        max_delay = int(travel_floor.max().item()) + 2
        T_out = T + max_delay
        Q_agg = torch.zeros(n_basins, T_out, device=device, dtype=dtype)
        
        # 向量化计算：使用掩膜和旅行时间矩阵
        # 1. 提取有效网格信息
        basin_flat = basin_id.flatten()      # (H*W,)
        floor_flat = travel_floor.flatten()  # (H*W,)
        frac_flat = travel_frac.flatten()    # (H*W,)
        
        valid_mask = basin_flat >= 0
        valid_basin = basin_flat[valid_mask]   # (n_valid,)
        valid_floor = floor_flat[valid_mask]   # (n_valid,)
        valid_frac = frac_flat[valid_mask]     # (n_valid,)
        
        # 2. 提取有效网格的产流序列
        runoff_flat = runoff.view(T, -1)[:, valid_mask]  # (T, n_valid)
        
        # 3. 构建时间索引矩阵
        t_idx = torch.arange(T, device=device, dtype=torch.long).unsqueeze(1)  # (T, 1)
        arrival_t1 = t_idx + valid_floor.unsqueeze(0)  # (T, n_valid)
        arrival_t2 = arrival_t1 + 1
        
        # 4. 计算分配到两个时间步的贡献
        weight1 = (1.0 - valid_frac).unsqueeze(0)  # (1, n_valid)
        weight2 = valid_frac.unsqueeze(0)          # (1, n_valid)
        q_contrib1 = runoff_flat * weight1         # (T, n_valid)
        q_contrib2 = runoff_flat * weight2         # (T, n_valid)
        
        # 5. 构建线性索引并使用 scatter_add_ 高效聚合
        valid_basin_exp = valid_basin.unsqueeze(0).expand(T, -1)  # (T, n_valid)
        linear_idx1 = valid_basin_exp * T_out + arrival_t1        # (T, n_valid)
        linear_idx2 = valid_basin_exp * T_out + arrival_t2.clamp(max=T_out - 1)
        
        Q_agg_flat = Q_agg.view(-1)
        Q_agg_flat.scatter_add_(0, linear_idx1.flatten(), q_contrib1.flatten())
        Q_agg_flat.scatter_add_(0, linear_idx2.flatten(), q_contrib2.flatten())
        
        Q_in = Q_agg[:, :T]  # 截取原始时间长度
        
        # 坡面汇流：线性水库处理
        K_t = K if isinstance(K, torch.Tensor) else torch.tensor(K, device=device, dtype=dtype)
        
        alpha = dt / (K_t + dt)
        beta = K_t / (K_t + dt)
        
        Q_prev = state.get("Q_prev", torch.zeros(n_basins, 1, device=device, dtype=dtype))
        
        outputs = []
        for t in range(T):
            q_in_t = Q_in[:, t:t+1]
            q_out_t = alpha * q_in_t + beta * Q_prev
            outputs.append(q_out_t)
            Q_prev = q_out_t
        
        Q_lateral = torch.cat(outputs, dim=1)
        state["Q_prev"] = Q_prev
        
        return {
            "Q_lateral": Q_lateral,      # (n_basins, T) 各子流域边界出流
            "Q_total": Q_lateral.sum(dim=0),
            "Q_in": Q_in,
        }, state
    
    def _define_parameters(self):
        return {
            "static": {
                "names": ["velocity", "K"],
                "bounds": [[100.0, 10000.0], [0.1, 10.0]],
                "init": [self.velocity, self.K],
                # 无延时无调蓄（velocity→∞: 瞬时到达，K=0: 无调蓄）
                "passthrough": [1e6, 0.0],
                "network_mapping": {
                    "velocity": "grid_runoff",        # 网格流速，由grid_runoff网络学习
                    "K": "hillslope_routing"          # 坡面汇流线性水库常数
                }
            },
            "dynamic": {"names": [], "bounds": []},
        }
