import torch
import numpy as np
from models.base import BaseModel
from models.registry import RIVER_REGISTRY, register

@register(RIVER_REGISTRY, "muskingum")
class Muskingum(BaseModel):
    """
    Muskingum 河道汇流法
    
    基于蓄水方程的河道汇流方法：
    dS/dt = Q_in - Q_out
    S = K * [X * Q_in + (1 - X) * Q_out]
    
    其中：
    S: 河道蓄水量
    K: 汇流时间常数（天）
    X: 蓄水权重系数（0 <= X <= 0.5）
    Q_in: 入流量
    Q_out: 出流量
    """

    def __init__(self, K=1.0, X=0.2, modules=None):
        """
        初始化Muskingum方法
        
        Args:
            K: 汇流时间常数（天）
            X: 蓄水系数（0.0-0.5，典型值0.1-0.3）
            modules: 模块字典
        """
        super().__init__()
        self.K = K
        self.X = X

    def forward(self, inputs, static_params, dynamic_params, state):
        """
        执行Muskingum河道汇流计算
        
        支持输入形状：
        - Q_in: [T] 单个河段时间序列
        - Q_in: [n_rivers, T] 多个河段批量计算
        
        Args:
            inputs: dict 包含 "Q_in" 上游入流量序列
            static_params: 静态参数字典
            dynamic_params: 动态参数字典
            state: 状态字典
            
        Returns:
            dict: 包含出流量序列 "Q"，维度统一为 [n_rivers, T]
            state: 更新后的状态字典
        """
        Q_in = inputs["Q_in"]
        
        # 检测输入维度并统一为 [n_rivers, T]
        is_single = Q_in.ndim == 1
        if is_single:
            Q_in = Q_in.unsqueeze(0)  # [T] -> [1, T]
        
        n_rivers, T = Q_in.shape

        # 获取静态参数（如果 DL 网络提供则覆盖默认值）
        K_val = static_params.get("K", self.K)
        X_val = static_params.get("X", self.X)
        K_t = K_val if isinstance(K_val, torch.Tensor) else torch.as_tensor(float(K_val), device=Q_in.device)
        X_t = X_val if isinstance(X_val, torch.Tensor) else torch.as_tensor(float(X_val), device=Q_in.device)
        
        # 扩展参数维度以支持批量计算 [n_rivers, 1]
        if K_t.ndim == 0:
            K_t = K_t.unsqueeze(0).unsqueeze(1)
        elif K_t.ndim == 1:
            K_t = K_t.unsqueeze(1)
        if X_t.ndim == 0:
            X_t = X_t.unsqueeze(0).unsqueeze(1)
        elif X_t.ndim == 1:
            X_t = X_t.unsqueeze(1)

        # 初始状态
        Q_out_prev = state.get("Q_out_prev", Q_in[:, 0:1] if T > 0 else torch.zeros(n_rivers, 1, device=Q_in.device))
        Q_in_prev = state.get("Q_in_prev", Q_in[:, 0:1] if T > 0 else torch.zeros(n_rivers, 1, device=Q_in.device))
        
        # 确保状态维度为 [n_rivers, 1]
        if Q_out_prev.ndim == 1:
            Q_out_prev = Q_out_prev.unsqueeze(1)
        if Q_in_prev.ndim == 1:
            Q_in_prev = Q_in_prev.unsqueeze(1)

        dt = torch.as_tensor(1.0, device=Q_in.device)

        denominator = 2 * K_t * (1 - X_t) + dt
        C0 = (dt - 2 * K_t * X_t) / denominator
        C1 = (dt + 2 * K_t * X_t) / denominator
        C2 = (2 * K_t * (1 - X_t) - dt) / denominator

        outputs = []
        for t in range(T):
            q_in_t = Q_in[:, t:t+1]  # [n_rivers, 1]
            if t == 0:
                q_new = C0 * q_in_t + C1 * Q_in_prev + C2 * Q_out_prev
            else:
                q_in_prev_t = Q_in[:, t-1:t]  # [n_rivers, 1]
                q_new = C0 * q_in_t + C1 * q_in_prev_t + C2 * outputs[-1]
            outputs.append(q_new)

        Q_out = torch.cat(outputs, dim=1) if outputs else torch.zeros_like(Q_in)  # [n_rivers, T]

        # 更新状态
        state["Q_out_prev"] = Q_out[:, -1:] if T > 0 else Q_out_prev  # [n_rivers, 1]
        state["Q_in_prev"] = Q_in[:, -1:] if T > 0 else Q_in_prev  # [n_rivers, 1]

        # 如果原始输入是单河段[T]，则输出也应该是[T]
        if is_single:
            Q_out = Q_out.squeeze(0)  # [1, T] -> [T]
            
        return {"Q": Q_out}, state

    def _define_parameters(self):
        return {
            "static": {
                "names": ["K", "X"],
                "bounds": [[0.1, 10.0], [0.0, 0.5]],
                "init": [self.K, self.X],
                # 无延时无调蓄的参数值（K=0: Q_out=Q_in，完全直通）
                "passthrough": [0.0, 0.0],
                # 参数分类：Muskingum所有参数属于river_routing（河道汇流）
                "network_mapping": {
                    "K": "river_routing",  # 汇流时间常数
                    "X": "river_routing"   # 蓄水权重系数
                }
            },
            "dynamic": {"names": [], "bounds": []},
        }


@register(RIVER_REGISTRY, "linear_reservoir")
class LinearReservoir(BaseModel):
    """
    线性水库法河道汇流
    
    假设河道作为线性水库，水量变化与出流量成正比：
    dS/dt = Q_in - Q_out
    S = K * Q_out
    
    其中：
    S: 河道蓄水量
    K: 水库常数（天）
    Q_in: 入流量
    Q_out: 出流量
    """

    def __init__(self, K=1.0, modules=None):
        """
        初始化线性水库法
        
        Args:
            K: 水库常数/滞后时间（天）
            modules: 模块字典
        """
        super().__init__()
        self.K = K

    def forward(self, inputs, static_params, dynamic_params, state):
        """
        执行线性水库法河道汇流计算
        
        支持输入形状：
        - Q_in: [T] 单个河段时间序列
        - Q_in: [n_rivers, T] 多个河段批量计算
        
        Args:
            inputs: dict 包含 "Q_in" 上游入流量序列
            static_params: 静态参数字典
            dynamic_params: 动态参数字典
            state: 状态字典
            
        Returns:
            dict: 包含出流量序列 "Q"，维度统一为 [n_rivers, T]
            state: 更新后的状态字典
        """
        Q_in = inputs["Q_in"]
        
        # 检测输入维度并统一为 [n_rivers, T]
        is_single = Q_in.ndim == 1
        if is_single:
            Q_in = Q_in.unsqueeze(0)  # [T] -> [1, T]
        
        n_rivers, T = Q_in.shape
        
        K = static_params.get("K", self.K)
        K_t = K if isinstance(K, torch.Tensor) else torch.as_tensor(float(K), device=Q_in.device)
        
        # 扩展参数维度以支持批量计算 [n_rivers, 1]
        if K_t.ndim == 0:
            K_t = K_t.unsqueeze(0).unsqueeze(1)
        elif K_t.ndim == 1:
            K_t = K_t.unsqueeze(1)
        
        # 初始状态
        Q_prev = state.get("Q_prev", Q_in[:, 0:1] if T > 0 else torch.zeros(n_rivers, 1, device=Q_in.device))
        
        # 确保状态维度为 [n_rivers, 1]
        if Q_prev.ndim == 1:
            Q_prev = Q_prev.unsqueeze(1)
        
        dt = 1.0  # 时间步长（天）
        # 线性水库递推系数
        alpha = dt / (K_t + dt)
        beta = K_t / (K_t + dt)

        outputs = []
        for t in range(T):
            q_in_t = Q_in[:, t:t+1]  # [n_rivers, 1]
            # 线性水库递推公式：Q_out[t] = alpha * Q_in[t] + beta * Q_out[t-1]
            q_out_t = alpha * q_in_t + beta * Q_prev
            outputs.append(q_out_t)
            Q_prev = q_out_t

        Q_out = torch.cat(outputs, dim=1) if outputs else torch.zeros_like(Q_in)  # [n_rivers, T]
        
        # 更新状态
        state["Q_prev"] = Q_prev  # [n_rivers, 1]
        
        # 注意：统一保持 [n_rivers, T] 维度，即使单个河段也是 [1, T]
        return {"Q": Q_out}, state

    def _define_parameters(self):
        return {
            "static": {
                "names": ["K"],
                "bounds": [[0.1, 10.0]],
                "init": [self.K],
                # 无延时无调蓄的参数值（K=0: alpha=1, beta=0, Q_out=Q_in）
                "passthrough": [0.0],
                # 参数分类：LinearReservoir所有参数属于river_routing（河道汇流）
                "network_mapping": {
                    "K": "river_routing"  # 水库常数/滞后时间
                }
            },
            "dynamic": {"names": [], "bounds": []},
        }


@register(RIVER_REGISTRY, "unit_hydrograph")
class UnitHydrograph(BaseModel):
    """
    单位线法河道/坡面汇流模型

    支持单位线类型：
        - scs      : SCS 单位线（默认）
        - nash     : Nash 级联单位线
        - triangle : 三角形单位线（经验型）

    参数说明：
        n       : Nash 级联级数（仅 nash / triangle 使用）
        k       : 滞后时间参数（天）
                  - SCS : 流域滞后时间 Tl
                  - Nash: 单水库时间常数
        uh_type : 单位线类型
    """

    def __init__(self, n=3, k=1.0, uh_type="scs", modules=None):
        super().__init__()

        self.n = int(n)
        self.k = float(k)
        self.uh_type = uh_type

        # 单位线长度（天）
        self.uh_length = 15

    # ------------------------------------------------------------------
    # 单位线生成
    # ------------------------------------------------------------------
    def _generate_unit_hydrograph(self, n, k, uh_type):
        """
        生成单位线数组（离散日尺度）

        Returns
        -------
        uh : ndarray, shape (uh_length,)
        """
        t = np.arange(1, self.uh_length + 1, dtype=float)

        # =========================
        # 1. Nash 级联单位线
        # =========================
        if uh_type == "nash":
            # Gamma(n) = (n-1)!
            gamma_n = 1.0
            for i in range(1, n):
                gamma_n *= i

            uh = (t / k) ** (n - 1) * np.exp(-t / k) / (k * gamma_n)

        # =========================
        # 2. 三角形单位线
        # =========================
        elif uh_type == "triangle":
            Tp = max(1.0, n * k)  # 峰值时间

            uh = np.zeros_like(t)
            rising = t <= Tp
            falling = t > Tp

            uh[rising] = t[rising] / (Tp ** 2)
            uh[falling] = (2 * Tp - t[falling]) / (Tp ** 2)
            uh[uh < 0.0] = 0.0

        # =========================
        # 3. SCS 单位线（默认）
        # =========================
        else:
            # SCS 三角形单位线（离散日尺度）
            # k = 流域滞后时间 Tl
            # Tp = Tl + D/2,  D = 1 天
            Tp = max(1.0, k + 0.5)

            uh = np.zeros_like(t)
            rising = t <= Tp
            falling = t > Tp

            uh[rising] = t[rising] / (Tp ** 2)
            uh[falling] = (2 * Tp - t[falling]) / (Tp ** 2)
            uh[uh < 0.0] = 0.0

        # =========================
        # 归一化（单位线体积 = 1）
        # =========================
        uh_sum = np.sum(uh)
        if uh_sum > 0:
            uh /= uh_sum

        return uh

    # ------------------------------------------------------------------
    # 前向计算
    # ------------------------------------------------------------------
    def forward(self, inputs, static_params, dynamic_params, state):
        """
        单位线汇流计算（卷积形式）
        
        支持输入形状：
        - Q_in: [T] 单个河段时间序列
        - Q_in: [n_rivers, T] 多个河段批量计算

        Parameters
        ----------
        inputs : dict
            必须包含：
                - "Q_in" : tensor, 入流时间序列
        state : dict
            - delay_queue : 延迟队列

        Returns
        -------
        outputs : dict
            - "Q" : 出流时间序列，维度统一为 [n_rivers, T]
        state : dict
            更新后的状态
        """
        Q_in = inputs["Q_in"]
        
        # 检测输入维度并统一为 [n_rivers, T]
        is_single = Q_in.ndim == 1
        if is_single:
            Q_in = Q_in.unsqueeze(0)  # [T] -> [1, T]
        
        n_rivers, T = Q_in.shape

        # 读取参数
        n = int(static_params.get("n", self.n))
        k = float(static_params.get("k", self.k))
        uh_type = static_params.get("uh_type", self.uh_type)

        # 生成单位线（numpy）并转换为torch
        uh_np = self._generate_unit_hydrograph(n, k, uh_type)
        uh = torch.from_numpy(uh_np).to(dtype=Q_in.dtype, device=Q_in.device)
        uh = uh.unsqueeze(0)  # [uh_length] -> [1, uh_length] for broadcasting

        # 初始化或读取延迟队列 [n_rivers, uh_length]
        delay_queue = state.get(
            "delay_queue",
            torch.zeros(n_rivers, self.uh_length, dtype=Q_in.dtype, device=Q_in.device)
        )
        
        # 确保delay_queue维度正确
        if delay_queue.ndim == 1:
            delay_queue = delay_queue.unsqueeze(0)

        Q_out_list = []

        # =========================
        # 卷积计算
        # =========================
        for t in range(T):
            # 入流对未来时刻的贡献 [n_rivers, uh_length]
            q_in_t = Q_in[:, t:t+1]  # [n_rivers, 1]
            delay_queue[:, :len(uh_np)] += q_in_t * uh  # broadcast: [n_rivers, 1] * [1, uh_length]

            # 当前时刻出流
            q_out_t = delay_queue[:, 0:1]  # [n_rivers, 1]
            Q_out_list.append(q_out_t)

            # 延迟队列前移
            delay_queue[:, :-1] = delay_queue[:, 1:]
            delay_queue[:, -1] = 0.0

        Q_out = torch.cat(Q_out_list, dim=1) if Q_out_list else torch.zeros_like(Q_in)  # [n_rivers, T]

        # 更新状态
        state["delay_queue"] = delay_queue  # [n_rivers, uh_length]

        # 注意：统一保持 [n_rivers, T] 维度，即使单个河段也是 [1, T]
        return {"Q": Q_out}, state

    # ------------------------------------------------------------------
    # 参数定义（用于生成）
    # ------------------------------------------------------------------
    def _define_parameters(self):
        return {
            "static": {
                "names": ["n", "k"],
                "bounds": [
                    [1, 10],     # n
                    [0.1, 10.0], # k (天)
                ],
                "init": [self.n, self.k],
                # 无延时无调蓄的参数值（n=1, k→0: 单位线近似delta函数，瞬时通过）
                "passthrough": [1, 0.01],
                # 参数分类：UnitHydrograph所有参数属于river_routing（河道汇流）
                "network_mapping": {
                    "n": "river_routing",  # Nash级联级数或单位线形状参数
                    "k": "river_routing"   # 滞后时间参数
                }
            },
            "dynamic": {
                "names": [],
                "bounds": [],
            },
        }

    # ------------------------------------------------------------------
    # 状态初始化
    # ------------------------------------------------------------------
    def init_state(self):
        return {
            "delay_queue": np.zeros(self.uh_length, dtype=float)
        }
