"""
潜在蒸发计算模块（PET Calculation Module）

包含多种潜在蒸发计算方法：
- Hargreaves-Samani
- Priestley-Taylor
- FAO Penman-Monteith
- Thornthwaite
- 简化方法

供 grid_runoff 和 subbasin_runoff 模块在无蒸发输入时调用
"""

import torch
import math


# ==============================================================================
# 潜在蒸发计算方法
# ==============================================================================

def calculate_pet_hargreaves(T_mean, T_max, T_min, Ra=None, day_of_year=None, latitude=None):
    """
    Hargreaves-Samani 公式计算潜在蒸发（PET）
    
    简化公式，仅需温度数据，适用于缺少辐射数据的情况。
    
    参数:
        T_mean: 日平均温度 (°C)
        T_max: 日最高温度 (°C)
        T_min: 日最低温度 (°C)
        Ra: 天顶辐射 (MJ/m2/day)，如果为None则根据latitude和day_of_year估算
        day_of_year: 年内日序号 (1-365)，用于计算Ra
        latitude: 纬度 (度)，用于计算Ra
    
    返回:
        PET: 潜在蒸发量 (mm/day)
    
    参考:
        Hargreaves & Samani (1985)
        PET = 0.0023 * Ra * (T_mean + 17.8) * (T_max - T_min)^0.5
    """
    # 温差
    T_range = torch.clamp(T_max - T_min, min=0.0)
    
    # 如果没有提供Ra，则根据纬度和日期估算
    if Ra is None:
        if latitude is not None and day_of_year is not None:
            Ra = _calculate_extraterrestrial_radiation(latitude, day_of_year)
        else:
            # 使用全球平均值近似
            Ra = 15.0  # MJ/m2/day (典型值范围 10-40)
    
    # Hargreaves公式
    # 0.408 将 MJ/m2/day 转换为 mm/day
    PET = 0.0023 * 0.408 * Ra * (T_mean + 17.8) * torch.sqrt(T_range)
    
    return torch.clamp(PET, min=0.0)


def calculate_pet_priestley_taylor(Rn, T_mean, alpha=1.26):
    """
    Priestley-Taylor 公式计算潜在蒸发（PET）
    
    适用于湿润地区，假设空气动力学项可以忽略。
    
    参数:
        Rn: 净辐射 (MJ/m²/day)
        T_mean: 日平均温度 (°C)
        alpha: Priestley-Taylor系数，默认1.26（湿润条件）
               干旱条件下可取1.0-1.1
    
    返回:
        PET: 潜在蒸发量 (mm/day)
    
    参考:
        Priestley & Taylor (1972)
        PET = alpha * (delta / (delta + gamma)) * Rn / lambda
    """
    # 饱和水汽压-温度曲线斜率 delta (kPa/°C)
    delta = 4098 * (0.6108 * torch.exp(17.27 * T_mean / (T_mean + 237.3))) / \
            torch.pow(T_mean + 237.3, 2)
    
    # 干湿表常数 gamma (kPa/°C)，假设海平面标准大气压
    gamma = 0.0665  # kPa/°C
    
    # 汽化潜热 lambda (MJ/kg)
    lambda_v = 2.501 - 0.002361 * T_mean
    
    # Priestley-Taylor公式
    PET = alpha * (delta / (delta + gamma)) * Rn / lambda_v
    
    return torch.clamp(PET, min=0.0)


def calculate_pet_penman_monteith(Rn, T_mean, u2, ea, es, G=0.0):
    """
    FAO Penman-Monteith 公式计算参考蒸发蒸腾（ET0）
    
    这是FAO推荐的标准方法，适用于各种气候条件。
    
    参数:
        Rn: 净辐射 (MJ/m²/day)
        T_mean: 日平均温度 (°C)
        u2: 2米高度风速 (m/s)
        ea: 实际水汽压 (kPa)
        es: 饱和水汽压 (kPa)
        G: 土壤热通量 (MJ/m2/day)，日尺度可忽略，默认0
    
    返回:
        ET0: 参考蒸发蒸腾量 (mm/day)
    
    参考:
        Allen et al. (1998) - FAO Irrigation and Drainage Paper 56
    """
    # 饱和水汽压-温度曲线斜率 delta (kPa/°C)
    delta = 4098 * (0.6108 * torch.exp(17.27 * T_mean / (T_mean + 237.3))) / \
            torch.pow(T_mean + 237.3, 2)
    
    # 干湿表常数 gamma (kPa/°C)
    gamma = 0.0665
    
    # 水汽压差
    VPD = es - ea
    VPD = torch.clamp(VPD, min=0.0)
    
    # FAO Penman-Monteith公式
    # 分子第一项：辐射项
    numerator_rad = 0.408 * delta * (Rn - G)
    # 分子第二项：空气动力学项
    numerator_aero = gamma * (900 / (T_mean + 273)) * u2 * VPD
    # 分母
    denominator = delta + gamma * (1 + 0.34 * u2)
    
    ET0 = (numerator_rad + numerator_aero) / denominator
    
    return torch.clamp(ET0, min=0.0)


def calculate_pet_thornthwaite(T_mean, day_length, heat_index=None, a_coef=None):
    """
    Thornthwaite 公式计算潜在蒸发（PET）
    
    经典的温度-蒸发经验公式，仅需月平均温度。
    适用于月尺度计算，日尺度需要考虑日长调整。
    
    参数:
        T_mean: 平均温度 (°C)
        day_length: 日照时长 (小时) 或月内日数调整因子
        heat_index: 热量指数 I = sum((T_i/5)^1.514)，如果为None则需要提供年温度数据计算
        a_coef: 系数 a = 6.75e-7*I^3 - 7.71e-5*I^2 + 1.79e-2*I + 0.49，如果为None则根据heat_index计算
    
    返回:
        PET: 潜在蒸发量 (mm/day 或 mm/month，取决于输入)
    
    参考:
        Thornthwaite (1948)
    """
    # 如果没有提供热量指数，使用简化的经验值
    if heat_index is None:
        # 使用全球平均热量指数近似
        heat_index = 50.0  # 典型值范围 20-80
    
    if a_coef is None:
        # 计算系数a
        I = heat_index
        a_coef = 6.75e-7 * I**3 - 7.71e-5 * I**2 + 1.79e-2 * I + 0.49
    
    # 基础PET（温度大于0时）
    T_positive = torch.clamp(T_mean, min=0.0)
    PET_base = 16.0 * torch.pow(10 * T_positive / heat_index, a_coef)
    
    # 日长调整（假设day_length为小时，标准日长为12小时）
    PET = PET_base * day_length / 12.0
    
    # 温度低于0时，PET为0
    PET = torch.where(T_mean > 0, PET, torch.zeros_like(PET))
    
    return torch.clamp(PET, min=0.0)


def calculate_pet_simple(T_mean, method="hargreaves_simple"):
    """
    简化的潜在蒸发估算（仅需平均温度）
    
    当只有温度数据时的简便估算方法。
    
    参数:
        T_mean: 日平均温度 (°C)
        method: 计算方法
            - "hargreaves_simple": 简化Hargreaves (假设日较差为T/3)
            - "blaney_criddle": Blaney-Criddle简化公式
            - "linear": 线性温度-蒸发关系
    
    返回:
        PET: 潜在蒸发量估计 (mm/day)
    """
    if method == "hargreaves_simple":
        # 简化Hargreaves：假设温差约为均温的1/3，Ra取典型值
        T_range = torch.clamp(T_mean / 3.0, min=1.0)  # 避免零温差
        Ra = 15.0  # MJ/m2/day 典型值
        PET = 0.0023 * 0.408 * Ra * (T_mean + 17.8) * torch.sqrt(T_range)
        
    elif method == "blaney_criddle":
        # Blaney-Criddle简化公式
        # PET = p * (0.46 * T + 8.13)，p为日照百分比，取0.27（全球平均）
        p = 0.27
        PET = p * (0.46 * T_mean + 8.13)
        
    elif method == "linear":
        # 简单线性关系：PET ≈ 0.2 * T (适用于温带)
        # 系数0.2来自经验拟合，可根据区域调整
        PET = 0.2 * torch.clamp(T_mean, min=0.0)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return torch.clamp(PET, min=0.0)


# ==============================================================================
# 辅助函数
# ==============================================================================

def _calculate_extraterrestrial_radiation(latitude, day_of_year):
    """
    计算天顶辐射 Ra (MJ/m2/day)
    
    参数:
        latitude: 纬度 (度)，正为北纬，负为南纬
        day_of_year: 年内日序号 (1-365)
    
    返回:
        Ra: 天顶辐射 (MJ/m²/day)
    """
    # 转换为弧度
    if isinstance(latitude, torch.Tensor):
        phi = latitude * math.pi / 180
    else:
        phi = torch.tensor(latitude * math.pi / 180)
    
    if isinstance(day_of_year, torch.Tensor):
        J = day_of_year
    else:
        J = torch.tensor(float(day_of_year))
    
    # 太阳赤纬
    delta = 0.409 * torch.sin(2 * math.pi * J / 365 - 1.39)
    
    # 日落时角
    ws = torch.acos(torch.clamp(-torch.tan(phi) * torch.tan(delta), min=-1.0, max=1.0))
    
    # 日地相对距离
    dr = 1 + 0.033 * torch.cos(2 * math.pi * J / 365)
    
    # 太阳常数 Gsc = 0.0820 MJ/m²/min
    Gsc = 0.0820
    
    # 天顶辐射
    Ra = (24 * 60 / math.pi) * Gsc * dr * (
        ws * torch.sin(phi) * torch.sin(delta) + 
        torch.cos(phi) * torch.cos(delta) * torch.sin(ws)
    )
    
    return Ra


def calculate_saturated_vapor_pressure(T):
    """
    计算饱和水汽压 es (kPa)
    
    参数:
        T: 温度 (°C)
    
    返回:
        es: 饱和水汽压 (kPa)
    """
    return 0.6108 * torch.exp(17.27 * T / (T + 237.3))


def calculate_actual_vapor_pressure_from_rh(T, RH):
    """
    根据相对湿度计算实际水汽压
    
    参数:
        T: 温度 (°C)
        RH: 相对湿度 (0-100%)
    
    返回:
        ea: 实际水汽压 (kPa)
    """
    es = calculate_saturated_vapor_pressure(T)
    ea = es * RH / 100.0
    return ea


# ==============================================================================
# PET计算器包装类
# ==============================================================================

class PETCalculator:
    """
    潜在蒸发计算器
    
    提供统一的接口来计算PET，支持多种方法。
    当没有蒸发观测数据时使用。
    
    支持的方法:
        - hargreaves: Hargreaves-Samani (推荐，仅需温度)
        - priestley_taylor: Priestley-Taylor (需要辐射)
        - penman_monteith: FAO Penman-Monteith (需要完整气象数据)
        - thornthwaite: Thornthwaite (需要月均温)
        - simple: 简化方法 (仅需平均温度)
    
    使用示例:
        pet_calc = PETCalculator(method='hargreaves')
        E = pet_calc.calculate(T_mean, T_max=T_max, T_min=T_min, doy=doy, lat=lat)
    """
    
    def __init__(self, method: str = 'hargreaves', **kwargs):
        """
        初始化PET计算器
        
        Args:
            method: 计算方法名称
            **kwargs: 方法特定参数（如Priestley-Taylor的alpha）
        """
        self.method = method.lower()
        self.params = kwargs
        
        valid_methods = ['hargreaves', 'priestley_taylor', 'penman_monteith', 
                        'thornthwaite', 'simple']
        if self.method not in valid_methods:
            raise ValueError(f"Unknown PET method: {method}. Valid: {valid_methods}")
    
    def calculate(self, T_mean, T_max=None, T_min=None, doy=None, lat=None,
                  Rn=None, u2=None, ea=None, es=None, RH=None, day_length=None):
        """
        计算潜在蒸发
        
        Args:
            T_mean: 日平均温度 (°C), shape: (T, H, W) 或 (T, N)
            T_max: 日最高温度 (可选)
            T_min: 日最低温度 (可选)
            doy: 年内日序号 (1-365)
            lat: 纬度 (度)
            Rn: 净辐射 (MJ/m²/day)
            u2: 2m风速 (m/s)
            ea: 实际水汽压 (kPa)
            es: 饱和水汽压 (kPa)
            RH: 相对湿度 (%)
            day_length: 日照时长 (小时)
            
        Returns:
            PET: 潜在蒸发量 (mm/day), 与T_mean同shape
        """
        # 转换为tensor
        if not isinstance(T_mean, torch.Tensor):
            T_mean = torch.tensor(T_mean, dtype=torch.float32)
        
        if self.method == 'hargreaves':
            # Hargreaves方法
            if T_max is None or T_min is None:
                # 没有最高最低温度，使用简化方法估算
                T_range = torch.clamp(T_mean * 0.3, min=2.0)  # 温差约为均温的30%
                T_max = T_mean + T_range / 2
                T_min = T_mean - T_range / 2
            
            if not isinstance(T_max, torch.Tensor):
                T_max = torch.tensor(T_max, dtype=torch.float32)
            if not isinstance(T_min, torch.Tensor):
                T_min = torch.tensor(T_min, dtype=torch.float32)
            
            PET = calculate_pet_hargreaves(
                T_mean, T_max, T_min,
                day_of_year=doy,
                latitude=lat
            )
            
        elif self.method == 'priestley_taylor':
            if Rn is None:
                raise ValueError("Priestley-Taylor requires net radiation (Rn)")
            alpha = self.params.get('alpha', 1.26)
            PET = calculate_pet_priestley_taylor(Rn, T_mean, alpha=alpha)
            
        elif self.method == 'penman_monteith':
            if any(x is None for x in [Rn, u2]):
                raise ValueError("Penman-Monteith requires Rn and u2")
            
            # 如果没有提供ea/es，尝试从RH计算
            if es is None:
                es = calculate_saturated_vapor_pressure(T_mean)
            if ea is None:
                if RH is not None:
                    ea = calculate_actual_vapor_pressure_from_rh(T_mean, RH)
                else:
                    # 假设相对湿度60%
                    ea = es * 0.6
            
            PET = calculate_pet_penman_monteith(Rn, T_mean, u2, ea, es)
            
        elif self.method == 'thornthwaite':
            if day_length is None:
                day_length = 12.0  # 默认12小时
            PET = calculate_pet_thornthwaite(
                T_mean, day_length,
                heat_index=self.params.get('heat_index'),
                a_coef=self.params.get('a_coef')
            )
            
        elif self.method == 'simple':
            sub_method = self.params.get('sub_method', 'hargreaves_simple')
            PET = calculate_pet_simple(T_mean, method=sub_method)
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return PET


def compute_pet(T_df, method='hargreaves', **kwargs):
    """
    便捷函数：从温度DataFrame计算PET DataFrame
    
    Args:
        T_df: pandas DataFrame，日平均温度 (°C)，索引为日期，列为子流域/网格
        method: PET计算方法 ('hargreaves', 'simple', 'thornthwaite')
        **kwargs: 传递给底层计算函数的额外参数
    
    Returns:
        pandas DataFrame: PET (mm/day)，与输入格式相同
    """
    import pandas as pd
    import numpy as np
    
    # 获取日期信息用于Hargreaves
    dates = T_df.index
    
    # 计算每日PET
    pet_values = np.zeros_like(T_df.values)
    
    for i, date in enumerate(dates):
        T_mean = torch.tensor(T_df.iloc[i].values, dtype=torch.float32)
        
        if method == 'hargreaves':
            # Hargreaves简化版本：假设日温差为T_mean的20%（典型值）
            T_range = torch.abs(T_mean) * 0.2 + 5.0  # 至少5度温差
            T_max = T_mean + T_range / 2
            T_min = T_mean - T_range / 2
            
            # 估算天顶辐射
            day_of_year = date.dayofyear
            lat = kwargs.get('latitude', 35.0)  # 默认纬度35度
            Ra = _calculate_extraterrestrial_radiation(lat, day_of_year)
            
            pet = calculate_pet_hargreaves(T_mean, T_max, T_min, Ra=Ra)
            
        elif method == 'simple':
            sub_method = kwargs.get('sub_method', 'hargreaves_simple')
            pet = calculate_pet_simple(T_mean, method=sub_method)
            
        elif method == 'thornthwaite':
            # Thornthwaite需要月数据，这里简化处理
            day_length = kwargs.get('day_length', 12.0)
            pet = calculate_pet_thornthwaite(T_mean, day_length)
            
        else:
            # 默认使用简单方法
            pet = calculate_pet_simple(T_mean, method='hargreaves_simple')
        
        pet_values[i, :] = pet.numpy()
    
    # 创建输出DataFrame
    pet_df = pd.DataFrame(pet_values, index=T_df.index, columns=T_df.columns)
    
    return pet_df
