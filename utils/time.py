"""
模拟时间控制工具：解析配置中的 start/end/dt，并提供时间索引。
"""

from datetime import datetime, timedelta
from typing import Iterator, List, Optional
from contextlib import contextmanager
import time

__all__ = [
    "now_str",
    "Timer",
    "timer",
    "parse_dt",
    "date_range",
    "TimeConfig",
    "build_time_config",
]

DATE_FMT = "%Y-%m-%d"


# -----------------------------
# 实用函数
# -----------------------------
def now_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """返回当前时间的格式化字符串。"""
    return datetime.now().strftime(fmt)


def parse_dt(dt: str) -> timedelta:
    """
    将类似 "1d" / "6h" / "30m" 的步长转为 timedelta。
    只支持天(d)、小时(h)、分钟(m)。
    """
    unit = dt[-1].lower()
    value = int(dt[:-1])
    if unit == "d":
        return timedelta(days=value)
    if unit == "h":
        return timedelta(hours=value)
    if unit == "m":
        return timedelta(minutes=value)
    raise ValueError(f"不支持的时间步长: {dt}")


def date_range(start: str, end: str, dt: str, fmt: str = DATE_FMT) -> List[datetime]:
    """
    生成 [start, end] 闭区间的时间列表（含端点）。
    """
    start_dt = datetime.strptime(start, fmt)
    end_dt = datetime.strptime(end, fmt)
    step = parse_dt(dt)

    if end_dt < start_dt:
        raise ValueError("end 早于 start")

    dates: List[datetime] = []
    cur = start_dt
    while cur <= end_dt:
        dates.append(cur)
        cur += step
    return dates


# -----------------------------
# 计时工具
# -----------------------------
class Timer:
    """简单计时器，可用作上下文管理器。"""

    def __init__(self):
        self.start_ts: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_ts = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.elapsed = time.time() - self.start_ts


@contextmanager
def timer() -> Iterator[float]:
    """计时上下文，yield 一个可获取耗时的 lambda（秒）。"""
    start_ts = time.time()
    try:
        yield lambda: time.time() - start_ts
    finally:
        pass


# -----------------------------
# 模拟时间配置
# -----------------------------
class TimeConfig:
    """
    用于模拟的时间轴工具：
    - 保存 start/end/dt
    - 提供索引/反索引
    - 提供时间序列长度
    """

    def __init__(self, start: str, end: str, dt: str, fmt: str = DATE_FMT):
        self.start_str = start
        self.end_str = end
        self.dt_str = dt
        self.fmt = fmt

        self.start = datetime.strptime(start, fmt)
        self.end = datetime.strptime(end, fmt)
        self.step = parse_dt(dt)

        if self.end < self.start:
            raise ValueError("end 早于 start")

        self._dates = date_range(start, end, dt, fmt)
        self.length = len(self._dates)

    @property
    def dates(self) -> List[datetime]:
        return self._dates

    def idx2time(self, idx: int) -> datetime:
        return self._dates[idx]

    def time2idx(self, dt: datetime) -> int:
        return self._dates.index(dt)

    def to_str_list(self) -> List[str]:
        return [d.strftime(self.fmt) for d in self._dates]


def build_time_config(cfg: dict, fmt: str = DATE_FMT) -> TimeConfig:
    """
    由配置字典构建 TimeConfig，配置示例：
    time:
      start: 2000-01-01
      end: 2005-12-31
      dt: 1d
    """
    return TimeConfig(
        start=cfg["start"],
        end=cfg["end"],
        dt=cfg["dt"],
        fmt=fmt,
    )