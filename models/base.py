# models/base.py

from typing import Dict, Any


class BaseModel:
    """
    所有水文模型的基类（最终父类）

    职责：
    - 定义参数信息（静态 / 动态）
    - 提供统一 forward / state 接口
    """

    def __init__(self, **cfg):
        """
        cfg:
            模型自身配置（非参数本身）
            如：模块开关、结构选项等
        """
        self.cfg = cfg

        # 使用延迟初始化，避免子类属性未初始化问题
        self._param_info = None

    @property
    def param_info(self):
        """延迟加载参数信息"""
        if self._param_info is None:
            self._param_info = self._define_parameters()
            self._check_param_info()
        return self._param_info

    # ======================
    # 参数定义（必须实现）
    # ======================
    def _define_parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        子类可覆盖。默认尝试兼容旧的 `_define_params`，否则返回空。

        统一格式：
        {
            "static": {"names": [...], "bounds": [...], "init": [...]},
            "dynamic": {"names": [...], "bounds": [...]}
        }
        """
        # 兼容旧实现
        if hasattr(self, "_define_params"):
            info = self._define_params()
            # 若旧格式直接返回 {"names","types","bounds","initial"}
            if "static" not in info and "dynamic" not in info:
                static_names = [
                    n for n, t in zip(info.get("names", []), info.get("types", []))
                    if t == "static"
                ]
                dynamic_names = [
                    n for n, t in zip(info.get("names", []), info.get("types", []))
                    if t == "dynamic"
                ]
                return {
                    "static": {
                        "names": static_names,
                        "bounds": info.get("bounds", [])[: len(static_names)],
                        "init": info.get("initial", [])[: len(static_names)],
                    },
                    "dynamic": {
                        "names": dynamic_names,
                        "bounds": info.get("bounds", [])[len(static_names):],
                    },
                }
            return info

        return {
            "static": {"names": [], "bounds": [], "init": []},
            "dynamic": {"names": [], "bounds": []},
        }

    def get_param_info(self) -> Dict[str, Dict[str, Any]]:
        return self.param_info

    # ======================
    # 状态管理
    # ======================
    def init_state(self) -> Dict[str, Any]:
        """
        初始化模型状态
        """
        return {}

    # ======================
    # 前向计算（必须实现）
    # ======================
    def forward(
        self,
        inputs: Dict[str, Any],
        static_params: Dict[str, Any],
        dynamic_params: Dict[str, Any],
        state: Dict[str, Any]
    ):
        """
        单时间步前向计算

        返回：
            output: dict
            new_state: dict
        """
        raise NotImplementedError

    # ======================
    # 内部校验
    # ======================
    def _check_param_info(self):
        for key in ["static", "dynamic"]:
            if key not in self.param_info:
                raise ValueError(f"param_info must contain '{key}'")

        s = self.param_info["static"]
        d = self.param_info["dynamic"]

        assert len(s["names"]) == len(s["bounds"]) == len(s["init"])
        assert len(d["names"]) == len(d["bounds"])
