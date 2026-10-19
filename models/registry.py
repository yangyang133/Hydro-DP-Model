# models/registry.py

from typing import Dict, Type
from models.base import BaseModel


class ModelRegistry:
    """
    通用模型注册表，可实例化模型并查询参数信息。
    """

    def __init__(self):
        self._registry: Dict[str, Type[BaseModel]] = {}

    def register(self, name: str):
        """
        装饰器方式注册模型：@register(REGISTRY, "model_name")
        """

        def decorator(model_cls: Type[BaseModel]):
            if not issubclass(model_cls, BaseModel):
                raise TypeError(
                    f"Model {model_cls.__name__} must inherit from BaseModel"
                )
            if name in self._registry:
                raise KeyError(f"Model '{name}' already registered")
            self._registry[name] = model_cls
            return model_cls

        return decorator

    def create(self, name: str, **cfg) -> BaseModel:
        """实例化模型。"""
        if name not in self._registry:
            raise KeyError(
                f"Model '{name}' not found. Available: {list(self._registry.keys())}"
            )
        return self._registry[name](**cfg)

    def get_param_info(self, name: str):
        """获取模型参数信息（不需要提前实例化）。"""
        model = self.create(name)
        return model.get_param_info()

    def list_models(self):
        return list(self._registry.keys())


# --------------------------------------------------
# 全局注册器 & 便捷注册函数
# --------------------------------------------------
GRID_REGISTRY = ModelRegistry()
SUBBASIN_REGISTRY = ModelRegistry()
HILLSLOPE_REGISTRY = ModelRegistry()
RIVER_REGISTRY = ModelRegistry()


def register(registry: ModelRegistry, name: str):
    """
    便捷装饰器：@register(GRID_REGISTRY, "saturation_excess")
    """
    return registry.register(name)


def build_model(registry: ModelRegistry, name: str, **cfg) -> BaseModel:
    """根据名称从指定注册表中创建模型实例。"""
    return registry.create(name, **cfg)
