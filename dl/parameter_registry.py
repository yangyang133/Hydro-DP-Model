#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
参数分类注册与汇总工具

提供四网络参数分类的全局注册和查询功能。
四个网络：
  - grid_runoff: 网格产流网络（未使用）
  - subbasin_runoff: 子流域产流网络（主要参数）
  - hillslope_routing: 坡面汇流网络（路由参数）
  - river_routing: 河道汇流网络（Muskingum等）
"""

from typing import Dict, List, Tuple

# 参数分类的四个网络类型
PARAM_NETWORKS = [
    "grid_runoff",
    "subbasin_runoff",
    "hillslope_routing",
    "river_routing"
]

PARAM_TYPES = ["static", "dynamic"]


class ParameterClassifier:
    """参数分类管理器"""
    
    def __init__(self):
        self.params_by_network: Dict[str, Dict[str, Tuple[str, List[str]]]] = {
            network: {} for network in PARAM_NETWORKS
        }
        self.model_registry: Dict[str, Dict] = {}
    
    def register_model(self, model_name: str, category: str, parameters: List[Dict]):
        """注册模型的参数分类"""
        self.model_registry[model_name] = {
            'category': category,
            'parameters': parameters
        }
        
        for param in parameters:
            network = param['network']
            param_type = param['type']
            param_name = param['name']
            
            if model_name not in self.params_by_network[network]:
                self.params_by_network[network][model_name] = {}
            
            if param_type not in self.params_by_network[network][model_name]:
                self.params_by_network[network][model_name][param_type] = []
            
            self.params_by_network[network][model_name][param_type].append(param_name)
    
    def get_network_params(self, network: str, model_name: str = None, param_type: str = None):
        """获取指定网络的参数"""
        if network not in self.params_by_network:
            return {}
        
        network_params = self.params_by_network[network]
        
        if model_name is None:
            return network_params
        
        if model_name not in network_params:
            return {}
        
        model_params = network_params[model_name]
        
        if param_type is None:
            return model_params
        
        return model_params.get(param_type, [])
    
    def get_summary(self):
        """获取参数分类总结"""
        summary = {}
        for network in PARAM_NETWORKS:
            summary[network] = {}
            network_params = self.params_by_network[network]
            for model_name, param_dict in network_params.items():
                summary[network][model_name] = {
                    'static': param_dict.get('static', []),
                    'dynamic': param_dict.get('dynamic', []),
                    'total': len(param_dict.get('static', [])) + len(param_dict.get('dynamic', []))
                }
        return summary
    
    def print_summary(self):
        """打印参数分类总结"""
        print("\n" + "=" * 80)
        print("参数分类总结 - 按四个网络类型组织")
        print("=" * 80)
        
        for network in PARAM_NETWORKS:
            network_params = self.params_by_network[network]
            if not network_params:
                continue
            
            print(f"\n【{network.upper()}】")
            print("-" * 80)
            
            for model_name, param_dict in sorted(network_params.items()):
                static_params = param_dict.get('static', [])
                dynamic_params = param_dict.get('dynamic', [])
                
                print(f"  {model_name}:")
                if static_params:
                    print(f"    静态参数 ({len(static_params)}): {', '.join(static_params)}")
                if dynamic_params:
                    print(f"    动态参数 ({len(dynamic_params)}): {', '.join(dynamic_params)}")


def register_all_subbasin_models(enable_cold_region: bool = False):
    """
    注册所有已实现的子流域径流模型的参数分类信息
    
    Args:
        enable_cold_region: 是否启用冰冻圈模式（将添加冰冻圈参数）
    """
    from models.components.subbasin_runoff import (
        XAJModel, TankModel,
        GR4JModel, SACSMAModel, HBVModel
    )
    
    classifier = ParameterClassifier()  # 修正：直接实例化
    
    # 所有五个子流域模型实例（根据 enable_cold_region 参数初始化）
    models = [
        ("xaj", XAJModel(enable_cold_region=enable_cold_region)),
        ("tank", TankModel(enable_cold_region=enable_cold_region)),
        ("gr4j", GR4JModel(enable_cold_region=enable_cold_region)),
        ("sacsma", SACSMAModel(enable_cold_region=enable_cold_region)),
        ("hbv", HBVModel(enable_cold_region=enable_cold_region)),
    ]
    
    # 逐个注册
    for model_name, model_instance in models:
        param_info = model_instance.param_info
        static_params = param_info.get("static", {})
        
        param_names = static_params.get("names", [])
        network_mapping = static_params.get("network_mapping", {})
        bounds_list = static_params.get("bounds", [])
        init_list = static_params.get("init", [])
        passthrough_list = static_params.get("passthrough", init_list)
        
        # 从param_info提取每个参数的详细信息
        parameters = []
        for i, param_name in enumerate(param_names):
            network = network_mapping.get(param_name, "subbasin_runoff")  # 默认子流域产流
            param_type = "static"  # 目前所有都是static
            
            bounds = bounds_list[i] if i < len(bounds_list) else [0, 1]
            init_val = init_list[i] if i < len(init_list) else 0.5
            passthrough_val = passthrough_list[i] if i < len(passthrough_list) else init_val
            
            parameters.append({
                "name": param_name,
                "network": network,
                "type": param_type,
                "bounds": bounds,
                "init": init_val,
                "passthrough": passthrough_val
            })
        
        # 注册整个模型的参数
        classifier.register_model(
            model_name=model_name,
            category="subbasin_runoff",  # 所有这些模型都属于子流域产流类别
            parameters=parameters
        )
    
    return classifier


def register_all_river_routing_models():
    """
    注册所有已实现的河道汇流模型的参数分类信息
    """
    from models.components.river_routing import (
        Muskingum, LinearReservoir, UnitHydrograph
    )
    
    classifier = ParameterClassifier()
    
    # 所有河道汇流模型实例
    models = [
        ("muskingum", Muskingum()),
        ("linear_reservoir", LinearReservoir()),
        ("unit_hydrograph", UnitHydrograph()),
    ]
    
    # 逐个注册
    for model_name, model_instance in models:
        param_info = model_instance.param_info
        static_params = param_info.get("static", {})
        
        param_names = static_params.get("names", [])
        network_mapping = static_params.get("network_mapping", {})
        bounds_list = static_params.get("bounds", [])
        init_list = static_params.get("init", [])
        passthrough_list = static_params.get("passthrough", init_list)
        
        # 从param_info提取每个参数的详细信息
        parameters = []
        for i, param_name in enumerate(param_names):
            network = network_mapping.get(param_name, "river_routing")  # 默认河道汇流
            param_type = "static"  # 目前所有都是static
            
            bounds = bounds_list[i] if i < len(bounds_list) else [0, 1]
            init_val = init_list[i] if i < len(init_list) else 0.5
            passthrough_val = passthrough_list[i] if i < len(passthrough_list) else init_val
            
            parameters.append({
                "name": param_name,
                "network": network,
                "type": param_type,
                "bounds": bounds,
                "init": init_val,
                "passthrough": passthrough_val
            })
        
        # 注册整个模型的参数
        classifier.register_model(
            model_name=model_name,
            category="river_routing",  # 所有这些模型都属于河道汇流类别
            parameters=parameters
        )
    
    return classifier


def register_all_models(enable_cold_region: bool = False, enable_precip_correct: bool = False):
    """
    注册所有模型（子流域产流 + 河道汇流）
    
    Args:
        enable_cold_region: 是否启用冰冻圈模式（将添加冰冻圈参数）
        enable_precip_correct: 是否启用降水校正系数
    """
    classifier = ParameterClassifier()
    
    # 注册子流域产流模型
    from models.components.subbasin_runoff import (
        XAJModel, TankModel,
        GR4JModel, SACSMAModel, HBVModel
    )
    
    # 所有五个子流域模型实例
    subbasin_models = [
        ("xaj", XAJModel(enable_cold_region=enable_cold_region, enable_precip_correct=enable_precip_correct)),
        ("tank", TankModel(enable_cold_region=enable_cold_region, enable_precip_correct=enable_precip_correct)),
        ("gr4j", GR4JModel(enable_cold_region=enable_cold_region, enable_precip_correct=enable_precip_correct)),
        ("sacsma", SACSMAModel(enable_cold_region=enable_cold_region, enable_precip_correct=enable_precip_correct)),
        ("hbv", HBVModel(enable_cold_region=enable_cold_region, enable_precip_correct=enable_precip_correct)),
    ]
    
    for model_name, model_instance in subbasin_models:
        param_info = model_instance.param_info
        static_params = param_info.get("static", {})
        
        param_names = static_params.get("names", [])
        network_mapping = static_params.get("network_mapping", {})
        bounds_list = static_params.get("bounds", [])
        init_list = static_params.get("init", [])
        passthrough_list = static_params.get("passthrough", init_list)
        
        parameters = []
        for i, param_name in enumerate(param_names):
            network = network_mapping.get(param_name, "subbasin_runoff")
            param_type = "static"
            
            bounds = bounds_list[i] if i < len(bounds_list) else [0, 1]
            init_val = init_list[i] if i < len(init_list) else 0.5
            passthrough_val = passthrough_list[i] if i < len(passthrough_list) else init_val
            
            parameters.append({
                "name": param_name,
                "network": network,
                "type": param_type,
                "bounds": bounds,
                "init": init_val,
                "passthrough": passthrough_val
            })
        
        classifier.register_model(
            model_name=model_name,
            category="subbasin_runoff",
            parameters=parameters
        )
    
    # 注册河道汇流模型
    from models.components.river_routing import (
        Muskingum, LinearReservoir, UnitHydrograph
    )
    
    river_models = [
        ("muskingum", Muskingum()),
        ("linear_reservoir", LinearReservoir()),
        ("unit_hydrograph", UnitHydrograph()),
    ]
    
    for model_name, model_instance in river_models:
        param_info = model_instance.param_info
        static_params = param_info.get("static", {})
        
        param_names = static_params.get("names", [])
        network_mapping = static_params.get("network_mapping", {})
        bounds_list = static_params.get("bounds", [])
        init_list = static_params.get("init", [])
        passthrough_list = static_params.get("passthrough", init_list)
        
        parameters = []
        for i, param_name in enumerate(param_names):
            network = network_mapping.get(param_name, "river_routing")
            param_type = "static"
            
            bounds = bounds_list[i] if i < len(bounds_list) else [0, 1]
            init_val = init_list[i] if i < len(init_list) else 0.5
            passthrough_val = passthrough_list[i] if i < len(passthrough_list) else init_val
            
            parameters.append({
                "name": param_name,
                "network": network,
                "type": param_type,
                "bounds": bounds,
                "init": init_val,
                "passthrough": passthrough_val
            })
        
        classifier.register_model(
            model_name=model_name,
            category="river_routing",
            parameters=parameters
        )
    
    # 注册坡面汇流模型（如d8_travel_time）
    from models.components.hillslope_routing import D8TravelTimeRouting
    
    hillslope_models = [
        ("d8_travel_time", D8TravelTimeRouting()),
    ]
    
    for model_name, model_instance in hillslope_models:
        param_info = model_instance.param_info
        static_params = param_info.get("static", {})
        
        param_names = static_params.get("names", [])
        network_mapping = static_params.get("network_mapping", {})
        bounds_list = static_params.get("bounds", [])
        init_list = static_params.get("init", [])
        passthrough_list = static_params.get("passthrough", init_list)
        
        parameters = []
        for i, param_name in enumerate(param_names):
            # 使用模型定义的 network_mapping
            network = network_mapping.get(param_name, "hillslope_routing")
            param_type = "static"
            
            bounds = bounds_list[i] if i < len(bounds_list) else [0, 1]
            init_val = init_list[i] if i < len(init_list) else 0.5
            passthrough_val = passthrough_list[i] if i < len(passthrough_list) else init_val
            
            parameters.append({
                "name": param_name,
                "network": network,
                "type": param_type,
                "bounds": bounds,
                "init": init_val,
                "passthrough": passthrough_val
            })
        
        classifier.register_model(
            model_name=model_name,
            category="hillslope_routing",
            parameters=parameters
        )
    
    return classifier


def get_all_parameters(enable_cold_region: bool = False, enable_precip_correct: bool = False):
    """
    获取所有已注册的参数信息
    
    Args:
        enable_cold_region: 是否启用冰冻圈模式
        enable_precip_correct: 是否启用降水校正系数
    
    Returns:
        list: 参数信息列表，每个元素包含:
              - name: 参数名
              - model: 所属模型
              - network_mapping: 网络分类
              - bounds: 参数范围
              - init: 初始值
    """
    classifier = register_all_models(enable_cold_region=enable_cold_region, enable_precip_correct=enable_precip_correct)
    
    all_params = []
    for model_name, model_info in classifier.model_registry.items():
        for param in model_info['parameters']:
            param_dict = {
                'name': param['name'],
                'model': model_name,
                'network_mapping': param['network'],
                'bounds': param['bounds'],
                'init': param['init'],
                'passthrough': param.get('passthrough', param['init']),
                'type': param['type']
            }
            all_params.append(param_dict)
    
    return all_params


def print_parameter_summary():
    """打印参数分类汇总报告"""
    classifier = register_all_models()  # 修改：注册所有模型
    
    print("=" * 90)
    print("四网络参数分类汇总报告（包含子流域产流 + 河道汇流）")
    print("=" * 90)
    print()
    
    # 1. 使用内置的打印方法
    classifier.print_summary()
    
    # 2. 更详细的统计
    print("\n" + "=" * 90)
    print("详细参数分布表")
    print("=" * 90)
    
    # 从 model_registry 提取详细信息
    for network in ["grid_runoff", "subbasin_runoff", "hillslope_routing", "river_routing"]:
        # 收集属于该网络的所有参数
        params_in_network = []
        for model_name, model_info in classifier.model_registry.items():
            for param in model_info['parameters']:
                if param['network'] == network:
                    params_in_network.append({
                        'model': model_name,
                        'param_name': param['name'],
                        'type': param['type'],
                        'bounds': param['bounds'],
                        'init': param['init']
                    })
        
        if not params_in_network:
            continue
        
        print(f"\n【{network}】网络 - 共 {len(params_in_network)} 个参数")
        print("-" * 90)
        
        # 按模型分组
        from collections import defaultdict
        params_by_model = defaultdict(list)
        for p in params_in_network:
            params_by_model[p['model']].append(p)
        
        for model_name in sorted(params_by_model.keys()):
            model_params = params_by_model[model_name]
            print(f"\n  模型: {model_name.upper()} ({len(model_params)} 个参数)")
            print(f"  {'-' * 86}")
            for param in model_params:
                bounds_str = f"[{param['bounds'][0]}, {param['bounds'][1]}]"
                init_str = f"{param['init']:.3f}" if isinstance(param['init'], float) else str(param['init'])
                print(f"    {param['param_name']:12s}  {bounds_str:25s}  初始: {init_str:10s}  类型: {param['type']}")
    
    print("\n" + "=" * 90)
    print("报告结束")
    print("=" * 90)


if __name__ == "__main__":
    print_parameter_summary()
