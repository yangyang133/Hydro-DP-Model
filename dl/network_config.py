"""
参数网络配置模块
集中管理参数网络的配置，消除硬编码
"""
from typing import Dict, List, Tuple
from dl.parameter_registry import get_all_parameters


class NetworkConfig:
    """网络配置管理器"""
    
    # 默认网络结构
    DEFAULT_HIDDEN_DIMS = [64, 32]
    DEFAULT_DROPOUT = 0.1
    DEFAULT_ACTIVATION = 'relu'
    
    # 特征维度配置（从数据文件自动推断，或手动指定）
    FEATURE_DIMS = {
        'basin': 10,      # data/dl_input/basin.csv 的特征数
        'hillslope': 5,   # data/dl_input/hillslope.csv 的特征数
        'river': 5        # data/dl_input/river.csv 的特征数
    }
    
    @staticmethod
    def get_network_config_for_model(model_name: str, river_model_name: str = None,
                                     enable_cold_region: bool = False) -> Dict[str, dict]:
        """
        获取指定模型的网络配置
        
        Args:
            model_name: 产流模型名称（如'xaj', 'gr4j'），None表示不创建产流网络
            river_model_name: 河道汇流模型名称（如'muskingum'），None表示不创建河道网络
            enable_cold_region: 是否启用冰冻圈模式
        
        Returns:
            dict: {network_type: {'input_dim': int, 'output_dim': int, 'params': list}}
        """
        # 如果两者都为None，返回空配置
        if model_name is None and river_model_name is None:
            return {}
        
        # 从参数注册表获取参数信息（传入enable_cold_region）
        params_registry = get_all_parameters(enable_cold_region=enable_cold_region)
        
        # 按网络类型组织参数
        network_configs = {}
        
        for param_info in params_registry:
            # 检查是否属于指定的模型（只有非None的模型才参与匹配）
            model_match = (model_name is not None and param_info['model'] == model_name)
            river_match = (river_model_name is not None and param_info['model'] == river_model_name)
            
            if model_match or river_match:
                network_type = param_info['network_mapping']
                
                if network_type == 'none':
                    continue
                
                if network_type not in network_configs:
                    network_configs[network_type] = {
                        'params': [],
                        'output_dim': 0
                    }
                
                network_configs[network_type]['params'].append(param_info)
                network_configs[network_type]['output_dim'] += 1
        
        # 添加输入维度
        input_dim_mapping = {
            'subbasin_runoff': 'basin',
            'hillslope_routing': 'hillslope',
            'river_routing': 'river',
            'grid_runoff': 'basin'  # 假设grid用basin特征
        }
        
        for network_type, config in network_configs.items():
            feature_type = input_dim_mapping.get(network_type, 'basin')
            config['input_dim'] = NetworkConfig.FEATURE_DIMS[feature_type]
        
        return network_configs
    
    @staticmethod
    def get_feature_files() -> Dict[str, str]:
        """获取特征文件路径"""
        return {
            'basin': 'data/dl_input/basin.csv',
            'hillslope': 'data/dl_input/hillslope.csv',
            'river': 'data/dl_input/river.csv'
        }
    
    @staticmethod
    def get_network_structure(network_type: str = None) -> Dict[str, any]:
        """获取网络结构配置"""
        return {
            'hidden_dims': NetworkConfig.DEFAULT_HIDDEN_DIMS,
            'dropout': NetworkConfig.DEFAULT_DROPOUT,
            'activation': NetworkConfig.DEFAULT_ACTIVATION
        }
    
    @staticmethod
    def update_feature_dim(feature_type: str, dim: int):
        """更新特征维度（用于自动推断）"""
        if feature_type in NetworkConfig.FEATURE_DIMS:
            NetworkConfig.FEATURE_DIMS[feature_type] = dim
    
    @staticmethod
    def infer_feature_dims_from_data(data_dir: str = 'data/dl_input') -> Dict[str, int]:
        """
        从数据文件自动推断特征维度
        
        Args:
            data_dir: 数据目录
        
        Returns:
            dict: {feature_type: dimension}
        """
        import pandas as pd
        from pathlib import Path
        
        dims = {}
        for feature_type, filename in NetworkConfig.get_feature_files().items():
            file_path = Path(data_dir) / Path(filename).name
            if file_path.exists():
                df = pd.read_csv(file_path)
                # 假设第一列是ID，其余是特征
                dims[feature_type] = df.shape[1] - 1
                NetworkConfig.update_feature_dim(feature_type, dims[feature_type])
        
        return dims


def get_network_save_path(output_dir: str, network_type: str) -> str:
    """获取网络保存路径"""
    from pathlib import Path
    return str(Path(output_dir) / f"{network_type}_net.pt")


def get_param_save_path(output_dir: str, network_type: str) -> str:
    """获取参数保存路径（CSV）"""
    from pathlib import Path
    return str(Path(output_dir) / f"{network_type}_params.csv")


if __name__ == "__main__":
    # 测试配置
    print("=== 网络配置测试 ===\n")
    
    print("1. 推断特征维度:")
    dims = NetworkConfig.infer_feature_dims_from_data()
    for ft, dim in dims.items():
        print(f"  {ft}: {dim}")
    
    print("\n2. XAJ + Muskingum 网络配置:")
    configs = NetworkConfig.get_network_config_for_model('xaj', 'muskingum')
    for net_type, config in configs.items():
        print(f"\n  {net_type}:")
        print(f"    输入维度: {config['input_dim']}")
        print(f"    输出维度: {config['output_dim']}")
        print(f"    参数列表: {[p['name'] for p in config['params']]}")
    
    print("\n3. 网络结构:")
    structure = NetworkConfig.get_network_structure()
    print(f"  隐藏层: {structure['hidden_dims']}")
    print(f"  Dropout: {structure['dropout']}")
