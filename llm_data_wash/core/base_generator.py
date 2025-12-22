from abc import ABC, abstractmethod
from typing import List, Dict

class BaseGenerator(ABC):
    """数据生成器抽象基类"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = config["data"]["output_dir"]
        self._init_output_dir()
    
    def _init_output_dir(self):
        """初始化输出目录"""
        import os
        os.makedirs(self.output_dir, exist_ok=True)
    
    @abstractmethod
    def process_single(self, conversation: Dict, idx: int) -> Dict:
        """处理单条对话（抽象方法，子类实现）"""
        pass
    
    @abstractmethod
    def process_batch(self, dataset: List[Dict]) -> List[Dict]:
        """批量处理对话（抽象方法，子类实现）"""
        pass
    
    @abstractmethod
    def save_batch(self, batch_data: List[Dict], batch_idx: int):
        """保存批次数据（抽象方法，子类实现）"""
        pass