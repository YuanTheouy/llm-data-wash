import yaml
import json
import os
from llm_data_wash import VLLMGenerator
from llm_data_wash.utils.logger import get_logger

logger = get_logger(__name__, log_file="./regenerate.log")

def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_local_dataset(data_path: str) -> list:
    """加载本地数据集（JSON/JSONL）"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据集文件不存在：{data_path}")
    
    data = []
    if data_path.endswith(".jsonl"):
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
    elif data_path.endswith(".json"):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError(f"不支持的数据集格式：{data_path}")
    
    logger.info(f"加载数据集完成：共{len(data)}条对话")
    return data

def main():
    # 1. 加载配置
    config = load_config("./configs/regenerator_vllm.yaml")
    
    # 2. 加载数据集
    dataset = load_local_dataset(config["data"]["input_path"])
    
    # 3. 初始化vLLM生成器
    generator = VLLMGenerator(config)
    
    # 4. 批量处理
    generator.process_batch(dataset)
    
    logger.info("全部流程执行完成！")

if __name__ == "__main__":
    main()