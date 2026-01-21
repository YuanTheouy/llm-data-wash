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
    
    # 数据标准化：检查是否为VQA格式并转换
    normalized_data = []
    for item in data:
        # 检查是否为VQA格式 (包含 image_path 和 question)
        if "image_path" in item and "question" in item:
             normalized_data.append({
                 "id": str(item.get("question_id", "")),
                 "conversations": [
                     {
                         "from": "human",
                         "value": item["question"],
                         "image_path": item["image_path"]
                     },
                     {
                         "from": "gpt",
                         "value": "" # 占位符，触发重新生成
                     }
                 ],
                 # 保留原始元数据
                 "original_answers": item.get("original_answers"),
                 "question_type": item.get("question_type")
             })
        else:
             normalized_data.append(item)
    
    logger.info(f"加载数据集完成：共{len(normalized_data)}条对话")
    return normalized_data

def main():
    # 1. 加载配置
    # config = load_config("./configs/regenerator_vllm.yaml")
    config = load_config("./configs/regenerator_qwen_vl.yaml")
    
    # 2. 加载数据集
    dataset = load_local_dataset(config["data"]["input_path"])
    
    # 3. 初始化vLLM生成器
    generator = VLLMGenerator(config)
    
    # 4. 批量处理
    generator.process_batch(dataset)
    
    logger.info("全部流程执行完成！")

if __name__ == "__main__":
    main()