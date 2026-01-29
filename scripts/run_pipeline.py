#!/usr/bin/env python3
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

def load_local_dataset(data_path: str, config: dict = None) -> list:
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
            # 处理VQAv2标准格式：{"questions": [...]}
            if isinstance(data, dict):
                if "questions" in data:
                    data = data["questions"]
                elif "annotations" in data: # VQAv2 annotations file
                    data = data["annotations"]
                else:
                    # 如果是字典但找不到已知列表键，记录警告并尝试查找任何列表类型的值
                    logger.warning(f"JSON文件加载为字典，但未找到'questions'键。可用键: {list(data.keys())}")
                    found_list = False
                    for key, value in data.items():
                        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                            logger.info(f"自动识别列表键: '{key}'，包含 {len(value)} 条数据")
                            data = value
                            found_list = True
                            break
                    
                    if not found_list:
                        logger.error("无法从JSON字典中提取数据列表。请检查JSON结构。")
                        # 如果确实无法提取，可能这个字典本身就是单个数据项？
                        # 但通常不可能。抛出错误避免后续 'str' object has no attribute 'get'
                        raise ValueError(f"JSON结构不符合预期，如果是字典，必须包含 'questions' 或其他列表字段。Keys: {list(data.keys())}")
    else:
        raise ValueError(f"不支持的数据集格式：{data_path}")
    
    # 获取图片路径配置
    image_dir = ""
    image_prefix = ""
    if config and "data" in config:
        image_dir = config["data"].get("image_dir", "")
        image_prefix = config["data"].get("image_prefix", "")

    # 数据标准化：检查是否为VQA格式并转换
    normalized_data = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            logger.warning(f"跳过非字典数据项 (index {i}): {item} (type: {type(item)})")
            continue

        # 1. 标准VQA格式 (包含 image_path 和 question)
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
        # 2. VQAv2格式 (包含 image_id 和 question)
        elif "image_id" in item and "question" in item:
             image_id = item["image_id"]
             # 构造图片路径
             # 默认COCO格式: COCO_train2014_000000524291.jpg (12位数字)
             if image_prefix:
                 image_filename = f"{image_prefix}{int(image_id):012d}.jpg"
             else:
                 # 如果没有配置前缀，尝试直接使用image_id (不太可能，但作为fallback)
                 image_filename = f"{image_id}.jpg"
            
             image_path = os.path.join(image_dir, image_filename)
             
             normalized_data.append({
                 "id": str(item.get("question_id", "")),
                 "conversations": [
                     {
                         "from": "human",
                         "value": item["question"],
                         "image_path": image_path
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
    # config = load_config("./configs/regenerator_qwen_vl.yaml")
    config = load_config("./configs/regenerator_qwen_vl_vqav2.yaml")
    
    # 2. 加载数据集
    dataset = load_local_dataset(config["data"]["input_path"], config)
    
    # 3. 初始化vLLM生成器
    generator = VLLMGenerator(config)
    
    # 4. 批量处理
    generator.process_batch(dataset)
    
    logger.info("全部流程执行完成！")

if __name__ == "__main__":
    main()
