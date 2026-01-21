import json
import os
import time
import requests
import threading
import base64
import concurrent.futures
from typing import List, Dict, Optional
from tqdm import tqdm
from llm_data_wash.core.base_generator import BaseGenerator
from llm_data_wash.utils.logger import get_logger

logger = get_logger(__name__)

class VLLMGenerator(BaseGenerator):
    """基于API调用的vLLM数据重生成器（对齐regen.py逻辑）"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        self.vllm_config = config["vllm"]
        self.concurrency_config = config["concurrency"]
        
        # API 配置
        self.vllm_url = self.vllm_config.get("vllm_url", "http://localhost:8000/v1/chat/completions")
        # 优先使用配置的 model_name，如果未配置则尝试使用 model_path，最后默认
        self.model_name = self.vllm_config.get("model_name") or self.vllm_config.get("model_path") or "Llama-3.1-8B-Instruct"
        self.timeout = self.vllm_config.get("timeout", 60)
        
        # 生成参数
        self.temperature = self.vllm_config.get("temperature", 0.7)
        self.max_tokens = self.vllm_config.get("max_tokens", 2048)
        
        # 并发控制
        self.num_threads = self.concurrency_config.get("num_threads", 4)
        self.batch_save = self.concurrency_config.get("batch_save", 10)
        
        # 速率限制锁
        self.api_rate_limit_lock = threading.Lock()
        self.last_api_call = 0
        self.min_api_interval = self.vllm_config.get("min_api_interval", 0.1)
        
        logger.info(f"初始化 VLLMGenerator (API模式): URL={self.vllm_url}, Model={self.model_name}, Threads={self.num_threads}")

    def encode_image(self, image_path: str) -> str:
        """将本地图片转换为Base64字符串"""
        if not os.path.exists(image_path):
            logger.warning(f"图片文件不存在: {image_path}")
            return ""
        
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"图片编码失败 {image_path}: {e}")
            return ""

    def call_vllm_api(self, messages: List[Dict]) -> str:
        """调用vLLM API生成回答"""
        try:
            # 简单的速率限制
            with self.api_rate_limit_lock:
                current_time = time.time()
                time_since_last_call = current_time - self.last_api_call
                if time_since_last_call < self.min_api_interval:
                    time.sleep(self.min_api_interval - time_since_last_call)
                self.last_api_call = time.time()

            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stop": []  # 可根据需要添加停止词
            }

            response = requests.post(self.vllm_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            
            # 适配不同的返回结构
            if "choices" in result and len(result["choices"]) > 0:
                message = result["choices"][0].get("message", {})
                return message.get("content", "")
            else:
                logger.warning(f"未知的API响应格式: {result}")
                return ""

        except Exception as e:
            logger.error(f"调用vLLM API时出错: {e}")
            return ""

    def process_single(self, conversation: Dict, idx: int) -> Dict:
        """处理单条对话，重新生成助手回答"""
        conversations_list = conversation.get("conversations", [])
        
        if not conversations_list:
            # 兼容其他格式
            if "items" in conversation:
                conversations_list = conversation["items"]
            else:
                conversations_list = conversation.get("messages", [])

        new_conversations = []
        # 维护用于API调用的消息列表
        api_messages = []

        for msg in conversations_list:
            role = msg.get("from") or msg.get("role")
            content = msg.get("value") or msg.get("content")
            image = msg.get("image") or msg.get("image_path")

            # 如果是人类消息，添加到消息列表
            if role in ["human", "user"]:
                # 构建API消息
                user_msg_api = {"role": "user", "content": content}
                if image:
                    # 如果有图片，构造多模态消息 (使用Base64)
                    base64_image = self.encode_image(image)
                    if base64_image:
                        user_msg_api["content"] = [
                            {"type": "text", "text": content},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    else:
                        # 图片加载失败，退化为纯文本
                        logger.warning(f"图片加载失败，退化为纯文本: {image}")
                
                # 如果没有图片，user_msg_api 保持 {"role": "user", "content": "..."} 的纯文本格式
                # Qwen2.5-VL 完全支持这种纯文本输入格式
                
                api_messages.append(user_msg_api)
                
                # 添加到新对话记录
                new_msg = {
                    "from": "human",
                    "value": content
                }
                if image:
                    new_msg["image"] = image
                new_conversations.append(new_msg)

            # 如果是助手消息，使用vLLM重新生成
            elif role in ["gpt", "assistant"]:
                # 调用vLLM生成新回答
                new_response = self.call_vllm_api(api_messages)
                
                # 将生成的回答加入历史，供后续轮次使用
                api_messages.append({
                    "role": "assistant",
                    "content": new_response
                })
                
                new_conversations.append({
                    "from": "gpt",
                    "value": new_response
                })
                
        return {
            "id": conversation.get("id", f"regenerated_{idx}"),
            "conversations": new_conversations
        }

    def process_batch(self, dataset: List[Dict]) -> List[Dict]:
        """使用线程池批量处理对话"""
        total_count = len(dataset)
        logger.info(f"开始使用线程池处理数据，总共 {total_count} 条对话，使用 {self.num_threads} 个线程")
        
        results = []
        
        # 准备任务
        tasks = [(idx, dataset[idx]) for idx in range(total_count)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # 提交所有任务
            future_to_idx = {
                executor.submit(self.process_single, conv, idx): (idx, i) 
                for i, (idx, conv) in enumerate(tasks)
            }
            
            # 处理完成的任务结果
            for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=total_count, desc="处理进度"):
                idx, task_idx = future_to_idx[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # 按照 regen.py 的逻辑，每 batch_save 个保存一次
                    # 注意：regen.py 的逻辑是按照完成顺序积攒，这里为了保持一致性，
                    # 我们也可以简单地每积攒一定数量就保存一次
                    if len(results) % self.batch_save == 0:
                        # 简单的批次保存，这里为了简化，我们保存当前累积的最新一批
                        # 实际生产中可能需要更复杂的逻辑来避免重复保存或覆盖
                        # 这里我们模拟 regen.py 的行为：保存一个批次文件
                        
                        # 计算当前批次范围（仅用于文件名，不一定完全对应索引顺序）
                        current_batch_size = self.batch_save
                        batch_start = max(0, len(results) - current_batch_size)
                        batch_end = len(results)
                        
                        # 获取这一批数据
                        batch_data = results[batch_start:batch_end]
                        self.save_batch(batch_data, len(results) // self.batch_save)
                        
                except Exception as e:
                    logger.error(f"处理任务 {idx} 时出错: {e}")
        
        # 排序结果
        try:
            results.sort(key=lambda x: int(x["id"].split("_")[-1]) if "_" in x["id"] and x["id"].split("_")[-1].isdigit() else 0)
        except Exception:
            pass # 如果ID格式不对，忽略排序错误
            
        # 合并保存最终结果
        self._merge_batches(results)
        return results

    def save_batch(self, batch_data: List[Dict], batch_idx: int):
        """保存批次数据"""
        try:
            batch_file = os.path.join(self.output_dir, f"batch_{batch_idx:04d}.jsonl")
            with open(batch_file, "w", encoding="utf-8") as f:
                for item in batch_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            logger.debug(f"批次 {batch_idx} 保存完成：{len(batch_data)}条")
        except Exception as e:
            logger.error(f"保存批次{batch_idx}失败：{e}")

    def _merge_batches(self, all_data: List[Dict]):
        """保存完整结果"""
        try:
            # 检查是否有自定义的最终输出路径
            final_output_path = self.config.get("data", {}).get("final_output_path")
            
            if final_output_path:
                # 如果指定了具体文件路径，直接保存到该路径
                merge_jsonl = final_output_path
                # 确保父目录存在
                os.makedirs(os.path.dirname(merge_jsonl), exist_ok=True)
            else:
                # 否则保存到 output_dir 下的默认文件名
                merge_jsonl = os.path.join(self.output_dir, "regenerated_complete.jsonl")
            
            # 保存 JSONL
            with open(merge_jsonl, "w", encoding="utf-8") as f:
                for item in all_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            # 只有在没有指定特定文件路径时，才保存额外的 JSON 格式备份，避免混乱
            if not final_output_path:
                merge_json = os.path.join(self.output_dir, "regenerated_complete.json")
                with open(merge_json, "w", encoding="utf-8") as f:
                    json.dump(all_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"所有数据处理完成！共 {len(all_data)} 条，已保存至 {merge_jsonl}")
        except Exception as e:
            logger.error(f"保存最终结果失败：{e}")
