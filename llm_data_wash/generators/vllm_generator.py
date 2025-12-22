import json
import requests
import time
import os
import threading
from typing import List, Dict
from llm_data_wash.core.base_generator import BaseGenerator
from llm_data_wash.utils.gpu_monitor import GPUMonitor
from llm_data_wash.utils.logger import get_logger
from llm_data_wash.utils.concurrency import run_thread_pool

logger = get_logger(__name__)

class VLLMGenerator(BaseGenerator):
    """多卡vLLM数据重生成器"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # 加载vLLM配置
        self.vllm_config = config["vllm"]
        self.vllm_url = self.vllm_config["url"]
        self.model_path = self.vllm_config["model_path"]
        self.temperature = self.vllm_config["temperature"]
        self.max_tokens = self.vllm_config["max_tokens"]
        self.timeout = self.vllm_config["timeout"]
        self.min_api_interval = self.vllm_config["min_api_interval"]
        
        # 加载并行配置
        self.num_workers = config["concurrency"]["num_workers"]
        self.batch_size = config["concurrency"]["batch_size"]
        
        # 线程安全
        self.api_lock = threading.Lock()
        self.result_lock = threading.Lock()
        self.last_api_call = 0
        
        # 初始化GPU监控
        self.gpu_monitor = GPUMonitor(gpu_count=8)
    
    def call_vllm_api(self, prompt: str) -> str:
        """调用vLLM API"""
        try:
            # API限流
            with self.api_lock:
                current_time = time.time()
                time_since_last = current_time - self.last_api_call
                if time_since_last < self.min_api_interval:
                    time.sleep(self.min_api_interval - time_since_last)
                self.last_api_call = current_time
            
            # 构造请求
            payload = {
                "model": self.model_path,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stop": [],
                "top_p": 0.9,
                "frequency_penalty": 0.0
            }
            
            response = requests.post(
                self.vllm_url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        
        except Exception as e:
            logger.error(f"vLLM API调用失败: {e}")
            return ""
    
    def process_single(self, conversation: Dict, idx: int) -> Dict:
        """处理单条对话（实现抽象方法）"""
        # 兼容不同对话格式
        convos = conversation.get("conversations",
                                conversation.get("items",
                                                conversation.get("messages", [])))
        new_convos = []
        context = ""
        
        for msg in convos:
            role = msg.get("from") or msg.get("role")
            content = msg.get("value") or msg.get("content")
            
            if role in ["human", "user"]:
                new_convos.append({"from": "human", "value": content})
                context += f"Human: {content}\n\n"
            elif role in ["gpt", "assistant"]:
                prompt = f"{context}Assistant:"
                new_answer = self.call_vllm_api(prompt)
                new_convos.append({"from": "gpt", "value": new_answer})
                context += f"Assistant: {new_answer}\n\n"
        
        return {
            "id": conversation.get("id", f"regenerated_{idx}"),
            "conversations": new_convos
        }
    
    def process_batch(self, dataset: List[Dict]) -> List[Dict]:
        """批量处理对话（实现抽象方法）"""
        total_count = len(dataset)
        logger.info(f"开始批量处理：共{total_count}条对话 | 并发数{self.num_workers}")
        
        # 启动GPU监控
        self.gpu_monitor.start_monitor(interval=10)
        
        # 构造线程池任务
        tasks = [(conv, idx) for idx, conv in enumerate(dataset)]
        results = run_thread_pool(
            func=self.process_single,
            tasks=tasks,
            max_workers=self.num_workers
        )
        
        # 按索引过滤有效结果
        valid_results = [r for r in results if r is not None]
        
        # 批次保存
        batch_idx = 0
        for i in range(0, len(valid_results), self.batch_size):
            batch_data = valid_results[i:i+self.batch_size]
            self.save_batch(batch_data, batch_idx)
            batch_idx += 1
        
        # 合并所有批次
        self._merge_batches(batch_idx)
        
        # 停止GPU监控
        self.gpu_monitor.stop_monitor()
        
        logger.info(f"批量处理完成：共生成{len(valid_results)}条有效数据")
        return valid_results
    
    def save_batch(self, batch_data: List[Dict], batch_idx: int):
        """保存批次数据（实现抽象方法）"""
        batch_file = os.path.join(self.output_dir, f"batch_{batch_idx:04d}.jsonl")
        with open(batch_file, "w", encoding="utf-8") as f:
            for item in batch_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"批次 {batch_idx} 保存完成：{len(batch_data)}条 | 路径：{batch_file}")
    
    def _merge_batches(self, total_batches: int):
        """合并所有批次为完整文件"""
        merge_file = os.path.join(self.output_dir, "regenerated_complete.jsonl")
        with open(merge_file, "w", encoding="utf-8") as out_f:
            for batch_idx in range(total_batches):
                batch_file = os.path.join(self.output_dir, f"batch_{batch_idx:04d}.jsonl")
                if os.path.exists(batch_file):
                    with open(batch_file, "r", encoding="utf-8") as in_f:
                        out_f.write(in_f.read())
        
        # 保存JSON格式
        json_merge_file = os.path.join(self.output_dir, "regenerated_complete.json")
        with open(merge_file, "r", encoding="utf-8") as f:
            json_data = [json.loads(line.strip()) for line in f if line.strip()]
        with open(json_merge_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"所有批次合并完成：{merge_file}")