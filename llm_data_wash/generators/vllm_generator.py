import json
import os
import threading
from typing import List, Dict
from llm_data_wash.core.base_generator import BaseGenerator
from llm_data_wash.utils.gpu_monitor import GPUMonitor
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from llm_data_wash.utils.logger import get_logger
from llm_data_wash.utils.concurrency import run_thread_pool
from tqdm import tqdm





logger = get_logger(__name__)

class VLLMGenerator(BaseGenerator):
    """多卡vLLM数据重生成器"""
    
    def __init__(self, config: Dict):
        # 绕过父类可能存在的 HTTP 初始化逻辑，如果父类强绑定了 HTTP，建议重构父类
        # 这里假设只继承必要的配置读取逻辑
        self.config = config
        self.output_dir = config["output_dir"]
        
        self.vllm_config = config["vllm"]
        self.model_path = self.vllm_config["model_path"]
        
        # 1. 初始化 vLLM 引擎
        # tensor_parallel_size=8 意味着模型参数被切分到 8 张卡上并行计算
        # 这种方式对于大模型推理吞吐量提升极其显著
        logger.info(f"正在初始化 vLLM 引擎，模型路径: {self.model_path}, TP=8...")
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=8, # 关键参数：利用 8 卡并行
            gpu_memory_utilization=0.90, # 预留一点显存防止 OOM
            trust_remote_code=True,
            max_model_len=self.vllm_config.get("max_tokens", 4096) * 2 # 确保上下文长度足够
        )
        
        # 2. 初始化 Tokenizer (用于应用 Chat Template)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        
        # 3. 设置采样参数
        self.sampling_params = SamplingParams(
            temperature=self.vllm_config["temperature"],
            max_tokens=self.vllm_config["max_tokens"],
            top_p=0.9,
            stop=[]
        )
        
        # 批量处理大小 (离线推理可以设得很大，vLLM 会自动切分)
        # 这里指一次性喂给 engine 的 prompt 数量
        self.process_batch_size = config["concurrency"].get("batch_size", 1000)

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
        """
        覆盖父类的处理逻辑，使用离线推理全量处理
        """
        total_count = len(dataset)
        logger.info(f"开始离线推理处理：共 {total_count} 条数据")
        
        results = []
        
        # 分块处理以避免内存一次性占用过大 (虽然 vLLM 可以处理，但 Python 侧列表过大也不好)
        for i in range(0, total_count, self.process_batch_size):
            batch_data = dataset[i : i + self.process_batch_size]
            batch_prompts = []
            batch_metadata = [] # 存储原始数据结构以便恢复
            
            # 1. 数据预处理 & 应用 Chat Template
            for conv_item in batch_data:
                # 获取对话列表
                convs = conv_item.get("conversations", conv_item.get("items", []))
                
                # 提取 User 的输入构建 Prompt
                # 注意：这里需要根据你的具体需求决定是保留历史还是只取最后一句
                # 假设这里是标准的 Chat 格式
                chat_messages = []
                for msg in convs:
                    if msg["from"] in ["human", "user"]:
                        chat_messages.append({"role": "user", "content": msg["value"]})
                    elif msg["from"] in ["gpt", "assistant"]:
                        # 如果是重生成任务，通常需要截断到最后一个 User，或者把前面的 Assistant 也带上
                        chat_messages.append({"role": "assistant", "content": msg["value"]})
                
                # 此时 chat_messages 应该是完整的历史。
                # 如果我们要让模型生成回复，通常移除最后一个 Assistant 的回复（如果是重写）或确保最后是 User
                # 这里假设你的逻辑是：给定 User 上下文，生成 Assistant 回复
                
                # 应用 Chat Template 转为 string
                # apply_chat_template 会自动处理 system prompt 和特殊 token
                prompt_str = self.tokenizer.apply_chat_template(
                    chat_messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                batch_prompts.append(prompt_str)
                batch_metadata.append(conv_item)
            
            # 2. 执行推理 (核心步骤)
            # vLLM 会自动处理 Padding, PagedAttention, Batching
            outputs = self.llm.generate(batch_prompts, self.sampling_params)
            
            # 3. 结果回填
            batch_results = []
            for j, output in enumerate(outputs):
                original_item = batch_metadata[j]
                generated_text = output.outputs[0].text
                
                # 构造新的对话结构
                new_convs = original_item.get("conversations", []).copy()
                # 简单追加或替换逻辑，根据你的业务需求修改
                new_convs.append({
                    "from": "gpt",
                    "value": generated_text
                })
                
                batch_results.append({
                    "id": original_item.get("id"),
                    "conversations": new_convs
                })
            
            results.extend(batch_results)
            
            # 实时保存当前批次
            self.save_batch(batch_results, i // self.process_batch_size)
            
        logger.info("所有数据推理完成")
        return results
    
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