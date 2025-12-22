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
        super().__init__(config)
        
        # 1. 保持原有结构读取
        self.vllm_config = config["vllm"]
        self.concurrency_config = config["concurrency"]
        
        # 2. 基础路径参数 (保持不变)
        self.model_path = self.vllm_config["model_path"]
        self.output_dir = config["data"]["output_dir"]
        
        # 3. 映射并发参数 (复用原有字段)
        # 离线推理中，batch_size 决定一次喂给 GPU 的数据量，复用 concurrency.batch_size
        self.batch_size = self.concurrency_config["batch_size"]
        
        # 4. 初始化 Tokenizer
        logger.info(f"正在加载 Tokenizer: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )

        # 5. 初始化 vLLM 引擎 (读取 Config 中的新参数)
        # 如果 config 里没写，我这里给了默认值作为兜底，但建议写在 config 里
        tp_size = self.vllm_config.get("tensor_parallel_size", 1)
        gpu_util = self.vllm_config.get("gpu_memory_utilization", 0.90)
        max_len = self.vllm_config.get("max_model_len", 4096) 
        
        logger.info(f"初始化 vLLM: TP={tp_size}, MemUtil={gpu_util}, MaxLen={max_len}")
        
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=tp_size,  # 关键：从配置读取 TP
            gpu_memory_utilization=gpu_util,
            max_model_len=max_len,
            trust_remote_code=True,
            dtype="auto"
        )
        
        # 6. 采样参数 (保持不变)
        self.sampling_params = SamplingParams(
            temperature=self.vllm_config["temperature"],
            max_tokens=self.vllm_config["max_tokens"],
            top_p=0.9
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
        优化：按轮次批量推理，兼顾多轮重生成和vLLM批量效率
        """
        total_count = len(dataset)
        logger.info(f"开始离线推理处理：共 {total_count} 条数据")
        
        results = []
        total_batches = (total_count + self.process_batch_size - 1) // self.process_batch_size
        
        with tqdm(total=total_count, desc="整体处理进度", unit="条") as pbar:
            for i in range(0, total_count, self.process_batch_size):
                batch_data = dataset[i : i + self.process_batch_size]
                batch_size = len(batch_data)
                
                # 步骤1：预处理所有样本，拆分多轮对话
                batch_rounds = []  # 每个样本的轮次信息：[(user1, _), (user2, _), ...]
                batch_chat_history = [[] for _ in range(batch_size)]  # 每个样本的对话历史
                batch_new_convs = [[] for _ in range(batch_size)]  # 每个样本的最终新对话
                
                max_rounds = 0  # 批次内样本的最大轮次数
                for idx, conv_item in enumerate(batch_data):
                    original_convs = conv_item.get("conversations", [])
                    # 拆分轮次（user+assistant为一轮）
                    rounds = []
                    current_user = None
                    for msg in original_convs:
                        if msg["from"] in ["human", "user"]:
                            current_user = msg["value"]
                        elif msg["from"] in ["gpt", "assistant"] and current_user is not None:
                            rounds.append((current_user, msg["value"]))
                            current_user = None
                    batch_rounds.append(rounds)
                    max_rounds = max(max_rounds, len(rounds))
                
                # 步骤2：按轮次批量推理（核心优化）
                for round_idx in range(max_rounds):
                    # 收集当前轮次需要推理的Prompt
                    round_prompts = []
                    round_sample_indices = []  # 记录哪些样本有当前轮次
                    
                    for sample_idx in range(batch_size):
                        rounds = batch_rounds[sample_idx]
                        if round_idx < len(rounds):
                            # 提取当前轮次的user内容
                            user_content, _ = rounds[round_idx]
                            # 构造Prompt（历史+当前user）
                            current_chat = batch_chat_history[sample_idx].copy()
                            current_chat.append({"role": "user", "content": user_content})
                            # 应用Chat Template
                            prompt_str = self.tokenizer.apply_chat_template(
                                current_chat,
                                tokenize=False,
                                add_generation_prompt=True
                            )
                            round_prompts.append(prompt_str)
                            round_sample_indices.append(sample_idx)
                    
                    # 批量推理当前轮次（关键：vLLM批量处理）
                    if round_prompts:
                        outputs = self.llm.generate(round_prompts, self.sampling_params)
                        # 回填当前轮次结果
                        for output_idx, output in enumerate(outputs):
                            sample_idx = round_sample_indices[output_idx]
                            new_assistant = output.outputs[0].text.strip()
                            # 更新该样本的对话历史
                            batch_chat_history[sample_idx].append({"role": "user", "content": batch_rounds[sample_idx][round_idx][0]})
                            batch_chat_history[sample_idx].append({"role": "assistant", "content": new_assistant})
                            # 更新该样本的最终对话
                            batch_new_convs[sample_idx].append({"from": "human", "value": batch_rounds[sample_idx][round_idx][0]})
                            batch_new_convs[sample_idx].append({"from": "gpt", "value": new_assistant})
                
                # 步骤3：构造批次结果
                batch_results = []
                for idx, conv_item in enumerate(batch_data):
                    batch_results.append({
                        "id": conv_item.get("id", f"regenerated_{i + idx}"),
                        "conversations": batch_new_convs[idx]
                    })
                
                # 保存当前批次
                self.save_batch(batch_results, i // self.process_batch_size)
                results.extend(batch_results)
                pbar.update(len(batch_data))
        
        # 合并所有批次
        self._merge_batches(total_batches)
        logger.info("所有数据推理完成并合并为完整文件")
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