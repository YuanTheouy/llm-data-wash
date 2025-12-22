import json
import os
import threading
from typing import List, Dict, Optional
from llm_data_wash.core.base_generator import BaseGenerator
from llm_data_wash.utils.gpu_monitor import GPUMonitor
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from llm_data_wash.utils.logger import get_logger
from llm_data_wash.utils.concurrency import run_thread_pool
from tqdm import tqdm

logger = get_logger(__name__)

class VLLMGenerator(BaseGenerator):
    """多卡vLLM数据重生成器（优化版）：支持多轮批量推理+超长Prompt处理+异常容错"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # 1. 保持原有结构读取
        self.vllm_config = config["vllm"]
        self.concurrency_config = config["concurrency"]
        
        # 2. 基础路径参数 (保持不变)
        self.model_path = self.vllm_config["model_path"]
        self.output_dir = config["data"]["output_dir"]
        
        # 3. 映射并发参数 (复用原有字段)
        self.batch_size = self.concurrency_config["batch_size"]
        
        # 4. 初始化 Tokenizer
        logger.info(f"正在加载 Tokenizer: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )

        # 5. 初始化 vLLM 引擎 (兼容旧版本vLLM)
        tp_size = self.vllm_config.get("tensor_parallel_size", 1)
        gpu_util = self.vllm_config.get("gpu_memory_utilization", 0.90)
        # 核心：读取配置文件的8192（已改好）
        max_len = self.vllm_config.get("max_model_len", 8192)  
        # 新增：可配置的生成参数和长度限制
        self.max_generate_tokens = self.vllm_config.get("max_tokens", 512)
        self.max_allowed_rounds = self.vllm_config.get("max_allowed_rounds", 5)  # 限制最大轮次
        self.single_turn_max_tokens = self.vllm_config.get("single_turn_max_tokens", 1024)  # 单轮内容限制
        
        logger.info(
            f"初始化 vLLM: TP={tp_size}, MemUtil={gpu_util}, MaxLen={max_len}, "
            f"MaxRounds={self.max_allowed_rounds}, SingleTurnMax={self.single_turn_max_tokens}"
        )
        
        # ========== 兼容旧版本vLLM的引擎初始化 ==========
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=gpu_util,
            max_model_len=max_len,
            trust_remote_code=True,
            dtype="auto",
            # 兼容旧版本的优化参数（删除enable_cuda_graph）
            enable_chunked_prefill=True,  # 分块处理超长Prompt
            max_num_batched_tokens=8192,  # 匹配max_model_len
            max_num_seqs=2048,  # 增大并发序列数
            swap_space=4  # 显存不足时用CPU内存
        )
        
        # 6. 采样参数 (增强：添加停止词)
        self.sampling_params = SamplingParams(
            temperature=self.vllm_config["temperature"],
            max_tokens=self.max_generate_tokens,
            top_p=0.9,
            stop_token_ids=[self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else None
        )
        
        # 批量处理大小 (8卡建议设为2000)
        self.process_batch_size = config["concurrency"].get("batch_size", 2000 if tp_size >=8 else 1000)
        
        # 创建输出目录（绝对路径，不会找不到）
        os.makedirs(self.output_dir, exist_ok=True)
        # 打印绝对路径，确保你能找到
        logger.info(f"✅ 输出目录已创建（绝对路径）：{os.path.abspath(self.output_dir)}")
        print(f"\n📌 数据将保存到：{os.path.abspath(self.output_dir)}\n")

    def process_single(self, conversation: Dict, idx: int) -> Dict:
        """处理单条对话（实现抽象方法，兼容原有接口）"""
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
    
    def call_vllm_api(self, prompt: str) -> str:
        """单条Prompt推理（兼容process_single方法）"""
        try:
            # 校验Prompt长度
            prompt_tokens = self.tokenizer.encode(prompt)
            max_prompt_len = self.llm.engine_config.max_model_len - self.max_generate_tokens
            if len(prompt_tokens) > max_prompt_len:
                prompt_tokens = prompt_tokens[-max_prompt_len:]
                prompt = self.tokenizer.decode(prompt_tokens, skip_special_tokens=False)
                logger.warning(f"单条Prompt过长，已截断至{max_prompt_len} Token")
            
            outputs = self.llm.generate([prompt], self.sampling_params)
            return outputs[0].outputs[0].text.strip()
        except Exception as e:
            logger.error(f"单条推理失败: {str(e)[:200]}")
            return ""
    
    def process_batch(self, dataset: List[Dict]) -> List[Dict]:
        """
        核心优化版：按轮次批量推理
        1. 超长Prompt自动截断
        2. 限制最大对话轮次/单轮长度
        3. 异常捕获（单样本失败不影响整批）
        4. 显存适配8卡环境
        """
        total_count = len(dataset)
        logger.info(f"开始离线推理处理：共 {total_count} 条数据，批次大小 {self.process_batch_size}")
        
        results = []
        total_batches = (total_count + self.process_batch_size - 1) // self.process_batch_size
        
        with tqdm(total=total_count, desc="整体处理进度", unit="条") as pbar:
            for batch_idx in range(total_batches):
                # 1. 切分当前批次
                start_idx = batch_idx * self.process_batch_size
                end_idx = min((batch_idx + 1) * self.process_batch_size, total_count)
                batch_data = dataset[start_idx:end_idx]
                current_batch_size = len(batch_data)
                
                # 2. 预处理：拆分多轮对话+初始化存储结构
                batch_rounds = []          # 每个样本的轮次信息 [(user1, _), (user2, _)...]
                batch_chat_history = [[] for _ in range(current_batch_size)]  # 对话历史
                batch_new_convs = [[] for _ in range(current_batch_size)]     # 最终生成的对话
                max_rounds = 0             # 批次内最大轮次数
                
                for sample_idx in range(current_batch_size):
                    conv_item = batch_data[sample_idx]
                    original_convs = conv_item.get("conversations", conv_item.get("items", []))
                    
                    # 拆分轮次（user+assistant为一轮）
                    rounds = []
                    current_user = None
                    for msg in original_convs:
                        role = msg.get("from") or msg.get("role")
                        content = msg.get("value") or msg.get("content")
                        
                        if role in ["human", "user"] and content.strip():
                            current_user = content.strip()
                        elif role in ["gpt", "assistant"] and current_user is not None:
                            rounds.append((current_user, content.strip()))
                            current_user = None
                    
                    # 限制最大轮次（避免历史过长）
                    if len(rounds) > self.max_allowed_rounds:
                        logger.warning(
                            f"样本 {start_idx+sample_idx} 轮次过多（{len(rounds)}→{self.max_allowed_rounds}），已截断"
                        )
                        rounds = rounds[-self.max_allowed_rounds:]
                    
                    batch_rounds.append(rounds)
                    max_rounds = max(max_rounds, len(rounds))
                
                # 3. 按轮次批量推理（核心优化）
                for round_idx in range(max_rounds):
                    round_prompts = []         # 当前轮次所有样本的Prompt
                    round_sample_indices = []  # 有当前轮次的样本索引
                    
                    # 构造当前轮次的批量Prompt
                    for sample_idx in range(current_batch_size):
                        rounds = batch_rounds[sample_idx]
                        if round_idx >= len(rounds):
                            continue
                        
                        # 提取当前轮次user内容+截断单轮超长内容
                        user_content, _ = rounds[round_idx]
                        user_tokens = self.tokenizer.encode(user_content)
                        if len(user_tokens) > self.single_turn_max_tokens:
                            user_content = self.tokenizer.decode(
                                user_tokens[:self.single_turn_max_tokens], 
                                skip_special_tokens=True
                            )
                            logger.warning(
                                f"样本 {start_idx+sample_idx} 第{round_idx}轮User内容过长，已截断"
                            )
                        
                        # 构造Prompt（历史+当前user）
                        current_chat = batch_chat_history[sample_idx].copy()
                        current_chat.append({"role": "user", "content": user_content})
                        
                        # 应用Chat Template
                        prompt_str = self.tokenizer.apply_chat_template(
                            current_chat,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        
                        # 核心修复：校验并截断超长Prompt
                        prompt_tokens = self.tokenizer.encode(prompt_str)
                        # 预留生成空间：max_model_len - 生成token数
                        max_prompt_len = self.llm.engine_config.max_model_len - self.max_generate_tokens
                        if len(prompt_tokens) > max_prompt_len:
                            truncated_tokens = prompt_tokens[-max_prompt_len:]
                            prompt_str = self.tokenizer.decode(truncated_tokens, skip_special_tokens=False)
                            logger.warning(
                                f"样本 {start_idx+sample_idx} 第{round_idx}轮Prompt过长 "
                                f"（{len(prompt_tokens)}→{max_prompt_len}），已截断"
                            )
                        
                        round_prompts.append(prompt_str)
                        round_sample_indices.append(sample_idx)
                    
                    # 4. 批量推理（带异常捕获）
                    outputs = []
                    if round_prompts:
                        try:
                            # 批量推理（8卡核心优势）
                            outputs = self.llm.generate(round_prompts, self.sampling_params)
                        except Exception as e:
                            logger.error(
                                f"批次{batch_idx}第{round_idx}轮批量推理失败：{str(e)[:200]}"
                            )
                            # 兜底：逐个推理失败的样本
                            outputs = []
                            for prompt in round_prompts:
                                try:
                                    single_output = self.llm.generate([prompt], self.sampling_params)
                                    outputs.append(single_output[0])
                                except Exception as e2:
                                    logger.error(f"单样本推理失败：{str(e2)[:200]}")
                                    outputs.append(None)
                    
                    # 5. 回填当前轮次结果
                    for output_idx, output in enumerate(outputs):
                        sample_idx = round_sample_indices[output_idx]
                        if output is None or not output.outputs:
                            # 失败兜底：空回复
                            new_assistant = ""
                            logger.warning(
                                f"样本 {start_idx+sample_idx} 第{round_idx}轮推理失败，使用空回复"
                            )
                        else:
                            new_assistant = output.outputs[0].text.strip()
                        
                        # 更新对话历史（用于下一轮Prompt）
                        current_user = batch_rounds[sample_idx][round_idx][0]
                        batch_chat_history[sample_idx].append({"role": "user", "content": current_user})
                        batch_chat_history[sample_idx].append({"role": "assistant", "content": new_assistant})
                        
                        # 更新最终对话结构
                        batch_new_convs[sample_idx].append({"from": "human", "value": current_user})
                        batch_new_convs[sample_idx].append({"from": "gpt", "value": new_assistant})
                
                # 6. 构造当前批次结果
                batch_results = []
                for sample_idx in range(current_batch_size):
                    conv_item = batch_data[sample_idx]
                    batch_results.append({
                        "id": conv_item.get("id", f"regenerated_{start_idx+sample_idx}"),
                        "conversations": batch_new_convs[sample_idx]
                    })
                
                # 7. 保存当前批次
                self.save_batch(batch_results, batch_idx)
                results.extend(batch_results)
                pbar.update(current_batch_size)
        
        # 8. 合并所有批次
        self._merge_batches(total_batches)
        logger.info(f"所有数据推理完成！共生成 {len(results)} 条数据，输出目录：{self.output_dir}")
        return results
    
    def save_batch(self, batch_data: List[Dict], batch_idx: int):
        """保存批次数据（增强：异常处理+中文兼容）"""
        try:
            batch_file = os.path.join(self.output_dir, f"batch_{batch_idx:04d}.jsonl")
            with open(batch_file, "w", encoding="utf-8") as f:
                for item in batch_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            logger.info(f"批次 {batch_idx} 保存完成：{len(batch_data)}条 | 路径：{batch_file}")
        except Exception as e:
            logger.error(f"保存批次{batch_idx}失败：{e}")
    
    def _merge_batches(self, total_batches: int):
        """合并所有批次（增强：清理临时文件+双格式保存）"""
        try:
            # 1. 合并为JSONL（推荐格式，适合大文件）
            merge_jsonl = os.path.join(self.output_dir, "regenerated_complete.jsonl")
            with open(merge_jsonl, "w", encoding="utf-8") as out_f:
                for batch_idx in range(total_batches):
                    batch_file = os.path.join(self.output_dir, f"batch_{batch_idx:04d}.jsonl")
                    if os.path.exists(batch_file):
                        with open(batch_file, "r", encoding="utf-8") as in_f:
                            out_f.write(in_f.read())
                        # 可选：删除临时批次文件
                        # os.remove(batch_file)
            
            # 2. 生成JSON格式（便于人工查看）
            merge_json = os.path.join(self.output_dir, "regenerated_complete.json")
            with open(merge_jsonl, "r", encoding="utf-8") as f:
                json_data = [json.loads(line.strip()) for line in f if line.strip()]
            with open(merge_json, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"所有批次合并完成：\n- JSONL: {merge_jsonl}\n- JSON: {merge_json}")
        except Exception as e:
            logger.error(f"合并批次失败：{e}")