# -*- coding: utf-8 -*-
"""
通用数据集处理：精准指定子数据集+流式下载+单文件比例严格50:25:15:10
nvidia/Nemotron-CC-Math-v1 仅从4plus子数据集采样，其他数据集均指定对应精品子集
输出：6个10B Parquet，单文件比例精准，与推荐数据集100%对齐
优化：使用临时文件缓存，避免60GB数据全部加载到内存导致OOM
"""
import os
import random
import math
import json
import shutil
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset
import argparse

# 固定随机种子，保证采样/拆分可复现
random.seed(42)

# ==================== 核心配置（已精准指定所有目标子数据集config_name）====================
DATASET_CONFIGS = [
    {
        "name": "nvidia/Nemotron-CC-Math-v1",    # 主数据集仓库
        "config_name": "4plus",                  # 精准指定：仅从4plus子数据集采样
        "global_ratio": 0.5,                     # 全局50% → 单文件5B，总30B
        "sample_type": "general_math",
        "text_fields": ["question", "answer", "text"],
    },
    {
        "name": "nvidia/Nemotron-Pretraining-Code-v1",  # 主数据集仓库
        "config_name": "synthetic_code_qa",             # 精准指定：合成代码QA子数据集
        "global_ratio": 0.25,                           # 全局25% → 单文件2.5B，总15B
        "sample_type": "general_code",
        "text_fields": ["question", "answer", "prompt", "completion", "text"],
    },
    {
        "name": "nvidia/Nemotron-CC-v2",         # 主数据集仓库
        "config_name": "High-Quality",           # 精准指定：High-Quality子数据集
        "global_ratio": 0.15,                    # 全局15% → 单文件1.5B，总9B
        "sample_type": "general_high_quality",
        "text_fields": ["text", "content", "sentence"],
    },
    {
        "name": "glaiveai/reasoning-v1-20m",     # 主数据集仓库
        "config_name": "default",                # 精准指定：default子数据集
        "global_ratio": 0.10,                    # 全局10% → 单文件1B，总6B
        "sample_type": "general_reasoning",
        "text_fields": ["question", "answer", "instruction", "output", "text"],
    },
]

# 总目标与拆分配置（固定，单文件比例严格50:25:15:10）
# 修正：用户确认 "60B" 为 60 GiB 物理大小
TOTAL_TARGET_BYTES = 60 * 1024**3  # 全局总60 GiB
SPLIT_NUM = 6                      # 拆分为6个文件
SINGLE_FILE_TOTAL = TOTAL_TARGET_BYTES / SPLIT_NUM  # 单个文件10 GiB

# 目录与运行配置
OUTPUT_DIR = "general_cpt_datasets_60B_exact_subset"
CACHE_DIR = "./hf_stream_cache_exact"
TEMP_DIR = "./temp_shards_cache"     # 临时文件目录，用于缓解内存压力
ESTIMATE_SAMPLE_CNT = 1000
STREAM_BATCH_SIZE = 1000
OUTPUT_PREFIX = "cpt_general_training_data_parquet_"

# 默认镜像地址（用户可覆盖）
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"

def parse_args():
    parser = argparse.ArgumentParser(description="精准指定子数据集-流式处理-单文件比例严格")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, type=str)
    parser.add_argument("--cache-dir", default=CACHE_DIR, type=str)
    parser.add_argument("--temp-dir", default=TEMP_DIR, type=str, help="临时文件存储目录")
    parser.add_argument("--estimate-samples", default=ESTIMATE_SAMPLE_CNT, type=int)
    parser.add_argument("--stream-batch-size", default=STREAM_BATCH_SIZE, type=int)
    parser.add_argument("--hf-token", type=str, help="Hugging Face Access Token (也可通过环境变量 HF_TOKEN 设置)")
    parser.add_argument("--hf-endpoint", default=DEFAULT_HF_ENDPOINT, type=str, help="Hugging Face 镜像地址 (也可通过环境变量 HF_ENDPOINT 设置)")
    return parser.parse_args()

def get_effective_text_field(example_keys, candidate_fields):
    """从第一条示例匹配有效文本字段（流式模式专用）"""
    for field in candidate_fields:
        if field in example_keys:
            return field
    for field in example_keys:
        if isinstance(example_keys[field], str):
            return field
    raise ValueError("未找到有效文本字段，请检查配置")

def estimate_avg_sample_size(ds_config, cache_dir, estimate_cnt, token=None):
    """流式预采样：仅从指定子数据集取样本，估算平均字节大小"""
    ds_name = ds_config["name"]
    config_name = ds_config["config_name"]
    text_candidates = ds_config["text_fields"]
    
    ds_stream = load_dataset(
        ds_name,
        config_name=config_name,
        split="train",
        streaming=True,
        cache_dir=cache_dir,
        # trust_remote_code=True,  # 已废弃，移除以避免报错
        token=token
    )
    first_example = next(iter(ds_stream))
    text_field = get_effective_text_field(first_example, text_candidates)
    
    ds_stream = load_dataset(
        ds_name,
        config_name=config_name,
        split="train",
        streaming=True,
        cache_dir=cache_dir,
        # trust_remote_code=True,  # 已废弃，移除以避免报错
        token=token
    )
    total_bytes = 0
    valid_count = 0
    for example in ds_stream:
        text = example[text_field].strip()
        if not text:
            continue
        total_bytes += len(text.encode("utf-8"))
        valid_count += 1
        if valid_count >= estimate_cnt:
            break
    if valid_count == 0:
        raise ValueError(f"{ds_name}[{config_name}] 子数据集无有效样本")
    avg_size = total_bytes / valid_count
    print(f"✅ {ds_name}[{config_name}] 预采样{valid_count}条，平均单样本：{avg_size:.2f} Bytes")
    return avg_size, text_field

def stream_collect_dataset_to_temp(ds_config, cache_dir, temp_dir, estimate_cnt, token=None):
    """
    流式采样并写入临时文件，避免内存溢出
    """
    ds_name = ds_config["name"]
    config_name = ds_config["config_name"]
    global_ratio = ds_config["global_ratio"]
    sample_type = ds_config["sample_type"]
    text_candidates = ds_config["text_fields"]
    
    global_target = int(TOTAL_TARGET_BYTES * global_ratio)
    single_shard_target = int(SINGLE_FILE_TOTAL * global_ratio)
    total_shard_target = [single_shard_target for _ in range(SPLIT_NUM)]
    total_shard_target[-1] = global_target - sum(total_shard_target[:-1])
    
    print(f"\n===== 开始处理：{ds_name} → 【{config_name}】子数据集 =====")
    print(f"📌 全局目标：{global_target/1024**3:.2f}GB | 单分片目标：{single_shard_target/1024**3:.2f}GB")
    
    avg_sample_size, text_field = estimate_avg_sample_size(ds_config, cache_dir, estimate_cnt)
    shard_required_samples = [math.ceil((t / avg_sample_size) * 1.1) for t in total_shard_target]
    print(f"📌 各分片需有效样本：{[f'{x:,}' for x in shard_required_samples]}")

    # 准备临时文件句柄
    os.makedirs(temp_dir, exist_ok=True)
    temp_files = {} # {shard_idx: file_handle}
    temp_filenames = {} # {shard_idx: filename}
    
    for i in range(SPLIT_NUM):
        fname = os.path.join(temp_dir, f"{config_name}_shard_{i}.jsonl")
        temp_files[i] = open(fname, 'w', encoding='utf-8')
        temp_filenames[i] = fname

    # 核心：streaming=True 实现流式下载
    ds_stream = load_dataset(
        ds_name,
        config_name=config_name,
        split="train",
        streaming=True,  # 关键：开启流式模式
        cache_dir=cache_dir,
        # trust_remote_code=True,  # 已废弃，移除以避免报错
        token=token
    )
    
    current_shard = 0
    collected_in_shard = 0
    pbar = tqdm(total=sum(shard_required_samples), desc=f"🔄 采样{config_name}")

    try:
        for example in ds_stream:
            text = example[text_field].strip()
            if not text:
                continue
            
            sample = {"text": text, "sample_type": sample_type}
            # 写入当前分片的临时文件
            temp_files[current_shard].write(json.dumps(sample, ensure_ascii=False) + "\n")
            
            collected_in_shard += 1
            pbar.update(1)
            
            if collected_in_shard >= shard_required_samples[current_shard]:
                print(f"\n✅ {config_name} 分片{current_shard}完成（{collected_in_shard:,}条），切换分片{current_shard+1}")
                current_shard += 1
                collected_in_shard = 0
                # 关键：达标即停，不再下载后续数据
                if current_shard >= SPLIT_NUM:
                    print(f"🎉 {config_name} 所有分片收集完毕，停止下载")
                    break
    finally:
        # 关闭所有临时文件
        for f in temp_files.values():
            f.close()
        pbar.close()

    return temp_filenames

def merge_temp_shards_and_save(all_ds_temp_files, output_dir, output_prefix):
    """
    读取临时文件，合并、打乱并保存为Parquet
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n===== 合并分片 → 6个文件（单文件约{SINGLE_FILE_TOTAL/1024**3:.2f}GB） =====")
    
    for shard_idx in range(SPLIT_NUM):
        print(f"\n🔄 正在处理分片 {shard_idx}...")
        shard_all_samples = []
        
        # 读取该分片对应的所有子数据集临时文件
        for ds_config in DATASET_CONFIGS:
            config_name = ds_config["config_name"]
            temp_file = all_ds_temp_files[config_name][shard_idx]
            
            if os.path.exists(temp_file):
                print(f"   - 读取 {os.path.basename(temp_file)} ...")
                with open(temp_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            shard_all_samples.append(json.loads(line))
            else:
                print(f"   ⚠️ 警告：找不到文件 {temp_file}")

        # 单文件内打乱
        print(f"   🔀 正在打乱 {len(shard_all_samples):,} 条数据...")
        random.shuffle(shard_all_samples)
        
        # 转换为DataFrame
        df_shard = pd.DataFrame(shard_all_samples)
        
        # 保存
        output_file = os.path.join(output_dir, f"{output_prefix}{shard_idx:05d}.parquet")
        df_shard.to_parquet(output_file, index=False)
        
        # 统计
        actual_size = os.path.getsize(output_file)
        type_ratio = (df_shard["sample_type"].value_counts() / len(df_shard) * 100).round(2)
        
        print(f"✅ 最终文件{shard_idx+1}/{SPLIT_NUM}：{os.path.basename(output_file)}")
        print(f"   📏 实际大小：{actual_size/1024**3:.2f}GB")
        print(f"    样本总数：{len(df_shard):,}条")
        print(f"   ⚖️  内部比例：")
        for tp, ratio in type_ratio.items():
            print(f"      {tp:20s}：{ratio:.2f}%")

    # 全局统计
    total_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in os.listdir(output_dir) if f.endswith(".parquet"))
    print(f"\n===== 所有文件生成完成！ =====")
    print(f"✅ 全局总大小：{total_size/1024**3:.2f}GB（目标60GB）")

def main():
    args = parse_args()
    
    # 设置环境变量：HF Mirror
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
        print(f"🌍 使用 Hugging Face 镜像：{os.environ['HF_ENDPOINT']}")
    
    # 获取 Token：优先命令行参数，其次环境变量
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        print("🔑 已检测到 Hugging Face Token")
    else:
        print("⚠️ 未检测到 Token，部分受限数据集可能会下载失败")

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)

    # 步骤1：逐个处理指定子数据集，写入临时文件
    all_ds_temp_files = {}  # {config_name: {shard_idx: temp_file_path}}
    
    for ds_config in DATASET_CONFIGS:
        try:
            temp_files_map = stream_collect_dataset_to_temp(
                ds_config=ds_config,
                cache_dir=args.cache_dir,
                temp_dir=args.temp_dir,
                estimate_cnt=args.estimate_samples,
                token=hf_token
            )
            all_ds_temp_files[ds_config["config_name"]] = temp_files_map
        except Exception as e:
            print(f"❌ 处理{ds_config['name']}失败：{str(e)}")
            # 发生错误时清理已生成的临时文件可能更好，但为了调试保留
            raise e

    # 步骤2：合并临时文件
    try:
        merge_temp_shards_and_save(
            all_ds_temp_files=all_ds_temp_files,
            output_dir=args.output_dir,
            output_prefix=OUTPUT_PREFIX
        )
    finally:
        # 可选：清理临时目录
        # shutil.rmtree(args.temp_dir)
        print(f"\nℹ️ 临时文件保留在 {args.temp_dir}，如需释放空间请手动删除")

if __name__ == "__main__":
    main()
