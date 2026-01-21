import os
import json
import glob
import argparse
from huggingface_hub import snapshot_download
import pyarrow.parquet as pq
import pandas as pd

# 设置HF镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def parse_args():
    parser = argparse.ArgumentParser(description="下载并处理 VQAv2 数据集")
    parser.add_argument("--output_dir", type=str, default="datasets/VQAv2", help="数据集保存目录")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 配置路径
    # 如果用户提供了绝对路径，就直接使用；如果是相对路径，则相对于当前工作目录
    dataset_dir = os.path.abspath(args.output_dir)
    
    image_dir = os.path.join(dataset_dir, "images")
    output_file = os.path.join(dataset_dir, "vqa_v2_validation.jsonl")
    
    print(f"数据将保存到: {dataset_dir}")
    os.makedirs(image_dir, exist_ok=True)
    
    print("正在直接下载/同步 Parquet 文件 (绕过 datasets 库)...")
    # 直接下载仓库文件，避免 datasets 库的复杂处理导致 Segfault
    # allow_patterns 只下载验证集的 parquet 文件
    try:
        local_dir = snapshot_download(
            repo_id="lmms-lab/VQAv2",
            repo_type="dataset",
            allow_patterns="*validation*",  # 假设文件名包含 validation
            local_dir=os.path.join(dataset_dir, "raw_files"),
            resume_download=True
        )
    except Exception as e:
        print(f"下载失败: {e}")
        # 如果下载失败，尝试更通用的 pattern
        local_dir = snapshot_download(
            repo_id="lmms-lab/VQAv2",
            repo_type="dataset",
            allow_patterns="*.parquet",
            local_dir=os.path.join(dataset_dir, "raw_files"),
            resume_download=True
        )

    print(f"Parquet 文件已准备在: {local_dir}")
    
    # 查找所有 parquet 文件
    parquet_files = []
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            if file.endswith(".parquet") and "validation" in file:
                parquet_files.append(os.path.join(root, file))
    
    if not parquet_files:
        print("未找到 validation 相关的 parquet 文件，尝试所有 parquet 文件...")
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                if file.endswith(".parquet"):
                    parquet_files.append(os.path.join(root, file))
    
    parquet_files.sort()
    print(f"找到 {len(parquet_files)} 个 Parquet 文件，开始处理...")

    count = 0
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for p_file in parquet_files:
            print(f"正在处理文件: {os.path.basename(p_file)}")
            try:
                # 使用 pyarrow 逐批读取，极大降低内存占用且避免 crash
                parquet_file = pq.ParquetFile(p_file)
                for batch in parquet_file.iter_batches(batch_size=100):
                    df = batch.to_pandas()
                    
                    for idx, row in df.iterrows():
                        try:
                            # 提取图像数据
                            # Parquet 中图像通常存储为 struct (bytes, path) 或直接 bytes
                            image_data = row.get("image")
                            
                            if image_data is None:
                                continue
                                
                            image_bytes = None
                            
                            # 解析图像数据结构
                            if isinstance(image_data, dict):
                                image_bytes = image_data.get("bytes")
                            elif isinstance(image_data, bytes):
                                image_bytes = image_data
                            
                            if not image_bytes:
                                # 如果没有 bytes，尝试从 path 读取（不太可能在 HF parquet 中，但以防万一）
                                print(f"警告: 记录缺少图像 bytes，跳过")
                                continue

                            image_id = row.get("image_id", f"unk_{count}")
                            image_filename = f"image_{image_id}.jpg"
                            image_path = os.path.join(image_dir, image_filename)
                            
                            # 写入图像文件
                            with open(image_path, "wb") as img_f:
                                img_f.write(image_bytes)
                            
                            # 构建记录
                            record = {
                                "question_id": row.get("question_id"),
                                "image_id": image_id,
                                "question": row.get("question"),
                                "question_type": row.get("question_type"),
                                "original_answers": row.get("answers").tolist() if hasattr(row.get("answers"), "tolist") else row.get("answers"),
                                "image_path": image_path
                            }
                            
                            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
                            
                            count += 1
                            if count % 1000 == 0:
                                print(f"已处理 {count} 条记录", end='\r')
                                
                        except Exception as inner_e:
                            # print(f"行处理错误: {inner_e}")
                            continue
                            
            except Exception as e:
                print(f"文件处理错误 {p_file}: {e}")
                continue

    print(f"\n全部完成！共处理 {count} 条记录。")

if __name__ == "__main__":
    main()
