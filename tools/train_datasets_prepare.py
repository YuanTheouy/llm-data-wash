import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset
import json
import numpy as np  # 处理可能的numpy类型（避免二次报错）

# 1. 加载数据集（保留你的原有配置）
# ds = load_dataset(
#     "anon8231489123/ShareGPT_Vicuna_unfiltered",
#     data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
#     cache_dir="/workspace/TrainingDatasets/ShareGPT"
# )

# ds = load_dataset(
#     "HuggingFaceH4/ultrachat_200k",
#     cache_dir="/workspace/TrainingDatasets/ultrachat_200k/"
# )

ds = load_dataset(
    "open-r1/OpenThoughts-114k-math",
    cache_dir="/workspace/TrainingDatasets/OpenThoughts-114k-math/"
)

# # 2. 关键修复1：将Dataset转为原生Python列表
# train_data = ds["train"].to_list()

# # 3. 关键修复2：递归转换numpy类型为原生Python类型（避免数字类型序列化失败）
# def convert_numpy_to_python(obj):
#     if isinstance(obj, np.integer):
#         return int(obj)
#     elif isinstance(obj, np.floating):
#         return float(obj)
#     elif isinstance(obj, np.ndarray):
#         return obj.tolist()
#     elif isinstance(obj, dict):
#         return {k: convert_numpy_to_python(v) for k, v in obj.items()}
#     elif isinstance(obj, list):
#         return [convert_numpy_to_python(i) for i in obj]
#     else:
#         return obj

# train_data = convert_numpy_to_python(train_data)

# # 4. 保存为JSON文件
# with open("/workspace/TrainingDatasets/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json", "w", encoding="utf-8") as f:
#     json.dump(train_data, f, ensure_ascii=False, indent=2)