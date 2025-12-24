import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# export HF_ENDPOINT=https://hf-mirror.com

from huggingface_hub import snapshot_download

MODEL_CONFIGS = [
    # {
    #     "model_name": "lmsys/vicuna-7b-v1.3",
    #     "local_dir": "../Models/vicuna-7b-v1.3/"
    # },
    # {
    #     "model_name": "yuhuili/EAGLE-Vicuna-7B-v1.3",
    #     "local_dir": "../Models/EAGLE-Vicuna-7B-v1.3/"
    # }
    # {
    #     "model_name": "Qwen/Qwen3-8B",
    #     "local_dir": "../Models/Qwen3-8B/"
    # },
    # {
    #     "model_name": "Tengyunw/qwen3_8b_eagle3",
    #     "local_dir": "../Models/qwen3_8b_eagle3/"
    # }
    {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "local_dir": "./Models/Llama-3.1-8B-Instruct/"
    },
    {
        "model_name": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
        "local_dir": "./Models/EAGLE3-LLaMA3.1-Instruct-8B/"
    },
        {
        "model_name": "GSAI-ML/LLaDA-V",
        "local_dir": "./Models/LLaDA-V/"
    }
]
HF_TOKEN = ""
for config in MODEL_CONFIGS:
    print(f"开始下载模型：{config['model_name']}")
    # 创建本地目录（不存在则新建）
    os.makedirs(config["local_dir"], exist_ok=True)
    # 下载模型到指定路径
    snapshot_download(
        repo_id=config["model_name"],
        repo_type="model",
        local_dir=config["local_dir"],
        local_dir_use_symlinks=False,  # 禁用软链接，直接保存文件
        resume_download=True,          # 断点续传
        token = HF_TOKEN
    )
    print(f"模型 {config['model_name']} 下载完成，保存路径：{config['local_dir']}")