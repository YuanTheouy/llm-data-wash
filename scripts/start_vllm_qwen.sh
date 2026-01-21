#!/bin/bash

# 使用部分显卡 (4张)，因为Qwen2.5-VL-7B的头数(28)无法被8整除，但可以被4整除
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Qwen2.5-VL-7B-Instruct 启动脚本
# 使用 --tensor-parallel-size 4
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/Models/Qwen2.5-VL-7B-Instruct \
  --served-model-name /workspace/Models/Qwen2.5-VL-7B-Instruct \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --port 8000 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --max-num-seqs 256 \
  --limit-mm-per-prompt '{"image": 1}' \
  --enforce-eager
