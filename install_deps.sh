#!/bin/bash
set -e  # 遇到错误立即停止，避免装到一半出问题

# ===================== 第一步：创建并激活虚拟环境 =====================
ENV_PATH="~/venvs/data"
# 展开波浪号为绝对路径（避免source识别不到）
ENV_PATH_ABS=$(eval echo "$ENV_PATH")

# 如果环境不存在，创建；如果存在，直接激活
if [ ! -d "$ENV_PATH_ABS" ]; then
    echo "🔧 正在创建虚拟环境：$ENV_PATH_ABS"
    python -m venv "$ENV_PATH_ABS"
else
    echo "✅ 虚拟环境已存在，直接激活"
fi

# 激活虚拟环境（兼容bash/zsh）
echo "🔧 正在激活虚拟环境..."
source "$ENV_PATH_ABS/bin/activate"

# 验证激活是否成功
echo "✅ 当前Python路径：$(which python)"
echo "✅ 当前Pip路径：$(which pip)"

# ===================== 第二步：装除vllm外的所有依赖 =====================
echo "🔧 开始安装基础依赖（避开vllm，防止NCCL卡点）..."
pip install \
  -r <(grep -v "vllm" requirements.txt) \
  --no-deps \
  --no-cache-dir \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

# ===================== 第三步：单独装vllm（内置NCCL） =====================
echo "🔧 开始安装vllm（单机多卡版，内置NCCL）..."
pip install "vllm==0.4.0" \
  --extra-index-url https://download.vllm.ai/whl/cu121 \
  --no-deps \
  --no-cache-dir \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

# ===================== 第四步：验证安装结果 =====================
echo -e "\n======= 🎉 依赖安装完成，验证核心包 ======="
python -c "
import sys
import requests, torch, vllm, huggingface_hub

print(f'Python版本：{sys.version.split()[0]}')
print(f'requests版本：{requests.__version__}')
print(f'torch版本：{torch.__version__}')
print(f'vllm版本：{vllm.__version__}')
print(f'huggingface_hub版本：{huggingface_hub.__version__}')
print('\n✅ 所有依赖安装成功！虚拟环境路径：$ENV_PATH_ABS')
"

echo -e "\n💡 下次使用只需执行：source $ENV_PATH_ABS/bin/activate"