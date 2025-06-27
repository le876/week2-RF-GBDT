#!/bin/bash

# 安装GPU加速依赖的脚本

echo "=========================================="
echo "开始安装GPU加速依赖"
echo "=========================================="

# 检查CUDA是否可用
echo "检查CUDA环境..."
nvcc -V
nvidia-smi

# 安装RAPIDS cuML库和CuPy
echo "安装GPU加速库..."
pip install cuml-cu12 cupy-cuda12x

# 验证安装
echo "验证安装..."
python -c "
try:
    import cuml
    import cupy
    print('✅ GPU加速库安装成功!')
    print(f'cuML版本: {cuml.__version__}')
    print(f'CuPy版本: {cupy.__version__}')
except ImportError as e:
    print(f'❌ GPU加速库安装失败: {e}')
"

echo "=========================================="
echo "GPU加速依赖安装完成"
echo "现在可以运行: bash scripts/train_gpu.sh"
echo "==========================================" 