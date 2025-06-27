#!/bin/bash

# 在当前mlenv环境中安装依赖的脚本

echo "=========================================="
echo "开始在mlenv环境中安装依赖"
echo "=========================================="

# 安装基本依赖
echo "安装基本依赖..."
pip install numpy pandas scikit-learn matplotlib seaborn joblib optuna

# 检查GPU是否可用
echo "检查CUDA环境..."
if nvidia-smi &> /dev/null; then
    echo "✅ 检测到NVIDIA GPU"
    
    # 尝试安装GPU加速库
    echo "尝试安装GPU加速库..."
    
    # 使用pip安装cupy (适用于CUDA 12.x)
    pip install --extra-index-url https://pypi.nvidia.com cupy-cuda12x
    
    # 尝试安装CUDA运行时库
    echo "尝试安装CUDA运行时库..."
    if [ -f "/usr/local/cuda/lib64/libnvrtc.so.12" ]; then
        echo "✅ CUDA运行时库已存在"
    else
        echo "⚠️ 未找到CUDA运行时库，将使用CPU模式"
        echo "提示: 可以通过安装CUDA工具包获取完整的运行时库"
    fi
    
    # 验证安装
    echo "验证GPU库安装..."
    python -c "
try:
    import cupy as cp
    # 测试CuPy功能
    try:
        test_array = cp.array([1, 2, 3])
        test_result = cp.sum(test_array)
        test_cpu = cp.asnumpy(test_result)
        print('✅ CuPy库安装成功并正常工作!')
        print(f'CuPy版本: {cp.__version__}')
    except Exception as e:
        print(f'⚠️ CuPy库安装成功但无法正常工作: {e}')
        print('将使用CPU模式运行')
except ImportError as e:
    print(f'⚠️ GPU加速库安装失败: {e}')
    print('将使用CPU模式运行')
"
else
    echo "⚠️ 未检测到NVIDIA GPU或驱动程序未正确安装"
    echo "将使用CPU模式运行"
fi

# 验证基本环境
echo "验证基本环境..."
python -c "
import numpy
import pandas
import sklearn
import matplotlib
import optuna
print('✅ 基本依赖安装成功!')
print(f'NumPy版本: {numpy.__version__}')
print(f'scikit-learn版本: {sklearn.__version__}')
print(f'Optuna版本: {optuna.__version__}')
"

echo "=========================================="
echo "依赖安装完成"
echo "现在可以运行: bash scripts/train_gpu.sh"
echo "==========================================" 