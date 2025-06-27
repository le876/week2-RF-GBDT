#!/bin/bash

# 安装Ackley随机森林模型所需的依赖项

echo "=========================================="
echo "安装Ackley随机森林模型依赖项"
echo "=========================================="

# 检查conda环境
if [ -z "$CONDA_PREFIX" ]; then
    echo "❌ 未检测到激活的conda环境"
    echo "请先激活conda环境，例如: conda activate mlenv"
    exit 1
else
    echo "✅ 使用conda环境: $CONDA_PREFIX"
fi

# 安装基本依赖项
echo "安装基本依赖项..."
conda install -y numpy=1.24.3 pandas scikit-learn matplotlib seaborn scipy statsmodels joblib

# 检查GPU是否可用
echo "检查GPU状态..."
if nvidia-smi &> /dev/null; then
    echo "✅ 检测到NVIDIA GPU"
    nvidia-smi
    
    # 检测CUDA版本
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d'.' -f1,2)
    echo "检测到CUDA版本: $CUDA_VERSION"
    
    # 安装CuPy
    echo "安装GPU加速库..."
    if [[ "$CUDA_VERSION" == "12."* ]]; then
        echo "安装CUDA 12.x兼容的CuPy..."
        conda install -y -c conda-forge cupy cudatoolkit=12.0
    elif [[ "$CUDA_VERSION" == "11."* ]]; then
        echo "安装CUDA 11.x兼容的CuPy..."
        conda install -y -c conda-forge cupy cudatoolkit=11.8
    else
        echo "⚠️ 不支持的CUDA版本: $CUDA_VERSION"
        echo "将尝试安装通用版本..."
        conda install -y -c conda-forge cupy
    fi
    
    # 验证CuPy安装
    if python -c "import cupy as cp; print('CuPy版本:', cp.__version__); test_array = cp.array([1, 2, 3]); print('测试成功!')" &> /dev/null; then
        echo "✅ CuPy安装成功"
    else
        echo "❌ CuPy安装失败"
        echo "将使用CPU模式"
    fi
else
    echo "⚠️ 未检测到NVIDIA GPU或驱动程序未正确安装"
    echo "将使用CPU模式"
fi

# 安装Optuna
echo "安装Optuna优化库..."
conda install -y -c conda-forge optuna

# 验证安装
echo "验证安装..."
python -c "
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import statsmodels
import joblib
import optuna
from sklearn.ensemble import RandomForestRegressor
print('✅ 所有基本依赖项已安装')
try:
    import cupy as cp
    print('✅ GPU加速库已安装')
except ImportError:
    print('⚠️ GPU加速库未安装，将使用CPU模式')
"

echo "=========================================="
echo "依赖项安装完成"
echo "=========================================="
echo "现在可以运行训练脚本: bash scripts/train_gpu.sh" 