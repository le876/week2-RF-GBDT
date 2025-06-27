#!/bin/bash

# 设置环境和安装依赖的脚本

echo "=========================================="
echo "开始设置GBDT训练环境"
echo "=========================================="

# 检查conda是否可用
if ! command -v conda &> /dev/null; then
    echo "❌ 未找到conda命令，请先安装Anaconda或Miniconda"
    exit 1
fi

# 创建新的conda环境
echo "创建conda环境 'ackley_env'..."
conda create -y -n ackley_env python=3.9

# 激活环境
echo "激活conda环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ackley_env

# 安装基本依赖
echo "安装基本依赖..."
conda install -y -c conda-forge numpy pandas scikit-learn matplotlib seaborn joblib optuna

# 检查GPU是否可用
echo "检查CUDA环境..."
if nvidia-smi &> /dev/null; then
    echo "✅ 检测到NVIDIA GPU"
    
    # 尝试安装GPU加速库
    echo "尝试安装GPU加速库..."
    
    # 方法1: 使用conda安装rapids
    conda install -y -c rapidsai -c conda-forge -c nvidia \
        cuml=24.6 cupy python=3.9 cuda-version=12.0
    
    # 验证安装
    echo "验证GPU库安装..."
    python -c "
try:
    import cuml
    import cupy
    print('✅ GPU加速库安装成功!')
    print(f'cuML版本: {cuml.__version__}')
    print(f'CuPy版本: {cupy.__version__}')
except ImportError as e:
    print(f'⚠️ GPU加速库安装部分失败: {e}')
    print('将使用CPU模式运行')
"
else
    echo "⚠️ 未检测到NVIDIA GPU或驱动程序未正确安装"
    echo "将使用CPU模式运行"
fi

# 安装项目依赖
echo "安装项目依赖..."
pip install -r requirements.txt

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
echo "环境设置完成"
echo "使用方法: conda activate ackley_env"
echo "然后运行: bash scripts/train.sh 或 bash scripts/train_gpu.sh"
echo "==========================================" 