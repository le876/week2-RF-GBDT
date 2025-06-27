#!/bin/bash

# Ackley函数GBDT回归GPU加速训练脚本

# 设置工作目录为项目根目录
cd "$(dirname "$0")/.."

# 创建必要的目录
mkdir -p data/raw data/processed models visualizations

echo "=========================================="
echo "开始Ackley函数GBDT GPU加速训练"
echo "=========================================="

# 检查GPU是否可用
echo "检查GPU状态..."
if nvidia-smi &> /dev/null; then
    nvidia-smi
    
    # 简化的GPU检测逻辑
    if python -c "
import sys
try:
    import cupy as cp
    # 简单测试CuPy功能
    test_array = cp.array([1, 2, 3])
    test_result = cp.sum(test_array)
    cp.asnumpy(test_result)
    print('CuPy测试成功')
    sys.exit(0)
except Exception as e:
    print(f'CuPy测试失败: {e}')
    sys.exit(1)
" &> /dev/null; then
        echo "✅ CuPy库可用，将使用GPU加速"
        USE_GPU=true
    else
        echo "⚠️ CuPy库不可用，将使用CPU模式"
        USE_GPU=false
    fi
else
    echo "⚠️ 未检测到NVIDIA GPU或驱动程序未正确安装"
    echo "将使用CPU模式运行"
    USE_GPU=false
fi

# 1. 数据预处理
echo "步骤1: 数据预处理"
python -c "
import sys
sys.path.append('.')
from data.load_data import load_data, preprocess_data, save_data

# 加载数据
data = load_data()

# 预处理数据
processed_data = preprocess_data(data)

# 保存处理后的数据
save_data(processed_data, 'data/processed/ackley_data.npz')
"

if [ $? -ne 0 ]; then
    echo "❌ 数据预处理失败，退出训练"
    exit 1
fi

# 2. 贝叶斯优化调优
echo "步骤2: 执行贝叶斯优化调优 (50次试验)"
if [ "$USE_GPU" = true ]; then
    echo "使用GPU加速..."
    USE_GPU_FLAG="True"
else
    echo "使用CPU模式..."
    USE_GPU_FLAG="False"
fi

python -c "
import sys
sys.path.append('.')
import os
import numpy as np
import time
from data.load_data import load_data, preprocess_data
from models.model_tuning import bayesian_optimization, save_tuning_results
from models.baseline_model import save_model

# 加载和预处理数据
data = load_data()
processed_data = preprocess_data(data)

# 记录开始时间
start_time = time.time()

# 贝叶斯优化调优
use_gpu = $USE_GPU_FLAG
print(f'GPU加速模式: {use_gpu}')

try:
    optuna_results = bayesian_optimization(
        processed_data,
        n_trials=50,
        timeout=None,
        use_gpu=use_gpu
    )

    # 计算耗时
    elapsed_time = time.time() - start_time
    print(f'\\n优化完成，总耗时: {elapsed_time:.2f}秒')

    # 保存调优结果
    os.makedirs('models/tuning_results', exist_ok=True)
    save_tuning_results(optuna_results, 'models/tuning_results/gpu_bayesian_opt')

    # 保存最佳模型
    save_model(optuna_results['best_model'], 'models/best_model_gpu.pkl', optuna_results['test_metrics'])

    # 打印最佳参数
    print('\\n最佳参数:')
    for param, value in optuna_results['best_params'].items():
        print(f'{param}: {value}')

    # 打印测试集Pearson系数
    test_pearson = optuna_results['test_metrics'].get('pearson_test', 0)
    print(f'\\n测试集Pearson系数: {test_pearson:.4f}')

    # 检查是否达到目标
    if test_pearson > 0.85:
        print('✅ 已达到目标 Pearson > 0.85!')
    else:
        print('❌ 未达到目标 Pearson > 0.85')
except Exception as e:
    print(f'\\n❌ 优化过程出错: {e}')
    print('将尝试使用基线模型')
    
    # 使用基线模型
    from models.baseline_model import train_baseline_model
    
    model, metrics = train_baseline_model(
        processed_data,
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=5
    )
    
    # 保存基线模型
    save_model(model, 'models/best_model_gpu.pkl', metrics)
    
    # 打印测试集Pearson系数
    test_pearson = metrics.get('pearson_test', 0)
    print(f'\\n基线模型测试集Pearson系数: {test_pearson:.4f}')
"

if [ $? -ne 0 ]; then
    echo "❌ 模型训练失败，退出训练"
    exit 1
fi

# 3. 最终评估
echo "步骤3: 最终模型评估"
python -c "
import sys
sys.path.append('.')
from data.load_data import load_data, preprocess_data
from models.baseline_model import load_model
from evaluation.metrics import evaluate_model, print_evaluation_report

# 加载和预处理数据
data = load_data()
processed_data = preprocess_data(data)

# 加载最佳模型
model = load_model('models/best_model_gpu.pkl')

# 评估模型
evaluation_results = evaluate_model(model, processed_data)

# 打印评估报告
print_evaluation_report(evaluation_results)
"

# 4. 生成可视化
echo "步骤4: 生成评估可视化"
python -c "
import sys
sys.path.append('.')
import matplotlib
# 设置matplotlib使用Agg后端，避免中文字体问题
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# 设置全局字体为无衬线字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

from data.load_data import load_data, preprocess_data
from models.baseline_model import load_model
from evaluation.residual_plot import generate_all_plots
import os

# 加载和预处理数据
data = load_data()
processed_data = preprocess_data(data)

# 加载最佳模型
model = load_model('models/best_model_gpu.pkl')

# 创建可视化目录
os.makedirs('visualizations/gpu_evaluation', exist_ok=True)

# 生成所有评估图表
generate_all_plots(model, processed_data, save_dir='visualizations/gpu_evaluation')

print('\\n✅ 可视化生成完成，所有图表已保存到 visualizations/gpu_evaluation/')
"

echo "=========================================="
echo "Ackley函数GBDT GPU加速训练完成"
echo "最佳模型已保存到: models/best_model_gpu.pkl"
echo "可视化结果已保存到: visualizations/gpu_evaluation/"
echo "==========================================" 