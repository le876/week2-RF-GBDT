#!/bin/bash

# 设置工作目录
cd "$(dirname "$0")"

# 创建必要的目录
mkdir -p models/tuning_results/bayesian_opt
mkdir -p visualizations

# 运行训练脚本
echo "开始训练Rosenbrock GBDT模型..."
python train.py --n_trials 50 --timeout 3600 --save_dir models/tuning_results/bayesian_opt

echo "训练完成！" 