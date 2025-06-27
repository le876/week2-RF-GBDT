#!/bin/bash

# 设置工作目录
cd "$(dirname "$0")"

# 创建必要的目录
mkdir -p models/results
mkdir -p visualizations

# 运行训练脚本
echo "开始训练Rosenbrock随机森林模型..."

# 尝试不同的参数组合
echo "尝试参数组合1: 500棵树，最大深度30"
python train.py --n_estimators 500 --max_depth 30 --save_dir models/results

echo "尝试参数组合2: 800棵树，最大深度40"
python train.py --n_estimators 800 --max_depth 40 --save_dir models/results

echo "尝试参数组合3: 1000棵树，最大深度None"
python train.py --n_estimators 1000 --max_depth None --save_dir models/results

echo "训练完成！" 