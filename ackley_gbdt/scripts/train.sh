#!/bin/bash

# Ackley函数GBDT回归训练脚本

# 设置工作目录为项目根目录
cd "$(dirname "$0")/.."

# 创建必要的目录
mkdir -p data/raw data/processed models visualizations

# 默认参数
EPOCHS=1000
EARLY_STOP=10
TUNING_METHOD="bayesian"  # 可选: grid, bayesian
VISUALIZE=true

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --early_stop)
      EARLY_STOP="$2"
      shift 2
      ;;
    --tuning)
      TUNING_METHOD="$2"
      shift 2
      ;;
    --no-visualize)
      VISUALIZE=false
      shift
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

echo "=========================================="
echo "开始Ackley函数GBDT回归训练"
echo "=========================================="
echo "参数设置:"
echo "  最大迭代次数: $EPOCHS"
echo "  早停轮数: $EARLY_STOP"
echo "  调优方法: $TUNING_METHOD"
echo "  可视化: $VISUALIZE"
echo "=========================================="

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
  echo "数据预处理失败，退出训练"
  exit 1
fi

# 2. 数据可视化
if [ "$VISUALIZE" = true ]; then
  echo "步骤2: 数据可视化"
  python -c "
import sys
sys.path.append('.')
from data.load_data import load_data, preprocess_data
from data.visualize_data import visualize_all

# 加载和预处理数据
data = load_data()
processed_data = preprocess_data(data)

# 生成数据可视化
visualize_all(processed_data, save_dir='visualizations/data')
"
  
  if [ $? -ne 0 ]; then
    echo "数据可视化失败，但继续训练"
  fi
fi

# 3. 训练基线模型
echo "步骤3: 训练基线模型"
python -c "
import sys
sys.path.append('.')
from data.load_data import load_data, preprocess_data
from models.baseline_model import cross_validate_model, train_baseline_model, save_model
import os

# 加载和预处理数据
data = load_data()
processed_data = preprocess_data(data)

# 交叉验证
cv_results = cross_validate_model(
    processed_data,
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    n_folds=5
)

# 训练基线模型
model, metrics = train_baseline_model(
    processed_data,
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3
)

# 保存模型
os.makedirs('models', exist_ok=True)
save_model(model, 'models/baseline_model.pkl', metrics)
"

if [ $? -ne 0 ]; then
  echo "基线模型训练失败，退出训练"
  exit 1
fi

# 4. 超参数调优
echo "步骤4: 超参数调优 (方法: $TUNING_METHOD)"
if [ "$TUNING_METHOD" = "grid" ]; then
  # 网格搜索调优
  python -c "
import sys
sys.path.append('.')
from data.load_data import load_data, preprocess_data
from models.model_tuning import grid_search_tuning, save_tuning_results
import os

# 加载和预处理数据
data = load_data()
processed_data = preprocess_data(data)

# 网格搜索调优
grid_results = grid_search_tuning(
    processed_data,
    param_grid={
        'n_estimators': [500, 1000, 1500],
        'learning_rate': [0.05, 0.1, 0.15],
        'max_depth': [4, 5, 6],
        'subsample': [0.8, 0.9],
        'min_samples_split': [2, 5]
    },
    cv=5
)

# 保存调优结果
os.makedirs('models/tuning_results', exist_ok=True)
save_tuning_results(grid_results, 'models/tuning_results/grid_search')

# 保存最佳模型
from models.baseline_model import save_model
save_model(grid_results['best_model'], 'models/best_model.pkl', grid_results['test_metrics'])
"
else
  # 贝叶斯优化调优
  python -c "
import sys
sys.path.append('.')
from data.load_data import load_data, preprocess_data
from models.model_tuning import bayesian_optimization, save_tuning_results
import os

# 加载和预处理数据
data = load_data()
processed_data = preprocess_data(data)

# 贝叶斯优化调优
optuna_results = bayesian_optimization(
    processed_data,
    n_trials=50,
    timeout=None
)

# 保存调优结果
os.makedirs('models/tuning_results', exist_ok=True)
save_tuning_results(optuna_results, 'models/tuning_results/bayesian_opt')

# 保存最佳模型
from models.baseline_model import save_model
save_model(optuna_results['best_model'], 'models/best_model.pkl', optuna_results['test_metrics'])
"
fi

if [ $? -ne 0 ]; then
  echo "超参数调优失败，使用基线模型继续"
  cp models/baseline_model.pkl models/best_model.pkl
fi

# 5. 最终评估
echo "步骤5: 最终模型评估"
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
try:
    model = load_model('models/best_model.pkl')
except FileNotFoundError:
    model = load_model('models/baseline_model.pkl')

# 评估模型
evaluation_results = evaluate_model(model, processed_data)

# 打印评估报告
print_evaluation_report(evaluation_results)
"

# 6. 生成可视化
if [ "$VISUALIZE" = true ]; then
  echo "步骤6: 生成评估可视化"
  python -c "
import sys
sys.path.append('.')
from data.load_data import load_data, preprocess_data
from models.baseline_model import load_model
from evaluation.residual_plot import generate_all_plots
import os

# 加载和预处理数据
data = load_data()
processed_data = preprocess_data(data)

# 加载最佳模型
try:
    model = load_model('models/best_model.pkl')
except FileNotFoundError:
    model = load_model('models/baseline_model.pkl')

# 创建可视化目录
os.makedirs('visualizations/evaluation', exist_ok=True)

# 生成所有评估图表
generate_all_plots(model, processed_data, save_dir='visualizations/evaluation')
"
  
  if [ $? -ne 0 ]; then
    echo "评估可视化生成失败，但训练已完成"
  fi
fi

echo "=========================================="
echo "Ackley函数GBDT回归训练完成"
echo "最佳模型已保存到: models/best_model.pkl"
echo "可视化结果已保存到: visualizations/"
echo "==========================================" 