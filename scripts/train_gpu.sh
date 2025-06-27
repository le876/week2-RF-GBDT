# 使用基线模型
python -c "
import sys
sys.path.append('.')
import os
import numpy as np
import time
from data.load_data import load_data, preprocess_data
from models.model_tuning import bayesian_optimization, save_tuning_results
from models.baseline_model import train_baseline_model, save_model

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
    print('将使用高性能基线模型')
    
    # 使用基线模型
    model, metrics = train_baseline_model(
        processed_data,
        n_estimators=600,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='log2',
        bootstrap=True,
        oob_score=True,
        random_state=42
    )
    
    # 保存基线模型
    save_model(model, 'models/best_model_gpu.pkl', metrics)
    
    # 打印测试集Pearson系数
    test_pearson = metrics.get('pearson_test', 0)
    print(f'\\n基线模型测试集Pearson系数: {test_pearson:.4f}')
    
    # 检查是否达到目标
    if test_pearson > 0.85:
        print('✅ 已达到目标 Pearson > 0.85!')
    else:
        print('❌ 未达到目标 Pearson > 0.85')
" 