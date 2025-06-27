#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
评估指标模块，实现模型评估和指标计算功能。
"""

import logging
import numpy as np
from typing import Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算回归评估指标。
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        包含各种评估指标的字典
    """
    # 计算基本指标
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    
    # 计算平均绝对百分比误差 (MAPE)
    # 避免除以零
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    # 自定义指标：R²和Pearson系数的调和平均
    if r2 > 0 and pearson_corr > 0:
        custom_metric = 2 * (r2 * pearson_corr) / (r2 + pearson_corr)
    else:
        custom_metric = 0
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson': pearson_corr,
        'mape': mape,
        'custom_metric': custom_metric
    }

def evaluate_model(model: Any, data: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """
    评估模型在训练集和测试集上的性能。
    
    Args:
        model: 训练好的模型
        data: 包含'X_train'、'y_train'、'X_test'和'y_test'的数据字典
        
    Returns:
        包含训练集和测试集评估指标的字典
    """
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    
    # 在训练集上评估
    y_pred_train = model.predict(X_train)
    train_metrics = calculate_metrics(y_train, y_pred_train)
    
    # 在测试集上评估
    y_pred_test = model.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_pred_test)
    
    # 计算过拟合指标
    overfitting_ratio = train_metrics['pearson'] / test_metrics['pearson'] if test_metrics['pearson'] > 0 else float('inf')
    
    return {
        'train': train_metrics,
        'test': test_metrics,
        'overfitting_ratio': overfitting_ratio
    }

def print_evaluation_report(evaluation_results: Dict[str, Dict[str, float]]) -> None:
    """
    打印评估报告。
    
    Args:
        evaluation_results: 评估结果字典
    """
    train_metrics = evaluation_results['train']
    test_metrics = evaluation_results['test']
    overfitting_ratio = evaluation_results['overfitting_ratio']
    
    print("\n" + "=" * 50)
    print("模型评估报告")
    print("=" * 50 + "\n")
    
    print("Train集评估:")
    print("-" * 30)
    print(f"MSE: {train_metrics['mse']:.4f}")
    print(f"RMSE: {train_metrics['rmse']:.4f}")
    print(f"MAE: {train_metrics['mae']:.4f}")
    print(f"R2: {train_metrics['r2']:.4f}")
    print(f"PEARSON: {train_metrics['pearson']:.4f}")
    print(f"MAPE: {train_metrics['mape']:.4f}")
    print(f"CUSTOM_METRIC: {train_metrics['custom_metric']:.4f}")
    
    # 检查是否达到目标
    if train_metrics['pearson'] > 0.85:
        print("\n✅ Pearson系数 ({:.4f}) > 0.85，达到目标!".format(train_metrics['pearson']))
    else:
        print("\n❌ Pearson系数 ({:.4f}) < 0.85，未达到目标!".format(train_metrics['pearson']))
    
    print("\nTest集评估:")
    print("-" * 30)
    print(f"MSE: {test_metrics['mse']:.4f}")
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    print(f"MAE: {test_metrics['mae']:.4f}")
    print(f"R2: {test_metrics['r2']:.4f}")
    print(f"PEARSON: {test_metrics['pearson']:.4f}")
    print(f"MAPE: {test_metrics['mape']:.4f}")
    print(f"CUSTOM_METRIC: {test_metrics['custom_metric']:.4f}")
    
    # 检查是否达到目标
    if test_metrics['pearson'] > 0.85:
        print("\n✅ Pearson系数 ({:.4f}) > 0.85，达到目标!".format(test_metrics['pearson']))
    else:
        print("\n❌ Pearson系数 ({:.4f}) < 0.85，未达到目标!".format(test_metrics['pearson']))
    
    print("\n" + "-" * 30)
    print("过拟合检测:")
    print(f"训练集/测试集Pearson比率: {overfitting_ratio:.4f}")
    
    if overfitting_ratio > 1.2:
        print("⚠️ 检测到过拟合 (比率 > 1.2)")
        print("建议: 增加正则化，减少模型复杂度，或增加训练数据")
    else:
        print("✅ 未检测到明显过拟合")
    
    print("-" * 30 + "\n")
    print("=" * 50)

if __name__ == "__main__":
    # 测试评估功能
    from sklearn.ensemble import RandomForestRegressor
    from data.load_data import load_data, preprocess_data
    
    # 加载数据
    data = load_data()
    processed_data = preprocess_data(data)
    
    # 训练简单模型
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(processed_data['X_train'], processed_data['y_train'])
    
    # 评估模型
    evaluation_results = evaluate_model(model, processed_data)
    
    # 打印评估报告
    print_evaluation_report(evaluation_results) 