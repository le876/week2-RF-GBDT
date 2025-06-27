#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
评估指标模块，实现各种评估指标的计算。
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

def calculate_regression_metrics(y_true: np.ndarray, 
                                y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算回归评估指标。
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        包含各种评估指标的字典
    """
    # 均方误差
    mse = mean_squared_error(y_true, y_pred)
    
    # 均方根误差
    rmse = np.sqrt(mse)
    
    # 平均绝对误差
    mae = mean_absolute_error(y_true, y_pred)
    
    # 决定系数
    r2 = r2_score(y_true, y_pred)
    
    # Pearson相关系数
    pearson, p_value = pearsonr(y_true, y_pred)
    
    # 平均绝对百分比误差
    # 避免除以零
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
    
    # 自定义加权指标: 0.7*R2 + 0.3*Pearson
    custom_metric = 0.7 * r2 + 0.3 * pearson
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson': pearson,
        'pearson_p_value': p_value,
        'mape': mape,
        'custom_metric': custom_metric
    }

def calculate_residuals(y_true: np.ndarray, 
                       y_pred: np.ndarray) -> np.ndarray:
    """
    计算残差。
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        残差数组
    """
    return y_true - y_pred

def calculate_relative_errors(y_true: np.ndarray, 
                             y_pred: np.ndarray) -> np.ndarray:
    """
    计算相对误差。
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        相对误差数组
    """
    # 避免除以零
    epsilon = 1e-10
    return np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon)) * 100

def calculate_metrics_by_range(y_true: np.ndarray, 
                              y_pred: np.ndarray, 
                              n_bins: int = 5) -> pd.DataFrame:
    """
    按真实值范围计算评估指标。
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        n_bins: 分箱数量
        
    Returns:
        包含各个范围评估指标的DataFrame
    """
    # 计算分箱边界
    bins = np.linspace(np.min(y_true), np.max(y_true), n_bins + 1)
    
    # 初始化结果列表
    results = []
    
    # 对每个分箱计算指标
    for i in range(n_bins):
        # 获取当前分箱的索引
        bin_indices = (y_true >= bins[i]) & (y_true < bins[i + 1])
        
        # 如果当前分箱有数据
        if np.sum(bin_indices) > 1:
            bin_y_true = y_true[bin_indices]
            bin_y_pred = y_pred[bin_indices]
            
            # 计算指标
            metrics = calculate_regression_metrics(bin_y_true, bin_y_pred)
            
            # 添加分箱信息
            metrics['bin_start'] = bins[i]
            metrics['bin_end'] = bins[i + 1]
            metrics['sample_count'] = np.sum(bin_indices)
            
            results.append(metrics)
    
    # 转换为DataFrame
    return pd.DataFrame(results)

def custom_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    实现R^2与Pearson的加权指标。
    
    Args:
        y_true: 真实值数组
        y_pred: 预测值数组
        
    Returns:
        float: 0.7*R2 + 0.3*Pearson
    """
    r2 = r2_score(y_true, y_pred)
    pearson, _ = pearsonr(y_true, y_pred)
    
    return 0.7 * r2 + 0.3 * pearson

def evaluate_model(model, data: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """
    全面评估模型性能。
    
    Args:
        model: 训练好的模型
        data: 包含'X_train', 'y_train', 'X_test', 'y_test'的数据字典
        
    Returns:
        包含训练集和测试集评估指标的嵌套字典
    """
    results = {}
    
    # 在训练集上评估
    X_train, y_train = data['X_train'], data['y_train']
    y_train_pred = model.predict(X_train)
    train_metrics = calculate_regression_metrics(y_train, y_train_pred)
    results['train'] = train_metrics
    
    # 在测试集上评估（如果有）
    if 'X_test' in data and 'y_test' in data:
        X_test, y_test = data['X_test'], data['y_test']
        y_test_pred = model.predict(X_test)
        test_metrics = calculate_regression_metrics(y_test, y_test_pred)
        results['test'] = test_metrics
    
    return results

def print_evaluation_report(metrics: Dict[str, Dict[str, float]]) -> None:
    """
    打印评估报告。
    
    Args:
        metrics: 包含训练集和测试集评估指标的嵌套字典
    """
    print("\n" + "="*50)
    print("模型评估报告")
    print("="*50)
    
    for dataset_name, dataset_metrics in metrics.items():
        print(f"\n{dataset_name.capitalize()}集评估:")
        print("-"*30)
        
        for metric_name, metric_value in dataset_metrics.items():
            if metric_name != 'pearson_p_value':  # 不打印p值
                print(f"{metric_name.upper()}: {metric_value:.4f}")
        
        # 特别强调Pearson系数
        pearson = dataset_metrics.get('pearson', 0)
        if pearson > 0.85:
            print(f"\n✅ Pearson系数 ({pearson:.4f}) > 0.85，达到目标!")
        else:
            print(f"\n❌ Pearson系数 ({pearson:.4f}) <= 0.85，未达到目标。")
    
    # 添加过拟合检测
    check_overfitting(metrics)
    
    print("\n" + "="*50)

def check_overfitting(metrics: Dict[str, Dict[str, float]]) -> None:
    """
    检测模型是否存在过拟合问题。
    
    Args:
        metrics: 包含训练集和测试集评估指标的嵌套字典
    """
    if 'train' in metrics and 'test' in metrics:
        train_pearson = metrics['train']['pearson']
        test_pearson = metrics['test']['pearson']
        
        ratio = test_pearson / train_pearson
        
        print("\n" + "-"*30)
        print("过拟合检测:")
        print(f"训练集/测试集Pearson比率: {ratio:.4f}")
        
        if ratio < 0.85:
            print("⚠️ 检测到潜在过拟合，建议:")
            print("  - 增加正则化参数（降低max_depth、增加min_samples_split）")
            print("  - 使用子采样（subsample < 1.0）")
            print("  - 减小学习率并增加树的数量")
        else:
            print("✅ 未检测到明显过拟合")
        
        print("-"*30)

if __name__ == "__main__":
    # 示例用法
    import sys
    sys.path.append('..')
    from data.load_data import load_data, preprocess_data
    from models.baseline_model import load_model
    
    # 加载和预处理数据
    data = load_data()
    processed_data = preprocess_data(data)
    
    # 假设已经有训练好的模型
    try:
        model = load_model("../trained_models/best_model.pkl")
        
        # 评估模型
        evaluation_results = evaluate_model(model, processed_data)
        
        # 打印评估报告
        print_evaluation_report(evaluation_results)
        
        # 检查过拟合
        check_overfitting(evaluation_results)
        
    except FileNotFoundError:
        print("模型文件不存在，请先训练模型。") 