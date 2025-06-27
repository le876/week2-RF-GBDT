#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
比较不同特征工程方法的结果
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from typing import Dict, List, Any
import joblib

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置matplotlib字体
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.unicode_minus': False
})

def load_results(result_dir: str) -> Dict[str, Any]:
    """
    加载优化结果
    
    Args:
        result_dir: 结果目录路径
        
    Returns:
        包含结果数据的字典
    """
    try:
        # 查找最新的结果目录
        subdirs = [os.path.join(result_dir, d) for d in os.listdir(result_dir) 
                  if os.path.isdir(os.path.join(result_dir, d))]
        if not subdirs:
            logger.warning(f"No result directories found in {result_dir}")
            return {}
        
        # 按修改时间排序，选择最新的
        latest_dir = max(subdirs, key=os.path.getmtime)
        logger.info(f"Loading results from {latest_dir}")
        
        # 加载指标数据
        metrics_path = os.path.join(latest_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
        else:
            # 尝试从test_metrics.csv加载
            test_metrics_path = os.path.join(latest_dir, 'test_metrics.csv')
            if os.path.exists(test_metrics_path):
                metrics_df = pd.read_csv(test_metrics_path)
                metrics = metrics_df.iloc[0].to_dict()
            else:
                metrics = {}
        
        # 确保特征数量信息存在
        if 'feature_count' not in metrics:
            # 尝试从best_model.pkl加载
            model_path = os.path.join(latest_dir, 'best_model.pkl')
            if os.path.exists(model_path):
                try:
                    model = joblib.load(model_path)
                    metrics['feature_count'] = model.n_features_in_
                except Exception as e:
                    logger.warning(f"Failed to load model: {str(e)}")
        
        # 加载试验数据
        trials_path = os.path.join(latest_dir, 'trials_history.csv')
        if os.path.exists(trials_path):
            trials_df = pd.read_csv(trials_path)
        else:
            trials_df = pd.DataFrame()
        
        # 加载趋势数据
        trend_path = os.path.join(latest_dir, 'metrics_trend.csv')
        if os.path.exists(trend_path):
            trend_df = pd.read_csv(trend_path)
        else:
            trend_df = pd.DataFrame()
        
        return {
            'metrics': metrics,
            'trials_df': trials_df,
            'trend_df': trend_df
        }
    except Exception as e:
        logger.error(f"Failed to load results: {str(e)}")
        return {}

def compare_results(results_dict: Dict[str, Dict[str, Any]]) -> None:
    """
    比较不同特征工程方法的结果
    
    Args:
        results_dict: 包含不同方法结果的字典
    """
    # 提取测试集指标进行比较
    test_metrics = {}
    for method, results in results_dict.items():
        if 'metrics' in results and results['metrics']:
            test_metrics[method] = {
                'test_mse': results['metrics'].get('test_mse', float('nan')),
                'test_pearson': results['metrics'].get('test_pearson', float('nan')),
                'test_r2': results['metrics'].get('test_r2', float('nan')),
                'feature_count': results['metrics'].get('feature_count', float('nan'))
            }
    
    if not test_metrics:
        logger.warning("没有找到有效的测试指标数据")
        return
        
    # 创建比较表格
    metrics_df = pd.DataFrame(test_metrics).T
    metrics_df = metrics_df.reset_index().rename(columns={'index': 'Feature Engineering Method'})
    
    logger.info("Comparison of test metrics for different feature engineering methods:")
    logger.info("\n" + metrics_df.to_string(index=False))
    
    # 保存比较结果
    os.makedirs('visualizations', exist_ok=True)
    metrics_df.to_csv('visualizations/feature_engineering_comparison.csv', index=False)
    
    # 绘制比较图表
    plot_comparison(metrics_df)
    
def plot_comparison(metrics_df: pd.DataFrame) -> None:
    """
    绘制不同特征工程方法的比较图表
    
    Args:
        metrics_df: 包含比较指标的DataFrame
    """
    # 设置图表风格
    sns.set_style("whitegrid")
    
    # 创建图表
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # MSE比较（越低越好）
    sns.barplot(x='Feature Engineering Method', y='test_mse', data=metrics_df, ax=axes[0])
    axes[0].set_title('Test MSE Comparison')
    axes[0].set_ylabel('MSE (lower is better)')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Pearson相关系数比较（越高越好）
    sns.barplot(x='Feature Engineering Method', y='test_pearson', data=metrics_df, ax=axes[1])
    axes[1].set_title('Test Pearson Correlation Comparison')
    axes[1].set_ylabel('Pearson Correlation (higher is better)')
    axes[1].tick_params(axis='x', rotation=45)
    
    # R²比较（越高越好）
    sns.barplot(x='Feature Engineering Method', y='test_r2', data=metrics_df, ax=axes[2])
    axes[2].set_title('Test R² Comparison')
    axes[2].set_ylabel('R² (higher is better)')
    axes[2].tick_params(axis='x', rotation=45)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('visualizations/feature_engineering_comparison.png', dpi=300)
    plt.close()
    
    # 绘制特征数量与性能关系图
    plt.figure(figsize=(10, 6))
    
    # 创建双轴图表
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 绘制特征数量与MSE的关系
    ax1.set_xlabel('Feature Engineering Method')
    ax1.set_ylabel('Feature Count', color='blue')
    ax1.bar(metrics_df['Feature Engineering Method'], metrics_df['feature_count'], color='blue', alpha=0.6, label='Feature Count')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.tick_params(axis='x', rotation=45)
    
    # 创建第二个Y轴
    ax2 = ax1.twinx()
    ax2.set_ylabel('Performance Metrics', color='red')
    ax2.plot(metrics_df['Feature Engineering Method'], metrics_df['test_pearson'], 'ro-', label='Pearson Correlation')
    ax2.plot(metrics_df['Feature Engineering Method'], metrics_df['test_r2'], 'go-', label='R²')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title('Feature Count vs Model Performance')
    plt.tight_layout()
    plt.savefig('visualizations/feature_count_vs_performance.png', dpi=300)
    plt.close()

def main():
    """主函数"""
    logger.info("开始比较不同特征工程方法的结果")
    
    # 定义结果目录
    result_dirs = {
        'Complete Features': 'models/tuning_results/bayesian_opt',
        'Simplified Features': 'models/tuning_results/simplified_features',
        'Squared Features Only': 'models/tuning_results/squared_features'
    }
    
    # 加载结果
    results = {}
    for method, dir_path in result_dirs.items():
        if os.path.exists(dir_path):
            logger.info(f"加载 {method} 的结果...")
            results[method] = load_results(dir_path)
        else:
            logger.warning(f"目录 {dir_path} 不存在，跳过 {method}")
    
    # 比较结果
    if results:
        compare_results(results)
        logger.info("比较完成，结果已保存到 visualizations 目录")
    else:
        logger.warning("没有找到有效的结果数据")

if __name__ == "__main__":
    main() 