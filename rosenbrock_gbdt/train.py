#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rosenbrock数据集的GBDT模型训练主脚本
"""

import os
import logging
import argparse
import numpy as np
from data.load_data import load_data, preprocess_data, get_feature_names
from models.model_tuning import bayesian_optimization, save_tuning_results

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练Rosenbrock数据集的GBDT模型')
    parser.add_argument('--n_trials', type=int, default=50, help='贝叶斯优化的试验次数')
    parser.add_argument('--timeout', type=int, default=3600, help='优化超时时间（秒）')
    parser.add_argument('--use_gpu', action='store_true', help='是否使用GPU加速（注意：GBDT不支持完全GPU加速）')
    parser.add_argument('--random_state', type=int, default=42, help='随机种子')
    parser.add_argument('--save_dir', type=str, default='models/tuning_results/bayesian_opt', help='结果保存目录')
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    logger.info("开始训练Rosenbrock数据集的GBDT模型")
    logger.info(f"参数: n_trials={args.n_trials}, timeout={args.timeout}, use_gpu={args.use_gpu}, random_state={args.random_state}")
    
    # 加载和预处理数据
    data = load_data()
    processed_data = preprocess_data(data)
    
    # 获取特征名称
    feature_names = get_feature_names(processed_data['X_train'].shape[1])
    
    # 贝叶斯优化
    optuna_results = bayesian_optimization(
        processed_data,
        n_trials=args.n_trials,
        timeout=args.timeout,
        use_gpu=args.use_gpu,
        random_state=args.random_state
    )
    
    # 添加特征名称到结果字典
    optuna_results['feature_names'] = feature_names
    
    # 添加训练数据到结果字典（用于生成训练集预测分析图）
    optuna_results['X_train'] = processed_data['X_train']
    optuna_results['y_train'] = processed_data['y_train']
    optuna_results['X_test'] = processed_data['X_test']
    optuna_results['y_test'] = processed_data['y_test']
    
    # 保存结果
    save_tuning_results(optuna_results, args.save_dir)
    
    # 打印最终结果
    logger.info("训练完成！")
    logger.info(f"最佳参数: {optuna_results['best_params']}")
    logger.info(f"最佳MSE: {optuna_results['best_score']:.4f}")
    
    if 'test_metrics' in optuna_results and optuna_results['test_metrics']:
        logger.info(f"测试集Pearson相关系数: {optuna_results['test_metrics']['pearson_test']:.4f}")
        logger.info(f"测试集R²: {optuna_results['test_metrics']['r2_test']:.4f}")
        
        # 检查是否达到目标
        if optuna_results['test_metrics']['pearson_test'] >= 0.85:
            logger.info("🎉 成功达到目标：Pearson相关系数 >= 0.85")
        else:
            logger.info(f"⚠️ 未达到目标：当前Pearson相关系数为 {optuna_results['test_metrics']['pearson_test']:.4f}，目标为 >= 0.85")
            logger.info("建议增加试验次数或调整超参数搜索空间")

if __name__ == "__main__":
    main() 