#!/bin/bash

# 设置工作目录
cd "$(dirname "$0")"

echo "开始比较不同特征工程方法的结果..."

# 运行比较脚本
python compare_feature_engineering.py

echo "比较完成，结果已保存到 visualizations 目录"
echo "可以查看以下文件:"
echo "- visualizations/feature_engineering_comparison.csv"
echo "- visualizations/feature_engineering_comparison.png"
echo "- visualizations/feature_count_vs_performance.png" 