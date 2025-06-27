#!/bin/bash

# 修复项目中的缩进错误和中文字符显示问题

echo "开始修复项目问题..."

# 1. 修复Python脚本中的缩进错误
echo "修复Python脚本中的缩进错误..."

# 修复数据可视化脚本中的缩进
sed -i 's/^    import sys/import sys/g' data/visualize_data.py
sed -i 's/^    import sys/import sys/g' models/model_tuning.py

# 2. 修复中文字符显示问题
echo "修复中文字符显示问题..."

# 创建matplotlib配置文件
mkdir -p ~/.config/matplotlib
cat > ~/.config/matplotlib/matplotlibrc << EOF
font.family: sans-serif
font.sans-serif: DejaVu Sans, Arial, Helvetica
axes.unicode_minus: False
EOF

echo "修复完成！"
echo "请重新运行优化训练脚本: bash scripts/train_optimized.sh" 