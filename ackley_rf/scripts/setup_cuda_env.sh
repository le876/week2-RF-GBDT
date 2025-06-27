#!/bin/bash

# 设置CUDA环境变量的脚本

echo "=========================================="
echo "设置CUDA环境变量"
echo "=========================================="

# 检测CUDA版本
CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d'.' -f1,2)
echo "检测到CUDA版本: $CUDA_VERSION"

# 查找libnvrtc.so文件
echo "查找libnvrtc.so文件..."
NVRTC_PATHS=$(find /usr -name "libnvrtc.so*" 2>/dev/null)

if [ -z "$NVRTC_PATHS" ]; then
    echo "⚠️ 未找到libnvrtc.so文件"
    echo "请安装CUDA运行时库或使用root权限运行install_cuda_libs.sh脚本"
else
    echo "找到以下libnvrtc.so文件:"
    for NVRTC_PATH in $NVRTC_PATHS; do
        echo "  - $NVRTC_PATH"
    done
    
    # 获取libnvrtc.so文件所在目录
    NVRTC_DIR=$(dirname "$(echo "$NVRTC_PATHS" | head -n1)")
    echo "使用目录: $NVRTC_DIR"
    
    # 设置环境变量
    export LD_LIBRARY_PATH=$NVRTC_DIR:$LD_LIBRARY_PATH
    echo "已设置环境变量: LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    
    # 创建环境变量设置脚本
    cat > cuda_env.sh << EOF
export LD_LIBRARY_PATH=$NVRTC_DIR:\$LD_LIBRARY_PATH
EOF
    
    echo "环境变量设置脚本已创建: cuda_env.sh"
    echo "在每次新会话中，请运行以下命令设置环境变量:"
    echo "source cuda_env.sh"
fi

echo "=========================================="
echo "CUDA环境变量设置完成"
echo "=========================================="

# 验证CuPy是否可用
echo "验证CuPy是否可用..."
python -c "
import sys
try:
    import cupy as cp
    # 简单测试CuPy功能
    test_array = cp.array([1, 2, 3])
    test_result = cp.sum(test_array)
    cp.asnumpy(test_result)
    print('✅ CuPy测试成功')
except Exception as e:
    print(f'❌ CuPy测试失败: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo "✅ CuPy可用，可以运行GPU加速训练"
    echo "运行命令: bash scripts/train_gpu.sh"
else
    echo "❌ CuPy不可用，请检查CUDA环境配置"
fi 