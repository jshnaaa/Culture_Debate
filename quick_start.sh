#!/bin/bash

# 快速启动多智能体系统
# 用法: ./quick_start.sh [test|batch|full]

set -e

echo "🚀 启动多智能体文化对齐系统..."

# 检查数据文件
if [ ! -f "data/normad.jsonl" ]; then
    echo "❌ 数据文件不存在: data/normad.jsonl"
    exit 1
fi

# 创建输出目录
mkdir -p output logs

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 获取运行模式
MODE=${1:-"test"}

case $MODE in
    "test")
        echo "🧪 测试模式: 处理前10项数据"
        OUTPUT_FILE="output/test_$(date +%Y%m%d_%H%M%S).jsonl"
        python run_multi_agent_inference.py \
            --input_path data/normad.jsonl \
            --output_path "$OUTPUT_FILE" \
            --max_items 10
        ;;
    "batch")
        echo "📊 批量模式: 处理前100项数据"
        OUTPUT_FILE="output/batch_$(date +%Y%m%d_%H%M%S).jsonl"
        python run_multi_agent_inference.py \
            --input_path data/normad.jsonl \
            --output_path "$OUTPUT_FILE" \
            --max_items 100
        ;;
    "full")
        echo "🚀 完整模式: 处理全部数据"
        OUTPUT_FILE="output/full_$(date +%Y%m%d_%H%M%S).jsonl"
        python run_multi_agent_inference.py \
            --input_path data/normad.jsonl \
            --output_path "$OUTPUT_FILE"
        ;;
    *)
        echo "❌ 无效模式: $MODE"
        echo "用法: $0 [test|batch|full]"
        exit 1
        ;;
esac

echo "✅ 处理完成! 结果保存在: $OUTPUT_FILE"