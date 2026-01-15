#!/bin/bash

# 批量处理NORMAD数据集
# 将大数据集分批处理，避免内存问题

set -e

echo "📊 批量处理NORMAD数据集"

# 配置参数
INPUT_FILE="data/normad.jsonl"
BATCH_SIZE=50
OUTPUT_DIR="output/batches"
LOG_DIR="logs"

# 检查输入文件
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ 数据文件不存在: $INPUT_FILE"
    exit 1
fi

# 创建目录
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# 获取总数据量
TOTAL_LINES=$(wc -l < "$INPUT_FILE")
TOTAL_BATCHES=$(( (TOTAL_LINES + BATCH_SIZE - 1) / BATCH_SIZE ))

echo "📋 数据统计:"
echo "  总数据量: $TOTAL_LINES 条"
echo "  批次大小: $BATCH_SIZE 条/批"
echo "  总批次数: $TOTAL_BATCHES 批"
echo

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 处理每个批次
for (( batch=0; batch<TOTAL_BATCHES; batch++ )); do
    start_idx=$((batch * BATCH_SIZE))
    batch_num=$((batch + 1))

    echo "🔄 处理批次 $batch_num/$TOTAL_BATCHES (从第 $start_idx 项开始)"

    # 输出文件名
    timestamp=$(date +%Y%m%d_%H%M%S)
    output_file="$OUTPUT_DIR/batch_${batch_num}_${timestamp}.jsonl"
    log_file="$LOG_DIR/batch_${batch_num}_${timestamp}.log"

    # 运行推理
    python run_multi_agent_inference.py \
        --input_path "$INPUT_FILE" \
        --output_path "$output_file" \
        --start_from $start_idx \
        --max_items $BATCH_SIZE \
        --log_level INFO \
        2>&1 | tee "$log_file"

    if [ $? -eq 0 ]; then
        echo "✅ 批次 $batch_num 完成: $output_file"
    else
        echo "❌ 批次 $batch_num 失败"
        exit 1
    fi

    # 短暂休息，让GPU降温
    echo "😴 休息5秒..."
    sleep 5
done

echo
echo "🎉 所有批次处理完成!"
echo "📁 结果文件位于: $OUTPUT_DIR/"
echo "📝 日志文件位于: $LOG_DIR/"

# 合并所有结果文件
echo "🔗 合并结果文件..."
final_output="$OUTPUT_DIR/merged_results_$(date +%Y%m%d_%H%M%S).jsonl"

for batch_file in "$OUTPUT_DIR"/batch_*.jsonl; do
    if [ -f "$batch_file" ]; then
        cat "$batch_file" >> "$final_output"
    fi
done

echo "✅ 合并完成: $final_output"

# 统计结果
total_results=$(wc -l < "$final_output")
echo "📊 最终统计: $total_results 条结果"