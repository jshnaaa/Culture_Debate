#!/bin/bash

# =============================================================================
# 多智能体文化对齐系统运行脚本
# =============================================================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_header() {
    echo -e "${BLUE}"
    echo "========================================================================"
    echo "🤖 多智能体文化对齐系统"
    echo "🎯 ACL 2025: Multiple LLM Agents Debate for Equitable Cultural Alignment"
    echo "========================================================================"
    echo -e "${NC}"
}

# 检查系统环境
check_environment() {
    print_info "检查系统环境..."

    # 检查Python
    if ! command -v python &> /dev/null; then
        print_error "未找到Python，请确保Python 3.8+已安装"
        exit 1
    fi

    PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_success "Python版本: $PYTHON_VERSION"

    # 检查GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
        print_success "GPU信息: $GPU_INFO"
    else
        print_warning "未检测到NVIDIA GPU，将使用CPU模式（速度较慢）"
    fi

    # 检查数据文件
    if [ ! -f "data/normad.jsonl" ]; then
        print_error "未找到数据文件 data/normad.jsonl"
        print_info "请确保NORMAD数据集文件存在"
        exit 1
    fi

    DATA_SIZE=$(wc -l < data/normad.jsonl)
    print_success "数据文件: data/normad.jsonl ($DATA_SIZE 条记录)"
}

# 安装依赖
install_dependencies() {
    print_info "检查Python依赖..."

    # 检查必要的包
    REQUIRED_PACKAGES=("torch" "transformers" "huggingface_hub" "accelerate" "yaml")
    MISSING_PACKAGES=()

    for package in "${REQUIRED_PACKAGES[@]}"; do
        if ! python -c "import $package" &> /dev/null; then
            MISSING_PACKAGES+=("$package")
        fi
    done

    if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
        print_warning "缺少以下Python包: ${MISSING_PACKAGES[*]}"
        read -p "是否自动安装? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "安装依赖包..."
            pip install torch transformers huggingface-hub accelerate pyyaml datasets
            print_success "依赖安装完成"
        else
            print_error "请手动安装依赖: pip install torch transformers huggingface-hub accelerate pyyaml datasets"
            exit 1
        fi
    else
        print_success "所有依赖已满足"
    fi
}

# 创建必要目录
setup_directories() {
    print_info "创建必要目录..."

    DIRS=("config" "output" "cache" "logs")
    for dir in "${DIRS[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_success "创建目录: $dir"
        fi
    done
}

# 检查HuggingFace Token
check_hf_token() {
    print_info "检查HuggingFace配置..."

    if [ -z "$HF_TOKEN" ] && [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
        print_warning "未设置HuggingFace Token"
        print_info "可以通过以下方式设置:"
        print_info "1. export HF_TOKEN='your_token'"
        print_info "2. huggingface-cli login"

        read -p "继续运行? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        print_success "HuggingFace Token已配置"
    fi
}

# 显示运行选项菜单
show_menu() {
    echo
    print_info "请选择运行模式:"
    echo "1) 🧪 测试模式 (处理前10项数据)"
    echo "2) 📊 小批量模式 (处理前100项数据)"
    echo "3) 🚀 完整模式 (处理全部数据)"
    echo "4) 🎯 自定义模式 (自定义参数)"
    echo "5) ❌ 退出"
    echo
}

# 运行测试模式
run_test_mode() {
    print_info "启动测试模式..."

    OUTPUT_FILE="output/test_results_$(date +%Y%m%d_%H%M%S).jsonl"

    python run_multi_agent_inference.py \
        --input_path data/normad.jsonl \
        --output_path "$OUTPUT_FILE" \
        --config_dir config \
        --log_level INFO \
        --max_items 10 \
        --start_from 0

    print_success "测试完成! 结果保存在: $OUTPUT_FILE"
}

# 运行小批量模式
run_batch_mode() {
    print_info "启动小批量模式..."

    OUTPUT_FILE="output/batch_results_$(date +%Y%m%d_%H%M%S).jsonl"

    python run_multi_agent_inference.py \
        --input_path data/normad.jsonl \
        --output_path "$OUTPUT_FILE" \
        --config_dir config \
        --log_level INFO \
        --max_items 100 \
        --start_from 0

    print_success "批量处理完成! 结果保存在: $OUTPUT_FILE"
}

# 运行完整模式
run_full_mode() {
    print_warning "完整模式将处理所有数据，可能需要数小时时间"
    read -p "确认继续? (y/N): " -n 1 -r
    echo

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        return
    fi

    print_info "启动完整模式..."

    OUTPUT_FILE="output/full_results_$(date +%Y%m%d_%H%M%S).jsonl"

    python run_multi_agent_inference.py \
        --input_path data/normad.jsonl \
        --output_path "$OUTPUT_FILE" \
        --config_dir config \
        --log_level INFO

    print_success "完整处理完成! 结果保存在: $OUTPUT_FILE"
}

# 运行自定义模式
run_custom_mode() {
    print_info "自定义模式配置..."

    # 获取用户输入
    read -p "输入文件路径 [data/normad.jsonl]: " INPUT_PATH
    INPUT_PATH=${INPUT_PATH:-"data/normad.jsonl"}

    read -p "输出文件路径 [output/custom_results.jsonl]: " OUTPUT_PATH
    OUTPUT_PATH=${OUTPUT_PATH:-"output/custom_results_$(date +%Y%m%d_%H%M%S).jsonl"}

    read -p "最大处理项数 (留空处理全部): " MAX_ITEMS

    read -p "开始位置 [0]: " START_FROM
    START_FROM=${START_FROM:-0}

    read -p "日志级别 [INFO]: " LOG_LEVEL
    LOG_LEVEL=${LOG_LEVEL:-"INFO"}

    # 构建命令
    CMD="python run_multi_agent_inference.py --input_path \"$INPUT_PATH\" --output_path \"$OUTPUT_PATH\" --config_dir config --log_level $LOG_LEVEL --start_from $START_FROM"

    if [ ! -z "$MAX_ITEMS" ]; then
        CMD="$CMD --max_items $MAX_ITEMS"
    fi

    print_info "执行命令: $CMD"

    eval $CMD

    print_success "自定义处理完成! 结果保存在: $OUTPUT_PATH"
}

# 显示系统信息
show_system_info() {
    print_info "系统信息:"
    echo "  📁 工作目录: $(pwd)"
    echo "  🐍 Python版本: $(python --version)"
    echo "  💾 可用内存: $(free -h 2>/dev/null | awk '/^Mem:/ {print $7}' || echo '未知')"
    echo "  🗂️  数据文件: data/normad.jsonl ($(wc -l < data/normad.jsonl) 条记录)"

    if command -v nvidia-smi &> /dev/null; then
        echo "  🎮 GPU信息:"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | while read line; do
            echo "    GPU$line"
        done
    fi
}

# 清理函数
cleanup() {
    print_info "清理临时文件..."
    # 这里可以添加清理逻辑
}

# 主函数
main() {
    # 设置陷阱以确保清理
    trap cleanup EXIT

    # 显示标题
    print_header

    # 检查环境
    check_environment
    install_dependencies
    setup_directories
    check_hf_token

    # 显示系统信息
    show_system_info

    # 设置Python路径
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"

    # 主循环
    while true; do
        show_menu
        read -p "请选择 (1-5): " choice

        case $choice in
            1)
                run_test_mode
                ;;
            2)
                run_batch_mode
                ;;
            3)
                run_full_mode
                ;;
            4)
                run_custom_mode
                ;;
            5)
                print_info "退出程序"
                exit 0
                ;;
            *)
                print_error "无效选择，请重新选择"
                ;;
        esac

        echo
        read -p "按回车键继续..." -r
    done
}

# 如果脚本被直接执行
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi