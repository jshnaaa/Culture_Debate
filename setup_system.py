"""
设置多智能体系统
创建必要的目录结构和配置文件
"""

import os
import yaml
import json
from pathlib import Path

def create_directories():
    """创建必要的目录"""
    directories = [
        "config",
        "output",
        "cache",
        "logs"
    ]

    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ 创建目录: {dir_name}")

def create_global_config():
    """创建全局配置文件"""
    config = {
        "hf_token": "",  # 需要用户填入HuggingFace token
        "cache_dir": "./cache",
        "max_active_agents": 3,
        "idle_timeout": 300.0,
        "memory_threshold": 0.8,
        "message_bus": {
            "max_queue_size": 1000,
            "message_timeout": 30.0,
            "retry_attempts": 3
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }

    config_path = Path("config/global_config.yaml")
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"✅ 创建全局配置: {config_path}")

def create_agent_configs():
    """创建智能体配置文件"""

    # 基督教文化智能体配置
    christian_config = {
        "model": {
            "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
            "cache_dir": "./cache",
            "torch_dtype": "bfloat16",
            "device_map": "auto",
            "max_new_tokens": 512,
            "temperature": 0.0,
            "max_input_length": 2048
        },
        "cultural": {
            "cultural_values": ["个人自由", "人权", "平等", "民主", "个人责任"],
            "social_norms": {
                "商务穿着": "正式场合穿着得体，日常可以相对随意",
                "社交互动": "直接沟通，握手问候，注重个人空间",
                "决策方式": "个人决策，考虑个人利益和权利"
            },
            "communication_style": {
                "直接性": "高",
                "正式程度": "中等"
            },
            "decision_factors": ["个人权利和自由", "法律和规则", "公平性"],
            "prompt_templates": {
                "initial_decision": "作为基督教文化代表，评估行为的社会可接受性...",
                "feedback": "基于基督教价值观提供反馈...",
                "final_decision": "做出最终判断..."
            }
        },
        "custom": {}
    }

    # 伊斯兰文化智能体配置
    islamic_config = {
        "model": {
            "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
            "cache_dir": "./cache",
            "torch_dtype": "bfloat16",
            "device_map": "auto",
            "max_new_tokens": 512,
            "temperature": 0.0,
            "max_input_length": 2048
        },
        "cultural": {
            "cultural_values": ["谦逊", "敬畏", "家庭责任", "社会秩序", "诚实"],
            "social_norms": {
                "穿着规范": "保守穿着，特别是公共场合",
                "社交互动": "同性握手，异性避免身体接触",
                "商务礼仪": "注重传统和尊重"
            },
            "communication_style": {
                "直接性": "中等",
                "正式程度": "高"
            },
            "decision_factors": ["宗教教义", "家庭和社区利益", "传统和习俗"],
            "prompt_templates": {
                "initial_decision": "作为伊斯兰文化代表，评估行为的社会可接受性...",
                "feedback": "基于伊斯兰价值观提供反馈...",
                "final_decision": "做出最终判断..."
            }
        },
        "custom": {}
    }

    # 其他文化智能体配置（简化版）
    other_configs = {
        "cultural_buddhist": {
            "model": {"model_id": "google/gemma-2-9b-it", "cache_dir": "./cache", "torch_dtype": "bfloat16"},
            "cultural": {"cultural_values": ["内心平静", "慈悲", "简朴", "和谐"]}
        },
        "cultural_hindu": {
            "model": {"model_id": "meta-llama/Meta-Llama-3-8B-Instruct", "cache_dir": "./cache", "torch_dtype": "bfloat16"},
            "cultural": {"cultural_values": ["达摩", "家庭责任", "精神修养", "传统仪式"]}
        },
        "cultural_traditional": {
            "model": {"model_id": "meta-llama/Meta-Llama-3-8B-Instruct", "cache_dir": "./cache", "torch_dtype": "bfloat16"},
            "cultural": {"cultural_values": ["自然和谐", "祖先崇拜", "部落团结", "传统智慧"]}
        }
    }

    # 保存配置文件
    configs = {
        "cultural_christian": christian_config,
        "cultural_islamic": islamic_config,
        **other_configs
    }

    for agent_type, config in configs.items():
        config_path = Path(f"config/{agent_type}_config.yaml")
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        print(f"✅ 创建智能体配置: {config_path}")

def create_run_script():
    """创建运行脚本"""
    script_content = '''#!/bin/bash

# 多智能体系统运行脚本

echo "🚀 启动多智能体文化对齐系统"

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ 未找到Python，请确保Python已安装"
    exit 1
fi

# 检查数据文件
if [ ! -f "data/normad.jsonl" ]; then
    echo "❌ 未找到数据文件 data/normad.jsonl"
    exit 1
fi

# 创建输出目录
mkdir -p output logs

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "📊 开始处理NORMAD数据集..."

# 运行推理（默认处理前10项用于测试）
python run_multi_agent_inference.py \\
    --input_path data/normad.jsonl \\
    --output_path output/multi_agent_results.jsonl \\
    --config_dir config \\
    --log_level INFO \\
    --max_items 10 \\
    --start_from 0

echo "✅ 处理完成！结果保存在 output/multi_agent_results.jsonl"
'''

    script_path = Path("run_inference.sh")
    with open(script_path, 'w') as f:
        f.write(script_content)

    # 设置执行权限
    os.chmod(script_path, 0o755)
    print(f"✅ 创建运行脚本: {script_path}")

def main():
    """主设置函数"""
    print("🔧 设置多智能体系统...")

    # 创建目录结构
    create_directories()

    # 创建配置文件
    create_global_config()
    create_agent_configs()

    # 创建运行脚本
    create_run_script()

    print("\n✅ 系统设置完成！")
    print("\n📋 下一步操作：")
    print("1. 编辑 config/global_config.yaml，填入你的HuggingFace token")
    print("2. 确保数据文件 data/normad.jsonl 存在")
    print("3. 运行测试：python test_system.py")
    print("4. 运行推理：./run_inference.sh 或 python run_multi_agent_inference.py")

    print("\n⚠️  注意事项：")
    print("- 确保有足够的GPU内存（建议16GB+）")
    print("- 首次运行会下载模型，需要网络连接")
    print("- 可以通过 --max_items 参数控制处理的数据量")

if __name__ == "__main__":
    main()