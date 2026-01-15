"""
多智能体系统在NORMAD数据集上进行推理
批量处理NORMAD.jsonl数据集，输出多智能体辩论结果
"""

import asyncio
import json
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, Any, List
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from agents.multi_agent_system import MultiAgentSystem


def setup_logging(log_level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('multi_agent_inference.log'),
            logging.StreamHandler()
        ]
    )


def load_normad_data(file_path: str) -> List[Dict[str, Any]]:
    """加载NORMAD数据集"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def convert_normad_to_scenario(normad_item: Dict[str, Any]) -> Dict[str, Any]:
    """将NORMAD数据项转换为多智能体场景格式"""
    return {
        "id": normad_item.get("ID"),
        "country": normad_item.get("Country", "").lower(),
        "story": normad_item.get("Story", ""),
        "rule_of_thumb": normad_item.get("Rule-of-Thumb", ""),
        "background": normad_item.get("Background", ""),
        "gold_label": normad_item.get("Gold Label", ""),
        "axis": normad_item.get("Axis", ""),
        "subaxis": normad_item.get("Subaxis", ""),
        "value": normad_item.get("Value", "")
    }


def extract_final_decisions(debate_result: Dict[str, Any]) -> Dict[str, str]:
    """提取最终决策结果"""
    final_decisions = {}

    for agent_type, response in debate_result.get('final_responses', {}).items():
        parsed_response = response.get('parsed_response', {})
        answer = parsed_response.get('answer', 'neither')
        final_decisions[agent_type] = answer

    return final_decisions


def calculate_majority_decision(decisions: Dict[str, str]) -> str:
    """计算多数决策"""
    if not decisions:
        return "neither"

    # 统计各答案的频次
    answer_counts = {}
    for answer in decisions.values():
        answer_counts[answer] = answer_counts.get(answer, 0) + 1

    # 找出最多的答案
    max_count = max(answer_counts.values())
    majority_answers = [answer for answer, count in answer_counts.items() if count == max_count]

    # 如果有平局，返回第一个
    return majority_answers[0]


async def process_single_item(mas: MultiAgentSystem, normad_item: Dict[str, Any]) -> Dict[str, Any]:
    """处理单个数据项"""
    scenario = convert_normad_to_scenario(normad_item)

    try:
        # 启动多智能体辩论
        debate_result = await mas.start_cultural_debate(scenario)

        # 提取最终决策
        final_decisions = extract_final_decisions(debate_result)
        majority_decision = calculate_majority_decision(final_decisions)

        # 构建输出结果
        result = {
            "ID": scenario["id"],
            "Country": scenario["country"],
            "Story": scenario["story"],
            "Rule-of-Thumb": scenario["rule_of_thumb"],
            "Gold Label": scenario["gold_label"],

            # 多智能体结果
            "Agent_Decisions": final_decisions,
            "Majority_Decision": majority_decision,
            "Conversation_ID": debate_result["conversation_id"],
            "Processing_Time": debate_result["duration"],

            # 详细响应（可选）
            "Initial_Responses": {
                agent_type: {
                    "answer": resp["parsed_response"].get("answer", "neither"),
                    "explanation": resp["parsed_response"].get("explanation", ""),
                    "confidence": resp["confidence"]
                }
                for agent_type, resp in debate_result.get("initial_responses", {}).items()
            },

            "Final_Responses": {
                agent_type: {
                    "answer": resp["parsed_response"].get("answer", "neither"),
                    "confidence": resp["confidence"]
                }
                for agent_type, resp in debate_result.get("final_responses", {}).items()
            }
        }

        return result

    except Exception as e:
        logging.error(f"处理数据项 {scenario['id']} 失败: {str(e)}")
        return {
            "ID": scenario["id"],
            "Country": scenario["country"],
            "Story": scenario["story"],
            "Gold Label": scenario["gold_label"],
            "Error": str(e),
            "Agent_Decisions": {},
            "Majority_Decision": "neither"
        }


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多智能体系统NORMAD推理")
    parser.add_argument("--input_path", type=str, default="data/normad.jsonl",
                       help="输入NORMAD数据集路径")
    parser.add_argument("--output_path", type=str, default="output/multi_agent_results.jsonl",
                       help="输出结果路径")
    parser.add_argument("--config_dir", type=str, default="config",
                       help="配置文件目录")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别")
    parser.add_argument("--max_items", type=int, default=None,
                       help="最大处理数据项数量（用于测试）")
    parser.add_argument("--start_from", type=int, default=0,
                       help="从第几项开始处理")

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # 确保输出目录存在
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # 加载数据
        logger.info(f"加载NORMAD数据集: {args.input_path}")
        normad_data = load_normad_data(args.input_path)
        logger.info(f"加载了 {len(normad_data)} 个数据项")

        # 处理数据范围
        start_idx = args.start_from
        end_idx = len(normad_data)
        if args.max_items:
            end_idx = min(start_idx + args.max_items, len(normad_data))

        process_data = normad_data[start_idx:end_idx]
        logger.info(f"将处理 {len(process_data)} 个数据项 (从 {start_idx} 到 {end_idx-1})")

        # 初始化多智能体系统
        logger.info("初始化多智能体系统...")
        mas = MultiAgentSystem(args.config_dir)
        success = await mas.initialize()

        if not success:
            logger.error("多智能体系统初始化失败")
            return

        logger.info("多智能体系统初始化成功")

        # 显示系统状态
        stats = mas.get_system_stats()
        logger.info(f"系统状态: {json.dumps(stats, indent=2)}")

        # 批量处理
        results = []
        start_time = time.time()

        for i, normad_item in enumerate(process_data):
            item_start = time.time()
            logger.info(f"处理第 {start_idx + i + 1}/{len(normad_data)} 项 (ID: {normad_item.get('ID')})")

            try:
                result = await process_single_item(mas, normad_item)
                results.append(result)

                item_time = time.time() - item_start
                logger.info(f"项目 {normad_item.get('ID')} 处理完成，耗时 {item_time:.2f}秒")

                # 显示决策结果
                decisions = result.get("Agent_Decisions", {})
                majority = result.get("Majority_Decision", "neither")
                gold_label = result.get("Gold Label", "unknown")

                logger.info(f"  决策: {decisions}")
                logger.info(f"  多数决策: {majority}, 金标准: {gold_label}")

                # 每10项保存一次中间结果
                if (i + 1) % 10 == 0:
                    logger.info(f"保存中间结果... ({i + 1} 项已完成)")
                    with open(output_path, 'w', encoding='utf-8') as f:
                        for result in results:
                            f.write(json.dumps(result, ensure_ascii=False) + '\n')

            except Exception as e:
                logger.error(f"处理项目 {normad_item.get('ID')} 时发生错误: {str(e)}")
                # 添加错误记录
                error_result = {
                    "ID": normad_item.get("ID"),
                    "Country": normad_item.get("Country", ""),
                    "Story": normad_item.get("Story", ""),
                    "Gold Label": normad_item.get("Gold Label", ""),
                    "Error": str(e),
                    "Agent_Decisions": {},
                    "Majority_Decision": "neither"
                }
                results.append(error_result)

        # 保存最终结果
        logger.info("保存最终结果...")
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        total_time = time.time() - start_time
        logger.info(f"批量处理完成！")
        logger.info(f"总处理时间: {total_time:.2f}秒")
        logger.info(f"平均每项: {total_time/len(process_data):.2f}秒")
        logger.info(f"结果保存到: {output_path}")

        # 计算准确率统计
        correct_count = 0
        total_count = 0

        for result in results:
            if "Error" not in result:
                total_count += 1
                majority_decision = result.get("Majority_Decision", "neither")
                gold_label = result.get("Gold Label", "unknown")

                if majority_decision == gold_label:
                    correct_count += 1

        if total_count > 0:
            accuracy = correct_count / total_count
            logger.info(f"准确率: {correct_count}/{total_count} = {accuracy:.3f}")

        # 显示各智能体的决策分布
        agent_stats = {}
        for result in results:
            if "Error" not in result:
                for agent_type, decision in result.get("Agent_Decisions", {}).items():
                    if agent_type not in agent_stats:
                        agent_stats[agent_type] = {"yes": 0, "no": 0, "neither": 0}
                    agent_stats[agent_type][decision] = agent_stats[agent_type].get(decision, 0) + 1

        logger.info("各智能体决策分布:")
        for agent_type, stats in agent_stats.items():
            total = sum(stats.values())
            logger.info(f"  {agent_type}: Yes={stats['yes']}, No={stats['no']}, Neither={stats['neither']} (总计={total})")

    except Exception as e:
        logger.error(f"批量处理失败: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        # 关闭系统
        logger.info("关闭多智能体系统...")
        await mas.shutdown()
        logger.info("系统关闭完成")


if __name__ == "__main__":
    asyncio.run(main())