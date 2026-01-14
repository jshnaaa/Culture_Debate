"""
多智能体框架使用示例
演示如何使用新的多智能体框架进行文化对齐辩论
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from agents.multi_agent_system import MultiAgentSystem


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def main():
    """主函数"""
    # 创建多智能体系统
    mas = MultiAgentSystem()

    try:
        # 初始化系统
        print("🚀 初始化多智能体系统...")
        success = await mas.initialize()
        if not success:
            print("❌ 系统初始化失败")
            return

        print("✅ 系统初始化成功")

        # 系统健康检查
        print("🔍 进行系统健康检查...")
        health = await mas.health_check()
        print(f"💊 系统健康状态: {'正常' if health else '异常'}")

        # 获取系统统计信息
        stats = mas.get_system_stats()
        print("📊 系统统计信息:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))

        # 示例场景：商务会议着装
        scenario = {
            "country": "egypt",
            "story": "在埃及开罗的一次重要商务会议中，一位西方商务人员穿着休闲装（牛仔裤和T恤）出席会议。",
            "rule_of_thumb": "在正式商务场合，适当的着装体现了对会议和与会者的尊重。"
        }

        print("\n🎭 开始文化对齐辩论...")
        print(f"📍 场景: {scenario['story']}")
        print(f"📏 规则: {scenario['rule_of_thumb']}")

        # 启动文化辩论
        result = await mas.start_cultural_debate(scenario)

        print(f"\n🏁 辩论完成! 对话ID: {result['conversation_id']}")
        print(f"⏱️  总耗时: {result['duration']:.2f}秒")

        # 显示结果摘要
        print("\n📋 辩论结果摘要:")
        print("=" * 60)

        # 初始决策
        print("\n1️⃣ 初始决策:")
        for agent_type, response in result['initial_responses'].items():
            parsed = response['parsed_response']
            print(f"   {agent_type}: {parsed.get('answer', 'unknown')} "
                  f"(置信度: {response['confidence']:.2f})")

        # 最终决策
        print("\n3️⃣ 最终决策:")
        for agent_type, response in result['final_responses'].items():
            parsed = response['parsed_response']
            print(f"   {agent_type}: {parsed.get('answer', 'unknown')} "
                  f"(置信度: {response['confidence']:.2f})")

        # 决策一致性分析
        final_answers = [resp['parsed_response'].get('answer', 'unknown')
                        for resp in result['final_responses'].values()]
        unique_answers = set(final_answers)

        print(f"\n🤝 决策一致性: {len(unique_answers)} 种不同观点")
        for answer in unique_answers:
            count = final_answers.count(answer)
            percentage = (count / len(final_answers)) * 100
            print(f"   {answer}: {count}个智能体 ({percentage:.1f}%)")

        # 详细响应（可选）
        show_details = input("\n🔍 是否显示详细响应? (y/N): ").lower().strip() == 'y'
        if show_details:
            print("\n📝 详细响应:")
            print("=" * 80)

            for stage_name, stage_responses in [
                ("初始决策", result['initial_responses']),
                ("反馈交换", result['feedback_responses']),
                ("最终决策", result['final_responses'])
            ]:
                print(f"\n{stage_name}:")
                print("-" * 40)

                for agent_type, response in stage_responses.items():
                    print(f"\n🤖 {agent_type}:")
                    print(f"   回应: {response['raw_response'][:200]}...")
                    print(f"   置信度: {response['confidence']:.3f}")
                    print(f"   处理时间: {response['processing_time']:.3f}秒")

        # 保存结果（可选）
        save_results = input("\n💾 是否保存结果到文件? (y/N): ").lower().strip() == 'y'
        if save_results:
            output_file = f"debate_result_{result['conversation_id']}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            print(f"✅ 结果已保存到: {output_file}")

    except Exception as e:
        print(f"❌ 发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        # 关闭系统
        print("\n🛑 关闭多智能体系统...")
        await mas.shutdown()
        print("✅ 系统关闭完成")


async def demo_simple_scenario():
    """简单演示场景"""
    mas = MultiAgentSystem()

    try:
        await mas.initialize()

        # 简单场景
        scenario = {
            "country": "united_states",
            "story": "在美国的一个家庭聚会上，客人没有脱鞋就进入了房屋。",
            "rule_of_thumb": "尊重主人的家庭习惯和文化传统是基本礼貌。"
        }

        result = await mas.start_cultural_debate(scenario)

        print("🎯 简单演示结果:")
        for agent_type, response in result['final_responses'].items():
            answer = response['parsed_response'].get('answer', 'unknown')
            print(f"  {agent_type}: {answer}")

    finally:
        await mas.shutdown()


if __name__ == "__main__":
    # 选择运行模式
    print("🌟 多智能体文化对齐系统演示")
    print("1. 完整演示")
    print("2. 简单演示")

    choice = input("请选择模式 (1/2): ").strip()

    if choice == "2":
        asyncio.run(demo_simple_scenario())
    else:
        asyncio.run(main())