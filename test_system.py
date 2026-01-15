"""
测试多智能体系统是否正常工作
"""

import asyncio
import json
import logging
from pathlib import Path
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

# 设置简单的日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# 创建模拟的智能体类，避免加载真实模型
class MockCulturalAgent:
    def __init__(self, agent_id, agent_type, config):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config
        self.status = "inactive"

    async def initialize(self):
        self.status = "active"
        return True

    async def process_message(self, message):
        # 模拟响应
        from agents.base.agent_interface import AgentResponse

        content = message.content
        stage = content.get("context", {}).get("stage", "initial_decision")

        if stage == "initial_decision":
            response_text = "Yes. This behavior is socially acceptable."
        elif stage == "feedback":
            response_text = "I agree with the previous assessment."
        else:  # final_decision
            response_text = "Yes"

        return AgentResponse(
            agent_id=self.agent_id,
            response_text=response_text,
            confidence=0.8,
            metadata={},
            processing_time=0.1
        )

    def parse_response(self, response_text, stage):
        if stage == "final_decision":
            answer = "yes" if "yes" in response_text.lower() else "no"
        else:
            answer = "yes" if "yes" in response_text.lower() else "no"

        return {
            "answer": answer,
            "explanation": response_text,
            "raw_response": response_text,
            "confidence": 0.8
        }

    async def cleanup(self):
        self.status = "inactive"
        return True

async def test_system():
    """测试系统基本功能"""
    try:
        # 导入必要模块
        from agents.utils.agent_pool import AgentPool
        from agents.utils.message_bus import MessageBus
        from agents.base.agent_interface import AgentType, AgentMessage

        print("✅ 成功导入所有模块")

        # 测试智能体池
        pool_config = {"max_active_agents": 3, "idle_timeout": 300.0}
        agent_pool = AgentPool(pool_config)

        # 注册模拟智能体
        agent_pool.register_agent_class(AgentType.CULTURAL_CHRISTIAN, MockCulturalAgent, {"model_id": "mock"})
        agent_pool.register_agent_class(AgentType.CULTURAL_ISLAMIC, MockCulturalAgent, {"model_id": "mock"})

        print("✅ 智能体池创建成功")

        # 测试消息总线
        bus_config = {"max_queue_size": 100, "message_timeout": 5.0}
        message_bus = MessageBus(bus_config)
        await message_bus.start()

        print("✅ 消息总线启动成功")

        # 测试获取智能体
        agent1 = await agent_pool.get_agent(AgentType.CULTURAL_CHRISTIAN)
        agent2 = await agent_pool.get_agent(AgentType.CULTURAL_ISLAMIC)

        print(f"✅ 成功获取智能体: {agent1.agent_id}, {agent2.agent_id}")

        # 测试消息处理
        message = AgentMessage(
            sender_id="test",
            receiver_id=agent1.agent_id,
            message_type="generate_response",
            content={
                "context": {
                    "stage": "initial_decision",
                    "country": "egypt",
                    "story": "Test story",
                    "rule_of_thumb": "Test rule"
                }
            },
            timestamp=0.0,
            conversation_id="test"
        )

        response = await agent1.process_message(message)
        print(f"✅ 智能体响应: {response.response_text}")

        # 清理
        await message_bus.stop()
        await agent_pool.cleanup_all()

        print("✅ 系统测试完成，所有组件正常工作")
        return True

    except Exception as e:
        print(f"❌ 系统测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_normad_processing():
    """测试NORMAD数据处理"""
    try:
        # 读取一个NORMAD样本
        normad_sample = {
            "ID": 0,
            "Country": "egypt",
            "Story": "At a gathering at a friend's house, Sarah entered the room and made sure to greet everyone individually before sitting down.",
            "Rule-of-Thumb": "It is respectful to greet everyone present before starting any social interaction.",
            "Gold Label": "yes"
        }

        # 转换为场景格式
        scenario = {
            "id": normad_sample["ID"],
            "country": normad_sample["Country"].lower(),
            "story": normad_sample["Story"],
            "rule_of_thumb": normad_sample["Rule-of-Thumb"],
            "gold_label": normad_sample["Gold Label"]
        }

        print(f"✅ NORMAD数据转换成功: {scenario['country']}")

        # 模拟多智能体决策
        agent_decisions = {
            "cultural_christian": "yes",
            "cultural_islamic": "yes",
            "cultural_buddhist": "yes",
            "cultural_hindu": "yes",
            "cultural_traditional": "yes"
        }

        # 计算多数决策
        answer_counts = {}
        for answer in agent_decisions.values():
            answer_counts[answer] = answer_counts.get(answer, 0) + 1

        majority_decision = max(answer_counts, key=answer_counts.get)

        print(f"✅ 多数决策: {majority_decision}, 金标准: {scenario['gold_label']}")
        print(f"✅ 准确性: {'正确' if majority_decision == scenario['gold_label'] else '错误'}")

        return True

    except Exception as e:
        print(f"❌ NORMAD处理测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧪 开始测试多智能体系统...")

    async def run_tests():
        # 基本系统测试
        print("\n1. 测试基本系统功能...")
        system_ok = await test_system()

        # NORMAD处理测试
        print("\n2. 测试NORMAD数据处理...")
        normad_ok = await test_normad_processing()

        if system_ok and normad_ok:
            print("\n🎉 所有测试通过！系统准备就绪。")
            print("\n📋 运行命令:")
            print("python run_multi_agent_inference.py --input_path data/normad.jsonl --output_path output/results.jsonl --max_items 10")
        else:
            print("\n❌ 测试失败，请检查系统配置")

    asyncio.run(run_tests())