"""
多智能体框架测试用例
验证智能体创建、消息传递、池化管理等核心功能
"""

import asyncio
import pytest
import time
import tempfile
import shutil
from pathlib import Path

from agents.base.agent_interface import AgentType, AgentMessage, AgentStatus
from agents.base.base_agent import BaseAgent
from agents.utils.agent_pool import AgentPool
from agents.utils.message_bus import MessageBus
from config.agent_config import AgentConfigManager


class MockAgent(BaseAgent):
    """测试用的模拟智能体"""

    async def initialize(self) -> bool:
        self.set_status(AgentStatus.ACTIVE)
        return True

    async def generate_response(self, prompt: str, context: dict) -> str:
        await asyncio.sleep(0.1)  # 模拟处理时间
        return f"Mock response to: {prompt}"

    async def cleanup(self) -> bool:
        self.set_status(AgentStatus.INACTIVE)
        return True


@pytest.fixture
async def temp_config_dir():
    """临时配置目录"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
async def config_manager(temp_config_dir):
    """配置管理器"""
    return AgentConfigManager(temp_config_dir)


@pytest.fixture
async def agent_pool():
    """智能体池"""
    config = {
        "max_active_agents": 2,
        "idle_timeout": 1.0,
        "memory_threshold": 0.8
    }
    pool = AgentPool(config)

    # 注册模拟智能体
    pool.register_agent_class(
        AgentType.CULTURAL_CHRISTIAN,
        MockAgent,
        {"model_id": "mock_model"}
    )

    yield pool
    await pool.cleanup_all()


@pytest.fixture
async def message_bus():
    """消息总线"""
    config = {
        "max_queue_size": 100,
        "message_timeout": 5.0,
        "retry_attempts": 2
    }
    bus = MessageBus(config)
    await bus.start()
    yield bus
    await bus.stop()


class TestAgentInterface:
    """测试智能体接口"""

    def test_agent_creation(self):
        """测试智能体创建"""
        config = {"model_id": "test_model"}
        agent = MockAgent("test_agent", AgentType.CULTURAL_CHRISTIAN, config)

        assert agent.agent_id == "test_agent"
        assert agent.agent_type == AgentType.CULTURAL_CHRISTIAN
        assert agent.status == AgentStatus.INACTIVE

    @pytest.mark.asyncio
    async def test_agent_lifecycle(self):
        """测试智能体生命周期"""
        config = {"model_id": "test_model"}
        agent = MockAgent("test_agent", AgentType.CULTURAL_CHRISTIAN, config)

        # 初始化
        success = await agent.initialize()
        assert success
        assert agent.status == AgentStatus.ACTIVE

        # 生成响应
        response = await agent.generate_response("test prompt", {})
        assert "Mock response" in response

        # 清理
        success = await agent.cleanup()
        assert success
        assert agent.status == AgentStatus.INACTIVE

    @pytest.mark.asyncio
    async def test_message_processing(self):
        """测试消息处理"""
        config = {"model_id": "test_model"}
        agent = MockAgent("test_agent", AgentType.CULTURAL_CHRISTIAN, config)
        await agent.initialize()

        message = AgentMessage(
            sender_id="sender",
            receiver_id="test_agent",
            message_type="generate_response",
            content={"prompt": "test prompt", "context": {}},
            timestamp=time.time(),
            conversation_id="test_conv"
        )

        response = await agent.process_message(message)
        assert response.agent_id == "test_agent"
        assert "Mock response" in response.response_text
        assert response.confidence > 0

        await agent.cleanup()


class TestAgentPool:
    """测试智能体池"""

    @pytest.mark.asyncio
    async def test_agent_pool_creation(self, agent_pool):
        """测试智能体池创建"""
        assert agent_pool.max_active_agents == 2
        assert len(agent_pool.active_agents) == 0

    @pytest.mark.asyncio
    async def test_agent_loading(self, agent_pool):
        """测试智能体加载"""
        agent = await agent_pool.get_agent(AgentType.CULTURAL_CHRISTIAN)
        assert agent is not None
        assert agent.agent_type == AgentType.CULTURAL_CHRISTIAN
        assert agent.status == AgentStatus.ACTIVE
        assert len(agent_pool.active_agents) == 1

    @pytest.mark.asyncio
    async def test_agent_caching(self, agent_pool):
        """测试智能体缓存"""
        # 第一次获取
        agent1 = await agent_pool.get_agent(AgentType.CULTURAL_CHRISTIAN)
        assert agent1 is not None

        # 第二次获取应该返回同一个实例
        agent2 = await agent_pool.get_agent(AgentType.CULTURAL_CHRISTIAN)
        assert agent2 is agent1

        stats = agent_pool.get_stats()
        assert stats.cache_hit_rate > 0

    @pytest.mark.asyncio
    async def test_lru_eviction(self, agent_pool):
        """测试LRU驱逐机制"""
        # 注册另一个智能体类型
        agent_pool.register_agent_class(
            AgentType.CULTURAL_ISLAMIC,
            MockAgent,
            {"model_id": "mock_model"}
        )

        agent_pool.register_agent_class(
            AgentType.CULTURAL_BUDDHIST,
            MockAgent,
            {"model_id": "mock_model"}
        )

        # 加载智能体直到达到上限
        agent1 = await agent_pool.get_agent(AgentType.CULTURAL_CHRISTIAN)
        agent2 = await agent_pool.get_agent(AgentType.CULTURAL_ISLAMIC)
        assert len(agent_pool.active_agents) == 2

        # 加载第三个智能体应该触发LRU驱逐
        agent3 = await agent_pool.get_agent(AgentType.CULTURAL_BUDDHIST)
        assert len(agent_pool.active_agents) == 2
        assert agent1.status == AgentStatus.INACTIVE  # 应该被驱逐

    @pytest.mark.asyncio
    async def test_health_check(self, agent_pool):
        """测试健康检查"""
        result = await agent_pool.health_check()
        assert result is True


class TestMessageBus:
    """测试消息总线"""

    @pytest.mark.asyncio
    async def test_message_sending(self, message_bus):
        """测试消息发送"""
        message = AgentMessage(
            sender_id="sender",
            receiver_id="receiver",
            message_type="test",
            content={"data": "test"},
            timestamp=time.time(),
            conversation_id="test_conv"
        )

        success = await message_bus.send_message(message)
        assert success

        stats = message_bus.get_stats()
        assert stats.total_sent == 1

    @pytest.mark.asyncio
    async def test_message_receiving(self, message_bus):
        """测试消息接收"""
        message = AgentMessage(
            sender_id="sender",
            receiver_id="receiver",
            message_type="test",
            content={"data": "test"},
            timestamp=time.time(),
            conversation_id="test_conv"
        )

        await message_bus.send_message(message)
        received = await message_bus.receive_message("receiver", timeout=1.0)

        assert received is not None
        assert received.sender_id == "sender"
        assert received.content["data"] == "test"

    @pytest.mark.asyncio
    async def test_message_subscription(self, message_bus):
        """测试消息订阅"""
        # 订阅消息类型
        message_bus.subscribe("agent1", ["broadcast_test"])
        message_bus.subscribe("agent2", ["broadcast_test"])

        # 发送广播消息
        message = AgentMessage(
            sender_id="broadcaster",
            receiver_id="*",
            message_type="broadcast_test",
            content={"data": "broadcast"},
            timestamp=time.time(),
            conversation_id="test_conv"
        )

        success = await message_bus.send_message(message)
        assert success

        # 检查订阅者是否收到消息
        msg1 = await message_bus.receive_message("agent1", timeout=1.0)
        msg2 = await message_bus.receive_message("agent2", timeout=1.0)

        assert msg1 is not None
        assert msg2 is not None
        assert msg1.content["data"] == "broadcast"
        assert msg2.content["data"] == "broadcast"

    @pytest.mark.asyncio
    async def test_queue_overflow(self, message_bus):
        """测试队列溢出处理"""
        # 发送大量消息
        for i in range(150):  # 超过队列大小
            message = AgentMessage(
                sender_id="sender",
                receiver_id="receiver",
                message_type="test",
                content={"data": f"message_{i}"},
                timestamp=time.time(),
                conversation_id="test_conv"
            )
            await message_bus.send_message(message)

        queue_status = message_bus.get_queue_status()
        receiver_queue = queue_status.get("receiver", {})
        assert receiver_queue.get("queue_size", 0) <= receiver_queue.get("max_size", 100)


class TestConfigManager:
    """测试配置管理器"""

    def test_config_manager_creation(self, config_manager):
        """测试配置管理器创建"""
        assert config_manager is not None
        global_config = config_manager.get_global_config()
        assert "hf_token" in global_config
        assert "cache_dir" in global_config

    def test_agent_config_loading(self, config_manager):
        """测试智能体配置加载"""
        config = config_manager.get_agent_config(AgentType.CULTURAL_CHRISTIAN)
        assert config is not None
        assert config.agent_type == AgentType.CULTURAL_CHRISTIAN.value
        assert config.model_config.model_id is not None

    def test_config_validation(self, config_manager):
        """测试配置验证"""
        # 应该验证通过
        result = config_manager.validate_config(AgentType.CULTURAL_CHRISTIAN)
        assert result is True

    def test_config_export_import(self, config_manager, temp_config_dir):
        """测试配置导出导入"""
        export_file = Path(temp_config_dir) / "exported_config.yaml"

        # 导出配置
        config_manager.export_config(str(export_file))
        assert export_file.exists()

        # 修改配置
        config_manager.update_global_config({"test_key": "test_value"})

        # 创建新的配置管理器并导入
        new_config_manager = AgentConfigManager(temp_config_dir + "_new")
        new_config_manager.import_config(str(export_file))

        # 验证导入的配置
        imported_config = new_config_manager.get_global_config()
        assert "test_key" not in imported_config  # 应该是导出时的原始配置


class TestIntegration:
    """集成测试"""

    @pytest.mark.asyncio
    async def test_full_agent_communication(self, agent_pool, message_bus):
        """测试完整的智能体通信流程"""
        # 获取智能体
        agent = await agent_pool.get_agent(AgentType.CULTURAL_CHRISTIAN)
        assert agent is not None

        # 订阅消息
        message_bus.subscribe(agent.agent_id, ["test_message"])

        # 发送消息给智能体
        test_message = AgentMessage(
            sender_id="test_sender",
            receiver_id=agent.agent_id,
            message_type="test_message",
            content={"prompt": "Hello", "context": {}},
            timestamp=time.time(),
            conversation_id="integration_test"
        )

        await message_bus.send_message(test_message)

        # 智能体接收并处理消息
        received_message = await message_bus.receive_message(agent.agent_id, timeout=2.0)
        assert received_message is not None

        response = await agent.process_message(received_message)
        assert response.agent_id == agent.agent_id
        assert "Mock response" in response.response_text

    @pytest.mark.asyncio
    async def test_multi_agent_scenario(self, agent_pool, message_bus):
        """测试多智能体场景"""
        # 注册多个智能体类型
        agent_pool.register_agent_class(
            AgentType.CULTURAL_ISLAMIC,
            MockAgent,
            {"model_id": "mock_model"}
        )

        # 获取两个不同的智能体
        agent1 = await agent_pool.get_agent(AgentType.CULTURAL_CHRISTIAN)
        agent2 = await agent_pool.get_agent(AgentType.CULTURAL_ISLAMIC)

        assert agent1 is not None
        assert agent2 is not None
        assert agent1.agent_id != agent2.agent_id

        # 设置消息订阅
        message_bus.subscribe(agent1.agent_id, ["cultural_discussion"])
        message_bus.subscribe(agent2.agent_id, ["cultural_discussion"])

        # 模拟文化讨论
        discussion_message = AgentMessage(
            sender_id="moderator",
            receiver_id="*",
            message_type="cultural_discussion",
            content={
                "scenario": "Business meeting attire",
                "country": "egypt",
                "question": "Is casual dress appropriate?"
            },
            timestamp=time.time(),
            conversation_id="cultural_debate"
        )

        await message_bus.send_message(discussion_message)

        # 两个智能体都应该收到消息
        msg1 = await message_bus.receive_message(agent1.agent_id, timeout=2.0)
        msg2 = await message_bus.receive_message(agent2.agent_id, timeout=2.0)

        assert msg1 is not None
        assert msg2 is not None

        # 处理消息并生成响应
        response1 = await agent1.process_message(msg1)
        response2 = await agent2.process_message(msg2)

        assert response1.agent_id == agent1.agent_id
        assert response2.agent_id == agent2.agent_id


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])