"""
多智能体系统管理器
统一管理多智能体框架的所有组件，提供高层API
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

from .base.agent_interface import AgentType, AgentMessage, AgentResponse
from .utils.agent_pool import AgentPool
from .utils.message_bus import MessageBus
from .cultural.christian_agent import ChristianCulturalAgent
from .cultural.islamic_agent import IslamicCulturalAgent
from .cultural.buddhist_agent import BuddhistCulturalAgent
from .cultural.hindu_agent import HinduCulturalAgent
from .cultural.traditional_agent import TraditionalCulturalAgent
from config.agent_config import AgentConfigManager


class MultiAgentSystem:
    """多智能体系统管理器"""

    def __init__(self, config_dir: str = "config"):
        self.logger = logging.getLogger("MultiAgentSystem")

        # 初始化组件
        self.config_manager = AgentConfigManager(config_dir)
        self.agent_pool = None
        self.message_bus = None

        # 系统状态
        self.is_running = False
        self.conversation_contexts: Dict[str, Dict[str, Any]] = {}

        # 注册智能体类型
        self.agent_classes = {
            AgentType.CULTURAL_CHRISTIAN: ChristianCulturalAgent,
            AgentType.CULTURAL_ISLAMIC: IslamicCulturalAgent,
            AgentType.CULTURAL_BUDDHIST: BuddhistCulturalAgent,
            AgentType.CULTURAL_HINDU: HinduCulturalAgent,
            AgentType.CULTURAL_TRADITIONAL: TraditionalCulturalAgent
        }

    async def initialize(self) -> bool:
        """初始化多智能体系统"""
        try:
            self.logger.info("开始初始化多智能体系统")

            # 初始化智能体池
            pool_config = self.config_manager.get_global_config()
            self.agent_pool = AgentPool(pool_config)

            # 注册所有智能体类型
            for agent_type, agent_class in self.agent_classes.items():
                agent_config = self.config_manager.get_agent_config(agent_type)
                if agent_config:
                    # 转换配置格式
                    config_dict = self._convert_config_to_dict(agent_config)
                    self.agent_pool.register_agent_class(agent_type, agent_class, config_dict)

            # 初始化消息总线
            bus_config = pool_config.get("message_bus", {})
            self.message_bus = MessageBus(bus_config)
            await self.message_bus.start()

            # 启动定期清理任务
            asyncio.create_task(self.agent_pool.periodic_cleanup())

            self.is_running = True
            self.logger.info("多智能体系统初始化完成")
            return True

        except Exception as e:
            self.logger.error(f"多智能体系统初始化失败: {str(e)}")
            return False

    async def shutdown(self):
        """关闭多智能体系统"""
        try:
            self.logger.info("开始关闭多智能体系统")

            self.is_running = False

            # 停止消息总线
            if self.message_bus:
                await self.message_bus.stop()

            # 清理所有智能体
            if self.agent_pool:
                await self.agent_pool.cleanup_all()

            self.logger.info("多智能体系统关闭完成")

        except Exception as e:
            self.logger.error(f"关闭多智能体系统时发生错误: {str(e)}")

    def _convert_config_to_dict(self, agent_config) -> Dict[str, Any]:
        """将AgentConfig转换为字典格式"""
        config_dict = {
            "model_id": agent_config.model_config.model_id,
            "cache_dir": agent_config.model_config.cache_dir,
            "torch_dtype": agent_config.model_config.torch_dtype,
            "device_map": agent_config.model_config.device_map,
            "max_new_tokens": agent_config.model_config.max_new_tokens,
            "temperature": agent_config.model_config.temperature,
            "max_input_length": agent_config.model_config.max_input_length,
            "hf_token": self.config_manager.get_global_config().get("hf_token", "")
        }

        # 添加文化配置
        if agent_config.cultural_config:
            cultural_config = agent_config.cultural_config
            config_dict.update({
                "cultural_values": cultural_config.cultural_values,
                "social_norms": cultural_config.social_norms,
                "communication_style": cultural_config.communication_style,
                "decision_factors": cultural_config.decision_factors,
                "prompt_templates": cultural_config.prompt_templates
            })

        # 添加自定义配置
        if agent_config.custom_config:
            config_dict.update(agent_config.custom_config)

        return config_dict

    async def start_cultural_debate(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """启动文化对齐辩论"""
        if not self.is_running:
            raise RuntimeError("多智能体系统未初始化")

        conversation_id = f"debate_{int(time.time())}"
        self.conversation_contexts[conversation_id] = {
            "scenario": scenario,
            "stage": "initial_decision",
            "participants": [],
            "responses": {},
            "start_time": time.time()
        }

        try:
            self.logger.info(f"开始文化对齐辩论: {conversation_id}")

            # 第一阶段：初始决策
            initial_responses = await self._conduct_initial_decisions(conversation_id, scenario)

            # 更新对话上下文
            context = self.conversation_contexts[conversation_id]
            context["responses"]["initial"] = initial_responses
            context["stage"] = "feedback"

            # 第二阶段：交换反馈
            feedback_responses = await self._conduct_feedback_exchange(conversation_id)

            # 更新对话上下文
            context["responses"]["feedback"] = feedback_responses
            context["stage"] = "final_decision"

            # 第三阶段：最终决策
            final_responses = await self._conduct_final_decisions(conversation_id)

            # 更新对话上下文
            context["responses"]["final"] = final_responses
            context["stage"] = "completed"
            context["end_time"] = time.time()

            self.logger.info(f"文化对齐辩论完成: {conversation_id}")

            return {
                "conversation_id": conversation_id,
                "scenario": scenario,
                "initial_responses": initial_responses,
                "feedback_responses": feedback_responses,
                "final_responses": final_responses,
                "duration": context["end_time"] - context["start_time"]
            }

        except Exception as e:
            self.logger.error(f"文化对齐辩论失败: {str(e)}")
            raise

    async def _conduct_initial_decisions(self, conversation_id: str, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """进行初始决策阶段"""
        self.logger.info("开始初始决策阶段")

        # 获取所有文化智能体
        cultural_agents = []
        for agent_type in [AgentType.CULTURAL_CHRISTIAN, AgentType.CULTURAL_ISLAMIC,
                          AgentType.CULTURAL_BUDDHIST, AgentType.CULTURAL_HINDU,
                          AgentType.CULTURAL_TRADITIONAL]:
            agent = await self.agent_pool.get_agent(agent_type)
            if agent:
                cultural_agents.append(agent)

        # 并行生成初始决策
        tasks = []
        for agent in cultural_agents:
            task = self._get_agent_initial_decision(agent, scenario, conversation_id)
            tasks.append(task)

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # 整理响应结果
        initial_responses = {}
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                self.logger.error(f"智能体 {cultural_agents[i].agent_id} 初始决策失败: {str(response)}")
            else:
                initial_responses[cultural_agents[i].agent_type.value] = response

        return initial_responses

    async def _get_agent_initial_decision(self, agent, scenario: Dict[str, Any], conversation_id: str) -> Dict[str, Any]:
        """获取单个智能体的初始决策"""
        try:
            message = AgentMessage(
                sender_id="system",
                receiver_id=agent.agent_id,
                message_type="generate_response",
                content={
                    "prompt": "",  # 将由智能体内部构建
                    "context": {
                        "stage": "initial_decision",
                        "country": scenario.get("country", ""),
                        "story": scenario.get("story", ""),
                        "rule_of_thumb": scenario.get("rule_of_thumb", "")
                    }
                },
                timestamp=time.time(),
                conversation_id=conversation_id
            )

            response = await agent.process_message(message)
            parsed_response = agent.parse_response(response.response_text, "initial_decision")

            return {
                "raw_response": response.response_text,
                "parsed_response": parsed_response,
                "confidence": response.confidence,
                "processing_time": response.processing_time
            }

        except Exception as e:
            self.logger.error(f"获取智能体 {agent.agent_id} 初始决策失败: {str(e)}")
            raise

    async def _conduct_feedback_exchange(self, conversation_id: str) -> Dict[str, Any]:
        """进行反馈交换阶段"""
        self.logger.info("开始反馈交换阶段")

        context = self.conversation_contexts[conversation_id]
        initial_responses = context["responses"]["initial"]
        scenario = context["scenario"]

        # 获取所有参与的智能体
        participating_agents = []
        for agent_type_str in initial_responses.keys():
            agent_type = AgentType(agent_type_str)
            agent = await self.agent_pool.get_agent(agent_type)
            if agent:
                participating_agents.append(agent)

        # 生成反馈（每个智能体对其他智能体的回应进行反馈）
        feedback_responses = {}
        for agent in participating_agents:
            agent_type_str = agent.agent_type.value
            your_response = initial_responses[agent_type_str]["raw_response"]

            # 选择另一个智能体的回应作为反馈目标
            other_responses = {k: v for k, v in initial_responses.items() if k != agent_type_str}
            if other_responses:
                other_agent_type, other_response_data = next(iter(other_responses.items()))
                other_response = other_response_data["raw_response"]

                feedback = await self._get_agent_feedback(
                    agent, scenario, your_response, other_response, conversation_id
                )
                feedback_responses[agent_type_str] = feedback

        return feedback_responses

    async def _get_agent_feedback(self, agent, scenario: Dict[str, Any], your_response: str,
                                 other_response: str, conversation_id: str) -> Dict[str, Any]:
        """获取智能体反馈"""
        try:
            message = AgentMessage(
                sender_id="system",
                receiver_id=agent.agent_id,
                message_type="generate_response",
                content={
                    "prompt": "",
                    "context": {
                        "stage": "feedback",
                        "country": scenario.get("country", ""),
                        "story": scenario.get("story", ""),
                        "rule_of_thumb": scenario.get("rule_of_thumb", ""),
                        "your_response": your_response,
                        "other_response": other_response
                    }
                },
                timestamp=time.time(),
                conversation_id=conversation_id
            )

            response = await agent.process_message(message)

            return {
                "raw_response": response.response_text,
                "confidence": response.confidence,
                "processing_time": response.processing_time
            }

        except Exception as e:
            self.logger.error(f"获取智能体 {agent.agent_id} 反馈失败: {str(e)}")
            raise

    async def _conduct_final_decisions(self, conversation_id: str) -> Dict[str, Any]:
        """进行最终决策阶段"""
        self.logger.info("开始最终决策阶段")

        context = self.conversation_contexts[conversation_id]
        initial_responses = context["responses"]["initial"]
        feedback_responses = context["responses"]["feedback"]
        scenario = context["scenario"]

        # 获取所有参与的智能体
        participating_agents = []
        for agent_type_str in initial_responses.keys():
            agent_type = AgentType(agent_type_str)
            agent = await self.agent_pool.get_agent(agent_type)
            if agent:
                participating_agents.append(agent)

        # 生成最终决策
        final_responses = {}
        for agent in participating_agents:
            agent_type_str = agent.agent_type.value

            # 准备完整的讨论上下文
            your_initial = initial_responses[agent_type_str]["raw_response"]
            your_feedback = feedback_responses.get(agent_type_str, {}).get("raw_response", "")

            # 获取其他智能体的回应（简化处理，取第一个其他智能体）
            other_responses = {k: v for k, v in initial_responses.items() if k != agent_type_str}
            other_initial = ""
            other_feedback = ""
            if other_responses:
                other_agent_type, other_data = next(iter(other_responses.items()))
                other_initial = other_data["raw_response"]
                other_feedback = feedback_responses.get(other_agent_type, {}).get("raw_response", "")

            final_decision = await self._get_agent_final_decision(
                agent, scenario, your_initial, other_initial, your_feedback, other_feedback, conversation_id
            )
            final_responses[agent_type_str] = final_decision

        return final_responses

    async def _get_agent_final_decision(self, agent, scenario: Dict[str, Any], your_response: str,
                                       other_response: str, your_feedback: str, other_feedback: str,
                                       conversation_id: str) -> Dict[str, Any]:
        """获取智能体最终决策"""
        try:
            message = AgentMessage(
                sender_id="system",
                receiver_id=agent.agent_id,
                message_type="generate_response",
                content={
                    "prompt": "",
                    "context": {
                        "stage": "final_decision",
                        "country": scenario.get("country", ""),
                        "story": scenario.get("story", ""),
                        "rule_of_thumb": scenario.get("rule_of_thumb", ""),
                        "your_response": your_response,
                        "other_response": other_response,
                        "your_feedback": your_feedback,
                        "other_feedback": other_feedback
                    }
                },
                timestamp=time.time(),
                conversation_id=conversation_id
            )

            response = await agent.process_message(message)
            parsed_response = agent.parse_response(response.response_text, "final_decision")

            return {
                "raw_response": response.response_text,
                "parsed_response": parsed_response,
                "confidence": response.confidence,
                "processing_time": response.processing_time
            }

        except Exception as e:
            self.logger.error(f"获取智能体 {agent.agent_id} 最终决策失败: {str(e)}")
            raise

    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        stats = {
            "is_running": self.is_running,
            "active_conversations": len(self.conversation_contexts)
        }

        if self.agent_pool:
            pool_stats = self.agent_pool.get_stats()
            stats["agent_pool"] = {
                "total_agents": pool_stats.total_agents,
                "active_agents": pool_stats.active_agents,
                "memory_usage": pool_stats.gpu_utilization,
                "cache_hit_rate": pool_stats.cache_hit_rate
            }

        if self.message_bus:
            bus_stats = self.message_bus.get_stats()
            stats["message_bus"] = {
                "total_sent": bus_stats.total_sent,
                "total_received": bus_stats.total_received,
                "failed_deliveries": bus_stats.failed_deliveries,
                "average_delivery_time": bus_stats.average_delivery_time
            }

        return stats

    async def health_check(self) -> bool:
        """系统健康检查"""
        try:
            if not self.is_running:
                return False

            # 检查智能体池
            if self.agent_pool:
                pool_health = await self.agent_pool.health_check()
                if not pool_health:
                    return False

            # 检查消息总线
            if self.message_bus:
                bus_health = await self.message_bus.health_check()
                if not bus_health:
                    return False

            return True

        except Exception as e:
            self.logger.error(f"健康检查失败: {str(e)}")
            return False