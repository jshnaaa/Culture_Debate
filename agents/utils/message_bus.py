"""
智能体消息总线
实现智能体间的消息传递、路由和广播功能
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Callable, Set
from collections import defaultdict, deque
from dataclasses import dataclass, field

from ..base.agent_interface import AgentMessage, AgentType


@dataclass
class MessageRoute:
    """消息路由信息"""
    sender_pattern: str
    receiver_pattern: str
    message_type: str
    handler: Optional[Callable] = None
    priority: int = 0


@dataclass
class MessageStats:
    """消息统计信息"""
    total_sent: int = 0
    total_received: int = 0
    total_broadcast: int = 0
    failed_deliveries: int = 0
    average_delivery_time: float = 0.0
    message_types: Dict[str, int] = field(default_factory=dict)


class MessageBus:
    """智能体消息总线"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("MessageBus")

        # 消息队列配置
        self.max_queue_size = config.get("max_queue_size", 1000)
        self.message_timeout = config.get("message_timeout", 30.0)
        self.retry_attempts = config.get("retry_attempts", 3)

        # 消息存储
        self.message_queues: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_queue_size))
        self.pending_messages: Dict[str, AgentMessage] = {}
        self.message_handlers: Dict[str, Callable] = {}

        # 路由配置
        self.routes: List[MessageRoute] = []
        self.subscribers: Dict[str, Set[str]] = defaultdict(set)  # message_type -> agent_ids

        # 统计信息
        self.stats = MessageStats()
        self.delivery_times: deque = deque(maxlen=100)

        # 异步任务
        self.running = False
        self.message_processor_task = None

    async def start(self):
        """启动消息总线"""
        self.running = True
        self.message_processor_task = asyncio.create_task(self._message_processor())
        self.logger.info("消息总线已启动")

    async def stop(self):
        """停止消息总线"""
        self.running = False
        if self.message_processor_task:
            self.message_processor_task.cancel()
            try:
                await self.message_processor_task
            except asyncio.CancelledError:
                pass
        self.logger.info("消息总线已停止")

    async def send_message(self, message: AgentMessage) -> bool:
        """发送消息"""
        try:
            message_id = str(uuid.uuid4())
            message.content["message_id"] = message_id

            # 记录发送时间
            send_time = time.time()
            message.timestamp = send_time

            # 添加到目标智能体队列
            if message.receiver_id == "*":
                # 广播消息
                return await self._broadcast_message(message)
            else:
                # 单播消息
                return await self._unicast_message(message)

        except Exception as e:
            self.logger.error(f"发送消息失败: {str(e)}")
            self.stats.failed_deliveries += 1
            return False

    async def _unicast_message(self, message: AgentMessage) -> bool:
        """单播消息"""
        try:
            # 检查队列是否已满
            queue = self.message_queues[message.receiver_id]
            if len(queue) >= self.max_queue_size:
                self.logger.warning(f"智能体 {message.receiver_id} 的消息队列已满")
                return False

            # 添加到队列
            queue.append(message)
            self.stats.total_sent += 1

            # 更新消息类型统计
            msg_type = message.message_type
            self.stats.message_types[msg_type] = self.stats.message_types.get(msg_type, 0) + 1

            self.logger.debug(f"消息已发送: {message.sender_id} -> {message.receiver_id} ({msg_type})")
            return True

        except Exception as e:
            self.logger.error(f"单播消息失败: {str(e)}")
            return False

    async def _broadcast_message(self, message: AgentMessage) -> bool:
        """广播消息"""
        try:
            # 获取订阅者列表
            subscribers = self.subscribers.get(message.message_type, set())
            if not subscribers:
                self.logger.warning(f"消息类型 {message.message_type} 没有订阅者")
                return True

            success_count = 0
            for agent_id in subscribers:
                if agent_id != message.sender_id:  # 不发送给自己
                    broadcast_msg = AgentMessage(
                        sender_id=message.sender_id,
                        receiver_id=agent_id,
                        message_type=message.message_type,
                        content=message.content.copy(),
                        timestamp=message.timestamp,
                        conversation_id=message.conversation_id
                    )

                    if await self._unicast_message(broadcast_msg):
                        success_count += 1

            self.stats.total_broadcast += 1
            self.logger.debug(f"广播消息已发送给 {success_count} 个订阅者")
            return success_count > 0

        except Exception as e:
            self.logger.error(f"广播消息失败: {str(e)}")
            return False

    async def receive_message(self, agent_id: str, timeout: float = None) -> Optional[AgentMessage]:
        """接收消息"""
        try:
            timeout = timeout or self.message_timeout
            queue = self.message_queues[agent_id]

            # 等待消息到达
            start_time = time.time()
            while not queue and (time.time() - start_time) < timeout:
                await asyncio.sleep(0.1)

            if queue:
                message = queue.popleft()
                self.stats.total_received += 1

                # 计算投递时间
                delivery_time = time.time() - message.timestamp
                self.delivery_times.append(delivery_time)
                self._update_average_delivery_time()

                self.logger.debug(f"消息已接收: {agent_id} <- {message.sender_id}")
                return message

            return None

        except Exception as e:
            self.logger.error(f"接收消息失败: {str(e)}")
            return None

    def subscribe(self, agent_id: str, message_types: List[str]):
        """订阅消息类型"""
        for msg_type in message_types:
            self.subscribers[msg_type].add(agent_id)
        self.logger.debug(f"智能体 {agent_id} 订阅了消息类型: {message_types}")

    def unsubscribe(self, agent_id: str, message_types: List[str] = None):
        """取消订阅"""
        if message_types is None:
            # 取消所有订阅
            for msg_type in list(self.subscribers.keys()):
                self.subscribers[msg_type].discard(agent_id)
        else:
            for msg_type in message_types:
                self.subscribers[msg_type].discard(agent_id)

        self.logger.debug(f"智能体 {agent_id} 取消订阅")

    def add_route(self, route: MessageRoute):
        """添加消息路由"""
        self.routes.append(route)
        self.logger.debug(f"添加路由: {route.sender_pattern} -> {route.receiver_pattern}")

    def register_handler(self, message_type: str, handler: Callable):
        """注册消息处理器"""
        self.message_handlers[message_type] = handler
        self.logger.debug(f"注册处理器: {message_type}")

    async def _message_processor(self):
        """消息处理器（后台任务）"""
        self.logger.info("消息处理器启动")

        while self.running:
            try:
                # 处理路由规则
                await self._process_routes()

                # 处理超时消息
                await self._cleanup_expired_messages()

                # 统计信息更新
                await self._update_stats()

                await asyncio.sleep(0.5)

            except Exception as e:
                self.logger.error(f"消息处理器错误: {str(e)}")
                await asyncio.sleep(1.0)

        self.logger.info("消息处理器停止")

    async def _process_routes(self):
        """处理路由规则"""
        # 简单实现，可以扩展为更复杂的路由逻辑
        pass

    async def _cleanup_expired_messages(self):
        """清理过期消息"""
        current_time = time.time()
        expired_count = 0

        for agent_id, queue in self.message_queues.items():
            # 移除过期消息
            while queue:
                message = queue[0]
                if (current_time - message.timestamp) > self.message_timeout:
                    queue.popleft()
                    expired_count += 1
                else:
                    break

        if expired_count > 0:
            self.logger.debug(f"清理了 {expired_count} 条过期消息")

    async def _update_stats(self):
        """更新统计信息"""
        # 更新平均投递时间
        self._update_average_delivery_time()

    def _update_average_delivery_time(self):
        """更新平均投递时间"""
        if self.delivery_times:
            self.stats.average_delivery_time = sum(self.delivery_times) / len(self.delivery_times)

    def get_stats(self) -> MessageStats:
        """获取统计信息"""
        return self.stats

    def get_queue_status(self) -> Dict[str, Dict[str, Any]]:
        """获取队列状态"""
        status = {}
        for agent_id, queue in self.message_queues.items():
            status[agent_id] = {
                "queue_size": len(queue),
                "max_size": self.max_queue_size,
                "usage_rate": len(queue) / self.max_queue_size
            }
        return status

    def clear_queue(self, agent_id: str):
        """清空指定智能体的消息队列"""
        if agent_id in self.message_queues:
            cleared_count = len(self.message_queues[agent_id])
            self.message_queues[agent_id].clear()
            self.logger.info(f"清空智能体 {agent_id} 的消息队列，清理了 {cleared_count} 条消息")

    def clear_all_queues(self):
        """清空所有消息队列"""
        total_cleared = sum(len(queue) for queue in self.message_queues.values())
        self.message_queues.clear()
        self.logger.info(f"清空所有消息队列，总共清理了 {total_cleared} 条消息")

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            # 检查消息处理器是否运行
            if not self.running or not self.message_processor_task:
                return False

            # 检查队列使用率
            for agent_id, queue in self.message_queues.items():
                usage_rate = len(queue) / self.max_queue_size
                if usage_rate > 0.9:  # 队列使用率超过90%
                    self.logger.warning(f"智能体 {agent_id} 队列使用率过高: {usage_rate:.2%}")

            # 检查投递失败率
            if self.stats.total_sent > 0:
                failure_rate = self.stats.failed_deliveries / self.stats.total_sent
                if failure_rate > 0.1:  # 失败率超过10%
                    self.logger.warning(f"消息投递失败率过高: {failure_rate:.2%}")

            return True

        except Exception as e:
            self.logger.error(f"健康检查失败: {str(e)}")
            return False