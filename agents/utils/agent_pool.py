"""
智能体池化管理器
实现智能体的动态加载、卸载和内存管理，支持多智能体并发运行
"""

import asyncio
import logging
import time
import torch
from typing import Dict, Any, Optional, List, Type
from collections import OrderedDict
from dataclasses import dataclass

from ..base.agent_interface import AgentInterface, AgentType, AgentStatus


@dataclass
class AgentPoolStats:
    """智能体池统计信息"""
    total_agents: int
    active_agents: int
    loading_agents: int
    memory_usage_mb: float
    gpu_utilization: float
    cache_hit_rate: float


class AgentPool:
    """智能体池化管理器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("AgentPool")

        # 池配置
        self.max_active_agents = config.get("max_active_agents", 3)
        self.idle_timeout = config.get("idle_timeout", 300.0)  # 5分钟
        self.memory_threshold = config.get("memory_threshold", 0.8)  # 80%

        # 智能体存储
        self.active_agents: OrderedDict[str, AgentInterface] = OrderedDict()
        self.agent_configs: Dict[AgentType, Dict[str, Any]] = {}
        self.agent_classes: Dict[AgentType, Type[AgentInterface]] = {}

        # 统计信息
        self.total_requests = 0
        self.cache_hits = 0
        self.load_operations = 0
        self.unload_operations = 0

        # 锁和信号量
        self.pool_lock = asyncio.Lock()
        self.loading_semaphore = asyncio.Semaphore(1)  # 同时只允许一个加载操作

    def register_agent_class(self, agent_type: AgentType, agent_class: Type[AgentInterface], config: Dict[str, Any]):
        """注册智能体类型和配置"""
        self.agent_classes[agent_type] = agent_class
        self.agent_configs[agent_type] = config
        self.logger.info(f"注册智能体类型: {agent_type.value}")

    async def get_agent(self, agent_type: AgentType) -> Optional[AgentInterface]:
        """获取智能体实例"""
        async with self.pool_lock:
            self.total_requests += 1
            agent_id = f"{agent_type.value}_instance"

            # 检查是否已在活跃池中
            if agent_id in self.active_agents:
                agent = self.active_agents[agent_id]
                # 移到最后（LRU更新）
                self.active_agents.move_to_end(agent_id)
                self.cache_hits += 1
                self.logger.debug(f"从缓存获取智能体: {agent_id}")
                return agent

            # 检查是否需要清理空间
            if len(self.active_agents) >= self.max_active_agents:
                await self._evict_lru_agent()

            # 检查内存使用情况
            if self._get_memory_usage() > self.memory_threshold:
                await self._cleanup_memory()

            # 加载新智能体
            return await self._load_agent(agent_type)

    async def _load_agent(self, agent_type: AgentType) -> Optional[AgentInterface]:
        """加载智能体"""
        async with self.loading_semaphore:
            try:
                if agent_type not in self.agent_classes:
                    self.logger.error(f"未注册的智能体类型: {agent_type.value}")
                    return None

                agent_id = f"{agent_type.value}_instance"
                self.logger.info(f"开始加载智能体: {agent_id}")

                # 创建智能体实例
                agent_class = self.agent_classes[agent_type]
                config = self.agent_configs[agent_type].copy()
                agent = agent_class(agent_id, agent_type, config)

                # 初始化智能体
                success = await agent.initialize()
                if not success:
                    self.logger.error(f"智能体初始化失败: {agent_id}")
                    return None

                # 添加到活跃池
                self.active_agents[agent_id] = agent
                self.load_operations += 1

                self.logger.info(f"智能体加载完成: {agent_id}")
                return agent

            except Exception as e:
                self.logger.error(f"加载智能体失败: {str(e)}")
                return None

    async def _evict_lru_agent(self):
        """驱逐最少使用的智能体"""
        if not self.active_agents:
            return

        # 获取最少使用的智能体（OrderedDict的第一个）
        lru_agent_id = next(iter(self.active_agents))
        lru_agent = self.active_agents[lru_agent_id]

        self.logger.info(f"驱逐LRU智能体: {lru_agent_id}")
        await self._unload_agent(lru_agent_id)

    async def _unload_agent(self, agent_id: str):
        """卸载智能体"""
        try:
            if agent_id not in self.active_agents:
                return

            agent = self.active_agents[agent_id]
            self.logger.info(f"开始卸载智能体: {agent_id}")

            # 清理智能体资源
            await agent.cleanup()

            # 从活跃池中移除
            del self.active_agents[agent_id]
            self.unload_operations += 1

            self.logger.info(f"智能体卸载完成: {agent_id}")

        except Exception as e:
            self.logger.error(f"卸载智能体失败: {str(e)}")

    async def _cleanup_memory(self):
        """清理内存"""
        self.logger.info("开始内存清理")

        # 查找空闲的智能体
        idle_agents = []
        current_time = time.time()

        for agent_id, agent in self.active_agents.items():
            if hasattr(agent, 'is_idle') and agent.is_idle(self.idle_timeout):
                idle_agents.append(agent_id)

        # 卸载空闲智能体
        for agent_id in idle_agents:
            await self._unload_agent(agent_id)

        # 强制垃圾回收
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info(f"内存清理完成，卸载了 {len(idle_agents)} 个空闲智能体")

    def _get_memory_usage(self) -> float:
        """获取GPU内存使用率"""
        if not torch.cuda.is_available():
            return 0.0

        try:
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            return allocated / total
        except Exception:
            return 0.0

    async def cleanup_all(self):
        """清理所有智能体"""
        self.logger.info("开始清理所有智能体")

        agent_ids = list(self.active_agents.keys())
        for agent_id in agent_ids:
            await self._unload_agent(agent_id)

        self.logger.info("所有智能体清理完成")

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            # 检查活跃智能体状态
            for agent_id, agent in self.active_agents.items():
                if agent.get_status() == AgentStatus.ERROR:
                    self.logger.warning(f"发现错误状态的智能体: {agent_id}")
                    await self._unload_agent(agent_id)

            # 检查内存使用
            memory_usage = self._get_memory_usage()
            if memory_usage > 0.9:  # 90%以上触发紧急清理
                self.logger.warning(f"内存使用率过高: {memory_usage:.2%}")
                await self._cleanup_memory()

            return True

        except Exception as e:
            self.logger.error(f"健康检查失败: {str(e)}")
            return False

    def get_stats(self) -> AgentPoolStats:
        """获取池统计信息"""
        active_count = len(self.active_agents)
        loading_count = sum(1 for agent in self.active_agents.values()
                           if agent.get_status() == AgentStatus.LOADING)

        cache_hit_rate = (self.cache_hits / self.total_requests
                         if self.total_requests > 0 else 0.0)

        return AgentPoolStats(
            total_agents=len(self.agent_classes),
            active_agents=active_count,
            loading_agents=loading_count,
            memory_usage_mb=self._get_memory_usage() * 1024,  # 转换为MB
            gpu_utilization=self._get_memory_usage(),
            cache_hit_rate=cache_hit_rate
        )

    def get_agent_list(self) -> List[Dict[str, Any]]:
        """获取智能体列表信息"""
        agents_info = []

        for agent_id, agent in self.active_agents.items():
            info = {
                "agent_id": agent_id,
                "agent_type": agent.agent_type.value,
                "status": agent.get_status().value,
                "last_activity": getattr(agent, 'last_activity_time', 0)
            }

            if hasattr(agent, 'get_performance_stats'):
                info.update(agent.get_performance_stats())

            agents_info.append(info)

        return agents_info

    async def periodic_cleanup(self):
        """定期清理任务"""
        while True:
            try:
                await asyncio.sleep(60)  # 每分钟执行一次
                await self.health_check()

                # 定期统计日志
                stats = self.get_stats()
                self.logger.info(f"池状态 - 活跃: {stats.active_agents}, "
                               f"内存: {stats.gpu_utilization:.2%}, "
                               f"缓存命中率: {stats.cache_hit_rate:.2%}")

            except Exception as e:
                self.logger.error(f"定期清理任务出错: {str(e)}")