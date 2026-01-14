"""
智能体基础接口定义
定义所有智能体必须实现的核心接口和抽象方法
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio


class AgentType(Enum):
    """智能体类型枚举"""
    CULTURAL_CHRISTIAN = "cultural_christian"
    CULTURAL_ISLAMIC = "cultural_islamic"
    CULTURAL_BUDDHIST = "cultural_buddhist"
    CULTURAL_HINDU = "cultural_hindu"
    CULTURAL_TRADITIONAL = "cultural_traditional"
    CONFLICT_DETECTOR = "conflict_detector"
    MEDIATOR = "mediator"
    DECISION_MAKER = "decision_maker"


class AgentStatus(Enum):
    """智能体状态枚举"""
    INACTIVE = "inactive"
    LOADING = "loading"
    ACTIVE = "active"
    PROCESSING = "processing"
    ERROR = "error"
    UNLOADING = "unloading"


@dataclass
class AgentMessage:
    """智能体消息数据结构"""
    sender_id: str
    receiver_id: str
    message_type: str
    content: Dict[str, Any]
    timestamp: float
    conversation_id: str


@dataclass
class AgentResponse:
    """智能体响应数据结构"""
    agent_id: str
    response_text: str
    confidence: float
    metadata: Dict[str, Any]
    processing_time: float


class AgentInterface(ABC):
    """智能体基础接口"""

    def __init__(self, agent_id: str, agent_type: AgentType, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config
        self.status = AgentStatus.INACTIVE
        self.conversation_history: List[AgentMessage] = []
        self.metadata: Dict[str, Any] = {}

    @abstractmethod
    async def initialize(self) -> bool:
        """
        初始化智能体，加载模型和配置

        Returns:
            bool: 初始化是否成功
        """
        pass

    @abstractmethod
    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """
        处理接收到的消息

        Args:
            message: 输入消息

        Returns:
            AgentResponse: 处理结果
        """
        pass

    @abstractmethod
    async def generate_response(self, prompt: str, context: Dict[str, Any]) -> str:
        """
        生成回应文本

        Args:
            prompt: 输入提示
            context: 上下文信息

        Returns:
            str: 生成的回应
        """
        pass

    @abstractmethod
    async def cleanup(self) -> bool:
        """
        清理智能体资源

        Returns:
            bool: 清理是否成功
        """
        pass

    def get_status(self) -> AgentStatus:
        """获取智能体状态"""
        return self.status

    def set_status(self, status: AgentStatus):
        """设置智能体状态"""
        self.status = status

    def add_message_to_history(self, message: AgentMessage):
        """添加消息到历史记录"""
        self.conversation_history.append(message)

        # 保持历史记录在合理范围内
        if len(self.conversation_history) > self.config.get("max_history_length", 100):
            self.conversation_history = self.conversation_history[-50:]

    def get_conversation_history(self, limit: Optional[int] = None) -> List[AgentMessage]:
        """获取对话历史"""
        if limit:
            return self.conversation_history[-limit:]
        return self.conversation_history

    def update_metadata(self, key: str, value: Any):
        """更新元数据"""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """获取元数据"""
        return self.metadata.get(key, default)