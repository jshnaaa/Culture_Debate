"""
多智能体框架包
"""

from .multi_agent_system import MultiAgentSystem
from .base.agent_interface import AgentType, AgentMessage, AgentResponse, AgentStatus

__all__ = [
    'MultiAgentSystem',
    'AgentType',
    'AgentMessage',
    'AgentResponse',
    'AgentStatus'
]