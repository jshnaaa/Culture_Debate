"""
佛教文化智能体
代表东亚和谐文化，强调内心平静、简朴和礼仪
"""

from typing import Dict, Any
from ..base.agent_interface import AgentType
from .cultural_agent_base import CulturalAgentBase


class BuddhistCulturalAgent(CulturalAgentBase):
    """佛教文化智能体"""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, AgentType.CULTURAL_BUDDHIST, config)

        # 佛教文化价值观
        self.cultural_values = [
            "内心平静", "慈悲", "智慧", "简朴", "和谐",
            "尊重", "谦逊", "节制", "正念", "中庸"
        ]

        # 社会规范
        self.social_norms = {
            "穿着风格": "简洁适度，避免奢华张扬",
            "社交礼仪": "深度鞠躬，双手合十问候",
            "长幼秩序": "严格的等级制度，尊敬长辈和上级",
            "商务行为": "注重礼仪和传统，避免激进表现",
            "冲突处理": "避免直接对抗，寻求和谐解决",
            "表达方式": "内敛含蓄，避免过度情感表达"
        }

        # 沟通风格
        self.communication_style = {
            "直接性": "低",
            "正式程度": "高",
            "情感表达": "内敛",
            "冲突处理": "回避冲突，寻求和谐"
        }

        # 决策考虑因素
        self.decision_factors = [
            "内心和谐",
            "集体利益",
            "传统礼仪",
            "长期后果",
            "道德修养"
        ]

    def _get_cultural_name(self) -> str:
        """获取文化名称"""
        return "佛教"

    def _get_cultural_context(self) -> str:
        """获取文化背景描述"""
        return """佛教文化强调内心平静、和谐与简朴。主要特征包括：
- 追求内心平静和精神修养
- 强调慈悲和智慧
- 重视简朴和节制的生活方式
- 严格的社会等级制度
- 深度尊重长辈和权威
- 避免冲突，寻求和谐
- 内敛含蓄的表达方式
- 注重礼仪和传统
- 商务中体现谦逊和尊重
- 强调集体利益胜过个人利益"""

    def _analyze_scenario_from_cultural_perspective(self, scenario: str, country: str) -> Dict[str, Any]:
        """从佛教文化角度分析场景"""
        analysis = {
            "inner_peace": self._assess_inner_peace(scenario),
            "compassion": self._assess_compassion(scenario),
            "simplicity": self._assess_simplicity(scenario),
            "hierarchy_respect": self._assess_hierarchy_respect(scenario),
            "harmony": self._assess_harmony(scenario)
        }

        return analysis

    def _assess_inner_peace(self, scenario: str) -> str:
        """评估内心平静"""
        disruptive_keywords = ["愤怒", "焦虑", "冲动", "激动", "暴躁"]
        peaceful_keywords = ["平静", "冷静", "思考", "沉着", "安详"]

        if any(keyword in scenario.lower() for keyword in disruptive_keywords):
            return "可能扰乱内心平静"
        elif any(keyword in scenario.lower() for keyword in peaceful_keywords):
            return "有助于内心平静"
        else:
            return "对内心平静影响中性"

    def _assess_compassion(self, scenario: str) -> str:
        """评估慈悲心"""
        compassionate_keywords = ["帮助", "关怀", "慈悲", "善良", "同情"]
        uncompassionate_keywords = ["伤害", "冷漠", "自私", "残忍", "无情"]

        if any(keyword in scenario.lower() for keyword in uncompassionate_keywords):
            return "缺乏慈悲心"
        elif any(keyword in scenario.lower() for keyword in compassionate_keywords):
            return "体现慈悲心"
        else:
            return "慈悲心表现中性"

    def _assess_simplicity(self, scenario: str) -> str:
        """评估简朴性"""
        extravagant_keywords = ["奢华", "炫耀", "浪费", "过度", "奢侈"]
        simple_keywords = ["简单", "朴素", "适度", "节俭", "简朴"]

        if any(keyword in scenario.lower() for keyword in extravagant_keywords):
            return "过于奢华，不够简朴"
        elif any(keyword in scenario.lower() for keyword in simple_keywords):
            return "体现简朴美德"
        else:
            return "简朴程度适中"

    def _assess_hierarchy_respect(self, scenario: str) -> str:
        """评估等级尊重"""
        disrespectful_keywords = ["不敬", "冒犯", "违抗", "无礼", "挑战权威"]
        respectful_keywords = ["尊敬", "恭敬", "礼貌", "服从", "尊重长辈"]

        if any(keyword in scenario.lower() for keyword in disrespectful_keywords):
            return "缺乏对等级的尊重"
        elif any(keyword in scenario.lower() for keyword in respectful_keywords):
            return "体现对等级的尊重"
        else:
            return "等级尊重表现中性"

    def _assess_harmony(self, scenario: str) -> str:
        """评估和谐性"""
        disharmonious_keywords = ["冲突", "争吵", "对立", "分歧", "破坏"]
        harmonious_keywords = ["和谐", "协调", "平衡", "统一", "融洽"]

        if any(keyword in scenario.lower() for keyword in disharmonious_keywords):
            return "可能破坏和谐"
        elif any(keyword in scenario.lower() for keyword in harmonious_keywords):
            return "促进和谐"
        else:
            return "对和谐影响中性"

    def _calculate_confidence(self, response: str) -> float:
        """计算佛教文化智能体的置信度"""
        base_confidence = super()._calculate_confidence(response)

        # 如果回答中包含佛教价值观关键词，提高置信度
        value_keywords = ["和谐", "平静", "尊重", "简朴", "慈悲", "智慧", "礼仪"]
        keyword_count = sum(1 for keyword in value_keywords if keyword in response.lower())

        confidence_boost = min(0.2, keyword_count * 0.05)
        return min(0.95, base_confidence + confidence_boost)

    async def _handle_cultural_consultation(self, message) -> str:
        """处理佛教文化咨询"""
        scenario = message.content.get("scenario", "")
        question = message.content.get("question", "")

        analysis = self._analyze_scenario_from_cultural_perspective(scenario, "")

        response = f"""从佛教文化（东亚和谐文化）的角度分析：

内心平静：{analysis['inner_peace']}
慈悲心：{analysis['compassion']}
简朴性：{analysis['simplicity']}
等级尊重：{analysis['hierarchy_respect']}
和谐性：{analysis['harmony']}

基于佛教文化的核心价值观（内心平静、慈悲、简朴、尊重、和谐），我的建议是：
基于上述分析，这个行为应该追求内心和谐与外在平衡。"""

        return response