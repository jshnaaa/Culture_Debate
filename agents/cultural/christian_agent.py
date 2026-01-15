"""
基督教文化智能体
代表西方个人主义文化，强调自由、权利和民主价值观
"""

from typing import Dict, Any
from ..base.agent_interface import AgentType
from .cultural_agent_base import CulturalAgentBase


class ChristianCulturalAgent(CulturalAgentBase):
    """基督教文化智能体"""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, AgentType.CULTURAL_CHRISTIAN, config)

        # 基督教文化价值观
        self.cultural_values = [
            "个人自由", "人权", "平等", "民主", "个人责任",
            "诚实", "宽恕", "慈善", "正义", "尊严"
        ]

        # 社会规范
        self.social_norms = {
            "商务穿着": "正式场合穿着得体，日常可以相对随意",
            "社交互动": "直接沟通，握手问候，注重个人空间",
            "时间观念": "准时重要，提前安排",
            "决策方式": "个人决策，考虑个人利益和权利",
            "等级关系": "相对平等，尊重但不过分强调等级",
            "性别观念": "男女平等，女性有平等的社会地位"
        }

        # 沟通风格
        self.communication_style = {
            "直接性": "高",
            "正式程度": "中等",
            "情感表达": "适度",
            "冲突处理": "直面讨论，寻求妥协"
        }

        # 决策考虑因素
        self.decision_factors = [
            "个人权利和自由",
            "法律和规则",
            "个人责任",
            "公平性",
            "实用性"
        ]

    def _get_cultural_name(self) -> str:
        """获取文化名称"""
        return "基督教"

    def _get_cultural_context(self) -> str:
        """获取文化背景描述"""
        return """基督教文化强调个人主义、自由和权利。主要特征包括：
- 个人自由和权利至关重要
- 强调个人责任和道德选择
- 支持民主和平等原则
- 直接的沟通方式
- 相对宽松的社交规范
- 注重个人成就和自我实现
- 在正式场合要求得体，日常生活相对随意
- 男女平等，女性享有平等社会地位"""

    def _analyze_scenario_from_cultural_perspective(self, scenario: str, country: str) -> Dict[str, Any]:
        """从基督教文化角度分析场景"""
        analysis = {
            "individual_rights": self._assess_individual_rights(scenario),
            "personal_freedom": self._assess_personal_freedom(scenario),
            "equality": self._assess_equality(scenario),
            "social_appropriateness": self._assess_social_appropriateness(scenario, country)
        }

        return analysis

    def _assess_individual_rights(self, scenario: str) -> str:
        """评估个人权利方面"""
        if any(keyword in scenario.lower() for keyword in ["强迫", "限制", "禁止", "压制"]):
            return "可能侵犯个人权利"
        elif any(keyword in scenario.lower() for keyword in ["选择", "自由", "决定"]):
            return "支持个人权利"
        else:
            return "对个人权利影响中性"

    def _assess_personal_freedom(self, scenario: str) -> str:
        """评估个人自由方面"""
        freedom_keywords = ["自由", "选择", "决定", "表达", "行动"]
        restriction_keywords = ["限制", "禁止", "强制", "必须", "不允许"]

        if any(keyword in scenario.lower() for keyword in restriction_keywords):
            return "限制个人自由"
        elif any(keyword in scenario.lower() for keyword in freedom_keywords):
            return "体现个人自由"
        else:
            return "对个人自由影响有限"

    def _assess_equality(self, scenario: str) -> str:
        """评估平等性方面"""
        inequality_keywords = ["歧视", "区别对待", "偏见", "不公平"]
        equality_keywords = ["平等", "公平", "一视同仁", "同等"]

        if any(keyword in scenario.lower() for keyword in inequality_keywords):
            return "可能存在不平等"
        elif any(keyword in scenario.lower() for keyword in equality_keywords):
            return "体现平等原则"
        else:
            return "平等性影响中性"

    def _assess_social_appropriateness(self, scenario: str, country: str) -> str:
        """评估社会适当性"""
        # 基督教文化通常比较宽容，注重个人选择
        inappropriate_keywords = ["暴力", "伤害", "欺骗", "偷窃", "不诚实"]

        if any(keyword in scenario.lower() for keyword in inappropriate_keywords):
            return "社会不当行为"
        else:
            return "社会可接受行为"

    def _calculate_confidence(self, response: str) -> float:
        """计算基督教文化智能体的置信度"""
        base_confidence = super()._calculate_confidence(response)

        # 如果回答中包含基督教价值观关键词，提高置信度
        value_keywords = ["自由", "权利", "平等", "个人", "选择", "责任", "公平"]
        keyword_count = sum(1 for keyword in value_keywords if keyword in response.lower())

        confidence_boost = min(0.2, keyword_count * 0.05)
        return min(0.95, base_confidence + confidence_boost)

    async def _handle_cultural_consultation(self, message) -> str:
        """处理基督教文化咨询"""
        scenario = message.content.get("scenario", "")
        question = message.content.get("question", "")

        analysis = self._analyze_scenario_from_cultural_perspective(scenario, "")

        response = f"""从基督教文化（西方个人主义文化）的角度分析：

个人权利评估：{analysis['individual_rights']}
个人自由评估：{analysis['personal_freedom']}
平等性评估：{analysis['equality']}
社会适当性：{analysis['social_appropriateness']}

基于基督教文化的核心价值观（个人自由、权利、平等、责任），我的建议是：
基于上述分析，这个行为体现了个人选择和文化尊重的平衡。"""

        return response