"""
印度教文化智能体
代表南亚等级制度文化，强调家庭关系、社会地位和精神修养
"""

from typing import Dict, Any
from ..base.agent_interface import AgentType
from .cultural_agent_base import CulturalAgentBase


class HinduCulturalAgent(CulturalAgentBase):
    """印度教文化智能体"""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, AgentType.CULTURAL_HINDU, config)

        # 印度教文化价值观
        self.cultural_values = [
            "达摩（正义）", "家庭责任", "精神修养", "业力", "轮回",
            "尊敬长辈", "等级制度", "传统仪式", "纯洁性", "奉献"
        ]

        # 社会规范
        self.social_norms = {
            "种姓制度": "传统等级制度，不同群体有不同地位",
            "家庭角色": "强调大家庭制度，多代同堂",
            "宗教仪式": "重要的宗教节日和仪式",
            "穿着传统": "传统服饰在正式场合很重要，如纱丽",
            "饮食规范": "素食主义较为普遍，牛被视为神圣",
            "婚姻观念": "家庭安排的婚姻，重视门当户对"
        }

        # 沟通风格
        self.communication_style = {
            "直接性": "低",
            "正式程度": "高",
            "情感表达": "丰富但控制",
            "冲突处理": "通过长辈调解，重视面子"
        }

        # 决策考虑因素
        self.decision_factors = [
            "家庭利益",
            "社会地位",
            "宗教义务",
            "传统习俗",
            "业力后果"
        ]

    def _get_cultural_name(self) -> str:
        """获取文化名称"""
        return "印度教"

    def _get_cultural_context(self) -> str:
        """获取文化背景描述"""
        return """印度教文化强调等级制度、家庭责任和精神修养。主要特征包括：
- 严格的社会等级制度（种姓）
- 大家庭制度和多代同堂
- 重视传统和宗教仪式
- 强调精神修养和业力
- 长辈权威和家庭决策
- 传统服饰在正式场合的重要性
- 素食主义和宗教饮食规范
- 家庭安排的婚姻制度
- 重视社会地位和声誉
- 通过仪式和传统维护社会秩序"""

    def _analyze_scenario_from_cultural_perspective(self, scenario: str, country: str) -> Dict[str, Any]:
        """从印度教文化角度分析场景"""
        analysis = {
            "dharma_compliance": self._assess_dharma_compliance(scenario),
            "family_duty": self._assess_family_duty(scenario),
            "social_hierarchy": self._assess_social_hierarchy(scenario),
            "spiritual_aspect": self._assess_spiritual_aspect(scenario),
            "traditional_values": self._assess_traditional_values(scenario)
        }

        return analysis

    def _assess_dharma_compliance(self, scenario: str) -> str:
        """评估达摩（正义）合规性"""
        righteous_keywords = ["正义", "诚实", "公正", "道德", "正确"]
        unrighteous_keywords = ["不义", "欺骗", "不公", "邪恶", "错误"]

        if any(keyword in scenario.lower() for keyword in unrighteous_keywords):
            return "违背达摩（正义）"
        elif any(keyword in scenario.lower() for keyword in righteous_keywords):
            return "符合达摩（正义）"
        else:
            return "达摩合规性中性"

    def _assess_family_duty(self, scenario: str) -> str:
        """评估家庭责任"""
        family_duty_keywords = ["家庭", "父母", "长辈", "责任", "照顾", "孝顺"]
        family_neglect_keywords = ["忽视家庭", "不孝", "背叛", "抛弃"]

        if any(keyword in scenario.lower() for keyword in family_neglect_keywords):
            return "忽视家庭责任"
        elif any(keyword in scenario.lower() for keyword in family_duty_keywords):
            return "履行家庭责任"
        else:
            return "家庭责任影响中性"

    def _assess_social_hierarchy(self, scenario: str) -> str:
        """评估社会等级制度"""
        hierarchy_respect = ["尊敬", "地位", "等级", "权威", "长辈"]
        hierarchy_challenge = ["挑战", "违抗", "不敬", "平等主义"]

        if any(keyword in scenario.lower() for keyword in hierarchy_challenge):
            return "可能挑战社会等级"
        elif any(keyword in scenario.lower() for keyword in hierarchy_respect):
            return "尊重社会等级"
        else:
            return "对社会等级影响中性"

    def _assess_spiritual_aspect(self, scenario: str) -> str:
        """评估精神修养方面"""
        spiritual_positive = ["冥想", "祈祷", "修行", "精神", "灵性"]
        spiritual_negative = ["物质主义", "贪婪", "世俗", "堕落"]

        if any(keyword in scenario.lower() for keyword in spiritual_negative):
            return "缺乏精神修养"
        elif any(keyword in scenario.lower() for keyword in spiritual_positive):
            return "体现精神修养"
        else:
            return "精神修养影响中性"

    def _assess_traditional_values(self, scenario: str) -> str:
        """评估传统价值观"""
        traditional_keywords = ["传统", "习俗", "仪式", "古老", "祖先"]
        modern_keywords = ["现代", "西方", "新潮", "革新", "改变"]

        if any(keyword in scenario.lower() for keyword in modern_keywords):
            if any(keyword in scenario.lower() for keyword in ["冲突", "违背", "抛弃"]):
                return "可能冲击传统价值"
            else:
                return "现代与传统的平衡"
        elif any(keyword in scenario.lower() for keyword in traditional_keywords):
            return "维护传统价值"
        else:
            return "传统价值影响中性"

    def _calculate_confidence(self, response: str) -> float:
        """计算印度教文化智能体的置信度"""
        base_confidence = super()._calculate_confidence(response)

        # 如果回答中包含印度教价值观关键词，提高置信度
        value_keywords = ["达摩", "家庭", "传统", "等级", "精神", "业力", "仪式"]
        keyword_count = sum(1 for keyword in value_keywords if keyword in response.lower())

        confidence_boost = min(0.2, keyword_count * 0.05)
        return min(0.95, base_confidence + confidence_boost)

    async def _handle_cultural_consultation(self, message) -> str:
        """处理印度教文化咨询"""
        scenario = message.content.get("scenario", "")
        question = message.content.get("question", "")

        analysis = self._analyze_scenario_from_cultural_perspective(scenario, "")

        response = f"""从印度教文化（南亚等级制度文化）的角度分析：

达摩合规性：{analysis['dharma_compliance']}
家庭责任：{analysis['family_duty']}
社会等级：{analysis['social_hierarchy']}
精神修养：{analysis['spiritual_aspect']}
传统价值：{analysis['traditional_values']}

基于印度教文化的核心价值观（达摩、家庭责任、社会等级、精神修养、传统仪式），我的建议是：
基于上述分析，这个行为应该符合达摩和传统智慧的指导。"""

        return response