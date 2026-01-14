"""
伊斯兰文化智能体
代表中东集体主义文化，强调谦逊、社会秩序和家庭价值
"""

from typing import Dict, Any
from ..base.agent_interface import AgentType
from .cultural_agent_base import CulturalAgentBase


class IslamicCulturalAgent(CulturalAgentBase):
    """伊斯兰文化智能体"""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, AgentType.CULTURAL_ISLAMIC, config)

        # 伊斯兰文化价值观
        self.cultural_values = [
            "谦逊", "敬畏", "家庭责任", "社会秩序", "诚实",
            "慷慨", "正义", "团结", "尊重长辈", "道德纯洁"
        ]

        # 社会规范
        self.social_norms = {
            "穿着规范": "保守穿着，特别是公共场合，女性需要适度遮蔽",
            "社交互动": "同性之间握手，异性之间避免身体接触",
            "饮食规范": "遵循清真饮食，禁酒",
            "祈祷时间": "每日五次祈祷，需要考虑祈祷时间安排",
            "家庭角色": "强调家庭责任，男性为家庭经济支柱",
            "商务礼仪": "注重传统和尊重，避免过于随意"
        }

        # 沟通风格
        self.communication_style = {
            "直接性": "中等",
            "正式程度": "高",
            "情感表达": "克制",
            "冲突处理": "寻求和谐，避免直接对抗"
        }

        # 决策考虑因素
        self.decision_factors = [
            "宗教教义",
            "家庭和社区利益",
            "传统和习俗",
            "道德纯洁性",
            "社会和谐"
        ]

    def _get_cultural_name(self) -> str:
        """获取文化名称"""
        return "伊斯兰"

    def _get_cultural_context(self) -> str:
        """获取文化背景描述"""
        return """伊斯兰文化强调集体主义、家庭价值和社会秩序。主要特征包括：
- 谦逊和敬畏是重要品德
- 家庭和社区利益优先于个人利益
- 严格的道德和行为规范
- 保守的穿着要求，特别是女性
- 强调传统和宗教教义
- 重视社会和谐与秩序
- 男女有不同的社会角色和责任
- 商务和社交中注重尊重和传统"""

    def _analyze_scenario_from_cultural_perspective(self, scenario: str, country: str) -> Dict[str, Any]:
        """从伊斯兰文化角度分析场景"""
        analysis = {
            "religious_compliance": self._assess_religious_compliance(scenario),
            "family_values": self._assess_family_values(scenario),
            "modesty": self._assess_modesty(scenario),
            "social_harmony": self._assess_social_harmony(scenario),
            "gender_roles": self._assess_gender_roles(scenario)
        }

        return analysis

    def _assess_religious_compliance(self, scenario: str) -> str:
        """评估宗教合规性"""
        haram_keywords = ["酒", "赌博", "利息", "猪肉", "不当接触"]
        halal_keywords = ["祈祷", "慈善", "诚实", "正义"]

        if any(keyword in scenario.lower() for keyword in haram_keywords):
            return "可能违反宗教规范"
        elif any(keyword in scenario.lower() for keyword in halal_keywords):
            return "符合宗教教义"
        else:
            return "宗教合规性中性"

    def _assess_family_values(self, scenario: str) -> str:
        """评估家庭价值观"""
        family_positive = ["家庭", "父母", "长辈", "责任", "照顾"]
        family_negative = ["背叛", "不孝", "分离", "忽视家庭"]

        if any(keyword in scenario.lower() for keyword in family_negative):
            return "可能损害家庭价值"
        elif any(keyword in scenario.lower() for keyword in family_positive):
            return "体现家庭价值"
        else:
            return "对家庭价值影响中性"

    def _assess_modesty(self, scenario: str) -> str:
        """评估谦逊和端庄"""
        immodest_keywords = ["暴露", "炫耀", "张扬", "不端庄"]
        modest_keywords = ["谦逊", "端庄", "朴素", "适度"]

        if any(keyword in scenario.lower() for keyword in immodest_keywords):
            return "缺乏谦逊端庄"
        elif any(keyword in scenario.lower() for keyword in modest_keywords):
            return "体现谦逊品德"
        else:
            return "谦逊程度适中"

    def _assess_social_harmony(self, scenario: str) -> str:
        """评估社会和谐"""
        disharmony_keywords = ["冲突", "争吵", "分裂", "对抗"]
        harmony_keywords = ["和谐", "团结", "合作", "和平"]

        if any(keyword in scenario.lower() for keyword in disharmony_keywords):
            return "可能破坏社会和谐"
        elif any(keyword in scenario.lower() for keyword in harmony_keywords):
            return "促进社会和谐"
        else:
            return "对社会和谐影响中性"

    def _assess_gender_roles(self, scenario: str) -> str:
        """评估性别角色"""
        # 伊斯兰文化中男女有不同的传统角色
        if "女性" in scenario.lower():
            if any(keyword in scenario.lower() for keyword in ["工作", "独立", "领导"]):
                return "需要考虑传统性别角色"
            elif any(keyword in scenario.lower() for keyword in ["家庭", "照顾", "教育子女"]):
                return "符合传统女性角色"
        elif "男性" in scenario.lower():
            if any(keyword in scenario.lower() for keyword in ["提供", "保护", "责任"]):
                return "符合传统男性角色"

        return "性别角色影响中性"

    def _calculate_confidence(self, response: str) -> float:
        """计算伊斯兰文化智能体的置信度"""
        base_confidence = super()._calculate_confidence(response)

        # 如果回答中包含伊斯兰价值观关键词，提高置信度
        value_keywords = ["谦逊", "家庭", "传统", "尊重", "社区", "道德", "秩序"]
        keyword_count = sum(1 for keyword in value_keywords if keyword in response.lower())

        confidence_boost = min(0.2, keyword_count * 0.05)
        return min(0.95, base_confidence + confidence_boost)

    async def _handle_cultural_consultation(self, message) -> str:
        """处理伊斯兰文化咨询"""
        scenario = message.content.get("scenario", "")
        question = message.content.get("question", "")

        analysis = self._analyze_scenario_from_cultural_perspective(scenario, "")

        response = f"""从伊斯兰文化（中东集体主义文化）的角度分析：

宗教合规性：{analysis['religious_compliance']}
家庭价值观：{analysis['family_values']}
谦逊端庄：{analysis['modesty']}
社会和谐：{analysis['social_harmony']}
性别角色：{analysis['gender_roles']}

基于伊斯兰文化的核心价值观（谦逊、家庭责任、社会秩序、道德纯洁），我的建议是：
{await self.generate_response(f"基于上述分析，请回答：{question}", {"scenario": scenario})}"""

        return response