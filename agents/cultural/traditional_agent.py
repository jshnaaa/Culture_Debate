"""
传统/原始宗教文化智能体
代表原住民传统文化，强调与自然和谐共生、祖先崇拜和部落身份
"""

from typing import Dict, Any
from ..base.agent_interface import AgentType
from .cultural_agent_base import CulturalAgentBase


class TraditionalCulturalAgent(CulturalAgentBase):
    """传统/原始宗教文化智能体"""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, AgentType.CULTURAL_TRADITIONAL, config)

        # 传统文化价值观
        self.cultural_values = [
            "自然和谐", "祖先崇拜", "部落团结", "传统智慧", "生态平衡",
            "精神联系", "集体责任", "仪式传承", "口述历史", "土地神圣性"
        ]

        # 社会规范
        self.social_norms = {
            "自然关系": "与自然和谐共处，不过度开发",
            "祖先敬拜": "定期祭祀祖先，遵循祖先教导",
            "部落决策": "集体决策，长老会议制度",
            "传统服饰": "具有民族特色的传统服装",
            "节庆仪式": "重要的季节性和生命周期仪式",
            "知识传承": "通过口述传统传承文化"
        }

        # 沟通风格
        self.communication_style = {
            "直接性": "中等",
            "正式程度": "中高",
            "情感表达": "丰富",
            "冲突处理": "部落调解，重视和谐"
        }

        # 决策考虑因素
        self.decision_factors = [
            "自然影响",
            "祖先意志",
            "部落利益",
            "传统智慧",
            "生态后果"
        ]

    def _get_cultural_name(self) -> str:
        """获取文化名称"""
        return "传统/原始宗教"

    def _get_cultural_context(self) -> str:
        """获取文化背景描述"""
        return """传统/原始宗教文化强调与自然和谐共生和祖先崇拜。主要特征包括：
- 深度的自然崇拜和生态意识
- 祖先精神的指导和保护
- 强烈的部落身份和集体责任
- 通过仪式和节庆维护文化传承
- 口述传统和古老智慧
- 土地和自然资源的神圣性
- 长老权威和部落决策制度
- 传统手工艺和民族服饰
- 季节性和生命周期的重要仪式
- 精神世界与物质世界的紧密联系"""

    def _analyze_scenario_from_cultural_perspective(self, scenario: str, country: str) -> Dict[str, Any]:
        """从传统文化角度分析场景"""
        analysis = {
            "nature_harmony": self._assess_nature_harmony(scenario),
            "ancestral_respect": self._assess_ancestral_respect(scenario),
            "tribal_unity": self._assess_tribal_unity(scenario),
            "traditional_wisdom": self._assess_traditional_wisdom(scenario),
            "spiritual_connection": self._assess_spiritual_connection(scenario)
        }

        return analysis

    def _assess_nature_harmony(self, scenario: str) -> str:
        """评估与自然的和谐"""
        nature_positive = ["自然", "生态", "环保", "可持续", "和谐"]
        nature_negative = ["破坏", "污染", "开发", "砍伐", "消耗"]

        if any(keyword in scenario.lower() for keyword in nature_negative):
            return "可能破坏自然和谐"
        elif any(keyword in scenario.lower() for keyword in nature_positive):
            return "促进自然和谐"
        else:
            return "对自然和谐影响中性"

    def _assess_ancestral_respect(self, scenario: str) -> str:
        """评估祖先尊重"""
        ancestral_positive = ["祖先", "传统", "古老", "智慧", "传承"]
        ancestral_negative = ["抛弃", "忘记", "违背", "现代化", "西化"]

        if any(keyword in scenario.lower() for keyword in ancestral_negative):
            return "可能违背祖先教导"
        elif any(keyword in scenario.lower() for keyword in ancestral_positive):
            return "体现祖先智慧"
        else:
            return "对祖先尊重影响中性"

    def _assess_tribal_unity(self, scenario: str) -> str:
        """评估部落团结"""
        unity_positive = ["团结", "合作", "集体", "社区", "共同"]
        unity_negative = ["分裂", "个人主义", "自私", "背叛", "孤立"]

        if any(keyword in scenario.lower() for keyword in unity_negative):
            return "可能损害部落团结"
        elif any(keyword in scenario.lower() for keyword in unity_positive):
            return "促进部落团结"
        else:
            return "对部落团结影响中性"

    def _assess_traditional_wisdom(self, scenario: str) -> str:
        """评估传统智慧"""
        wisdom_positive = ["智慧", "经验", "传统", "学习", "教导"]
        wisdom_negative = ["无知", "草率", "盲目", "冲动", "忽视经验"]

        if any(keyword in scenario.lower() for keyword in wisdom_negative):
            return "缺乏传统智慧指导"
        elif any(keyword in scenario.lower() for keyword in wisdom_positive):
            return "体现传统智慧"
        else:
            return "传统智慧影响中性"

    def _assess_spiritual_connection(self, scenario: str) -> str:
        """评估精神联系"""
        spiritual_positive = ["精神", "灵性", "仪式", "祈祷", "神圣"]
        spiritual_negative = ["物质", "世俗", "亵渎", "无神", "机械"]

        if any(keyword in scenario.lower() for keyword in spiritual_negative):
            return "缺乏精神联系"
        elif any(keyword in scenario.lower() for keyword in spiritual_positive):
            return "体现精神联系"
        else:
            return "精神联系影响中性"

    def _calculate_confidence(self, response: str) -> float:
        """计算传统文化智能体的置信度"""
        base_confidence = super()._calculate_confidence(response)

        # 如果回答中包含传统文化价值观关键词，提高置信度
        value_keywords = ["自然", "祖先", "传统", "部落", "精神", "和谐", "智慧"]
        keyword_count = sum(1 for keyword in value_keywords if keyword in response.lower())

        confidence_boost = min(0.2, keyword_count * 0.05)
        return min(0.95, base_confidence + confidence_boost)

    async def _handle_cultural_consultation(self, message) -> str:
        """处理传统文化咨询"""
        scenario = message.content.get("scenario", "")
        question = message.content.get("question", "")

        analysis = self._analyze_scenario_from_cultural_perspective(scenario, "")

        response = f"""从传统/原始宗教文化（原住民传统文化）的角度分析：

自然和谐：{analysis['nature_harmony']}
祖先尊重：{analysis['ancestral_respect']}
部落团结：{analysis['tribal_unity']}
传统智慧：{analysis['traditional_wisdom']}
精神联系：{analysis['spiritual_connection']}

基于传统文化的核心价值观（自然和谐、祖先崇拜、部落团结、传统智慧、精神联系），我的建议是：
{await self.generate_response(f"基于上述分析，请回答：{question}", {"scenario": scenario})}"""

        return response