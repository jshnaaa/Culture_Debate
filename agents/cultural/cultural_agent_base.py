"""
文化智能体基础类
为所有文化智能体提供共同的功能和接口实现
"""

import re
from typing import Dict, Any, List
from ..base.base_agent import BaseAgent
from ..base.agent_interface import AgentType, AgentMessage


class CulturalAgentBase(BaseAgent):
    """文化智能体基础类"""

    def __init__(self, agent_id: str, agent_type: AgentType, config: Dict[str, Any]):
        super().__init__(agent_id, agent_type, config)

        # 文化特征配置
        self.cultural_values = config.get("cultural_values", [])
        self.social_norms = config.get("social_norms", {})
        self.communication_style = config.get("communication_style", {})
        self.decision_factors = config.get("decision_factors", [])

        # Prompt模板
        self.prompt_templates = config.get("prompt_templates", {})

    def _build_prompt(self, prompt: str, context: Dict[str, Any]) -> str:
        """构建包含文化背景的prompt"""
        # 获取场景信息
        country = context.get("country", "")
        story = context.get("story", "")
        rule_of_thumb = context.get("rule_of_thumb", "")
        stage = context.get("stage", "initial_decision")

        # 根据阶段选择合适的模板
        if stage == "initial_decision":
            return self._build_initial_decision_prompt(country, story, rule_of_thumb)
        elif stage == "feedback":
            return self._build_feedback_prompt(context)
        elif stage == "final_decision":
            return self._build_final_decision_prompt(context)
        else:
            return self._build_custom_prompt(prompt, context)

    def _build_initial_decision_prompt(self, country: str, story: str, rule_of_thumb: str) -> str:
        """构建初始决策prompt"""
        cultural_context = self._get_cultural_context()

        template = self.prompt_templates.get("initial_decision", """
作为{cultural_background}文化的代表，请评估以下行为在{country}是否社会可接受。

文化背景: {cultural_context}

规则参考: {rule_of_thumb}
场景描述: {story}

请基于你的文化价值观和对{country}文化的理解，回答这个行为是否可接受。
回答格式：Yes/No/Neither，然后用不超过3句话解释你的理由。

回答：""")

        return template.format(
            cultural_background=self._get_cultural_name(),
            cultural_context=cultural_context,
            country=country,
            rule_of_thumb=rule_of_thumb,
            story=story
        )

    def _build_feedback_prompt(self, context: Dict[str, Any]) -> str:
        """构建反馈prompt"""
        country = context.get("country", "")
        story = context.get("story", "")
        rule_of_thumb = context.get("rule_of_thumb", "")
        your_response = context.get("your_response", "")
        other_response = context.get("other_response", "")

        template = self.prompt_templates.get("feedback", """
作为{cultural_background}文化的代表，你正在与其他文化背景的讨论者讨论以下场景在{country}的社会可接受性。

文化背景: {cultural_context}
规则参考: {rule_of_thumb}
场景描述: {story}

你的观点: {your_response}
对方观点: {other_response}

请基于你的文化价值观，对对方的观点提供反馈。用不超过3句话回应。

反馈：""")

        return template.format(
            cultural_background=self._get_cultural_name(),
            cultural_context=self._get_cultural_context(),
            country=country,
            rule_of_thumb=rule_of_thumb,
            story=story,
            your_response=your_response,
            other_response=other_response
        )

    def _build_final_decision_prompt(self, context: Dict[str, Any]) -> str:
        """构建最终决策prompt"""
        country = context.get("country", "")
        story = context.get("story", "")
        rule_of_thumb = context.get("rule_of_thumb", "")
        your_response = context.get("your_response", "")
        other_response = context.get("other_response", "")
        your_feedback = context.get("your_feedback", "")
        other_feedback = context.get("other_feedback", "")

        template = self.prompt_templates.get("final_decision", """
作为{cultural_background}文化的代表，基于以下完整讨论，请做出最终决策。

文化背景: {cultural_context}
规则参考: {rule_of_thumb}
场景描述: {story}

讨论过程：
你的初始观点: {your_response}
对方初始观点: {other_response}
你的反馈: {your_feedback}
对方反馈: {other_feedback}

请综合考虑讨论内容和你的文化价值观，做出最终判断。
只需回答：Yes、No 或 Neither

最终答案：""")

        return template.format(
            cultural_background=self._get_cultural_name(),
            cultural_context=self._get_cultural_context(),
            country=country,
            rule_of_thumb=rule_of_thumb,
            story=story,
            your_response=your_response,
            other_response=other_response,
            your_feedback=your_feedback,
            other_feedback=other_feedback
        )

    def _build_custom_prompt(self, prompt: str, context: Dict[str, Any]) -> str:
        """构建自定义prompt"""
        cultural_context = self._get_cultural_context()
        return f"""
作为{self._get_cultural_name()}文化的代表：

文化背景: {cultural_context}

{prompt}
"""

    def _get_cultural_name(self) -> str:
        """获取文化名称"""
        # 子类必须重写此方法
        raise NotImplementedError("子类必须实现 _get_cultural_name 方法")

    def _get_cultural_context(self) -> str:
        """获取文化背景描述"""
        # 子类必须重写此方法
        raise NotImplementedError("子类必须实现 _get_cultural_context 方法")

    def parse_response(self, response: str, stage: str = "initial_decision") -> Dict[str, Any]:
        """解析智能体响应"""
        if stage == "final_decision":
            return self._parse_final_answer(response)
        else:
            return self._parse_detailed_response(response)

    def _parse_final_answer(self, response: str) -> Dict[str, Any]:
        """解析最终答案（只有Yes/No/Neither）"""
        response = response.strip().lower()

        # 查找答案
        if "yes" in response:
            answer = "yes"
        elif "no" in response:
            answer = "no"
        elif "neither" in response:
            answer = "neither"
        else:
            answer = "neither"  # 默认值

        return {
            "answer": answer,
            "raw_response": response,
            "confidence": self._calculate_confidence(response)
        }

    def _parse_detailed_response(self, response: str) -> Dict[str, Any]:
        """解析详细响应（包含解释）"""
        lines = response.strip().split('\n')
        answer_line = ""
        explanation = ""

        # 查找答案行
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ["yes", "no", "neither"]):
                answer_line = line
                break

        # 提取其余部分作为解释
        if answer_line:
            explanation = response.replace(answer_line, "").strip()
        else:
            explanation = response.strip()

        # 解析答案
        answer_line_lower = answer_line.lower()
        if "yes" in answer_line_lower:
            answer = "yes"
        elif "no" in answer_line_lower:
            answer = "no"
        elif "neither" in answer_line_lower:
            answer = "neither"
        else:
            answer = "neither"

        return {
            "answer": answer,
            "explanation": explanation,
            "raw_response": response,
            "confidence": self._calculate_confidence(response)
        }

    def get_cultural_similarity(self, other_agent_type: AgentType) -> float:
        """计算与其他文化智能体的相似度"""
        # 简单的相似度计算，子类可以重写
        similarity_matrix = {
            (AgentType.CULTURAL_CHRISTIAN, AgentType.CULTURAL_BUDDHIST): 0.3,
            (AgentType.CULTURAL_CHRISTIAN, AgentType.CULTURAL_HINDU): 0.2,
            (AgentType.CULTURAL_CHRISTIAN, AgentType.CULTURAL_ISLAMIC): 0.4,
            (AgentType.CULTURAL_CHRISTIAN, AgentType.CULTURAL_TRADITIONAL): 0.1,
            (AgentType.CULTURAL_ISLAMIC, AgentType.CULTURAL_HINDU): 0.3,
            (AgentType.CULTURAL_ISLAMIC, AgentType.CULTURAL_BUDDHIST): 0.2,
            (AgentType.CULTURAL_ISLAMIC, AgentType.CULTURAL_TRADITIONAL): 0.2,
            (AgentType.CULTURAL_BUDDHIST, AgentType.CULTURAL_HINDU): 0.6,
            (AgentType.CULTURAL_BUDDHIST, AgentType.CULTURAL_TRADITIONAL): 0.4,
            (AgentType.CULTURAL_HINDU, AgentType.CULTURAL_TRADITIONAL): 0.3,
        }

        key1 = (self.agent_type, other_agent_type)
        key2 = (other_agent_type, self.agent_type)

        return similarity_matrix.get(key1, similarity_matrix.get(key2, 0.1))

    async def _handle_custom_message(self, message: AgentMessage) -> str:
        """处理文化智能体特定的消息类型"""
        if message.message_type == "cultural_consultation":
            return await self._handle_cultural_consultation(message)
        elif message.message_type == "value_assessment":
            return await self._handle_value_assessment(message)
        else:
            return await super()._handle_custom_message(message)

    async def _handle_cultural_consultation(self, message: AgentMessage) -> str:
        """处理文化咨询"""
        scenario = message.content.get("scenario", "")
        question = message.content.get("question", "")

        prompt = f"""
基于{self._get_cultural_name()}文化的价值观和社会规范，请回答以下问题：

场景：{scenario}
问题：{question}

请提供你的文化视角和建议。
"""

        return await self.generate_response(prompt, message.content)

    async def _handle_value_assessment(self, message: AgentMessage) -> str:
        """处理价值观评估"""
        values = message.content.get("values", [])
        context = message.content.get("context", "")

        assessments = []
        for value in values:
            if value.lower() in [v.lower() for v in self.cultural_values]:
                importance = "高"
            elif any(value.lower() in norm.lower() for norm in self.social_norms.values()):
                importance = "中"
            else:
                importance = "低"

            assessments.append(f"{value}: {importance}重要性")

        return f"从{self._get_cultural_name()}文化角度的价值观评估：\n" + "\n".join(assessments)