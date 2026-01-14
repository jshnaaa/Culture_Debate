"""
智能体基础实现类
提供智能体的通用功能实现，包括模型加载、内存管理、日志记录等
"""

import torch
import logging
import time
import asyncio
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub.hf_api import HfFolder

from .agent_interface import AgentInterface, AgentType, AgentStatus, AgentMessage, AgentResponse


class BaseAgent(AgentInterface):
    """智能体基础实现类"""

    def __init__(self, agent_id: str, agent_type: AgentType, config: Dict[str, Any]):
        super().__init__(agent_id, agent_type, config)

        # 模型相关属性
        self.model = None
        self.tokenizer = None
        self.device = None

        # 性能监控
        self.total_requests = 0
        self.total_processing_time = 0.0
        self.last_activity_time = time.time()

        # 日志配置
        self.logger = logging.getLogger(f"Agent.{self.agent_id}")

        # 从配置中获取模型参数
        self.model_id = config.get("model_id")
        self.cache_dir = config.get("cache_dir", "")
        self.torch_dtype = getattr(torch, config.get("torch_dtype", "bfloat16"))
        self.device_map = config.get("device_map", "auto")
        self.max_new_tokens = config.get("max_new_tokens", 512)
        self.temperature = config.get("temperature", 0.0)

    async def initialize(self) -> bool:
        """初始化智能体"""
        try:
            self.set_status(AgentStatus.LOADING)
            self.logger.info(f"开始初始化智能体 {self.agent_id}")

            # 设置HuggingFace token
            hf_token = self.config.get("hf_token")
            if hf_token:
                HfFolder.save_token(hf_token)

            # 加载tokenizer
            self.logger.info(f"加载tokenizer: {self.model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir
            )

            # 加载模型
            self.logger.info(f"加载模型: {self.model_id}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                cache_dir=self.cache_dir,
                device_map=self.device_map,
                trust_remote_code=True
            )

            # 获取设备信息
            if hasattr(self.model, 'device'):
                self.device = self.model.device
            else:
                self.device = next(self.model.parameters()).device

            self.set_status(AgentStatus.ACTIVE)
            self.logger.info(f"智能体 {self.agent_id} 初始化完成，设备: {self.device}")
            return True

        except Exception as e:
            self.logger.error(f"智能体 {self.agent_id} 初始化失败: {str(e)}")
            self.set_status(AgentStatus.ERROR)
            return False

    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """处理消息"""
        start_time = time.time()

        try:
            self.set_status(AgentStatus.PROCESSING)
            self.add_message_to_history(message)
            self.last_activity_time = start_time

            # 根据消息类型处理
            if message.message_type == "generate_response":
                response_text = await self.generate_response(
                    message.content.get("prompt", ""),
                    message.content.get("context", {})
                )
            else:
                response_text = await self._handle_custom_message(message)

            processing_time = time.time() - start_time
            self.total_requests += 1
            self.total_processing_time += processing_time

            response = AgentResponse(
                agent_id=self.agent_id,
                response_text=response_text,
                confidence=self._calculate_confidence(response_text),
                metadata={
                    "processing_time": processing_time,
                    "message_type": message.message_type,
                    "timestamp": time.time()
                },
                processing_time=processing_time
            )

            self.set_status(AgentStatus.ACTIVE)
            return response

        except Exception as e:
            self.logger.error(f"处理消息时发生错误: {str(e)}")
            self.set_status(AgentStatus.ERROR)

            return AgentResponse(
                agent_id=self.agent_id,
                response_text=f"处理消息时发生错误: {str(e)}",
                confidence=0.0,
                metadata={"error": str(e), "timestamp": time.time()},
                processing_time=time.time() - start_time
            )

    async def generate_response(self, prompt: str, context: Dict[str, Any]) -> str:
        """生成回应"""
        if not self.model or not self.tokenizer:
            raise RuntimeError(f"智能体 {self.agent_id} 未正确初始化")

        try:
            # 构建完整的输入文本
            full_prompt = self._build_prompt(prompt, context)

            # Tokenize输入
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.get("max_input_length", 2048)
            ).to(self.device)

            # 生成回应
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # 解码输出
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()

            return response

        except Exception as e:
            self.logger.error(f"生成回应时发生错误: {str(e)}")
            raise

    async def cleanup(self) -> bool:
        """清理资源"""
        try:
            self.set_status(AgentStatus.UNLOADING)
            self.logger.info(f"开始清理智能体 {self.agent_id}")

            # 清理模型
            if self.model:
                del self.model
                self.model = None

            # 清理tokenizer
            if self.tokenizer:
                del self.tokenizer
                self.tokenizer = None

            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.set_status(AgentStatus.INACTIVE)
            self.logger.info(f"智能体 {self.agent_id} 清理完成")
            return True

        except Exception as e:
            self.logger.error(f"清理智能体 {self.agent_id} 时发生错误: {str(e)}")
            self.set_status(AgentStatus.ERROR)
            return False

    def _build_prompt(self, prompt: str, context: Dict[str, Any]) -> str:
        """构建完整的prompt"""
        # 子类可以重写此方法来自定义prompt构建逻辑
        return prompt

    async def _handle_custom_message(self, message: AgentMessage) -> str:
        """处理自定义消息类型"""
        # 子类可以重写此方法来处理特定的消息类型
        return f"收到消息类型: {message.message_type}"

    def _calculate_confidence(self, response: str) -> float:
        """计算回应的置信度"""
        # 简单的置信度计算，子类可以重写
        if not response or len(response.strip()) < 10:
            return 0.1
        return min(0.9, len(response.strip()) / 100.0)

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        avg_processing_time = (
            self.total_processing_time / self.total_requests
            if self.total_requests > 0 else 0.0
        )

        return {
            "total_requests": self.total_requests,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_processing_time,
            "last_activity_time": self.last_activity_time,
            "status": self.status.value,
            "device": str(self.device) if self.device else None
        }

    def is_idle(self, idle_threshold: float = 300.0) -> bool:
        """判断智能体是否空闲"""
        return (time.time() - self.last_activity_time) > idle_threshold