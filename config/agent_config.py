"""
智能体配置管理
提供智能体配置的加载、验证和管理功能
"""

import json
import yaml
import logging
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agents.base.agent_interface import AgentType


@dataclass
class ModelConfig:
    """模型配置"""
    model_id: str
    cache_dir: str = ""
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    max_new_tokens: int = 512
    temperature: float = 0.0
    max_input_length: int = 2048


@dataclass
class CulturalConfig:
    """文化配置"""
    cultural_values: List[str]
    social_norms: Dict[str, str]
    communication_style: Dict[str, str]
    decision_factors: List[str]
    prompt_templates: Dict[str, str]


@dataclass
class AgentConfig:
    """智能体配置"""
    agent_type: str
    model_config: ModelConfig
    cultural_config: Optional[CulturalConfig] = None
    custom_config: Dict[str, Any] = None


class AgentConfigManager:
    """智能体配置管理器"""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.logger = logging.getLogger("AgentConfigManager")

        # 确保配置目录存在
        self.config_dir.mkdir(exist_ok=True)

        # 配置缓存
        self.agent_configs: Dict[AgentType, AgentConfig] = {}
        self.global_config: Dict[str, Any] = {}

        # 加载配置
        self._load_global_config()
        self._load_agent_configs()

    def _load_global_config(self):
        """加载全局配置"""
        global_config_path = self.config_dir / "global_config.yaml"

        if global_config_path.exists():
            try:
                with open(global_config_path, 'r', encoding='utf-8') as f:
                    self.global_config = yaml.safe_load(f)
                self.logger.info("全局配置加载成功")
            except Exception as e:
                self.logger.error(f"加载全局配置失败: {str(e)}")
                self.global_config = self._get_default_global_config()
        else:
            self.global_config = self._get_default_global_config()
            self._save_global_config()

    def _get_default_global_config(self) -> Dict[str, Any]:
        """获取默认全局配置"""
        return {
            "hf_token": "",
            "cache_dir": "./cache",
            "max_active_agents": 3,
            "idle_timeout": 300.0,
            "memory_threshold": 0.8,
            "message_bus": {
                "max_queue_size": 1000,
                "message_timeout": 30.0,
                "retry_attempts": 3
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }

    def _save_global_config(self):
        """保存全局配置"""
        try:
            global_config_path = self.config_dir / "global_config.yaml"
            with open(global_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.global_config, f, default_flow_style=False, allow_unicode=True)
            self.logger.info("全局配置保存成功")
        except Exception as e:
            self.logger.error(f"保存全局配置失败: {str(e)}")

    def _load_agent_configs(self):
        """加载所有智能体配置"""
        # 加载文化智能体配置
        cultural_agents = [
            AgentType.CULTURAL_CHRISTIAN,
            AgentType.CULTURAL_ISLAMIC,
            AgentType.CULTURAL_BUDDHIST,
            AgentType.CULTURAL_HINDU,
            AgentType.CULTURAL_TRADITIONAL
        ]

        for agent_type in cultural_agents:
            config = self._load_cultural_agent_config(agent_type)
            if config:
                self.agent_configs[agent_type] = config

        # 加载其他智能体配置
        other_agents = [
            AgentType.CONFLICT_DETECTOR,
            AgentType.MEDIATOR,
            AgentType.DECISION_MAKER
        ]

        for agent_type in other_agents:
            config = self._load_other_agent_config(agent_type)
            if config:
                self.agent_configs[agent_type] = config

    def _load_cultural_agent_config(self, agent_type: AgentType) -> Optional[AgentConfig]:
        """加载文化智能体配置"""
        config_file = self.config_dir / f"{agent_type.value}_config.yaml"

        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                return self._parse_agent_config(agent_type, config_data)
            except Exception as e:
                self.logger.error(f"加载 {agent_type.value} 配置失败: {str(e)}")

        # 创建默认配置
        default_config = self._get_default_cultural_config(agent_type)
        self._save_agent_config(agent_type, default_config)
        return default_config

    def _load_other_agent_config(self, agent_type: AgentType) -> Optional[AgentConfig]:
        """加载其他智能体配置"""
        config_file = self.config_dir / f"{agent_type.value}_config.yaml"

        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                return self._parse_agent_config(agent_type, config_data)
            except Exception as e:
                self.logger.error(f"加载 {agent_type.value} 配置失败: {str(e)}")

        # 创建默认配置
        default_config = self._get_default_other_config(agent_type)
        self._save_agent_config(agent_type, default_config)
        return default_config

    def _parse_agent_config(self, agent_type: AgentType, config_data: Dict[str, Any]) -> AgentConfig:
        """解析智能体配置"""
        # 解析模型配置
        model_data = config_data.get("model", {})
        model_config = ModelConfig(
            model_id=model_data.get("model_id", "meta-llama/Meta-Llama-3-8B-Instruct"),
            cache_dir=model_data.get("cache_dir", self.global_config.get("cache_dir", "")),
            torch_dtype=model_data.get("torch_dtype", "bfloat16"),
            device_map=model_data.get("device_map", "auto"),
            max_new_tokens=model_data.get("max_new_tokens", 512),
            temperature=model_data.get("temperature", 0.0),
            max_input_length=model_data.get("max_input_length", 2048)
        )

        # 解析文化配置（如果存在）
        cultural_config = None
        if "cultural" in config_data:
            cultural_data = config_data["cultural"]
            cultural_config = CulturalConfig(
                cultural_values=cultural_data.get("cultural_values", []),
                social_norms=cultural_data.get("social_norms", {}),
                communication_style=cultural_data.get("communication_style", {}),
                decision_factors=cultural_data.get("decision_factors", []),
                prompt_templates=cultural_data.get("prompt_templates", {})
            )

        return AgentConfig(
            agent_type=agent_type.value,
            model_config=model_config,
            cultural_config=cultural_config,
            custom_config=config_data.get("custom", {})
        )

    def _get_default_cultural_config(self, agent_type: AgentType) -> AgentConfig:
        """获取默认文化智能体配置"""
        # 根据智能体类型设置不同的模型
        model_mapping = {
            AgentType.CULTURAL_CHRISTIAN: "meta-llama/Meta-Llama-3-8B-Instruct",
            AgentType.CULTURAL_ISLAMIC: "meta-llama/Meta-Llama-3-8B-Instruct",
            AgentType.CULTURAL_BUDDHIST: "google/gemma-2-9b-it",
            AgentType.CULTURAL_HINDU: "microsoft/DialoGPT-medium",
            AgentType.CULTURAL_TRADITIONAL: "microsoft/DialoGPT-medium"
        }

        model_config = ModelConfig(
            model_id=model_mapping.get(agent_type, "meta-llama/Meta-Llama-3-8B-Instruct"),
            cache_dir=self.global_config.get("cache_dir", ""),
            torch_dtype="bfloat16",
            device_map="auto",
            max_new_tokens=512,
            temperature=0.0
        )

        # 获取文化特定配置
        cultural_configs = self._get_cultural_specific_configs()
        cultural_config = cultural_configs.get(agent_type)

        return AgentConfig(
            agent_type=agent_type.value,
            model_config=model_config,
            cultural_config=cultural_config,
            custom_config={}
        )

    def _get_default_other_config(self, agent_type: AgentType) -> AgentConfig:
        """获取其他智能体的默认配置"""
        model_config = ModelConfig(
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            cache_dir=self.global_config.get("cache_dir", ""),
            torch_dtype="bfloat16",
            device_map="auto",
            max_new_tokens=512,
            temperature=0.0
        )

        return AgentConfig(
            agent_type=agent_type.value,
            model_config=model_config,
            cultural_config=None,
            custom_config={}
        )

    def _get_cultural_specific_configs(self) -> Dict[AgentType, CulturalConfig]:
        """获取文化特定配置"""
        return {
            AgentType.CULTURAL_CHRISTIAN: CulturalConfig(
                cultural_values=["个人自由", "人权", "平等", "民主", "个人责任"],
                social_norms={
                    "商务穿着": "正式场合穿着得体，日常可以相对随意",
                    "社交互动": "直接沟通，握手问候，注重个人空间",
                    "决策方式": "个人决策，考虑个人利益和权利"
                },
                communication_style={"直接性": "高", "正式程度": "中等"},
                decision_factors=["个人权利和自由", "法律和规则", "公平性"],
                prompt_templates={
                    "initial_decision": "作为基督教文化代表，评估行为的社会可接受性...",
                    "feedback": "基于基督教价值观提供反馈...",
                    "final_decision": "做出最终判断..."
                }
            ),
            AgentType.CULTURAL_ISLAMIC: CulturalConfig(
                cultural_values=["谦逊", "敬畏", "家庭责任", "社会秩序", "诚实"],
                social_norms={
                    "穿着规范": "保守穿着，特别是公共场合",
                    "社交互动": "同性握手，异性避免身体接触",
                    "商务礼仪": "注重传统和尊重"
                },
                communication_style={"直接性": "中等", "正式程度": "高"},
                decision_factors=["宗教教义", "家庭和社区利益", "传统和习俗"],
                prompt_templates={
                    "initial_decision": "作为伊斯兰文化代表，评估行为的社会可接受性...",
                    "feedback": "基于伊斯兰价值观提供反馈...",
                    "final_decision": "做出最终判断..."
                }
            )
            # 其他文化配置...
        }

    def _save_agent_config(self, agent_type: AgentType, config: AgentConfig):
        """保存智能体配置"""
        try:
            config_file = self.config_dir / f"{agent_type.value}_config.yaml"
            config_dict = asdict(config)

            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

            self.logger.info(f"智能体配置保存成功: {agent_type.value}")
        except Exception as e:
            self.logger.error(f"保存智能体配置失败: {str(e)}")

    def get_agent_config(self, agent_type: AgentType) -> Optional[AgentConfig]:
        """获取智能体配置"""
        return self.agent_configs.get(agent_type)

    def get_global_config(self) -> Dict[str, Any]:
        """获取全局配置"""
        return self.global_config.copy()

    def update_global_config(self, updates: Dict[str, Any]):
        """更新全局配置"""
        self.global_config.update(updates)
        self._save_global_config()

    def update_agent_config(self, agent_type: AgentType, updates: Dict[str, Any]):
        """更新智能体配置"""
        if agent_type in self.agent_configs:
            config = self.agent_configs[agent_type]
            # 这里可以实现更复杂的配置更新逻辑
            self._save_agent_config(agent_type, config)

    def validate_config(self, agent_type: AgentType) -> bool:
        """验证智能体配置"""
        config = self.get_agent_config(agent_type)
        if not config:
            return False

        # 验证模型配置
        if not config.model_config.model_id:
            self.logger.error(f"智能体 {agent_type.value} 缺少模型ID")
            return False

        # 验证文化配置（如果需要）
        if agent_type.value.startswith("cultural_") and not config.cultural_config:
            self.logger.error(f"文化智能体 {agent_type.value} 缺少文化配置")
            return False

        return True

    def get_all_agent_types(self) -> List[AgentType]:
        """获取所有已配置的智能体类型"""
        return list(self.agent_configs.keys())

    def export_config(self, output_file: str):
        """导出配置到文件"""
        try:
            export_data = {
                "global_config": self.global_config,
                "agent_configs": {
                    agent_type.value: asdict(config)
                    for agent_type, config in self.agent_configs.items()
                }
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(export_data, f, default_flow_style=False, allow_unicode=True)

            self.logger.info(f"配置导出成功: {output_file}")
        except Exception as e:
            self.logger.error(f"配置导出失败: {str(e)}")

    def import_config(self, input_file: str):
        """从文件导入配置"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                import_data = yaml.safe_load(f)

            if "global_config" in import_data:
                self.global_config.update(import_data["global_config"])
                self._save_global_config()

            if "agent_configs" in import_data:
                for agent_type_str, config_data in import_data["agent_configs"].items():
                    try:
                        agent_type = AgentType(agent_type_str)
                        config = self._parse_agent_config(agent_type, config_data)
                        self.agent_configs[agent_type] = config
                        self._save_agent_config(agent_type, config)
                    except ValueError:
                        self.logger.warning(f"未知的智能体类型: {agent_type_str}")

            self.logger.info(f"配置导入成功: {input_file}")
        except Exception as e:
            self.logger.error(f"配置导入失败: {str(e)}")