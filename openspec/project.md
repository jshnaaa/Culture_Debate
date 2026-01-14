# Project Context

## Purpose
多智能体文化对齐研究项目 - 实现基于协作冲突调解的文化对齐系统，通过多个文化背景的LLM智能体进行辩论和调解，解决跨文化场景中的社会规范冲突。

## Tech Stack
- Python 3.8+
- PyTorch
- Transformers (HuggingFace)
- LLaMA 3.1 (多个智能体实例)
- JSONL数据格式
- CUDA (GPU加速)

## Project Conventions

### Code Style
- 使用中文注释和日志输出
- 函数命名采用snake_case
- 类命名采用PascalCase
- 常量使用UPPER_CASE
- 每个智能体一个独立的Python模块

### Architecture Patterns
- 多智能体协作架构
- 基于prompt template的统一接口
- 阶段化辩论流程（初始决策→冲突检测→辩论反馈→调解→最终决策）
- 模块化智能体设计（文化智能体、冲突智能体、调解智能体、决策智能体）

### Testing Strategy
- 基于NORMAD-ETI数据集的评估
- 准确率和文化群体平等性指标
- 跨文化场景的全面测试覆盖

### Git Workflow
- main分支用于稳定版本
- 特性分支用于新智能体开发
- commit信息使用中文描述

## Domain Context

### 文化智能体分类
1. **基督教文化智能体** - 代表西方个人主义文化
2. **伊斯兰文化智能体** - 代表中东集体主义文化
3. **佛教文化智能体** - 代表东亚和谐文化
4. **印度教文化智能体** - 代表南亚等级制度文化
5. **传统宗教文化智能体** - 代表原住民自然崇拜文化

### 关键概念
- **文化冲突检测** - 识别不同文化背景下的规范差异
- **辩论僵局判断** - 检测讨论停滞和重复循环
- **调解机制** - 提供新视角和折中方案
- **文化对齐** - 在尊重文化差异基础上达成共识

## Important Constraints
- 必须支持75个国家的文化背景
- 响应长度限制（初始决策≤3句话，反馈≤3句话）
- 确定性生成（temperature=0.0）
- GPU内存限制（需要同时加载多个8B-9B模型）

## External Dependencies
- HuggingFace Model Hub
- NORMAD-ETI数据集
- Cultural Atlas社会礼仪规范
- CUDA运行时环境
