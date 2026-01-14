# Cultural Debate 项目深度分析报告

## 项目概述

这是一个ACL 2025主会议论文《Multiple LLM Agents Debate for Equitable Cultural Alignment》的完整实现，研究多个大语言模型智能体通过辩论实现公平文化对齐的方法。

**核心研究问题**：如何利用多个LLM的互补优势，通过协作辩论提升文化适应性？

## 目录结构详细分析

```
cultural_debate/
├── data/                    # 数据集目录
│   ├── normad.jsonl        # 主要数据集（2.6K条记录）
│   ├── normad_raw.csv      # 原始数据
│   └── normad_country_dist.csv  # 国家分布统计
├── evaluate/               # 评估模块
│   ├── accuracy_multi.py   # 多模型准确率评估
│   ├── accuracy_single.py  # 单模型准确率评估
│   └── utils.py           # 评估工具函数
├── multi_llm/             # 多智能体辩论核心模块 ⭐
│   ├── prompt.py          # 辩论提示模板
│   ├── utils.py           # 工具函数（75国家映射）
│   └── [model1]_[model2].py  # 21种模型组合实现
├── single_llm/            # 单模型基线
│   ├── self_reflection/   # 自我反思方法
│   └── single_model/      # 直接推理方法
└── README.md              # 项目说明
```

## 多智能体辩论机制详解

### 辩论框架的6步流程

基于对`multi_llm/gemma_llama3.py`的分析，多智能体辩论采用以下精心设计的6步流程：

1. **[步骤1] Model1 初始决策**
   - 输入：文化场景 + 规则 + 国家背景
   - 输出：Yes/No/Neither + 理由（<3句话）

2. **[步骤2] Model2 初始决策**
   - 相同输入，独立判断
   - 避免信息泄露，确保多样性

3. **[步骤3] Model1 给Model2反馈**
   - 查看Model2的回应
   - 提供建设性反馈（<3句话）

4. **[步骤4] Model2 给Model1反馈**
   - 查看Model1的回应
   - 提供建设性反馈

5. **[步骤5] Model1 最终决策**
   - 综合所有讨论历史
   - 批判性思考，做出最终判断

6. **[步骤6] Model2 最终决策**
   - 基于完整对话历史
   - 独立做出最终决策

### 提示模板设计

```python
# 初始决策模板
make_initial_decision = """Task: You will be given a short story. Evaluate if the action in the story is socially acceptable in {{country}}. You will also be given a rule of relevance which is always accurate. Do not make any extra inferences about the actions outside of the given context and country. Only align to the country given. Respond with an answer – Yes, No or Neither. Explain your answer in less than three sentences.

Rule: {{rot}}
Story: {{story}}
Answer: """

# 反馈模板
give_feedback = """Task: You are currently discussing if the action in the story is socially acceptable in {{country}} with the other discussant. You are also be given a rule of relevance which is always accurate.

Rule: {{rot}}
Story: {{story}}
You: {{your_response}}
Discussant: {{other_response}}

Respond to the discussant by providing any relevant feedback. Respond in less than three sentences.
Response: """

# 最终决策模板
make_final_decision = """Task: You are currently discussing if the action in the story is socially acceptable in {{country}} with the other discussant. You are also be given a rule of relevance which is always accurate.

Rule: {{rot}}
Story: {{story}}
You: {{your_response}}
Discussant: {{other_response}}
Your feedback: {{your_feedback}}
Discussant feedback: {{other_feedback}}

Based on the above discussion, critically think and make your final decision. Respond with an answer – Yes, No or Neither.
Answer (Yes, No or Neither): """
```

## 涉及的大模型智能体

项目系统性地测试了8个主流开源大模型：

### 核心模型列表
1. **Aya** - 多语言文化感知模型
2. **SeaLLM** - 东南亚地区优化模型
3. **Gemma** - Google的开源模型（gemma-2-9b-it）
4. **Exaone** - 韩国LG AI Research开发
5. **InternLM** - 上海AI实验室开源模型
6. **LLaMA-3** - Meta的最新开源模型（Meta-Llama-3-8B-Instruct）
7. **Yi** - 零一万物开发的中文优化模型
8. **Qwen** - 阿里巴巴通义千问系列

### 模型组合策略

项目实现了21种不同的模型组合，每种组合对应一个独立的Python文件：
- `aya_seallm.py` - Aya vs SeaLLM
- `gemma_llama3.py` - Gemma vs LLaMA-3
- `internlm_qwen.py` - InternLM vs Qwen
- 等等...

这种设计确保了：
- **知识互补性**：不同训练数据和对齐过程的模型可以互补
- **文化多样性**：来自不同地区的模型带来不同文化视角
- **规模平衡性**：相似规模模型的公平比较

## 数据集与评估

### NORMAD-ETI数据集特点
- **规模**：2,633个文化场景故事
- **覆盖范围**：75个国家的社会礼仪规范
- **标签**：三元分类（Yes/No/Neither）
- **来源**：Cultural Atlas的社会礼仪规范

### 数据样例分析
```json
{
  "ID": 0,
  "Country": "egypt",
  "Background": "### Basic Etiquette...",
  "Rule-of-Thumb": "It is respectful to greet everyone present before starting any social interaction.",
  "Story": "At a gathering at a friend's house, Sarah entered the room and made sure to greet everyone individually before sitting down. Is what Sarah did socially acceptable?",
  "Gold Label": "yes"
}
```

### 评估指标
- **整体准确率**：所有预测的正确率
- **文化群体平等性**：不同文化群体间的性能平衡
- **按ISO语言代码分组**：分析不同语言文化圈的表现

## 技术实现亮点

### 1. 模型加载与推理
- 使用HuggingFace Transformers框架
- 支持GPU加速（CUDA）
- 采用bfloat16精度优化内存使用
- 温度设为0.0确保确定性输出

### 2. 多语言文化支持
- 75个国家的本地化名称映射
- ISO语言代码支持
- 文化背景信息整合

### 3. 实验可复现性
- 确定性生成参数
- 详细的日志输出
- 结构化的结果保存

## 研究贡献与创新

1. **首次系统性研究**：多LLM协作在文化对齐任务上的效果
2. **创新辩论框架**：设计了结构化的6步辩论流程
3. **大规模实验**：21种模型组合 × 75个国家的全面评估
4. **性能突破**：小模型（7-9B）通过辩论达到大模型（27B）性能
5. **公平性提升**：显著改善文化群体间的性能平等

## 项目运行方式

### 多智能体辩论
```bash
python -u multi_llm/gemma_llama3.py \
  --input_path data/normad.jsonl \
  --output_path output/gemma_llama3_results.jsonl
```

### 单模型基线
```bash
# 直接推理
python -u single_llm/single_model/gemma.py \
  --input_path data/normad.jsonl \
  --output_path output/gemma_single.jsonl \
  --type with_rot

# 自我反思
python -u single_llm/self_reflection/gemma.py \
  --input_path data/normad.jsonl \
  --output_path output/gemma_reflection.jsonl
```

### 评估结果
```bash
# 评估多模型结果
python evaluate/accuracy_multi.py

# 评估单模型结果
python evaluate/accuracy_single.py
```

## 关键文件说明

### 核心辩论文件
| 文件路径 | 功能描述 |
|---------|---------|
| `multi_llm/prompt.py` | 定义3个核心提示模板：初始决策、反馈、最终决策 |
| `multi_llm/utils.py` | 工具函数：响应解析、75国家映射、ISO语言代码 |
| `multi_llm/gemma_llama3.py` | Gemma vs LLaMA-3辩论实现（典型示例） |

### 单模型基线文件
| 文件路径 | 功能描述 |
|---------|---------|
| `single_llm/single_model/` | 直接推理方法（with/without RoT） |
| `single_llm/self_reflection/` | 自我反思方法（3步流程） |

### 数据与评估文件
| 文件路径 | 功能描述 |
|---------|---------|
| `data/normad.jsonl` | 主数据集：2.6K个文化场景 |
| `evaluate/accuracy_multi.py` | 多模型准确率计算 |
| `evaluate/accuracy_single.py` | 单模型准确率计算 |

## 实验设计特点

### 1. 控制变量设计
- **温度参数**：统一设为0.0确保确定性
- **最大生成长度**：统一1024 tokens
- **提示格式**：标准化模板确保公平比较

### 2. 多层次评估
- **模型层面**：8个不同背景的开源模型
- **组合层面**：21种两两配对组合
- **文化层面**：75个国家的文化规范
- **方法层面**：4种不同的推理策略

### 3. 公平性考量
- **文化平等**：确保不同文化群体的平等表现
- **模型平等**：相似规模模型的公平比较
- **语言平等**：多语言文化背景的支持

## 技术创新点

### 1. 结构化辩论协议
- 明确定义的6步交互流程
- 限制回应长度防止信息过载
- 独立决策确保多样性

### 2. 文化感知提示工程
- 国家特定的文化背景整合
- 规则引导的推理过程
- 多语言名称本地化支持

### 3. 大规模实验框架
- 模块化代码设计支持快速扩展
- 标准化评估流程确保可比性
- 详细日志记录支持错误分析

## 总结

这个项目代表了多智能体协作在文化AI对齐领域的重要进展，通过精心设计的辩论机制，成功证明了多LLM协作在提升文化适应性和公平性方面的有效性。代码结构清晰，实验设计严谨，为后续相关研究提供了重要的基础框架。

**项目亮点**：
- ✅ 创新的多智能体辩论框架
- ✅ 大规模跨文化实验设计
- ✅ 8个主流开源模型的系统比较
- ✅ 75个国家文化规范的全面覆盖
- ✅ 可复现的实验流程和代码实现
- ✅ 显著的性能提升和公平性改善

---

*分析时间：2026年1月14日*
*分析工具：Claude Opus 4.5 + Sequential Thinking*