# 设计方案总结

## 一、你的4个问题的完整回答

### 1. 国家权重是否可学习？

**是的，权重完全可学习**。我设计了一个两阶段方案：

#### Stage 1: 初始化（基于文化知识）
- 定义了8个文化原型（Nordic, East Asian, Middle Eastern等）
- 为75个国家生成初始权重向量 $\mathbf{w}_c \in \mathbb{R}^5$
- 约束：$\sum w_i = 1, w_i \geq 0$
- 已生成并保存在 `data/country_weights_init.json`

#### Stage 2: 学习优化（基于NORMAD训练集）
使用**投影梯度下降（Projected Gradient Descent）**：

**损失函数**：
```
L = L_accuracy + λ_reg * L_reg

其中：
- L_accuracy: 交叉熵损失（预测 vs gold label）
- L_reg = α||W||² + β*Σ(-vi*log(vi)) + γ*Σ||w_i-w_j||²
  - L2正则：防止权重过大
  - 熵正则：保持权重多样性（避免one-hot）
  - 文化一致性：相似国家权重接近
```

**优化算法**：
```python
# 每次迭代
1. 梯度下降: W = W - lr * grad_W
2. 投影到simplex: W[i] = project_to_simplex(W[i]) for all i
```

**Simplex投影**（Euclidean projection）：
- 保证 $\sum w_i = 1, w_i \geq 0$
- 已实现在 `utils/weight_learner.py` 的 `project_to_simplex()` 函数

**实现位置**：
- 核心代码：`utils/weight_learner.py`
- 包含 `CountryWeightLearner` 类和 `WeightLearningLoss` 类
- 支持保存/加载学习后的权重

---

### 2. 价值标签提取方案

**使用 Qwen2.5-7B 提取**（和Conflict Analyzer共用模型）

**提取Prompt**（已实现）：
```python
VALUE_EXTRACTION_PROMPT = """
Analyze the following cultural perspective and rate how much it emphasizes
each value dimension on a 0-1 scale.

Response: {agent_response}

Value Dimensions:
1. Autonomy: self-direction, independence, personal choice
2. Order & Security: stability, safety, rules, protection
3. Tradition: respect for customs, cultural heritage
4. Care & Universalism: welfare, equality, justice, compassion
5. Achievement & Power: success, influence, status, competence

Output JSON format (no explanation):
{
  "autonomy": 0.0-1.0,
  "order_security": 0.0-1.0,
  "tradition": 0.0-1.0,
  "care_universalism": 0.0-1.0,
  "achievement_power": 0.0-1.0
}
"""
```

**Pipeline整合**：
```
Cultural Agent生成 (Llama3.1-8B)
  ↓
价值标签提取 (Qwen2.5-7B, batch处理5个回答)
  ↓
Conflict分析 (Qwen2.5-7B, 复用已加载模型)
  ↓
Mediator调解 (Qwen2.5-14B, 切换模型)
```

**实现位置**：
- 核心代码：`utils/value_extractor.py`
- 包含 `extract_value_tags()` 和 `batch_extract_value_tags()` 函数
- 自动解析JSON输出，失败时回退到均匀分布

---

### 3. 第一阶段聚焦（NORMAD选择题）

**确认：只考虑第一阶段**
- 数据集：NORMAD 2,633个选择题样本
- 评估指标：
  1. 准确率（Accuracy）：与gold label对比
  2. 冲突下降率：$(C_0 - C_f) / C_0$
  3. 多样性保持度：$D_f / D_0$（应>0.8）
  4. 调解成功率：成功降低冲突的样本占比
  5. 文化群体公平性：各文化圈准确率方差

**第二阶段（开放问答）暂不实现**，代码框架已预留扩展接口。

---

### 4. 2×48GB GPU推理方案

**硬件配置分析**：
- Llama3.1-8B (bfloat16)：~16GB
- Qwen2.5-7B (bfloat16)：~14GB
- Qwen2.5-14B (bfloat16)：~28GB
- 如果同时加载：58GB > 48GB（单卡OOM）

**推荐方案：按需加载（Sequential Loading）**

```python
class ModelManager:
    def load_model(self, model_name):
        # 1. 卸载当前模型
        if self.current_model is not None:
            del self.current_model
            torch.cuda.empty_cache()

        # 2. 加载新模型
        self.current_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device  # 单卡即可
        )
```

**推理流程**：
```python
# Round 0: 生成5个agent回答
model_mgr.load_model("cultural_agent")  # 16GB
responses = model_mgr.batch_generate([prompt1, ..., prompt5])

# Round 1: 提取标签 + 冲突分析
model_mgr.load_model("conflict_analyzer")  # 14GB
value_tags = batch_extract_value_tags(responses)
conflict_report = analyze_conflict(responses, value_tags)

# Round 2: 调解（如果需要）
if conflict_report['conflict_score'] > threshold:
    model_mgr.load_model("mediator")  # 28GB
    mediation = generate_mediation(responses, conflict_report)
```

**优点**：
- 内存峰值：28GB（单卡可放下）
- 不需要vllm的tensor parallelism
- 代码简单，易调试
- 支持batch inference（5个agent并行生成）

**实现位置**：
- 核心代码：`utils/model_manager.py`
- 包含 `ModelManager` 类
- 支持单个/批量生成
- 自动内存管理和清理

---

## 二、已生成的文件清单

### 核心实现文件

1. **`data/country_weights_init.json`**
   - 75个国家的初始5维权重向量
   - 基于8个文化原型生成
   - 约束：sum=1, all≥0

2. **`utils/weight_learner.py`**
   - `CountryWeightLearner` 类：可学习权重模块
   - `project_to_simplex()` 函数：Simplex投影
   - `WeightLearningLoss` 类：损失函数（accuracy + 正则化）
   - `train_weights()` 函数：训练流程

3. **`utils/model_manager.py`**
   - `ModelManager` 类：按需加载模型
   - `generate()` 方法：单个生成
   - `batch_generate()` 方法：批量生成
   - 自动内存管理

4. **`utils/value_extractor.py`**
   - `extract_value_tags()` 函数：提取单个回答的价值标签
   - `batch_extract_value_tags()` 函数：批量提取
   - `compute_value_distance()` 函数：计算价值距离（用于冲突分析）

### 文档文件

5. **`README.md`**
   - 完整的使用指南
   - 快速开始教程
   - API文档
   - 常见问题解答

6. **`quick_start.py`**
   - 端到端示例代码
   - 演示完整pipeline（从样本到最终决策）
   - 可直接运行（更新模型路径后）

7. **`config/model_paths.json.template`**
   - 模型路径配置模板
   - 包含硬件需求说明

8. **`DESIGN_SUMMARY.md`**（本文件）
   - 设计方案总结
   - 回答你的4个问题
   - 实现细节说明

---

## 三、核心技术要点

### 3.1 权重学习的关键

**为什么需要投影？**
- 梯度下降可能导致权重违反约束（sum≠1或负值）
- Simplex投影保证每次更新后权重合法
- 投影是凸优化问题，有唯一解

**为什么需要熵正则？**
- 防止权重退化到one-hot（如[1,0,0,0,0]）
- 保持文化多样性（鼓励多个维度都有贡献）
- 权衡：λ_entropy太大会导致权重过于均匀

**为什么需要文化一致性？**
- 相似国家（如北欧三国）应有相似权重
- 避免过拟合到单个国家的噪声数据
- 提高泛化能力

### 3.2 按需加载的关键

**为什么不用vllm的tensor parallelism？**
- 你的模型最大28GB，单卡48GB足够
- Tensor parallelism增加通信开销
- 按需加载更简单，调试更容易

**如何优化加载速度？**
- 使用 `torch.cuda.empty_cache()` 确保内存释放
- 缓存tokenizer（轻量级，不需要卸载）
- 使用 `gc.collect()` 强制垃圾回收

**Batch inference的优势？**
- 5个Cultural Agent的回答可以一次生成
- 减少模型加载次数
- 提高GPU利用率

### 3.3 价值标签提取的关键

**为什么用LLM而不是规则？**
- 规则难以覆盖所有表达方式
- LLM可以理解隐含的价值强调
- 更灵活，适应不同风格的回答

**如何处理提取失败？**
- 正则表达式提取JSON
- 验证key的完整性
- 失败时回退到均匀分布[0.2, 0.2, 0.2, 0.2, 0.2]

**为什么用Jaccard距离？**
- 适合连续值的集合相似度
- 归一化到[0,1]，易于组合
- 计算高效

---

## 四、下一步工作

### 立即可做

1. **更新模型路径**
   ```bash
   cp config/model_paths.json.template config/model_paths.json
   # 编辑 model_paths.json，填入你的本地路径
   ```

2. **测试权重投影**
   ```bash
   python utils/weight_learner.py
   # 应该输出投影前后的权重，验证sum=1
   ```

3. **测试模型加载**
   ```bash
   python utils/model_manager.py
   # 需要先更新脚本中的模型路径
   ```

4. **运行快速开始**
   ```bash
   python quick_start.py
   # 完整演示一个样本的处理流程
   ```

### 需要实现的模块

以下模块在设计文档中提到，但代码尚未实现（你可以根据需要逐步添加）：

1. **`agents/cultural_agent.py`**
   - `CulturalAgent` 类
   - 包含prompt模板和生成逻辑

2. **`agents/conflict_analyzer.py`**
   - `ConflictAnalyzer` 类
   - 计算冲突分数（embedding + value + polarity）

3. **`agents/mediator_agent.py`**
   - `MediatorAgent` 类
   - 生成调解建议

4. **`agents/fairness_evaluator.py`**
   - `FairnessEvaluator` 类
   - 检测调解偏向

5. **`utils/embedding_utils.py`**
   - Sentence-Transformer封装
   - 余弦相似度计算

6. **`utils/metrics.py`**
   - 评估指标计算函数

7. **`utils/visualization.py`**
   - 可视化函数（冲突演化、热力图等）

8. **`experiments/run_full_system.py`**
   - 主实验脚本
   - 在2,633个样本上运行完整系统

### 实验计划

按优先级排序：

1. **验证pipeline**（1-2天）
   - 在10个样本上测试完整流程
   - 确保所有模块正常工作

2. **实现缺失模块**（3-5天）
   - 按上述列表逐个实现
   - 每个模块写单元测试

3. **权重初始化验证**（1天）
   - 在训练集上验证初始权重的准确率
   - 如果太低（<50%），需要调整初始权重

4. **权重学习实验**（2-3天）
   - 在训练集上训练权重
   - 在验证集上评估
   - 调优超参数（lr, λ, μ）

5. **Baseline对比**（3-5天）
   - 运行Single Model, Self-Reflection, Two-Agent
   - 对比准确率

6. **Ablation实验**（3-5天）
   - 6个配置（见README）
   - 分析各组件贡献

7. **可视化和分析**（2-3天）
   - 生成所有图表
   - 撰写实验报告

**总计：约3-4周完成全部实验**

---

## 五、关键决策总结

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 权重是否可学习 | **是**，用投影梯度下降 | 提高准确率，适应数据 |
| 价值标签提取 | **LLM**（Qwen2.5-7B） | 灵活，适应多样表达 |
| 数据集范围 | **仅第一阶段**（NORMAD） | 简化实验，聚焦核心 |
| GPU推理方案 | **按需加载**（单卡） | 内存充足，代码简单 |
| 文化原型数量 | **8个** | 覆盖主要文化圈，可解释 |
| 冲突分数计算 | **线性组合**（α=0.4, β=0.4, γ=0.2） | 可解释，易调参 |
| 调解轮次上限 | **3轮** | 平衡效果和效率 |
| 多样性度量 | **分布熵** | 直观，计算简单 |

---

## 六、常见问题预判

### Q1: 权重学习会不会过拟合？

**A**: 有3个机制防止过拟合：
1. L2正则化（λ_l2=0.01）
2. 熵正则化（鼓励权重分散）
3. 文化一致性正则（相似国家权重接近）

如果仍然过拟合，可以：
- 增大正则化权重
- 使用早停（early stopping）
- 在验证集上选择最优权重

### Q2: 如何验证权重学习有效？

**A**: 对比实验：
1. **固定初始权重**：不训练，直接用初始权重
2. **学习后权重**：训练后的权重
3. **对比指标**：训练集准确率、验证集准确率、测试集准确率

预期：学习后权重在训练集和验证集上都更好，测试集略有提升。

### Q3: 按需加载会不会太慢？

**A**: 时间成本分析：
- 模型加载时间：~10-30秒/次
- 每个样本需要加载3次（Cultural → Conflict → Mediator）
- 总时间：2,633样本 × 3次 × 20秒 ≈ 44小时

优化方案：
- 批处理样本（每次加载处理100个样本）
- 减少加载次数（如果冲突低，跳过Mediator）
- 预期可降低到10-15小时

### Q4: 如何调试价值标签提取失败？

**A**: 分步检查：
1. 打印LLM原始输出，检查是否包含JSON
2. 如果没有JSON，调整prompt（增加few-shot示例）
3. 如果JSON格式错误，增强正则表达式
4. 记录失败率，如果>10%，需要改进prompt

### Q5: 初始权重是否合理？

**A**: 验证方法：
1. 检查文化圈内一致性（如北欧国家权重应接近）
2. 在训练集上测试准确率（应>40%，否则初始化有问题）
3. 可视化权重分布（t-SNE降维，检查聚类）

如果不合理：
- 重新设计文化原型
- 用GPT-4重新生成初始权重
- 参考Hofstede/GLOBE数据集

---

## 七、联系和后续支持

如果在实现过程中遇到问题，可以：

1. **检查代码注释**：所有模块都有详细注释
2. **运行测试**：每个模块的 `if __name__ == "__main__"` 部分有测试代码
3. **参考README**：包含完整的使用示例
4. **查看quick_start.py**：端到端示例

**祝实验顺利！🚀**
