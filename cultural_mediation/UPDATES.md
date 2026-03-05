# 更新说明 - 双卡并行版本

## 主要变更

### 1. 架构变更：双卡Pipeline Parallelism

**之前的设计**（单卡按需加载）：
- 单卡48GB
- 3个模型按需加载：Llama3.1-8B → Qwen2.5-7B → Qwen2.5-14B
- 每次切换模型需要卸载和加载

**现在的设计**（双卡并行）：
- **GPU0 (cuda:0)**: Llama3.1-8B（Cultural Agent，常驻内存）
- **GPU1 (cuda:1)**: Qwen2.5-14B（价值提取 + 冲突分析 + 调解，常驻内存）
- 两个模型同时加载，不需要频繁切换

**优势**：
- ✅ 避免模型加载时间（节省~20秒/样本）
- ✅ 充分利用双卡资源（GPU0: 16GB, GPU1: 28GB）
- ✅ 代码更简洁（不需要手动管理加载/卸载）

### 2. 模型配置简化

**之前**：3个模型
- Cultural Agent: Llama3.1-8B
- Conflict Analyzer: Qwen2.5-7B
- Mediator: Qwen2.5-14B

**现在**：2个模型
- Cultural Agent: Llama3.1-8B (GPU0)
- Qwen Unified: Qwen2.5-14B (GPU1) - 用于所有GPU1任务

**理由**：
- Qwen2.5-14B能力更强，冲突分析更准确
- 简化架构，减少模型数量
- GPU1的48GB足够运行14B模型

### 3. NORMAD数据集格式适配

**实际格式**：
```json
{
  "instruction": "### Question: ... Background: ... Rule-of-Thumb: ... Story: ...",
  "input": "",
  "output": "1",  // 1=yes, 2=no, 3=neutral
  "country": "egypt"
}
```

**解析逻辑**：
- 用正则表达式从instruction提取Background、Rule-of-Thumb、Story
- 将output (1/2/3) 映射到 (yes/no/neutral)
- 已实现在 `utils/normad_loader.py`

### 4. 模型路径配置

**实际路径**：
```
Llama3.1-8B: /root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct
Qwen2.5-14B: /root/autodl-tmp/CultureMoE/Culture_Alignment/Qwen2.5-14B-Instruct
NORMAD数据: /root/autodl-fs/normad_merge_gen.json
```

已更新到 `config/model_paths.json`

---

## 新增文件

### 1. `utils/normad_loader.py` ✨ NEW
- **功能**：加载和解析NORMAD数据集
- **类**：
  - `NORMADSample`: 表示单个样本，自动解析instruction
  - `NORMADLoader`: 加载整个数据集
- **用法**：
  ```python
  loader = NORMADLoader("/root/autodl-fs/normad_merge_gen.json")
  sample = loader.get_sample(0)
  print(sample.country, sample.gold_label, sample.story)
  ```

### 2. `utils/dual_gpu_manager.py` ✨ NEW
- **功能**：双卡模型管理器
- **类**：`DualGPUModelManager`
- **方法**：
  - `generate_cultural_responses(prompts)`: GPU0批量生成
  - `generate_with_qwen(prompt)`: GPU1单个生成
  - `batch_generate_with_qwen(prompts)`: GPU1批量生成
  - `get_memory_usage()`: 查看GPU内存
- **用法**：
  ```python
  mgr = DualGPUModelManager()
  responses = mgr.generate_cultural_responses([prompt1, ..., prompt5])
  conflict = mgr.generate_with_qwen(conflict_prompt)
  ```

### 3. `config/model_paths.json` ✨ NEW
- **功能**：模型路径和设备配置
- **内容**：包含实际的模型路径和GPU分配

---

## 修改文件

### 1. `utils/value_extractor.py` 🔧 UPDATED
- **变更**：适配 `DualGPUModelManager`
- **函数签名**：
  ```python
  # 之前
  extract_value_tags(response, model_manager)

  # 现在
  extract_value_tags(response, dual_gpu_manager)
  ```
- **调用方式**：使用 `dual_gpu_manager.generate_with_qwen()`

### 2. `quick_start.py` 🔧 UPDATED
- **变更**：完全重写，适配双卡和新数据格式
- **主要改动**：
  - 使用 `DualGPUModelManager` 替代 `ModelManager`
  - 使用 `NORMADLoader` 加载数据
  - 适配新的instruction解析逻辑
  - 只测试一个样本（不是完整数据集）
- **运行方式**：
  ```bash
  python quick_start.py
  ```

### 3. `README.md` 🔧 UPDATED
- **变更**：更新硬件要求、模型配置、使用示例
- **主要章节**：
  - 硬件要求：改为双卡
  - 配置模型路径：更新为实际路径
  - 运行单个样本：更新代码示例

---

## 保持不变的文件

以下文件无需修改，仍然可用：

- ✅ `data/country_weights_init.json` - 75国家初始权重
- ✅ `utils/weight_learner.py` - 权重学习模块
- ✅ `DESIGN_SUMMARY.md` - 设计方案总结

---

## 使用指南

### 快速测试

1. **测试NORMAD加载**：
   ```bash
   cd utils
   python normad_loader.py
   ```
   预期输出：解析示例样本，显示Background、RoT、Story

2. **测试双卡管理器**：
   ```bash
   cd utils
   python dual_gpu_manager.py
   ```
   预期输出：加载两个模型，显示GPU内存使用

3. **运行完整示例**：
   ```bash
   python quick_start.py
   ```
   预期输出：
   - 加载模型（GPU0 + GPU1）
   - 加载NORMAD样本
   - 生成5个agent回答
   - 提取价值标签
   - 分析冲突
   - （如果冲突高）生成调解
   - 计算最终答案
   - 显示准确率

### 预期运行时间

- 模型加载：~30秒（一次性）
- 单个样本处理：~10-15秒
  - Cultural Agent生成（5个）：~5秒
  - 价值标签提取（5个）：~3秒
  - 冲突分析：~1秒
  - 调解（如果需要）：~3秒

### 内存使用

- GPU0: ~16GB（Llama3.1-8B）
- GPU1: ~28GB（Qwen2.5-14B）
- 总计: ~44GB / 96GB可用

---

## 常见问题

### Q1: 如果只有单卡怎么办？

**A**: 可以回退到按需加载模式：
1. 使用之前的 `utils/model_manager.py`
2. 修改 `config/model_paths.json`，所有模型都用 `cuda:0`
3. 但会牺牲速度（需要频繁加载模型）

### Q2: 如何验证双卡都在工作？

**A**: 运行 `quick_start.py` 后，查看GPU使用：
```bash
nvidia-smi
```
应该看到：
- GPU0: 16GB左右（Llama）
- GPU1: 28GB左右（Qwen）

### Q3: NORMAD数据集格式不对怎么办？

**A**: 检查 `normad_loader.py` 的解析逻辑：
```bash
python utils/normad_loader.py
```
如果解析失败，调整正则表达式。

### Q4: 模型路径找不到？

**A**: 检查 `config/model_paths.json` 中的路径：
```bash
ls /root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct
ls /root/autodl-tmp/CultureMoE/Culture_Alignment/Qwen2.5-14B-Instruct
```
确保路径正确且模型文件存在。

---

## 下一步工作

现在基础设施已完成，可以开始实现核心agents：

1. **`agents/cultural_agent.py`** - Cultural Agent类（封装prompt）
2. **`agents/conflict_analyzer.py`** - Conflict Analyzer类
3. **`agents/mediator_agent.py`** - Mediator Agent类
4. **`agents/fairness_evaluator.py`** - Fairness Evaluator类
5. **`experiments/run_full_system.py`** - 在完整数据集上运行

参考 `quick_start.py` 中的实现逻辑。

---

## 总结

✅ **完成的工作**：
- 双卡Pipeline Parallelism架构
- NORMAD数据集加载和解析
- 双卡模型管理器
- 更新价值标签提取
- 完整的quick_start示例

✅ **优势**：
- 速度提升：避免模型加载时间
- 内存优化：充分利用双卡
- 架构简化：只需2个模型

✅ **可以开始**：
- 运行 `quick_start.py` 测试
- 实现剩余agents模块
- 在完整数据集上实验

**祝实验顺利！🚀**
