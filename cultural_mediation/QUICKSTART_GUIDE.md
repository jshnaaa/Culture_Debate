# 快速开始指南

## 🚀 立即开始（3步）

### Step 1: 测试组件

```bash
python cultural_mediation/test_components.py
```

**预期输出**：
```
✓ normad_loader: PASS
✓ weight_learner: PASS
✓ model_paths: PASS
⊘ dual_gpu_manager: SKIPPED (optional)
✓ value_extractor: PASS
```

如果有FAIL，检查错误信息并修复。

### Step 2: 运行完整示例

```bash
python quick_start.py
```

**预期流程**：
1. 加载模型（GPU0 + GPU1）~30秒
2. 加载NORMAD样本
3. 生成5个agent回答 ~5秒
4. 提取价值标签 ~3秒
5. 分析冲突 ~1秒
6. （如果冲突高）调解 ~3秒
7. 计算最终答案
8. 显示结果

**预期输出示例**：
```
[Step 1] Initializing dual-GPU model manager...
[GPU0] Loading Cultural Agent: /root/.../Meta-Llama-3.1-8B-Instruct
✓ GPU0 memory allocated: 16.23 GB
[GPU1] Loading Qwen: /root/.../Qwen2.5-14B-Instruct
✓ GPU1 memory allocated: 28.45 GB

[Step 2] Loading NORMAD dataset...
✓ Loaded 2633 samples from normad_merge_gen.json
✓ Loaded sample from country: egypt
✓ Gold label: yes

[Step 4] Generating cultural agent responses (GPU0: Llama3.1-8B)...
  Agent 1 (Autonomy, w=0.11): Yes
  Agent 2 (Order & Security, w=0.26): Yes
  Agent 3 (Tradition, w=0.39): Yes
  Agent 4 (Care & Universalism, w=0.16): Yes
  Agent 5 (Achievement & Power, w=0.08): Neither

[Step 6] Analyzing conflict (GPU1: Qwen2.5-14B)...
✓ Overall conflict score: 0.342
  (Threshold for mediation: 0.6)

[Step 7] Conflict below threshold, no mediation needed

[Step 8] Computing final decision...
✓ Final answer: Yes
✓ Gold label: yes
✓ Prediction: ✓ CORRECT
```

### Step 3: 检查GPU使用

在另一个终端运行：
```bash
watch -n 1 nvidia-smi
```

应该看到：
- **GPU 0**: ~16GB（Llama3.1-8B）
- **GPU 1**: ~28GB（Qwen2.5-14B）

---

## 📁 文件结构

```
cultural_mediation/
├── config/
│   └── model_paths.json          # ✓ 已配置实际路径
├── data/
│   └── country_weights_init.json # ✓ 75国家初始权重
├── utils/
│   ├── normad_loader.py          # ✓ NORMAD数据加载
│   ├── dual_gpu_manager.py       # ✓ 双卡模型管理
│   ├── weight_learner.py         # ✓ 权重学习
│   └── value_extractor.py        # ✓ 价值标签提取
├── test_components.py            # ✓ 组件测试
├── quick_start.py                # ✓ 完整示例
├── QUICKSTART_GUIDE.md           # ← 你在这里
└── UPDATES.md                    # 详细变更说明
```

---

## 🔧 常见问题

### Q1: 模型加载失败

**错误**：`FileNotFoundError: /root/.../Meta-Llama-3.1-8B-Instruct`

**解决**：
```bash
# 检查路径
ls /root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct
ls /root/autodl-tmp/CultureMoE/Culture_Alignment/Qwen2.5-14B-Instruct

# 如果路径不对，编辑config/model_paths.json
vim config/model_paths.json
```

### Q2: NORMAD数据集加载失败

**错误**：`FileNotFoundError: /root/autodl-fs/normad_merge_gen.json`

**解决**：
```bash
# 检查文件
ls /root/autodl-fs/normad_merge_gen.json

# 如果路径不对，编辑config/model_paths.json的_data_paths部分
```

### Q3: GPU内存不足

**错误**：`CUDA out of memory`

**解决**：
```bash
# 检查GPU使用
nvidia-smi

# 如果有其他进程占用，kill掉
kill -9 <PID>

# 或者重启Python进程
```

### Q4: 解析失败

**错误**：`Warning: Failed to parse sample`

**解决**：
```bash
# 测试NORMAD加载
python utils/normad_loader.py

# 查看具体哪个字段解析失败
# 可能需要调整正则表达式
```

### Q5: 生成结果不合理

**问题**：所有agent都回答"Neither"

**可能原因**：
1. Prompt格式不对
2. 模型温度设置过高（应该是0.0）
3. 权重向量异常

**调试**：
```python
# 打印prompt
print(prompts[0])

# 打印原始生成结果
print(agent_responses_raw[0])

# 检查权重
print(country_weights)
```

---

## 📊 性能基准

### 单样本处理时间

| 阶段 | 时间 | GPU |
|------|------|-----|
| 模型加载 | ~30秒 | 一次性 |
| Cultural Agent生成 | ~5秒 | GPU0 |
| 价值标签提取 | ~3秒 | GPU1 |
| 冲突分析 | ~1秒 | GPU1 |
| 调解（如需要） | ~3秒 | GPU1 |
| **总计** | **~12秒/样本** | - |

### 完整数据集（2633样本）

- **预计时间**：~9小时（12秒/样本 × 2633）
- **可优化**：
  - 批处理多个样本（GPU利用率更高）
  - 跳过冲突低的样本的调解步骤
  - 预期可降到~5-6小时

### GPU内存使用

- **GPU0**: 16GB / 48GB (33%)
- **GPU1**: 28GB / 48GB (58%)
- **总计**: 44GB / 96GB (46%)

---

## 🎯 下一步工作

### 1. 验证基础功能（今天）

- [x] 测试组件加载
- [x] 运行单样本示例
- [ ] 验证输出合理性
- [ ] 检查GPU使用正常

### 2. 实现核心Agents（1-2天）

需要实现以下模块（参考quick_start.py中的逻辑）：

- [ ] `agents/cultural_agent.py` - 封装Cultural Agent的prompt和生成
- [ ] `agents/conflict_analyzer.py` - 封装冲突分析逻辑
- [ ] `agents/mediator_agent.py` - 封装调解逻辑
- [ ] `agents/fairness_evaluator.py` - 公平性检查

### 3. 权重学习实验（2-3天）

- [ ] 准备训练集/验证集/测试集划分
- [ ] 实现训练循环（`experiments/train_weights.py`）
- [ ] 在验证集上调优超参数
- [ ] 保存学习后的权重

### 4. 完整实验（3-5天）

- [ ] Baseline对比（Single Model, Self-Reflection, Two-Agent）
- [ ] Ablation实验（6个配置）
- [ ] 在完整数据集上运行
- [ ] 生成可视化图表

---

### 性能优化

1. **批处理**：一次处理多个样本
2. **早停**：冲突低时跳过调解
3. **缓存**：相同prompt的结果缓存
4. **异步**：GPU0和GPU1并行工作（高级）

