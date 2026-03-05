# 使用指南 - 单模型架构

## 🎯 架构说明

### 核心设计

**单模型架构**：所有智能体（Cultural Agents, Conflict Analyzer, Mediator）共享同一个模型实例。

**支持的模型**：
- **Llama3.1-8B**: `/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct`
- **Qwen3-8B**: `/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-3-8B-Instruct`

**答案格式**：
- 直接使用 `"1"/"2"/"3"` 格式
- `1` = yes (socially acceptable)
- `2` = no (not socially acceptable)
- `3` = neutral (depends on context)

**评估指标**：
- 准确率（Accuracy）：预测答案与gold label的匹配率

---

## 🚀 快速开始

### 1. 测试组件

```bash
cd /root/autodl-tmp/CultureMoE/Culture_Alignment/cultural_mediation
python test_components.py
```

**预期输出**：
- ✓ normad_loader: PASS
- ✓ weight_learner: PASS
- ✓ model_paths: PASS
- ⊘ unified_model_manager: SKIPPED (可选，需要确认)
- ✓ value_extractor: PASS

### 2. 测试单个样本

#### 使用Llama模型
```bash
./run_test.sh llama 0
# 或者
python quick_start.py --model llama --sample_idx 0
```

#### 使用Qwen模型
```bash
./run_test.sh qwen 0
# 或者
python quick_start.py --model qwen --sample_idx 0
```

### 3. 批量测试

#### 测试10个样本（Llama）
```bash
./run_batch_test.sh llama 10
```

#### 测试50个样本（Qwen）
```bash
./run_batch_test.sh qwen 50
```

---

## 📝 脚本说明

### `run_test.sh` - 单样本测试

**语法**：
```bash
./run_test.sh [MODEL] [SAMPLE_IDX]
```

**参数**：
- `MODEL`: `llama`（默认）或 `qwen`
- `SAMPLE_IDX`: 样本索引（默认：0）

**示例**：
```bash
./run_test.sh                 # llama, 样本0
./run_test.sh llama           # llama, 样本0
./run_test.sh qwen            # qwen, 样本0
./run_test.sh llama 5         # llama, 样本5
./run_test.sh qwen 10         # qwen, 样本10
```

**输出**：
- 显示完整的处理流程
- 显示5个agent的回答
- 显示冲突分数
- 显示最终预测和准确性
- 退出码：0=正确，1=错误

### `run_batch_test.sh` - 批量测试

**语法**：
```bash
./run_batch_test.sh [MODEL] [NUM_SAMPLES]
```

**参数**：
- `MODEL`: `llama`（默认）或 `qwen`
- `NUM_SAMPLES`: 测试样本数（默认：10）

**示例**：
```bash
./run_batch_test.sh              # llama, 10样本
./run_batch_test.sh llama 20     # llama, 20样本
./run_batch_test.sh qwen 50      # qwen, 50样本
./run_batch_test.sh qwen 100     # qwen, 100样本
```

**输出**：
- 每个样本的测试结果
- 总体准确率
- 结果保存到 `results/batch_test_<model>_<num>samples.txt`

---

## 📊 预期性能

### 单样本处理时间

| 阶段 | 时间 | 说明 |
|------|------|------|
| 模型加载 | ~30秒 | 一次性（首次运行） |
| 5个Agent生成 | ~8-10秒 | Batch生成 |
| 冲突分析 | <1秒 | 简单计算 |
| 最终决策 | <1秒 | 加权投票 |
| **总计** | **~10秒/样本** | 模型加载后 |

### 内存使用

- **单卡模式**: ~16GB VRAM（8B模型）
- **可用性**: 单张48GB GPU完全足够

### 准确率基准

需要在完整数据集上测试，预期：
- Baseline（单模型）: ~60-70%
- 多智能体（无调解）: ~65-75%
- 完整系统（有调解）: ~70-80%

---

## 🔧 文件结构

```
cultural_mediation/
├── config/
│   └── model_paths.json              # 模型配置（llama/qwen）
├── data/
│   └── country_weights_init.json     # 75国家初始权重
├── utils/
│   ├── unified_model_manager.py      # ✨ 单模型管理器
│   ├── normad_loader.py              # 🔧 NORMAD加载（答案1/2/3）
│   ├── weight_learner.py             # 权重学习
│   └── value_extractor.py            # 价值提取
├── quick_start.py                    # 🔧 主测试脚本
├── run_test.sh                       # ✨ 单样本测试脚本
├── run_batch_test.sh                 # ✨ 批量测试脚本
├── test_components.py                # 组件测试
└── results/                          # 测试结果目录
```

**图例**：
- ✨ 新建文件
- 🔧 修改文件

---

## 💡 使用技巧

### 1. 快速验证

测试前5个样本，快速验证系统工作：
```bash
./run_batch_test.sh llama 5
```

### 2. 对比两个模型

```bash
# 测试Llama
./run_batch_test.sh llama 20

# 测试Qwen
./run_batch_test.sh qwen 20

# 对比结果
cat results/batch_test_llama_20samples.txt
cat results/batch_test_qwen_20samples.txt
```

### 3. 调试单个样本

如果某个样本预测错误，单独测试查看详细输出：
```bash
./run_test.sh llama 42  # 假设样本42预测错误
```

### 4. 监控GPU使用

在另一个终端运行：
```bash
watch -n 1 nvidia-smi
```

应该看到单卡~16GB使用。

### 5. 后台运行批量测试

```bash
nohup ./run_batch_test.sh llama 100 > test_llama_100.log 2>&1 &
```

---

## 🐛 常见问题

### Q1: 模型路径找不到

**错误**：`FileNotFoundError: /root/.../Meta-Llama-3.1-8B-Instruct`

**解决**：
```bash
# 检查路径
ls /root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct
ls /root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-3-8B-Instruct

# 如果路径不对，编辑配置
vim config/model_paths.json
```

### Q2: 权限错误

**错误**：`Permission denied: ./run_test.sh`

**解决**：
```bash
chmod +x run_test.sh
chmod +x run_batch_test.sh
```

### Q3: CUDA out of memory

**错误**：`CUDA out of memory`

**解决**：
```bash
# 检查GPU使用
nvidia-smi

# Kill其他进程
kill -9 <PID>

# 或者清理缓存
python -c "import torch; torch.cuda.empty_cache()"
```

### Q4: 答案解析失败

**问题**：Agent回答格式不对，无法解析出1/2/3

**调试**：
```python
# 在quick_start.py中添加打印
print(f"Raw response: {response_raw}")
print(f"Parsed answer: {answer}")
```

**可能原因**：
- 模型输出格式不稳定
- Prompt需要调整
- Temperature设置不对（应该是0.0）

### Q5: 准确率过低

**问题**：准确率<50%，接近随机

**可能原因**：
1. 权重向量不合理 → 检查 `country_weights_init.json`
2. Prompt设计问题 → 调整prompt模板
3. 答案解析错误 → 检查 `parse_agent_answer()` 函数
4. 模型能力不足 → 尝试另一个模型

---

## 📈 实验流程

### Phase 1: 验证系统（今天）

- [x] 测试组件加载
- [ ] 运行单样本（llama）
- [ ] 运行单样本（qwen）
- [ ] 验证输出合理性
- [ ] 批量测试10个样本

### Phase 2: 基准测试（1-2天）

- [ ] 在100个样本上测试llama
- [ ] 在100个样本上测试qwen
- [ ] 对比两个模型的准确率
- [ ] 分析错误样本

### Phase 3: 完整实验（3-5天）

- [ ] 在完整数据集（2633样本）上运行
- [ ] 计算各国家的准确率
- [ ] 分析冲突模式
- [ ] 生成可视化图表

### Phase 4: 权重学习（2-3天）

- [ ] 实现权重学习训练循环
- [ ] 在训练集上优化权重
- [ ] 在测试集上评估
- [ ] 对比学习前后的准确率

---

## 🎯 下一步工作

### 立即可做

1. **运行测试**：
   ```bash
   ./run_test.sh llama 0
   ./run_test.sh qwen 0
   ```

2. **批量测试**：
   ```bash
   ./run_batch_test.sh llama 10
   ./run_batch_test.sh qwen 10
   ```

3. **对比结果**：
   ```bash
   cat results/batch_test_llama_10samples.txt
   cat results/batch_test_qwen_10samples.txt
   ```

### 需要实现的功能

1. **完整数据集测试脚本** - 运行所有2633个样本
2. **权重学习模块** - 实现训练循环
3. **结果分析脚本** - 生成统计报告和图表
4. **Ablation实验** - 测试各组件的贡献

---

## 📞 获取帮助

**文档**：
- `USAGE_GUIDE.md` (本文件) - 使用指南
- `README.md` - 完整文档
- `DESIGN_SUMMARY.md` - 设计方案

**测试**：
```bash
python test_components.py  # 测试所有组件
./run_test.sh llama 0      # 测试单个样本
```

**日志**：
- 查看脚本输出
- 检查 `results/` 目录
- 使用 `--help` 查看参数

---

**祝实验顺利！🚀**

如有问题，先运行 `test_components.py` 排查组件问题，再运行 `run_test.sh` 测试完整流程。
