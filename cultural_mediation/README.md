# Cultural Conflict Mediation System

基于多智能体冲突调解的文化对齐系统，用于NORMAD数据集的文化价值判断任务。

## 核心创新

1. **冲突调解范式**：从"辩论达成共识"转向"调解管理冲突"
2. **可学习权重**：75个国家的文化价值权重可通过投影梯度下降优化
3. **多样性保持**：在降低冲突的同时保留文化特色
4. **公平性验证**：确保调解agent不偏向特定文化

## 系统架构

```
Cultural Agent Pool (5 agents)
    ↓
Value Tag Extraction (Qwen2.5-7B)
    ↓
Conflict Analyzer (Qwen2.5-7B)
    ↓
Mediator Agent (Qwen2.5-14B)
    ↓
Fairness Evaluator
    ↓
Iterative Refinement (max 3 rounds)
```

## 硬件要求

- **GPU**: 2×48GB (双卡并行，Pipeline Parallelism)
- **模型**: Llama3.1-8B (GPU0), Qwen2.5-14B (GPU1)
- **内存**: GPU0: ~16GB, GPU1: ~28GB, 总计~44GB VRAM

## 快速开始

### 1. 环境配置

```bash
conda create -n cultural_mediation python=3.10
conda activate cultural_mediation
pip install torch transformers sentence-transformers numpy pandas jsonlines matplotlib seaborn
```

### 2. 配置模型路径

模型路径已配置在 `config/model_paths.json`:
```json
{
  "cultural_agent": {
    "path": "/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct",
    "device": "cuda:0"
  },
  "qwen_unified": {
    "path": "/root/autodl-tmp/CultureMoE/Culture_Alignment/Qwen2.5-14B-Instruct",
    "device": "cuda:1"
  }
}
```

### 3. 初始化国家权重

```python
from utils.weight_learner import CountryWeightLearner

# 加载初始权重
learner = CountryWeightLearner(
    init_weights_path="data/country_weights_init.json"
)

# 查看特定国家权重
print(learner.get_country_weight('egypt'))
# Output: tensor([0.11, 0.26, 0.39, 0.16, 0.08])
#         [Autonomy, Order&Security, Tradition, Care&Universalism, Achievement&Power]
```

### 4. 运行单个样本

```python
from utils.dual_gpu_manager import DualGPUModelManager
from utils.normad_loader import NORMADLoader
from utils.weight_learner import CountryWeightLearner

# 初始化双卡管理器（自动加载两个模型）
model_mgr = DualGPUModelManager()

# 加载NORMAD数据集
loader = NORMADLoader("/root/autodl-fs/normad_merge_gen.json")
sample = loader.get_sample(0)

# 加载权重
learner = CountryWeightLearner(init_weights_path="data/country_weights_init.json")
country_weights = learner.get_country_weight(sample.country)

# 生成5个cultural agent的回答（GPU0，batch生成）
prompts = [build_prompt(sample, country_weights[i], dimension=i) for i in range(5)]
responses = model_mgr.generate_cultural_responses(prompts)

# 提取价值标签（GPU1）
from utils.value_extractor import batch_extract_value_tags
value_tags = batch_extract_value_tags(responses, model_mgr)

# 冲突分析（GPU1）
conflict_score = compute_conflict(responses, value_tags)
print(f"Conflict score: {conflict_score:.3f}")

# 如果冲突高，进行调解（GPU1）
if conflict_score > 0.6:
    mediation_prompt = build_mediation_prompt(responses, conflict_score)
    mediation = model_mgr.generate_with_qwen(mediation_prompt)
    print(f"Mediation: {mediation}")
```

## 权重学习

### 训练可学习权重

```python
from utils.weight_learner import (
    CountryWeightLearner,
    WeightLearningLoss,
    define_similar_country_pairs,
    train_weights
)
import torch
from torch.utils.data import DataLoader

# 初始化learner
learner = CountryWeightLearner(
    init_weights_path="data/country_weights_init.json"
)

# 定义相似国家对(用于一致性正则化)
similar_pairs = define_similar_country_pairs(learner.country_names)

# 定义损失函数
loss_fn = WeightLearningLoss(
    lambda_l2=0.01,
    lambda_entropy=0.05,
    lambda_consistency=0.02,
    similar_countries=similar_pairs
)

# 准备训练数据(从NORMAD训练集)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 训练
trained_learner = train_weights(
    learner,
    train_loader,
    loss_fn,
    num_epochs=10,
    lr=0.01,
    device='cuda:0'
)

# 保存学习后的权重
trained_learner.save_weights("data/country_weights_learned.json")
```

### 投影到Simplex

权重在每次梯度更新后自动投影到simplex约束 $\{\mathbf{w}: \sum w_i = 1, w_i \geq 0\}$:

```python
# 梯度更新
optimizer.step()

# 投影(自动调用)
learner.project_weights()

# 验证约束
for country in learner.country_names:
    w = learner.get_country_weight(country)
    assert abs(w.sum().item() - 1.0) < 1e-5  # sum = 1
    assert (w >= 0).all()  # all non-negative
```

## 实验配置

### Baseline对比

```bash
# 1. Single Model
python experiments/baseline_single.py --model llama3.1-8b

# 2. Self-Reflection
python experiments/baseline_reflection.py --model llama3.1-8b

# 3. Two-Agent Debate
python experiments/baseline_two_agent.py --model1 llama3.1-8b --model2 qwen2.5-7b

# 4. Five-Agent Debate (无调解)
python experiments/baseline_five_agent.py
```

### Ablation实验

```bash
# 无冲突惩罚
python experiments/ablation.py --no-conflict-penalty

# 无多样性奖励
python experiments/ablation.py --no-diversity-reward

# 无调解agent
python experiments/ablation.py --no-mediator

# 无公平性检查
python experiments/ablation.py --no-fairness-check

# 单轮调解
python experiments/ablation.py --single-round

# 完整系统
python experiments/run_full_system.py
```

### 超参数调优

```bash
# 网格搜索: λ ∈ {0.2, 0.4, 0.6, 0.8}, μ ∈ {0.1, 0.3, 0.5}
python experiments/hyperparameter_search.py \
    --lambda-values 0.2 0.4 0.6 0.8 \
    --mu-values 0.1 0.3 0.5 \
    --output results/hyperparam_search.json
```

## 评估指标

```python
from utils.metrics import (
    compute_accuracy,
    compute_conflict_reduction,
    compute_diversity_preservation,
    compute_mediation_success_rate
)

# 准确率
accuracy = compute_accuracy(predictions, gold_labels)

# 冲突下降率
conflict_reduction = compute_conflict_reduction(
    initial_conflicts, final_conflicts
)

# 多样性保持度
diversity_preservation = compute_diversity_preservation(
    initial_diversity, final_diversity
)

# 调解成功率
success_rate = compute_mediation_success_rate(
    conflict_scores, threshold=0.6
)

print(f"Accuracy: {accuracy:.2%}")
print(f"Conflict Reduction: {conflict_reduction:.2%}")
print(f"Diversity Preservation: {diversity_preservation:.2%}")
print(f"Mediation Success Rate: {success_rate:.2%}")
```

## 可视化

```python
from utils.visualization import (
    plot_conflict_evolution,
    plot_value_space_distribution,
    plot_cultural_conflict_heatmap,
    plot_diversity_conflict_tradeoff
)

# 冲突演化曲线
plot_conflict_evolution(
    conflict_history,
    save_path="results/figures/conflict_evolution.png"
)

# 价值空间分布(t-SNE)
plot_value_space_distribution(
    agent_embeddings_initial,
    agent_embeddings_final,
    save_path="results/figures/value_space.png"
)

# 文化冲突热力图
plot_cultural_conflict_heatmap(
    country_conflict_matrix,
    save_path="results/figures/conflict_heatmap.png"
)

# 多样性-冲突权衡曲线
plot_diversity_conflict_tradeoff(
    diversity_scores,
    conflict_scores,
    lambda_values,
    save_path="results/figures/tradeoff_curve.png"
)
```

## 文件结构

```
cultural_mediation/
├── agents/
│   ├── cultural_agent.py          # 5个文化agent
│   ├── conflict_analyzer.py       # 冲突检测
│   ├── mediator_agent.py          # 调解agent
│   └── fairness_evaluator.py      # 公平性验证
├── utils/
│   ├── model_manager.py           # 模型按需加载
│   ├── weight_learner.py          # 权重学习
│   ├── value_extractor.py         # 价值标签提取
│   ├── prompt_templates.py        # Prompt模板
│   ├── embedding_utils.py         # Embedding计算
│   ├── metrics.py                 # 评估指标
│   └── visualization.py           # 可视化
├── experiments/
│   ├── baseline_single.py
│   ├── baseline_reflection.py
│   ├── baseline_two_agent.py
│   ├── baseline_five_agent.py
│   ├── ablation.py
│   ├── run_full_system.py
│   └── hyperparameter_search.py
├── data/
│   ├── country_weights_init.json  # 初始权重(75国家)
│   └── normad.jsonl               # NORMAD数据集
├── config/
│   └── model_paths.json           # 模型路径配置
├── results/
│   ├── outputs/                   # 实验输出
│   ├── figures/                   # 可视化图表
│   └── tables/                    # 结果表格
└── README.md
```

## 常见问题

### Q1: 如何处理OOM错误?

**A**: 使用按需加载策略,确保每次只加载一个模型:
```python
# 错误: 同时加载多个模型
model1 = load_model("llama3.1-8b")
model2 = load_model("qwen2.5-14b")  # OOM!

# 正确: 按需加载
model_mgr.load_model("cultural_agent")  # 加载Llama
# ... 使用 ...
model_mgr.load_model("mediator")  # 自动卸载Llama,加载Qwen
```

### Q2: 权重学习收敛慢怎么办?

**A**: 调整学习率和正则化权重:
```python
# 增大学习率(小心过拟合)
optimizer = torch.optim.Adam(learner.parameters(), lr=0.05)

# 减小正则化(如果loss主要来自正则项)
loss_fn = WeightLearningLoss(
    lambda_l2=0.005,  # 减半
    lambda_entropy=0.025,
    lambda_consistency=0.01
)
```

### Q3: 如何解释学习后的权重?

**A**: 对比初始权重和学习后权重:
```python
import json

init_weights = json.load(open("data/country_weights_init.json"))
learned_weights = json.load(open("data/country_weights_learned.json"))

country = "egypt"
print(f"Initial:  {init_weights['weights'][country]}")
print(f"Learned:  {learned_weights['weights'][country]}")
print(f"Change:   {[l-i for i, l in zip(init_weights['weights'][country],
                                          learned_weights['weights'][country])]}")
```

## 引用

如果使用本代码,请引用:

```bibtex
@inproceedings{cultural-mediation-2025,
  title={Cultural Conflict Mediation in Multi-Agent LLM Systems: A Game-Theoretic Approach},
  author={Your Name},
  booktitle={Proceedings of ACL 2025},
  year={2025}
}
```

## 许可证

MIT License
