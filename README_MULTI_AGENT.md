# 多智能体文化对齐系统

## 系统概述

本项目实现了基于多智能体协作冲突调解的文化对齐系统，包含：

- **5个文化智能体**: 基督教、伊斯兰、佛教、印度教、传统宗教文化
- **智能体池化管理**: 动态加载/卸载，内存优化
- **消息传递机制**: 异步消息总线，支持广播和单播
- **配置管理系统**: 灵活的智能体配置和参数管理
- **6阶段辩论流程**: 初始决策 → 反馈交换 → 最终决策

## 目录结构

```
cultural_debate/
├── agents/                     # 多智能体框架
│   ├── base/                   # 基础接口和实现
│   ├── cultural/               # 5个文化智能体
│   ├── utils/                  # 工具模块(池化、消息总线)
│   └── multi_agent_system.py   # 主系统管理器
├── config/                     # 配置管理
├── tests/                      # 测试用例
├── data/                       # 数据集
│   └── normad.jsonl           # NORMAD-ETI数据集
└── run_multi_agent_inference.py # 主推理脚本
```

## 环境要求

### 硬件要求
- GPU: 16GB+ VRAM (推荐RTX 4090或A100)
- RAM: 32GB+ 系统内存
- 存储: 50GB+ 可用空间(用于模型缓存)

### 软件依赖
```bash
# 核心依赖
torch>=1.9.0
transformers>=4.20.0
huggingface-hub>=0.10.0
accelerate>=0.12.0
pyyaml>=6.0
datasets>=2.0.0

# 可选依赖
pytest>=7.0.0
pytest-asyncio>=0.20.0
```

## 安装和设置

### 1. 安装依赖
```bash
pip install torch transformers huggingface-hub accelerate pyyaml datasets
```

### 2. 配置HuggingFace Token
```bash
# 方法1: 环境变量
export HF_TOKEN="your_huggingface_token"

# 方法2: 修改配置文件
# 编辑 config/global_config.yaml 中的 hf_token 字段
```

### 3. 创建必要目录
```bash
mkdir -p config output cache logs
```

## 使用方法

### 基本用法

```bash
# 处理完整数据集
python run_multi_agent_inference.py \
    --input_path data/normad.jsonl \
    --output_path output/multi_agent_results.jsonl

# 测试运行(前10项)
python run_multi_agent_inference.py \
    --input_path data/normad.jsonl \
    --output_path output/test_results.jsonl \
    --max_items 10

# 从指定位置开始处理
python run_multi_agent_inference.py \
    --input_path data/normad.jsonl \
    --output_path output/results_batch2.jsonl \
    --start_from 100 \
    --max_items 50
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input_path` | 输入NORMAD数据集路径 | `data/normad.jsonl` |
| `--output_path` | 输出结果路径 | `output/multi_agent_results.jsonl` |
| `--config_dir` | 配置文件目录 | `config` |
| `--log_level` | 日志级别 | `INFO` |
| `--max_items` | 最大处理项数 | `None`(全部) |
| `--start_from` | 开始处理位置 | `0` |

### 输出格式

```json
{
  "ID": 0,
  "Country": "egypt",
  "Story": "场景描述...",
  "Rule-of-Thumb": "规则参考...",
  "Gold Label": "yes",
  "Agent_Decisions": {
    "cultural_christian": "yes",
    "cultural_islamic": "yes",
    "cultural_buddhist": "yes",
    "cultural_hindu": "yes",
    "cultural_traditional": "yes"
  },
  "Majority_Decision": "yes",
  "Conversation_ID": "debate_1705123456",
  "Processing_Time": 15.23,
  "Initial_Responses": {...},
  "Final_Responses": {...}
}
```

## 系统特性

### 1. 智能体池化管理
- **动态加载**: 按需加载智能体，最多同时激活3个
- **LRU驱逐**: 自动卸载最少使用的智能体
- **内存监控**: 实时监控GPU内存使用率
- **健康检查**: 定期检查智能体状态

### 2. 异步消息传递
- **消息队列**: 每个智能体独立的消息队列
- **广播支持**: 支持消息广播和订阅机制
- **超时处理**: 自动清理过期消息
- **统计监控**: 详细的消息传递统计

### 3. 6阶段辩论流程
1. **初始决策阶段**: 各文化智能体独立评估
2. **冲突检测阶段**: 识别文化观点差异
3. **反馈交换阶段**: 智能体间交换观点
4. **僵局检测阶段**: 判断是否需要调解
5. **调解介入阶段**: 提供新视角和折中方案
6. **最终决策阶段**: 综合讨论做出最终判断

### 4. 文化智能体特性

| 智能体 | 文化背景 | 核心价值观 | 决策特点 |
|--------|----------|------------|----------|
| 基督教文化 | 西方个人主义 | 自由、权利、平等 | 注重个人权利和法律规则 |
| 伊斯兰文化 | 中东集体主义 | 谦逊、家庭、秩序 | 重视传统和社会和谐 |
| 佛教文化 | 东亚和谐文化 | 平静、慈悲、简朴 | 追求内心和谐与平衡 |
| 印度教文化 | 南亚等级制度 | 达摩、家庭、精神 | 遵循传统智慧和等级 |
| 传统宗教 | 原住民文化 | 自然、祖先、部落 | 强调自然和谐共生 |

## 性能优化

### 内存优化
- **模型量化**: 支持INT8量化减少50%内存
- **智能缓存**: 缓存常用推理结果
- **动态清理**: 自动清理空闲智能体

### 并发优化
- **异步处理**: 非阻塞智能体通信
- **并行推理**: 支持多智能体并行生成
- **流水线处理**: overlap计算和I/O操作

## 监控和调试

### 性能监控
```python
# 获取系统统计信息
stats = mas.get_system_stats()
print(f"活跃智能体: {stats['agent_pool']['active_agents']}")
print(f"内存使用率: {stats['agent_pool']['memory_usage']:.2%}")
print(f"缓存命中率: {stats['agent_pool']['cache_hit_rate']:.2%}")
```

### 日志配置
```yaml
# config/global_config.yaml
logging:
  level: INFO  # DEBUG, INFO, WARNING, ERROR
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### 健康检查
```bash
# 系统健康检查
python -c "
import asyncio
from agents.multi_agent_system import MultiAgentSystem

async def check():
    mas = MultiAgentSystem()
    await mas.initialize()
    health = await mas.health_check()
    print(f'系统健康状态: {health}')
    await mas.shutdown()

asyncio.run(check())
"
```

## 故障排除

### 常见问题

1. **GPU内存不足**
   - 减少 `max_active_agents` 参数
   - 启用模型量化
   - 使用更小的模型

2. **模型加载失败**
   - 检查HuggingFace token配置
   - 确保网络连接正常
   - 验证模型ID正确性

3. **推理速度慢**
   - 增加GPU数量
   - 启用缓存机制
   - 调整批处理大小

### 错误码说明

- `RuntimeError: 多智能体系统未初始化` - 需要先调用 `initialize()`
- `MemoryError` - GPU内存不足，需要优化配置
- `TimeoutError` - 推理超时，检查模型状态

## 扩展开发

### 添加新文化智能体
1. 继承 `CulturalAgentBase` 类
2. 实现文化特定的 `_get_cultural_context()` 方法
3. 在配置文件中添加智能体配置
4. 注册到 `MultiAgentSystem`

### 自定义辩论流程
1. 修改 `MultiAgentSystem.start_cultural_debate()` 方法
2. 添加新的辩论阶段
3. 实现相应的消息处理逻辑

## 引用

如果使用本系统，请引用：

```bibtex
@inproceedings{cultural-alignment-2025,
  title={Multiple LLM Agents Debate for Equitable Cultural Alignment},
  author={...},
  booktitle={ACL 2025},
  year={2025}
}
```