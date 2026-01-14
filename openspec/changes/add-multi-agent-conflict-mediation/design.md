# 多智能体协作冲突调解系统设计

## Context

当前文化对齐项目需要从简单的双模型对话升级为复杂的多智能体协作系统。系统需要支持8个智能体（5个文化+冲突检测+调解+决策）的协同工作，处理75个国家的文化差异，并在GPU内存限制下实现高效运行。

**约束条件:**
- GPU内存限制：需要同时运行多个8B-9B模型
- 响应时间要求：每轮对话不超过30秒
- 准确性要求：文化对齐准确率需要提升15%以上
- 可扩展性：支持动态添加新的文化智能体

## Goals / Non-Goals

**Goals:**
- 实现真正的多智能体协作机制
- 建立有效的冲突检测和调解系统
- 优化GPU内存使用，支持8个智能体并发
- 提供可扩展的文化知识图谱
- 实现僵局检测和自动调解介入

**Non-Goals:**
- 不改变NORMAD-ETI数据集格式
- 不支持实时流式对话（保持批处理模式）
- 不实现分布式多机部署（单机多GPU方案）

## Decisions

### 1. 智能体架构设计
**决策**: 采用基于角色的智能体系统，每个智能体有特定的文化背景和职责
**原因**:
- 明确的角色分工便于管理和调试
- 符合文化对齐研究的理论基础
- 支持独立的prompt engineering和优化

**替代方案考虑**:
- 单一多角色智能体：复杂度高，难以调优
- 完全分布式智能体：资源开销大，通信复杂

### 2. 内存管理策略
**决策**: 智能体池化 + 动态加载 + 模型量化
**原因**:
- 池化机制可以复用模型实例，减少内存占用
- 动态加载支持按需激活智能体
- INT8量化可减少50%内存使用

**技术实现**:
```python
class AgentPool:
    def __init__(self, max_active_agents=3):
        self.pool = {}
        self.active_agents = {}
        self.max_active = max_active_agents

    async def get_agent(self, agent_type):
        if agent_type in self.active_agents:
            return self.active_agents[agent_type]

        if len(self.active_agents) >= self.max_active:
            await self._evict_lru_agent()

        agent = await self._load_agent(agent_type)
        self.active_agents[agent_type] = agent
        return agent
```

### 3. 冲突检测算法
**决策**: 多层次冲突检测：语义层 + 文化规范层 + 情感层
**原因**:
- 语义层检测观点对立
- 文化规范层识别文化差异
- 情感层监测讨论氛围

**核心算法**:
```python
class ConflictDetector:
    def detect_conflicts(self, responses):
        semantic_conflicts = self._semantic_analysis(responses)
        cultural_conflicts = self._cultural_norm_analysis(responses)
        emotional_conflicts = self._emotional_analysis(responses)

        return {
            'semantic': semantic_conflicts,
            'cultural': cultural_conflicts,
            'emotional': emotional_conflicts,
            'severity': self._calculate_severity(semantic_conflicts, cultural_conflicts, emotional_conflicts)
        }
```

### 4. 僵局检测机制
**决策**: 基于内容重复性 + 观点收敛性 + 冲突解决进展的综合判断
**触发条件**:
- 连续3轮语义相似度 > 0.85
- 观点变化幅度 < 0.1 且持续5轮
- 冲突点数量不减少超过3轮

### 5. 调解策略设计
**决策**: 基于僵局类型的差异化调解策略
**策略矩阵**:
- 观点固化 → 视角重构
- 文化对立 → 共性挖掘
- 逻辑循环 → 框架突破
- 情绪对抗 → 情绪调节

## Risks / Trade-offs

### 高风险项
1. **GPU内存不足风险**
   - **缓解措施**: 实现智能体池化和动态加载，最多同时激活3个智能体
   - **监控指标**: GPU内存使用率，OOM错误频率

2. **系统复杂度风险**
   - **缓解措施**: 分阶段实现，先实现核心流程再优化性能
   - **监控指标**: 代码复杂度指标，测试覆盖率

3. **调解算法效果不确定性**
   - **缓解措施**: 建立A/B测试框架，对比调解前后的效果
   - **监控指标**: 文化对齐准确率，僵局解决成功率

### 性能权衡
- **内存 vs 响应时间**: 动态加载会增加响应时间，但显著减少内存占用
- **准确性 vs 复杂度**: 多智能体系统提升准确性，但增加系统复杂度
- **可扩展性 vs 性能**: 插件化设计便于扩展，但可能影响运行性能

## Migration Plan

### Phase 1: 核心架构重构 (4周)
1. 重构现有双智能体系统为多智能体框架
2. 实现智能体池化和基础消息传递
3. 迁移现有prompt模板到新架构
4. **验证**: 新架构能正确运行现有测试用例

### Phase 2: 冲突检测系统 (3周)
1. 实现语义冲突检测算法
2. 构建75国文化规范知识图谱
3. 开发僵局检测机制
4. **验证**: 冲突检测准确率 > 80%

### Phase 3: 调解和决策系统 (3周)
1. 实现调解智能体和介入策略
2. 开发决策智能体综合算法
3. 集成完整的6阶段协作流程
4. **验证**: 端到端流程正常运行

### Phase 4: 性能优化 (2周)
1. GPU内存和计算优化
2. 并发处理优化
3. 系统测试和调优
4. **验证**: 性能指标达到预期

### 回滚策略
- 保持现有系统并行运行，直到新系统稳定
- 数据格式向后兼容，确保可以回退到旧系统
- 关键节点设置回滚检查点

## Open Questions

1. **文化知识图谱的数据源和更新机制**
   - 如何确保75个国家文化规范的准确性和时效性？
   - 是否需要专家审核机制？

2. **调解策略的个性化程度**
   - 是否需要为不同文化组合设计专门的调解策略？
   - 如何平衡通用性和针对性？

3. **系统性能的可接受范围**
   - 8个智能体的响应时间上限是多少？
   - GPU内存使用率的安全阈值？

4. **评估指标的完整性**
   - 除了准确率，还需要哪些评估维度？
   - 如何量化文化对齐的"公平性"？