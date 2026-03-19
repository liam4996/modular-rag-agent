# 多智能体 RAG 系统 - 项目概述

## 📋 项目信息

- **项目名称**: Multi-Agent RAG System
- **版本**: v2.0 (生产增强版)
- **SPEC 文档**: `docs/MULTI_AGENT_SPEC.md`
- **创建时间**: 2026-03-18

## 🎯 设计目标

构建一个**生产级**的多智能体 RAG 系统，在原有架构基础上增加：

1. ✅ **容错机制**：最大重试次数 + 兜底策略
2. ✅ **并行融合**：支持多 Agent 并行检索
3. ✅ **溯源与忠实度**：答案必须标注来源，严格基于检索内容

## 🏗️ 核心架构

```
Router Agent → Parallel Fusion Controller → [Search Agent + Web Agent] → Eval Agent → Generate Agent
                                              ↓
                                         Blackboard (共享状态)
```

## 📦 核心组件

| 组件 | 职责 | 位置 |
|------|------|------|
| Router Agent | 意图识别 + 路由决策 | `src/agent/multi_agent/router_agent.py` |
| Parallel Fusion Controller | 并行融合控制器 | `src/agent/multi_agent/parallel_controller.py` |
| Search Agent | 本地知识库检索 | `src/agent/multi_agent/search_agent.py` |
| Web Agent | 联网搜索 | `src/agent/multi_agent/web_agent.py` |
| Eval Agent | 质量评估 + 重试控制 | `src/agent/multi_agent/eval_agent.py` |
| Refine Agent | 查询优化 | `src/agent/multi_agent/refine_agent.py` |
| Generate Agent | 最终生成 + 溯源 + 兜底 | `src/agent/multi_agent/generate_agent.py` |
| Blackboard | 共享状态容器 | `src/agent/multi_agent/state.py` |

## 🚀 实施阶段

### Phase 1: 核心架构（1-2 天）
- [ ] 创建 `AgentState` 增强版
- [ ] 实现 `RouterAgent` 支持并行路由
- [ ] 实现 `ParallelFusionController`
- [ ] 更新 `SearchAgent` 和 `WebAgent`

### Phase 2: 容错机制（1-2 天）
- [ ] 实现 `EvalAgent` 重试控制
- [ ] 实现 `RefineAgent` 重试计数
- [ ] 实现 `GenerateAgent` 兜底回复
- [ ] 添加 `max_retries` 配置

### Phase 3: 溯源与忠实度（1 天）
- [ ] 实现 `Citation` 数据类
- [ ] 更新 `GenerateAgent` 溯源逻辑
- [ ] 添加忠实度检查

### Phase 4: 测试与优化（1-2 天）
- [ ] 单元测试
- [ ] 集成测试
- [ ] 性能测试
- [ ] 边界场景测试（兜底）

### Phase 5: 文档与演示（1 天）
- [ ] 更新架构文档
- [ ] 创建演示脚本
- [ ] 编写使用指南

## 📝 关键特性

### 1. 共享状态（Blackboard Pattern）
所有 Agent 共享同一个 `AgentState`，通过 `blackboard` 读写数据。

### 2. 并行融合检索
Router 可以决定同时调用多个 Agent，通过 `ThreadPoolExecutor` 并行执行。

### 3. 容错机制
- `retry_count`: 当前重试次数
- `max_retries`: 最大重试次数（默认 2）
- `fallback_triggered`: 是否触发兜底
- `fallback_reason`: 兜底原因

### 4. 溯源与忠实度
- 每个 claim 必须有 citation
- 严格基于检索内容，不臆造信息
- 兜底回复礼貌且有帮助

## 🎯 成功标准

- ✅ 所有 Phase 完成
- ✅ 通过所有测试
- ✅ 性能指标达标（响应时间、准确率等）
- ✅ 文档完整

## 📚 参考文档

- **主 SPEC**: `docs/MULTI_AGENT_SPEC.md`
- **现有架构**: `docs/LANGGRAPH_AGENT.md`
- **工作流文档**: `docs/LANGGRAPH_WORKFLOW.md`
