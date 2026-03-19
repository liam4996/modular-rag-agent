# 实施任务调度

## 📋 任务状态说明

- ⬜ **NOT_STARTED** - 未开始
- 🔶 **IN_PROGRESS** - 进行中
- ✅ **COMPLETED** - 已完成

---

## Phase 2: 容错机制（1-2 天）

### P2-T1: 实现 Eval Agent 重试控制 ✅
**任务 ID**: P2-T1  
**优先级**: HIGH  
**预估时间**: 3-4 小时  
**前置任务**: Phase 1 完成  
**文件**: 
- 创建：`src/agent/multi_agent/eval_agent.py`
- 修改：`src/agent/multi_agent/multi_agent_system.py` (集成)
**状态**: ✅ COMPLETED

**任务描述**:
实现增强版 `EvalAgent`，支持：
- 多维度评估：Relevance, Diversity, Coverage, Confidence
- 强制规则应用：
  1. 相关性 < 0.2 → 自动触发兜底
  2. 达到最大重试次数 → 自动触发兜底
  3. 不可能回答的问题 → 自动触发兜底
  4. 置信度 < 0.7 → 需要优化
- 模式识别：识别不可能回答的问题（如"我昨天晚饭吃了什么"）
- 返回 `EvaluationResult` 包含所有评估指标和决策建议

**验收标准**:
- [x] `EvalAgent` 类创建完成
- [x] `EvaluationResult` 数据类定义正确
- [x] 4 个强制规则实现
- [x] 不可能回答的问题模式识别（14+ 个模式）
- [x] 单元测试通过（使用 Mock LLM）

**完成时间**: 2026-03-18

---

### P2-T2: 实现 Refine Agent 重试计数 ✅
**任务 ID**: P2-T2  
**优先级**: HIGH  
**预估时间**: 2-3 小时  
**前置任务**: P2-T1  
**文件**:
- 创建：`src/agent/multi_agent/refine_agent.py`

**任务描述**:
实现 `RefineAgent` 类：
- 分析 Eval Agent 的反馈
- 改写查询使其更具体、更有效
- 优化策略：
  - 添加时间限定（"2025 年最新进展"）
  - 添加详细程度（"详细说明"、"原理解析"）
  - 添加同义词扩展
  - 添加上下文信息
- 返回 `RefinementResult` 包含优化后的查询和改动说明
- 默认优化策略（LLM 响应解析失败时使用）

**验收标准**:
- [x] `RefineAgent` 类创建完成
- [x] `RefinementResult` 数据类定义正确
- [x] 优化策略实现完整
- [x] 默认优化策略实现
- [x] 单元测试通过

**完成时间**: 2026-03-18

---

### P2-T3: 实现 Generate Agent 兜底回复 ✅
**任务 ID**: P2-T3  
**优先级**: HIGH  
**预估时间**: 2-3 小时  
**前置任务**: P2-T1, P2-T2  
**文件**:
- 修改：`src/agent/multi_agent/multi_agent_system.py`

**任务描述**:
增强 `Generate Agent` 节点：
- 判断是否进入兜底模式（`state.should_fallback`）
- 根据兜底原因生成不同的回复
- 兜底回复模板：
  - 礼貌说明无法找到答案
  - 显示检索详情（次数、结果数）
  - 提供 3 条具体建议
  - 建议联系管理员
- 正常回复模式（留待 Phase 3 实现溯源）

**验收标准**:
- [x] `_generate_fallback_response` 方法实现
- [x] 4 种兜底原因支持
- [x] 回复模板友好且信息完整
- [x] 集成测试通过

**完成时间**: 2026-03-18

---

### P2-T4: 添加 max_retries 配置 ✅
**任务 ID**: P2-T4  
**优先级**: MEDIUM  
**预估时间**: 1-2 小时  
**前置任务**: P2-T1  
**文件**:
- 修改：`src/agent/multi_agent/state.py` (已有)
- 修改：`src/agent/multi_agent/multi_agent_system.py` (集成)

**任务描述**:
完善重试控制配置：
- `AgentState.max_retries` 字段（默认 2）
- `state.should_fallback` 属性判断逻辑
- `state.retry_count` 递增机制
- `state.fallback_triggered` 和 `state.fallback_reason` 标志
- 工作流集成：Eval → Refine → Search 循环

**验收标准**:
- [x] `max_retries` 可配置
- [x] `should_fallback` 判断正确
- [x] 重试计数正确递增
- [x] 工作流循环正确实现
- [x] 单元测试通过

**完成时间**: 2026-03-18

---

## Phase 2 完成总结 ✅

**完成时间**: 2026-03-18  
**状态**: ✅ 完成  
**新增文件**:
- `src/agent/multi_agent/eval_agent.py`
- `src/agent/multi_agent/refine_agent.py`
- `examples/test_phase2_simple.py`
- `docs/PHASE2_COMPLETION_REPORT.md`

**修改文件**:
- `src/agent/multi_agent/__init__.py`
- `src/agent/multi_agent/multi_agent_system.py`
- `src/agent/multi_agent/state.py` (已有字段)

**核心能力**:
- ✅ 智能重试机制（最多 2 次优化重试）
- ✅ 多维度评估（Relevance, Diversity, Coverage, Confidence）
- ✅ 强制规则应用（4 个自动触发兜底规则）
- ✅ 查询优化策略（基于反馈针对性优化）
- ✅ 兜底回复（4 种原因，友好回复模板）
- ✅ 不可能回答的问题识别（14+ 个模式）

**测试覆盖**:
- ✅ Eval Agent 强制规则（4 个场景）
- ✅ Refine Agent 基本功能（解析失败处理）
- ✅ AgentState 重试控制（完整流程）
- ✅ 不可能回答的问题模式识别（14 个模式）
- ✅ 所有测试通过

---

## Phase 1: 核心架构（1-2 天）

### P1-T1: 创建增强版 AgentState ✅
**任务 ID**: P1-T1  
**优先级**: HIGH  
**预估时间**: 2-3 小时  
**前置任务**: 无  
**文件**: 
- 创建：`src/agent/multi_agent/state.py`
**状态**: ✅ COMPLETED

**任务描述**:
创建增强版 `AgentState` 数据类，包含：
- 输入：`user_input`, `conversation_history`
- 黑板：`blackboard` (Dict[str, Any])
- 重试控制：`retry_count`, `max_retries`, `fallback_triggered`, `fallback_reason`
- 执行追踪：`execution_log`, `execution_trace`, `metrics`
- 输出：`final_answer`
- 辅助方法：`add_to_blackboard()`, `read_from_blackboard()`, `increment_retry()`, `trigger_fallback()`, `get_all_context()`

**验收标准**:
- [x] `AgentState` 数据类创建完成
- [x] 所有字段类型正确
- [x] 辅助方法实现完整
- [x] 导入测试通过

---

### P1-T2: 实现 Router Agent 支持并行路由 ⬜
**任务 ID**: P1-T2  
**优先级**: HIGH  
**预估时间**: 3-4 小时  
**前置任务**: P1-T1  
**文件**:
- 创建：`src/agent/multi_agent/router_agent.py`
- 修改：`src/agent/intent_classifier.py` (可选)

**任务描述**:
实现增强版 `RouterAgent`，支持：
- 识别 5 种意图：CHAT, LOCAL_SEARCH, WEB_SEARCH, HYBRID_SEARCH, UNKNOWN
- 返回 `RoutingDecision` 包含 `agents_to_invoke` (可以是多个 Agent)
- 支持并行标记 `parallel: bool`
- 复杂查询识别（如"结合内部文档和网上资料"）

**验收标准**:
- [ ] `RouterAgent` 类创建完成
- [ ] 支持返回多个 Agent
- [ ] 支持并行标记
- [ ] 单元测试通过
- [ ] 能正确识别 HYBRID_SEARCH 意图

---

### P1-T3: 实现 Parallel Fusion Controller ⬜
**任务 ID**: P1-T3  
**优先级**: HIGH  
**预估时间**: 3-4 小时  
**前置任务**: P1-T1, P1-T2  
**文件**:
- 创建：`src/agent/multi_agent/parallel_controller.py`

**任务描述**:
实现 `ParallelFusionController` 类：
- `execute_parallel_search(state, agents_to_invoke)` 方法
- 使用 `ThreadPoolExecutor` 并行执行多个 Agent
- 等待所有 Agent 完成
- 将结果全部写入 Blackboard
- 错误处理和日志记录

**验收标准**:
- [ ] `ParallelFusionController` 类创建完成
- [ ] 支持并行执行 2 个以上 Agent
- [ ] 结果正确写入 Blackboard
- [ ] 错误处理完善
- [ ] 单元测试通过

---

### P1-T4: 更新 Search Agent ⬜
**任务 ID**: P1-T4  
**优先级**: MEDIUM  
**预估时间**: 2-3 小时  
**前置任务**: P1-T1  
**文件**:
- 创建：`src/agent/multi_agent/search_agent.py`
- 保留：`src/agent/tool_caller.py` (向后兼容)

**任务描述**:
创建新的 `SearchAgent` 类（基于现有 `ToolCaller`）：
- 封装现有的 `hybrid_search` 功能
- 返回格式适配新的 `AgentState`
- 支持从 Blackboard 读取上下文
- 写入结果到 `blackboard["local_results"]`

**验收标准**:
- [ ] `SearchAgent` 类创建完成
- [ ] 与现有 `ToolCaller` 兼容
- [ ] 正确读写 Blackboard
- [ ] 单元测试通过

---

### P1-T5: 创建 Web Agent ⬜
**任务 ID**: P1-T5  
**优先级**: MEDIUM  
**预估时间**: 3-4 小时  
**前置任务**: P1-T1  
**文件**:
- 创建：`src/agent/multi_agent/web_agent.py`
- 创建：`src/agent/tools/web_search.py`

**任务描述**:
创建 `WebSearchAgent` 类：
- 支持 DuckDuckGo 搜索（免费）
- 支持 Google/Bing API（可选）
- 返回格式：`List[WebSearchResult]`
- 写入结果到 `blackboard["web_results"]`
- 可以从 Blackboard 读取 `local_results` 用于优化查询

**验收标准**:
- [ ] `WebSearchAgent` 类创建完成
- [ ] DuckDuckGo 搜索可用
- [ ] 返回格式正确
- [ ] 正确读写 Blackboard
- [ ] 单元测试通过

---

### P1-T6: 创建多智能体编排器 ⬜
**任务 ID**: P1-T6  
**优先级**: HIGH  
**预估时间**: 4-5 小时  
**前置任务**: P1-T1 ~ P1-T5  
**文件**:
- 创建：`src/agent/multi_agent/multi_agent_system.py`

**任务描述**:
创建 `MultiAgentRAG` 主类：
- 使用 LangGraph 编排所有 Agent
- 实现条件路由逻辑
- 集成 `ParallelFusionController`
- 实现完整工作流：
  - Router → [Search + Web (并行)] → Eval → Generate
  - 支持 Chat 直接到 Generate
- 提供统一的 `run()` API

**验收标准**:
- [ ] `MultiAgentRAG` 类创建完成
- [ ] LangGraph 工作流正确
- [ ] 支持并行检索
- [ ] 支持条件路由
- [ ] 集成测试通过
- [ ] 演示脚本可用

---

### P1-T7: 创建演示脚本 ⬜
**任务 ID**: P1-T7  
**优先级**: MEDIUM  
**预估时间**: 2-3 小时  
**前置任务**: P1-T6  
**文件**:
- 创建：`examples/multi_agent_demo.py`

**任务描述**:
创建演示脚本展示 Phase 1 功能：
- 简单查询（单次检索）
- 复杂查询（并行融合检索）
- 对比 SimpleAgent vs MultiAgentRAG
- 展示执行日志和指标

**验收标准**:
- [ ] 演示脚本可运行
- [ ] 展示所有 Phase 1 功能
- [ ] 输出清晰易懂
- [ ] 无错误

---

## Phase 2: 容错机制（1-2 天）

### P2-T1: 实现 Eval Agent 重试控制 ⬜
**任务 ID**: P2-T1  
**优先级**: HIGH  
**前置任务**: P1-T6  
**预估时间**: 3-4 小时

### P2-T2: 实现 Refine Agent 重试计数 ⬜
**任务 ID**: P2-T2  
**优先级**: HIGH  
**前置任务**: P2-T1  
**预估时间**: 2-3 小时

### P2-T3: 实现 Generate Agent 兜底回复 ⬜
**任务 ID**: P2-T3  
**优先级**: HIGH  
**前置任务**: P2-T1, P2-T2  
**预估时间**: 3-4 小时

### P2-T4: 添加 max_retries 配置 ⬜
**任务 ID**: P2-T4  
**优先级**: MEDIUM  
**前置任务**: P2-T3  
**预估时间**: 1-2 小时

---

## Phase 3: 溯源与忠实度（1 天）

### P3-T1: 实现 Citation 数据类 ⬜
**任务 ID**: P3-T1  
**优先级**: HIGH  
**预估时间**: 1-2 小时

### P3-T2: 更新 Generate Agent 溯源逻辑 ⬜
**任务 ID**: P3-T2  
**优先级**: HIGH  
**前置任务**: P3-T1  
**预估时间**: 3-4 小时

### P3-T3: 添加忠实度检查 ⬜
**任务 ID**: P3-T3  
**优先级**: MEDIUM  
**前置任务**: P3-T2  
**预估时间**: 2-3 小时

---

## Phase 4: 测试与优化（1-2 天）

### P4-T1: 单元测试 ⬜
### P4-T2: 集成测试 ⬜
### P4-T3: 性能测试 ⬜
### P4-T4: 边界场景测试（兜底） ⬜

---

## Phase 5: 文档与演示（1 天）

### P5-T1: 更新架构文档 ⬜
### P5-T2: 创建演示脚本 ⬜
### P5-T3: 编写使用指南 ⬜

---

## 📊 总体进度

| Phase | 总任务 | 已完成 | 进行中 | 未开始 | 完成率 |
|-------|--------|--------|--------|--------|--------|
| Phase 1 | 7 | 0 | 0 | 7 | 0% |
| Phase 2 | 4 | 0 | 0 | 4 | 0% |
| Phase 3 | 3 | 0 | 0 | 3 | 0% |
| Phase 4 | 4 | 0 | 0 | 4 | 0% |
| Phase 5 | 3 | 0 | 0 | 3 | 0% |
| **总计** | **21** | **0** | **0** | **21** | **0%** |

---

## 🎯 当前目标

**Phase 1: 核心架构**
- 下一个任务：P1-T1 (创建增强版 AgentState)
- 预计完成时间：1-2 天

---

**更新时间**: 2026-03-18  
**状态**: 准备开始 Phase 1
