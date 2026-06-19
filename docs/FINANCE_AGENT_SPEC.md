# 金融智能体改造 SPEC

> 基于现有 `Modular RAG MCP Server` 多 Agent 架构，新增金融领域能力。
>
> **状态：设计文档 + 实施完成。** 本文档是先于代码编写的架构设计稿。
> 实际实现已全部完成（Phase 1-5），代码在 `feat/finance-agent` 分支。
> 运行 `python examples/test_finance_agent_e2e.py` 可验证全部功能。
> 
> **学习建议：** 
> - 先通读本 SPEC，理解架构决策（为什么 Plan-then-Execute 而不是 ReAct）
> - 对照源码看各 Agent 的输入/输出接口（多 Agent 间靠 blackboard 通信）
> - 重点看 `multi_agent_system.py` 的 `_build_graph()`——13 个节点的连接关系
> - E2E 测试文件验证了所有模块间的协作

---

## 一、改造总览

### 1.1 当前架构

```
User → Router → Plan/Retrieve/Web → Read → Eval → Refine/Generate
```

5 个 Agent 节点，核心能力是**文档检索 + 追问优化**。

### 1.2 目标架构

```
                        ┌─────────────────────────────────────┐
                        │        Supervisor Agent             │
                        │  意图识别 · 任务拆分 · 路由          │
                        │  结果聚合 · 流程管控                │
                        └────────────────┬────────────────────┘
                                         │
          ┌──────────────────────┬───────┴───────┬──────────────────────┐
          │                      │               │                      │
┌─────────▼──────────┐ ┌───────▼────────┐ ┌─────▼──────────┐ ┌────────▼─────────┐
│   RAG Agent         │ │  Web Agent    │ │ Finance Data    │ │ Business Compute │
│                     │ │               │ │ Agent           │ │ Agent            │
│ 本地知识库检索       │ │ 联网搜索      │ │                 │ │                  │
│ 文档溯源            │ │ 实时资讯      │ │ MCP 行情查询     │ │ 指标 · 清洗      │
│                     │ │               │ │ 股票/指数/基本面 │ │ 图表 · 报告      │
└─────────┬───────────┘ └───────┬───────┘ └────────┬────────┘ └────────┬─────────┘
          │                     │                   │                   │
          │   ┌─────────────────┘                   │                   │
          │   │                                     │                   │
          │   ▼                                     │                   │
          │  ┌──────────────────────────────┐       │                   │
          │  │  RAG / Web 结果               │       │                   │
          │  │  → Read → Eval → 质量不够?     │       │                   │
          │  │    ├── Refine → 重新检索       │       │                   │
          │  │    ├── 升级联网                │       │  ← 原有闭环保留   │
          │  │    └── 质量OK → 写入黑板       │       │                   │
          │  └──────────────────────────────┘       │                   │
          │                     │                   │                   │
          └─────────────────────┼───────────────────┘                   │
                                │                                       │
                                ▼                                       │
                    ┌───────────────────────┐                           │
                    │  Business Compute Agent│  ←───────── 接收各个Agent│
                    │                       │            的输出         │
                    │  ┌─────────────────┐  │                           │
                    │  │ 📊 图表 (PNG)    │  │  多模态输出:              │
                    │  │ 📝 文本 (MD)     │  │  三合一，交给 Generate   │
                    │  │ 📋 表格 (Table)  │  │                           │
                    │  └─────────────────┘  │                           │
                    └───────────┬───────────┘                           │
                                │                                       │
                    ┌───────────▼───────────┐                           │
                    │    Eval Agent         │                           │
                    │   整体质量评估         │                           │
                    │   (包括图表/计算质量)  │                           │
                    └───────────┬───────────┘                           │
                                │                                       │
                    ┌───────────▼───────────┐                           │
                    │   Generate Agent      │                           │
                    │                       │                           │
                    │ ┌───────────────────┐ │                           │
                    │ │ 金融意图?          │ │                           │
                    │ │ → 研报模式 🆕     │ │  ← 全上下文注入            │
                    │ │   (文档+行情+指标  │ │    LLM 撰写投资评级        │
                    │ │    +图表+对比)     │ │    + 财务分析 + 风险提示   │
                    │ │                   │ │                           │
                    │ │ 非金融意图?        │ │                           │
                    │ │ → 原 RAG 回答     │ │  ← 不变                   │
                    │ └───────────────────┘ │                           │
                    └───────────────────────┘                           │
```

> **保留：** `Read → Eval → Refine → Retry` 闭环仍然作用于 RAG Agent 和 Web Agent 的检索质量。
> **新增：** Supervisor 全局管控、Finance Data Agent、Business Compute Agent 的多模态输出。
> **🆕 新增：** Generate Agent 金融意图自动走研报模式（全上下文注入 + CFA Analyst Prompt）。

### 1.3 文件变更清单（实施后）

> ✅ = 已实现，所有 Phase 1-5 完成

| 操作 | 文件 | 说明 | 状态 |
|------|------|------|:--:|
| **新增** | `src/agent/multi_agent/supervisor_agent.py` | 主编排器（Plan-then-Execute） | ✅ |
| **新增** | `src/agent/multi_agent/finance_data_agent.py` | 金融数据 Agent（MCP 行情） | ✅ |
| **新增** | `src/agent/multi_agent/business_compute_agent.py` | 业务计算 Agent | ✅ |
| **新增** | `src/agent/multi_agent/tools/__init__.py` | Agent 内部工具包 | ✅ |
| **新增** | `src/agent/multi_agent/tools/financial_calculator.py` | 财务指标计算（12 公式） | ✅ |
| **新增** | `src/agent/multi_agent/tools/data_processor.py` | Pandas 清洗/对比/排名 | ✅ |
| **新增** | `src/agent/multi_agent/tools/data_visualizer.py` | Matplotlib K线/柱状/饼图 | ✅ |
| **新增** | `src/agent/multi_agent/tools/report_renderer.py` | Jinja2 报告模板渲染 | ✅ |
| **新增** | `src/mcp_server/tools/query_market_data.py` | MCP Tool：akshare + yfinance | ✅ |
| **新增** | `examples/test_finance_agent_e2e.py` | 10 场景端到端测试 | ✅ |
| **修改** | `src/agent/multi_agent/state.py` | 扩展 6 个金融属性 | ✅ |
| **修改** | `src/agent/multi_agent/multi_agent_system.py` | 13 节点 + 研报生成 | ✅ |
| **修改** | `src/agent/multi_agent/__init__.py` | 导出新模块 | ✅ |
| **修改** | `src/agent/multi_agent/router_agent.py` | 扩展金融意图 | ✅ |
| **修改** | `src/mcp_server/protocol_handler.py` | 注册 `query_market_data` | ✅ |
| **修改** | `config/settings.yaml` | 新增 finance 配置段 | ✅ |

### 1.4 向后兼容

**新架构是加法的，不是替换的。** 非金融意图完全走原有路径。详见 [1.4 原地稿].

---

## 二、编排模式：Plan-then-Execute + Conditional Re-plan

（详见原始 SPEC 完整内容）

---

## 五-C、金融研报生成（Financial Research Report Generation）🆕

### 5C.1 一句话结论

**Router 判断为金融意图 → Generate 节点自动走研报模式。** LLM 一次性拿到文档原文 + 行情数据 + 计算指标 + 行业对比 + 图表 + 模板化初稿，用分析师 prompt 撰写专业研报。

### 5C.2 上下文注入清单

**非金融意图的 Generate：** 只喂 `local_results` + `web_results`

**金融意图的 Generate（研报模式）：**

| 数据来源 | 黑板 Key | 用途 |
|---------|---------|------|
| RAG Agent | `local_results` | 财报原文、管理层讨论、风险提示 |
| Finance Data Agent | `market_data.fundamentals` | PE/PB/ROE/毛利率/净利率/EPS |
| Finance Data Agent | `market_data.quotes` | 股价/涨跌幅/市值 |
| Business Compute | `computed_results.metrics` | 计算后的完整指标 JSON |
| Business Compute | `computed_results.comparison_table` | 行业对比 Markdown 表格 |
| Business Compute | `chart_paths` | 图表文件路径列表 |
| Business Compute | `generated_report` | Jinja2 模板化分析初稿 |

### 5C.3 研报结构（Analyst Prompt 驱动）

```
1. 投资要点（投资评级: 买入/增持/中性/减持/卖出 + 核心逻辑 + 目标价）
2. 财务分析（营收/利润/毛利率归因 + 文档原文证据 [1][2]）
3. 行业对比（与竞品/行业均值对比 + 领先/落后标注）
4. 估值分析（PE/PB 历史区间对比 + 合理判断）
5. 风险提示（至少 3 条具体风险）
6. 图表引用（正文中自然引用已生成的 PNG 图表）
```

### 5C.4 生成模式选择（Generate Node 内部）

```python
def _generate_node(self, state):
    if intent.startswith("financial_"):
        return self._generate_financial_report(state)  # 🆕 研报模式
    else:
        return self._generate_normal_response(state)    # 原 RAG 模式
```

### 5C.5 兜底策略

如果 LLM 调用失败，`_generate_fallback_report` 直接从黑板拼接数据输出 Markdown 表格 + 图表引用。

---

## 十一、实施顺序（Phase 计划）

### Phase 1：基础管线 ✅
```
1. query_market_data MCP Tool（akshare + yfinance） ✅
2. FinancialCalculator（12 公式） ✅
3. FinanceDataAgent ✅
4. settings.yaml ✅
5. DataProcessor（Pandas 清洗/对比/排名/异常检测） ✅
6. DataVisualizer（Matplotlib K线/柱状/饼图） ✅
```

### Phase 2：编排改造 ✅
```
7. Supervisor Agent（Plan-then-Execute + DAG 拓扑排序） ✅
8. multi_agent_system.py LangGraph 图（13 节点） ✅
9. intent 分类（financial_analysis/market/report） ✅
10. AgentState（6 个金融属性） ✅
11. 向后兼容 ✅
```

### Phase 3：业务计算 ✅
```
12. BusinessComputeAgent ✅
13. ReportRenderer（Jinja2 模板） ✅
14. Agent 间数据流调通 ✅
```

### Phase 4：端到端验证 ✅
```
15. E2E 测试 10 场景 ✅
16. 异常处理 + NaN/None/空状态 ✅
17. MCP 回归测试 6/6 ✅
```

### Phase 5：研报生成 🆕 ⬅ 本次完成
```
18. Generate Node 金融意图路由 → _generate_financial_report ✅
19. 全上下文注入（7 路数据源 → LLM analyst prompt） ✅
20. 兜底报告策略（LLM 故障时直接输出数据） ✅
```

---

## 十二、风险与注意事项

| 风险 | 应对 |
|------|------|
| akshare/yfinance 接口不稳定 | 返回兜底数据 + 错误消息透传，不阻塞主流程 |
| 图表生成耗时 | 子进程中执行 + 超时控制（30s） |
| LLM 拆分子任务不合理 | Plan 阶段校验 + Eval 阶段纠错 |
| Business Compute 计算结果依赖数据质量 | 每个函数加输入校验 + NaN/None 处理 |
| 与现有 RAG 系统兼容性 | 新增节点不删除旧节点，渐进式替换 |
| 研报生成质量依赖文档质量 | RAG 检索结果越差，研报越空洞——根源在数据，不在架构 |
