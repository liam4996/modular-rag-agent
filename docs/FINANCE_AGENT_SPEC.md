# 金融智能体改造 SPEC

> 基于现有 `Modular RAG MCP Server` 多 Agent 架构，新增金融领域能力。

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
                    │   多模态最终回答       │                           │
                    │   图表+表格+文本+溯源  │                           │
                    └───────────────────────┘                           │
```

> **保留：** `Read → Eval → Refine → Retry` 闭环仍然作用于 RAG Agent 和 Web Agent 的检索质量。
> **新增：** Supervisor 全局管控、Finance Data Agent、Business Compute Agent 的多模态输出。

### 1.3 文件变更清单

| 操作 | 文件 | 说明 |
|------|------|------|
| **新增** | `src/agent/multi_agent/supervisor_agent.py` | 主编排器（替代现有 Router） |
| **新增** | `src/agent/multi_agent/finance_data_agent.py` | 金融数据 Agent（MCP 行情） |
| **新增** | `src/agent/multi_agent/business_compute_agent.py` | 业务计算 Agent |
| **新增** | `src/agent/multi_agent/tools/__init__.py` | Agent 内部工具包 |
| **新增** | `src/agent/multi_agent/tools/financial_calculator.py` | 财务指标计算工具 |
| **新增** | `src/agent/multi_agent/tools/data_visualizer.py` | 可视化生成工具 |
| **新增** | `src/agent/multi_agent/tools/report_renderer.py` | 报告模板渲染工具 |
| **新增** | `src/mcp_server/tools/query_market_data.py` | 新增 MCP Tool：行情查询 |
| **修改** | `src/agent/multi_agent/state.py` | 扩展 AgentState 字段 |
| **修改** | `src/agent/multi_agent/multi_agent_system.py` | 集成 Supervisor + 新 Agent |
| **修改** | `src/agent/multi_agent/__init__.py` | 导出新模块 |
| **修改** | `src/agent/multi_agent/router_agent.py` | 扩展意图枚举 + 金融意图 |
| **修改** | `src/mcp_server/protocol_handler.py` | 注册 `query_market_data` |
| **修改** | `config/settings.yaml` | 新增 finance 配置段 |

### 1.4 向后兼容：纯文档检索场景

**新架构是加法的，不是替换的。** 当用户只想搜公司内部文档时，流程和现在一模一样：

```
用户: "公司最新的人事制度怎么规定的？"

  intent → document_search  （Router 识别为非金融意图）
    ↓
  Supervisor: 无需拆子任务，直接路由
    ↓
  RAG Agent → Read → Eval → Refine → Generate
  ↑ 完全走原有闭湾，不触发任何新 Agent
```

不需要 Finance Data Agent，不需要 Business Compute Agent，不需要 `query_market_data`——因为 Supervisor 看到 `intent=document_search` 就直接派到 RAG Agent，其他 Agent 根本不启动。

**新老路径对照：**

| 用户意图 | 改前路径 | 改后路径 | 走了新增 Agent？ |
|---------|---------|---------|:--:|
| "合同违约怎么算" | Router → Retrieve → Eval → Generate | 完全相同 | ❌ |
| "查一下宁德时代" | Router → Retrieve/Web → Eval → Generate | RAG + Web + Finance + Compute + Eval → Generate | ✅ |
| "公司出差报销流程" | Router → Retrieve → Eval → Generate | 完全相同 | ❌ |
| "财务报表帮我出图" | 不支持 | Plan → RAG + Finance + Compute → Generate | ✅ |
| 闲聊 | Router → Generate | 完全相同 | ❌ |

**一句话：原来的功能一个没少，金融能力是可选的附加项。**

---

## 二、编排模式选择：Plan-then-Execute + Conditional Re-plan

### 2.1 为什么不选 ReAct

| 维度 | ReAct | Plan-then-Execute | Plan + Feedback（本项目） |
|------|-------|-------------------|-------------------------|
| LLM 调用次数 | 每步 1 次 | 1 次（规划） | 1 次（规划）+ N 次（Eval 纠错） |
| 并行能力 | 强制串行 | 无依赖全并行 | 无依赖全并行 |
| 延迟 | 高（5 步 ≈ 5×LLM延迟） | 低（1×LLM延迟） | 低（1 + 偶尔纠错） |
| 适应意外 | 天然适应 | 不懂变通 | Eval 兜底变通 |
| 适合场景 | 探索型任务 | 结构型任务 | 结构型 + 容错 |

金融分析任务高度结构化——你知道要做"搜文档 → 拿行情 → 算指标 → 画图"，不需要每步重新想。Plan-then-Execute 一步到位，Eval 兜底纠错。

### 2.2 执行流程

```
用户: "分析宁德时代 Q3 并对比行业，出图"

  ① Plan（LLM 一次性规划）:
     ┌──────────────────────────────────────────┐
     │ task_1: RAG 搜财报        → 无依赖，并行  │
     │ task_2: 查行情 API        → 无依赖，并行  │
     │ task_3: 计算 + 行业对比   → 依赖 1,2      │
     │ task_4: 画图              → 依赖 3        │
     └──────────────────────────────────────────┘

  ② Execute（并行 + 顺序混合）:
     task_1 ──┐
              ├──(并行)──→ task_3 ──→ task_4
     task_2 ──┘

  ③ Global Eval（安全网）:
     计算结果合理？ROE=180%？→ 重新计算
     文档相关度低？→ replan（重拆任务）
     数据缺失？→ 标记 fallback，继续
```

### 2.3 LangGraph 层面的实现

```python
# 这不是 ReAct 的 Thought→Action→Observation 循环
# 而是 Plan→Dispatch→Execute→Eval→(Replan|Generate)

class SupervisorAgent:
    
    def plan(self, state: AgentState) -> AgentState:
        """Plan 阶段：一次 LLM 调用，拆出所有子任务"""
        ...
    
    def dispatch(self, state: AgentState) -> str:
        """按 depends_on 拓扑排序，无依赖的并行派发，有依赖的等上游完成"""
        ...
    
    def aggregate(self, state: AgentState) -> AgentState:
        """收集所有 sub-agent 的输出，合并为统一上下文"""
        ...
    
    def evaluate(self, state: AgentState) -> str:
        """Global Eval：检查质量 → generate / replan / retry"""
        ...
```

---

## 三、Supervisor Agent（主编排器）

### 3.1 职责

| 职责 | 实现方式 |
|------|---------|
| **意图识别** | 扩展 Router 分类：chat / rag / web / finance · 复合意图支持 |
| **任务拆分** | LLM 将复杂金融问题拆为子任务列表，写入 `blackboard["task_plan"]` |
| **路由决策** | 条件边：子任务类型 → 对应 Agent |
| **结果聚合** | 各 Agent 写完 blackboard → Supervisor 合并 → 交给 Generate |
| **流程管控** | 控制重试、超时、并发执行 |

### 3.2 意图分类体系（扩展后）

```python
class IntentType(Enum):
    CHAT = "chat"                        # 闲聊
    DOC_SEARCH = "document_search"       # 文档检索（原 search）
    WEB_SEARCH = "web_search"            # 联网搜索
    FINANCIAL_ANALYSIS = "financial_analysis"  # 金融分析（新增）
    FINANCIAL_MARKET = "financial_market"      # 行情查询（新增）
    FINANCIAL_REPORT = "financial_report"      # 报告生成（新增）
    HYBRID = "hybrid"                    # 复合意图
```

### 3.3 任务拆分 Schema

```python
# blackboard["task_plan"] 结构
{
    "original_query": "分析宁德时代Q3财报并对比行业平均，用图表展示",
    "subtasks": [
        {
            "id": "task_1",
            "type": "document_search",       # → RAG Agent
            "query": "宁德时代 2024 Q3 财报 营收 利润 毛利率",
            "collection": "quarterly_earnings",
            "depends_on": []
        },
        {
            "id": "task_2",
            "type": "financial_market",     # → Finance Data Agent
            "query": "新能源电池行业平均ROE 毛利率 2024Q3",
            "symbols": ["300750.SZ", "300014.SZ", "002074.SZ"],
            "depends_on": []
        },
        {
            "id": "task_3",
            "type": "financial_computation", # → Business Compute Agent
            "operation": "compare_with_industry",
            "input_from": ["task_1", "task_2"],
            "depends_on": ["task_1", "task_2"]
        },
        {
            "id": "task_4",
            "type": "financial_computation",
            "operation": "visualize",
            "chart_type": "bar_compare",
            "input_from": ["task_3"],
            "depends_on": ["task_3"]
        }
    ]
}
```

### 3.4 LangGraph 图结构

(详见 SPEC 完整版)

### 3.5 聚合逻辑

(详见 SPEC 完整版)

---

## 四、Finance Data Agent（金融数据 MCP Agent）

### 4.1 职责

通过一个 MCP Tool `query_market_data` 获取实时行情和基本面数据。

### 4.2 统合性 MCP Tool

```yaml
name: query_market_data
description: >
  统一的金融市场数据查询接口。覆盖 A 股 / 港股 / 美股。
  数据类型：quote, fundamentals, history, industry_comparison
parameters:
  symbols: array[string]
  data_types: array[string] (quote|fundamentals|history|industry_comparison)
  period: string (1mo|3mo|6mo|1y|3y|5y)
```

数据源：akshare（A 股）+ yfinance（美港股），均免费。

---

## 五、Business Compute Agent（业务计算 Agent）

### 5.1 核心定位

> **用 Python 代码执行 LLM 无法完成的业务逻辑。**
> 是一组**真实的 Python 函数**，操作 Finance Data Agent 和 RAG Agent 返回的结构化数据。

### 5.2 多模态输出能力

Business Compute Agent 的最终产出是**三模态合一**的结果集：

| 模态 | 生成方式 | 存储位置 | 最终呈现 |
|------|---------|---------|---------|
| **图表 PNG** | Matplotlib/Plotly 纯 Python | `data/charts/{session}/xxx.png` | Generate 以 MCP ImageContent 返回 |
| **文本 MD** | FinancialCalculator 数值 → f-string 格式化 | `blackboard["computed_results"]` | Generate 拼接进回答正文 |
| **表格 Table** | Pandas DataFrame → `to_markdown()` | `blackboard["computed_results"]` | Generate 嵌入为 Markdown 表格 |

### 5.3 四大子模块

1. **FinancialCalculator** — 纯 Python 函数：ROE, ROA, PE, PB, 资产负债率, 同比/环比
2. **DataProcessor** — Pandas 数据清洗、多维度对比、排名、异常检测
3. **DataVisualizer** — Matplotlib/Plotly 生成 K 线图、趋势图、对比柱状图、饼图
4. **ReportRenderer** — Jinja2 模板渲染标准化财报分析报告

---

## 五-B、Eval Agent 与 Refine Agent（保留原有闭环）

### 5B.1 一句话结论

**Eval Agent 和 Refine Agent 完整保留，不做任何删除。**

| 层面 | 作用 | 说明 |
|------|------|------|
| **RAG/Web 检索闭环** | Eval → Refine → Retry | 和现在一模一样 |
| **全局结果审核** | Eval 检查所有 Agent 的输出 | 新增：数据完整性、计算合理性、图文一致性 |

### 5B.2 文件层面

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/agent/multi_agent/eval_agent.py` | **保留不动** | RAG 检索评估逻辑不变 |
| `src/agent/multi_agent/refine_agent.py` | **保留不动** | 查询改写逻辑不变 |
| `src/agent/multi_agent/multi_agent_system.py` | 新增一个 `_global_eval_node` | 全局审核入口 |

---

## 六、AgentState 扩展

新增 blackboard keys：`task_plan`, `task_results`, `market_data`, `computed_results`, `chart_paths`, `generated_report`, `aggregated_context`。

---

## 七、config/settings.yaml 新增配置

```yaml
finance:
  market_data:
    a_share:
      provider: "akshare"
    us_hk:
      provider: "yfinance"
  computation:
    visualization:
      output_dir: "data/charts"
      default_format: "png"
      default_dpi: 150
    report:
      template_dir: "templates/finance"
      output_dir: "data/reports"
```

---

## 八、新增 MCP Tool 定义汇总

### 8.1 `query_market_data`（唯一新增 MCP Tool）

| 字段 | 值 |
|------|-----|
| 名称 | `query_market_data` |
| 类型 | MCP Tool（外部 API 调用） |
| 数据源 | akshare（A 股）/ yfinance（美港股） |
| 输入 | `symbols`, `data_types`, `period` |
| 输出 | 统一 schema 的行情/基本面 JSON |
| 注册位置 | `src/mcp_server/protocol_handler.py` |

---

## 九、LangGraph 边路由决策表

```
当前节点 → 下一节点逻辑：

supervisor:
  - intent=chat                    → generate
  - intent=document_search         → rag
  - intent=web_search              → web
  - intent=financial_*             → plan (拆解子任务)
  - complexity=complex             → plan

plan:
  - 遍历 task_plan.subtasks，按类型并行派发
  - 全部完成                       → supervisor (聚合)

rag / web:
  - 执行完成                       → read → eval_rag（内部闭环）
  ↑ 原有 Eval/Refine 闭环保留，不动

finance_data / business_compute:
  - 执行完成                       → supervisor

supervisor (聚合后):
  - 全部完成                       → global_eval

global_eval (新增):
  - 数据完整 + 计算合理            → generate
  - 计算异常 + 可重算              → business_compute
  - 图文不一致                     → generate (标注 attention)

generate:
  - 多模态聚合输出                 → END
```

---

## 十、依赖清单

```toml
[project.optional-dependencies]
finance = [
    "akshare>=1.14.0",
    "yfinance>=0.2.40",
    "pandas>=2.0.0",
    "matplotlib>=3.8.0",
    "plotly>=5.18.0",
    "mplfinance>=0.12.10b0",
    "jinja2>=3.1.0",
    "numpy>=1.24.0",
]
```

---

## 十一、实施顺序（Phase 计划）

### Phase 1：基础管线
1. 新增 query_market_data MCP Tool
2. 新增 FinancialCalculator（纯 Python 函数）
3. 新增 FinanceDataAgent
4. 扩展 settings.yaml

### Phase 2：编排改造
5. 新增 Supervisor Agent
6. 改造 multi_agent_system.py 的 LangGraph 图
7. 扩展 intent 分类 + AgentState
8. 兼容现有 RAG / Web Agent 逻辑

### Phase 3：业务计算
9. 新增 BusinessComputeAgent + DataProcessor + DataVisualizer + ReportRenderer
10. Agent 间数据流调通

### Phase 4：端到端验证
11. 端到端测试 + 异常处理

---

## 十二、风险与注意事项

| 风险 | 应对 |
|------|------|
| akshare/yfinance 接口不稳定 | 返回兜底数据 + 错误消息透传，不阻塞主流程 |
| 图表生成耗时 | 子进程中执行 + 超时控制（30s） |
| LLM 拆分子任务不合理 | Plan 阶段校验 + Eval 阶段纠错 |
| Business Compute 计算结果依赖数据质量 | 每个函数加输入校验 + NaN/None 处理 |
| 与现有 RAG 系统兼容性 | 新增节点不删除旧节点，渐进式替换 |
