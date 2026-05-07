# 智能投研助手 — 基于多 Agent 架构的 AI 投研平台

> 基于现有 `Modular RAG MCP Server` 多 Agent 架构，构建全链路 AI 投研平台。
>
> 痛点：研究员读财报、研报、公告、纪要耗时极长
> 
> 解决：RAG 文档检索 + 实时行情数据 + 财务指标计算 + 图表可视化 + 研报生成
> 
> 落地形态：AI 投研助手、个股问答机器人、自动研报生成、行业对比分析
>
> **状态：设计文档。** 本文档是先于代码编写的架构设计稿。
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

### 1.3 文件变更清单（实施后）

> ✅ = 已实现，所有 Phase 1-4 完成

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
| **修改** | `src/agent/multi_agent/multi_agent_system.py` | 13 节点 LangGraph 图 | ✅ |
| **修改** | `src/agent/multi_agent/__init__.py` | 导出新模块 | ✅ |
| **修改** | `src/agent/multi_agent/router_agent.py` | 扩展金融意图 | ✅ |
| **修改** | `src/mcp_server/protocol_handler.py` | 注册 `query_market_data` | ✅ |
| **修改** | `config/settings.yaml` | 新增 finance 配置段 | ✅ |

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

```python
def _build_graph(self) -> StateGraph:
    workflow = StateGraph(AgentState)
    
    # ========== 节点 ==========
    workflow.add_node("supervisor", self._supervisor_node)    # 新：主编排
    workflow.add_node("plan", self._plan_node)                # 改：支持金融子任务
    workflow.add_node("rag", self._rag_node)                  # 原 retrieve
    workflow.add_node("web", self._web_node)                  # 原 web
    workflow.add_node("finance_data", self._finance_data_node) # 新
    workflow.add_node("business_compute", self._business_compute_node) # 新
    workflow.add_node("read", self._read_node)                # 保留
    workflow.add_node("eval", self._eval_node)                # 保留
    workflow.add_node("refine", self._refine_node)            # 保留
    workflow.add_node("generate", self._generate_node)        # 改：支持多源聚合
    
    workflow.set_entry_point("supervisor")
    
    # Supervisor → Plan / RAG / Web / Finance Data / Generate
    workflow.add_conditional_edges(
        "supervisor",
        self._route_from_supervisor,
        {
            "plan": "plan",
            "rag": "rag",
            "web": "web",
            "finance_data": "finance_data",
            "generate": "generate",
        }
    )
    
    # Plan → 并行派发到各 Agent
    workflow.add_conditional_edges(
        "plan",
        self._dispatch_subtasks,
        {
            "rag": "rag",
            "web": "web",
            "finance_data": "finance_data",
            "business_compute": "business_compute",
            "generate": "generate",
        }
    )
    
    # 各 Agent 执行完 → 回到 Supervisor 聚合
    workflow.add_edge("rag", "supervisor")
    workflow.add_edge("web", "supervisor")
    workflow.add_edge("finance_data", "supervisor")
    workflow.add_edge("business_compute", "supervisor")
    
    # 保留原有回路
    workflow.add_edge("refine", "rag")
    workflow.add_edge("generate", END)
    
    return workflow.compile()
```

### 3.5 聚合逻辑

```python
def _aggregate_results(self, state: AgentState) -> AgentState:
    """
    从 blackboard 中收集所有 Agent 的输出，合并为统一上下文。
    
    blackboard keys:
      - local_results: RAG Agent 输出
      - web_results: Web Agent 输出
      - market_data: Finance Data Agent 输出
      - computed_results: Business Compute Agent 输出
      - chart_paths: 可视化图片路径
    """
    aggregated = {
        "local_docs": state.blackboard.get("local_results", []),
        "web_info": state.blackboard.get("web_results", []),
        "market": state.blackboard.get("market_data", {}),
        "computation": state.blackboard.get("computed_results", {}),
        "charts": state.blackboard.get("chart_paths", []),
        "report": state.blackboard.get("generated_report", None),
    }
    state.blackboard["aggregated_context"] = aggregated
    return state
```

---

## 四、Finance Data Agent（金融数据 MCP Agent）

### 4.1 职责

通过一个 MCP Tool `query_market_data` 获取实时行情和基本面数据。

### 4.2 不是"多个 tool"，是一个统合性的 MCP Tool

```yaml
name: query_market_data
description: >
  统一的金融市场数据查询接口。覆盖 A 股 / 港股 / 美股。
  
  数据类型：
  - quote: 实时报价（股价、涨跌幅、换手率、市值）
  - fundamentals: 基本面（PE/PB/ROE/营收/利润/资产负债率）
  - history: 历史 K 线（日/周/月）
  - industry_comparison: 行业对比数据
  
parameters:
  symbols:
    type: array[string]
    description: 股票代码列表，如 ["300750.SZ", "000001.SZ", "AAPL"]
  data_types:
    type: array[string]
    enum: [quote, fundamentals, history, industry_comparison]
    description: 需要获取的数据类型
  period:
    type: string
    enum: [1mo, 3mo, 6mo, 1y, 3y, 5y]
    default: 3mo
```

### 4.3 实现方案

```python
# src/mcp_server/tools/query_market_data.py

class MarketDataSource(Enum):
    A_SHARE = "akshare"     # 免费 A 股数据
    US_HK = "yfinance"      # 免费美股/港股

class QueryMarketDataTool:
    """
    数据源：
    - A 股：akshare（免费，pip install akshare）
    - 美股/港股：yfinance（免费，pip install yfinance）
    
    返回统一 schema，屏蔽底层差异。
    """
    
    async def execute(
        self,
        symbols: List[str],
        data_types: List[str],
        period: str = "3mo",
    ) -> types.CallToolResult:
        # 1. 按市场分组（300750.SZ → a_share, AAPL → us）
        # 2. 并行调用对应数据源
        # 3. 统一返回格式
        ...
```

### 4.4 Agent 封装

```python
# src/agent/multi_agent/finance_data_agent.py

class FinanceDataAgent:
    """
    金融数据 Agent。
    
    本身不直接调 API，而是通过 MCP Tool `query_market_data` 获取数据。
    拿到原始数据后做基本清洗和格式化，写入 blackboard["market_data"]。
    
    与 RAG Agent 的关系：
    - RAG Agent 回答"为什么"（财报原文、研报观点）
    - Finance Data Agent 回答"是什么"（股价、PE、行业均值）
    - 两者结果汇合后交给 Business Compute Agent 做交叉分析
    """
    
    def query_market(
        self,
        symbols: List[str],
        data_types: List[str],
        period: str = "3mo",
    ) -> Dict[str, Any]:
        """
        调用 MCP query_market_data tool，返回结构化行情数据。
        """
        ...
```

---

## 五、Business Compute Agent（业务计算 Agent）

### 5.1 核心定位

> **用 Python 代码执行 LLM 无法完成的业务逻辑。**
>
> 不做"把 LLM 能做的事包装成 tool"。它是一组**真实的 Python 函数**，
> 操作的是 Finance Data Agent 和 RAG Agent 返回的结构化数据。

### 5.2 多模态输出能力

Business Compute Agent 的最终产出是**三模态合一**的结果集：

```
                    Business Compute Agent
                    ┌─────────────────────────────────┐
                    │                                 │
     行情JSON ──────┤  ┌─────────────────────────┐   │
     财报数据 ──────┤  │   FinancialCalculator    │   │
     研报文本 ──────┤  │   (纯Python, 不调LLM)     │   │
                    │  └───────────┬─────────────┘   │
                    │              │                 │
                    │    ┌─────────▼─────────┐       │
                    │    │   DataProcessor   │       │
                    │    │   (Pandas清洗/聚合) │       │
                    │    └─────────┬─────────┘       │
                    │              │                 │
                    │  ┌───────────┼───────────┐     │
                    │  │           │           │     │
                    │  ▼           ▼           ▼     │
                    │ ┌──────┐ ┌──────┐ ┌──────┐    │
                    │ │📊图表 │ │📝文本 │ │📋表格 │    │
                    │ │ PNG  │ │  MD  │ │Table │    │
                    │ └──┬───┘ └──┬───┘ └──┬───┘    │
                    │    │        │        │        │
                    └────┼────────┼────────┼────────┘
                         │        │        │
                         ▼        ▼        ▼
                  ┌─────────────────────────────┐
                  │     ComputedResult          │
                  │                             │
                  │  .charts: List[str]         │  ← PNG 文件路径
                  │  .text: str                  │  ← Markdown 分析文字
                  │  .tables: List[Dict]         │  ← 结构化表格数据
                  │  .report: Optional[str]      │  ← 完整渲染报告
                  │                             │
                  │  全部写入 blackboard         │
                  │  → Generate Agent 聚合输出   │
                  └─────────────────────────────┘
```

**三模态数据流：**

| 模态 | 生成方式 | 存储位置 | 最终呈现 |
|------|---------|---------|---------|
| **图表 PNG** | Matplotlib/Plotly 纯 Python | `data/charts/{session}/xxx.png` | Generate 以 MCP ImageContent 返回 |
| **文本 MD** | FinancialCalculator 数值 → f-string 格式化 | `blackboard["computed_results"]` | Generate 拼接进回答正文 |
| **表格 Table** | Pandas DataFrame → `to_markdown()` | `blackboard["computed_results"]` | Generate 嵌入为 Markdown 表格 |

**输出结构：**

```python
@dataclass
class ComputedResult:
    """Business Compute Agent 的统一输出"""
    
    # 图表
    charts: List[ChartOutput] = field(default_factory=list)
    
    # 文本
    analysis_text: str = ""                    # 计算结果的文字描述
    key_findings: List[str] = field(default_factory=list)  # 关键发现
    
    # 表格
    metric_tables: List[TableOutput] = field(default_factory=list)
    
    # 完整报告（可选）
    rendered_report: Optional[str] = None      # Jinja2 渲染的完整 Markdown 报告

@dataclass
class ChartOutput:
    """单张图表"""
    title: str
    chart_type: str              # "kline" / "trend_line" / "bar_compare" / "pie"
    file_path: str               # "data/charts/sess_001/kline_300750.png"
    description: str             # 图表说明文字（给 LLM 理解用的）

@dataclass
class TableOutput:
    """单张表格"""
    title: str
    headers: List[str]
    rows: List[List[Any]]
    markdown_str: str            # 预渲染的 Markdown 表格
```

### 5.3 内部架构

```
Finance Data Agent 输出          RAG Agent 输出
(行情 JSON)                     (文档原文 + 元数据)
       │                              │
       └──────────┬───────────────────┘
                  ▼
     ┌──────────────────────────┐
     │  Business Compute Agent   │
     │                          │
     │  ┌────────────────────┐  │
     │  │ FinancialCalculator│  │  ← 财务指标计算
     │  │ (纯 Python 函数)    │  │
     │  └────────────────────┘  │
     │  ┌────────────────────┐  │
     │  │ DataProcessor      │  │  ← Pandas 清洗/聚合
     │  │ (Pandas)           │  │
     │  └────────────────────┘  │
     │  ┌────────────────────┐  │
     │  │ DataVisualizer     │  │  ← Matplotlib/Plotly
     │  │ (图表生成)         │  │
     │  └────────────────────┘  │
     │  ┌────────────────────┐  │
     │  │ ReportRenderer     │  │  ← Jinja2 模板渲染
     │  │ (报告生成)         │  │
     │  └────────────────────┘  │
     └──────────┬───────────────┘
                ▼
     blackboard["computed_results"]
     blackboard["chart_paths"]
     blackboard["generated_report"]
```

### 5.4 子模块 1：FinancialCalculator（财务计算引擎）

```python
# src/agent/multi_agent/tools/financial_calculator.py

"""
纯 Python 函数，不依赖 LLM，不依赖外部 API。
输入：RAG 提取的财务数据 或 行情 API 返回的基本面
输出：计算后的指标
"""

@dataclass
class FinancialMetrics:
    """标准化财务指标输出"""
    profitability: Dict[str, float]    # ROE, ROA, gross_margin, net_margin
    solvency: Dict[str, float]         # current_ratio, quick_ratio, debt_to_asset
    efficiency: Dict[str, float]       # inventory_turnover, receivable_turnover
    valuation: Dict[str, float]        # PE, PB, PS, EV_EBITDA
    growth: Dict[str, float]           # revenue_yoy, profit_yoy, eps_yoy

def calculate_roe(net_income: float, equity: float) -> float:
    """ROE = 净利润 / 净资产 × 100%"""
    ...

def calculate_debt_to_asset(total_liabilities: float, total_assets: float) -> float:
    """资产负债率 = 总负债 / 总资产 × 100%"""
    ...

def calculate_yoy(current_value: float, previous_value: float) -> float:
    """同比增长率 = (当期 - 同期) / |同期| × 100%"""
    ...

def calculate_qoq(current_quarter: float, previous_quarter: float) -> float:
    """环比增长率"""
    ...

def calculate_pe(price: float, eps: float) -> float:
    """市盈率 PE"""
    ...

def calculate_pb(price: float, book_value_per_share: float) -> float:
    """市净率 PB"""
    ...

def compute_all_metrics(
    financial_data: Dict[str, Any],
    market_data: Dict[str, Any],
) -> FinancialMetrics:
    """
    批量计算所有财务指标。
    输入是 RAG 提取 + 行情 API 的 JSON，输出标准化 FinancialMetrics。
    """
    ...
```

### 5.5 子模块 2：DataProcessor（Pandas 数据处理）

```python
# src/agent/multi_agent/tools/data_processor.py

"""
用 Pandas 做数据清洗和结构化分析。
"""

def clean_financial_data(raw_data: List[Dict]) -> pd.DataFrame:
    """将各 Agent 的异构输出清洗为统一 DataFrame"""
    ...

def multi_dimension_compare(
    df: pd.DataFrame,
    dimensions: List[str],     # e.g. ["company", "period", "metric"]
    metrics: List[str],        # e.g. ["revenue", "roe", "gross_margin"]
) -> pd.DataFrame:
    """多维度对比分析"""
    ...

def rank_companies(
    df: pd.DataFrame,
    by_metric: str,
    ascending: bool = False,
) -> pd.DataFrame:
    """按指标排名"""
    ...

def detect_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = "iqr",       # "iqr" or "zscore"
) -> List[int]:
    """异常值检测"""
    ...
```

### 5.6 子模块 3：DataVisualizer（图表生成）

```python
# src/agent/multi_agent/tools/data_visualizer.py

"""
用 Matplotlib / Plotly 生成金融图表。

输出路径：data/charts/{session_id}/{chart_name}.png
"""

class DataVisualizer:
    def __init__(self, output_dir: str = "data/charts"):
        self.output_dir = Path(output_dir)
    
    def kline_chart(
        self,
        df: pd.DataFrame,           # columns: date, open, high, low, close, volume
        title: str = "K线图",
        save_path: Optional[str] = None,
    ) -> str:
        """K线图（用 mplfinance 或 Plotly）"""
        ...
    
    def trend_chart(
        self,
        df: pd.DataFrame,
        x: str,
        y: Union[str, List[str]],
        chart_type: str = "line",   # line / bar / area
        title: str = "趋势图",
        save_path: Optional[str] = None,
    ) -> str:
        """趋势折线图 / 柱状图"""
        ...
    
    def comparison_chart(
        self,
        data: Dict[str, pd.DataFrame],  # {"公司A": df, "公司B": df}
        metric: str,                     # "ROE" / "revenue" etc.
        chart_type: str = "grouped_bar",
        title: str = "对比图",
        save_path: Optional[str] = None,
    ) -> str:
        """多公司横向对比图"""
        ...
    
    def pie_chart(
        self,
        labels: List[str],
        values: List[float],
        title: str = "占比图",
        save_path: Optional[str] = None,
    ) -> str:
        """饼图 / 环形图"""
        ...
```

**注意**：图表生成是**纯 Python 代码**，不经过 LLM。LLM 只负责理解用户意图 → 选择图表类型 → 传入数据 → 得到图片路径。

### 5.7 子模块 4：ReportRenderer（报告模板渲染）

```python
# src/agent/multi_agent/tools/report_renderer.py

"""
用 Jinja2 模板 + 结构化数据 → 生成标准化金融分析报告。
"""

class ReportRenderer:
    TEMPLATES = {
        "earnings_review": "templates/finance/earnings_review.md.j2",
        "company_analysis": "templates/finance/company_analysis.md.j2",
        "industry_comparison": "templates/finance/industry_comparison.md.j2",
        "risk_alert": "templates/finance/risk_alert.md.j2",
    }
    
    def render(
        self,
        template_name: str,
        data: Dict[str, Any],      # FinancialMetrics + market_data + computed
        charts: List[str] = None,  # 图表文件路径列表
        output_path: Optional[str] = None,
    ) -> str:
        """
        渲染报告为 Markdown。
        
        报告结构：
        1. 标题 + 日期
        2. 核心指标摘要表
        3. 营收/利润分析（文本 + 图表）
        4. 估值分析
        5. 风险评估
        6. 结论
        """
        ...
```

### 5.8 Business Compute Agent 主类

```python
# src/agent/multi_agent/business_compute_agent.py

class BusinessComputeAgent:
    """
    业务计算 Agent。
    
    不调 LLM。它的输入是其他 Agent 产出的结构化数据，
    输出是计算结果 + 图表路径 + 报告 Markdown。
    """
    
    def __init__(self, settings=None):
        self.calculator = FinancialCalculator()
        self.processor = DataProcessor()
        self.visualizer = DataVisualizer()
        self.renderer = ReportRenderer()
    
    def execute(self, state: AgentState) -> AgentState:
        """
        根据 task_plan 中 business_compute 类型的子任务，
        执行相应的计算/可视化/报告操作。
        """
        subtasks = state.blackboard.get("task_plan", {}).get("subtasks", [])
        compute_tasks = [t for t in subtasks if t["type"] == "financial_computation"]
        
        for task in compute_tasks:
            operation = task["operation"]
            
            if operation == "calculate_metrics":
                result = self._do_calculate_metrics(state)
            elif operation == "compare_companies":
                result = self._do_compare(state, task)
            elif operation == "visualize":
                result = self._do_visualize(state, task)
            elif operation == "generate_report":
                result = self._do_render_report(state, task)
            
            state.add_to_blackboard(f"task_result_{task['id']}", result, "business_compute")
        
        return state
```

---

## 五-B、Eval Agent 与 Refine Agent（保留原有闭环）

### 5B.1 一句话结论

**Eval Agent 和 Refine Agent 完整保留，不做任何删除。** 它们在两个层面继续发挥作用：

| 层面 | 作用 | 说明 |
|------|------|------|
| **RAG/Web 检索闭环** | Eval → Refine → Retry | 和现在一模一样，检索质量不够就改写查询重新搜 |
| **全局结果审核** | Eval 检查所有 Agent 的输出 | 新增：评估 Finance Data 的数据完整性、Business Compute 的计算合理性 |

### 5B.2 原有闭环（不变）

```
RAG Agent 检索
  → Read Node（阅读结果）
    → Eval Agent（评估检索质量）
      ├── relevance < 0.4  → Refine Agent（改写查询）→ 回到 RAG 重新检索
      ├── 需要联网        → Web Agent（联网搜索）→ Read → Eval
      └── 质量 OK          → 写入 blackboard，交给下一步
```

**这个闭环是现在 `multi_agent_system.py` 里 Run 得好好的东西，一根毛都不动。**

### 5B.3 全局审核（新增）

当 Supervisor 收集完所有 Agent 的输出后，在做最终 Generate 之前，Eval 多做一层全局检查：

```python
def _global_eval_node(self, state: AgentState) -> AgentState:
    """
    全局审核节点。
    
    检查：
    1. Finance Data Agent 返回的行情数据是否完整？
       → 缺失关键字段 → 标记 fallback，不阻塞，用已有数据
    2. Business Compute Agent 的计算结果是否合理？
       → ROE > 100% 或 < -100% → 标记 anomaly
    3. RAG Agent 检索结果是否与行情数据矛盾？
       → 财报说营收增长 20% 但行情显示股价暴跌 30% → 标记 attention
    """
    ...
```

### 5B.4 在 LangGraph 中的位置

```
Business Compute Agent 完成
        │
        ▼
┌───────────────────┐
│  Global Eval Node  │  ← 检查所有 Agent 输出的质量
│                    │
│  检查项:           │
│  · 数据完整性      │
│  · 计算结果合理性  │
│  · 图文一致性      │
│  · 来源可信度      │
└────────┬──────────┘
         │
    ┌────┴────┐
    │          │
    ▼          ▼
 质量OK    需要修正
    │          │
    │     ┌────▼────┐
    │     │  Refine  │  ← 重新调用相关 Agent
    │     │  /Retry  │
    │     └────┬────┘
    │          │
    ▼          ▼
  Generate  Agent
```

### 5B.5 文件层面

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/agent/multi_agent/eval_agent.py` | **保留不动** | RAG 检索评估逻辑不变 |
| `src/agent/multi_agent/refine_agent.py` | **保留不动** | 查询改写逻辑不变 |
| `src/agent/multi_agent/multi_agent_system.py` | 新增一个 `_global_eval_node` | 全局审核入口，内部复用 eval_agent |

---

## 五-C、金融研报生成（Financial Research Report Generation）🆕

### 5C.1 一句话结论

**Router 判断为金融意图 → Generate 节点自动走研报模式。** LLM 一次性拿到文档原文 + 行情数据 + 计算指标 + 行业对比 + 图表 + 模板化初稿，用分析师 prompt 撰写专业研报。

### 5C.2 上下文注入清单

**非金融意图的 Generate：** 只喂 `local_results` + `web_results`

**金融意图的 Generate（研报模式）：**

| 数据来源 | 黑版 Key | 用途 |
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
研报模式 Prompt 要求的输出结构:

1. 投资要点（开篇 3-5 句话）
   - 投资评级: 买入/增持/中性/减持/卖出
   - 核心逻辑: 不超过 3 条
   - 目标价估算 (数据充分时)

2. 财务分析
   - 营收/利润/毛利率归因分析
   - 引用文档原文证据 [1][2]

3. 行业对比
   - 与行业均值/竞品对比
   - 标注领先/落后指标

4. 估值分析
   - PE/PB 与历史区间、行业均值对比
   - 当前估值合理判断

5. 风险提示
   - 至少 3 条具体风险
   - 从文档原文提取

6. 图表引用
   - 正文中自然引用已生成的 PNG 图表
```

### 5C.4 生成模式选择（Generate Node 内部）

```python
def _generate_node(self, state):
    intent = (state.intent or "").lower()

    if intent.startswith("financial_"):
        return self._generate_financial_report(state)  # 🆕 研报模式
    else:
        return self._generate_normal_response(state)    # 原 RAG 模式
```

### 5C.5 兜底策略

如果 LLM 调用失败（网络超时等），`_generate_fallback_report` 直接从黑版中拼接数据输出 Markdown 表格 + 图表引用——保证即使 LLM 挂了也不白跑。

---

## 六、AgentState 扩展

### 6.1 新增 Blackboard Keys

```python
# 在现有 blackboard 基础上，金融场景新增以下 key：

blackboard = {
    # ========== 现有 ==========
    "intent": "...",
    "local_results": [...],
    "web_results": [...],
    "evaluation": {...},
    "refined_query": "...",
    
    # ========== 金融新增 ==========
    "task_plan": {              # Supervisor 拆解的子任务
        "original_query": "...",
        "subtasks": [
            {"id": "task_1", "type": "document_search", ...},
            {"id": "task_2", "type": "financial_market", ...},
            {"id": "task_3", "type": "financial_computation", ...},
        ]
    },
    "task_results": {           # 各子任务执行结果
        "task_1": {...},
        "task_2": {...},
    },
    "market_data": {            # Finance Data Agent 输出
        "300750.SZ": {
            "quote": {"price": 187.5, "change_pct": -2.3, ...},
            "fundamentals": {"pe": 23.4, "pb": 4.2, "roe": 18.2, ...},
        }
    },
    "computed_results": {       # Business Compute Agent 输出
        "metrics": {...},       # FinancialMetrics dict
        "comparison_table": [...],
        "outliers": [...],
    },
    "chart_paths": [            # 图表文件路径
        "data/charts/sess_001/kline_300750.png",
        "data/charts/sess_001/roe_comparison.png",
    ],
    "generated_report": "## 宁德时代 Q3 财报分析\n\n...",  # 报告 Markdown
    
    "aggregated_context": {},   # Supervisor 聚合后的统一上下文
}
```

### 6.2 新增 AgentState 属性

```python
# state.py 中新增 @property

@property
def task_plan(self) -> Dict:
    return self.blackboard.get("task_plan", {})

@property
def market_data(self) -> Dict:
    return self.blackboard.get("market_data", {})

@property
def computed_results(self) -> Dict:
    return self.blackboard.get("computed_results", {})

@property
def chart_paths(self) -> List[str]:
    return self.blackboard.get("chart_paths", [])

@property
def generated_report(self) -> Optional[str]:
    return self.blackboard.get("generated_report")
```

---

## 七、config/settings.yaml 新增配置

```yaml
# =============================================================================
# Finance Configuration (NEW)
# =============================================================================
finance:
  # 行情数据源
  market_data:
    a_share:
      provider: "akshare"
    us_hk:
      provider: "yfinance"
    
  # 业务计算
  computation:
    # 可视化
    visualization:
      output_dir: "data/charts"
      default_format: "png"
      default_dpi: 150
      chart_themes: ["default", "dark", "report"]
    
    # 报告
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
  - 遍历 task_plan.subtasks，按类型并行派发：
    - type=document_search         → rag
    - type=web_search              → web
    - type=financial_market        → finance_data
    - type=financial_computation   → business_compute
  - 全部完成                       → supervisor (聚合)

rag / web:
  - 执行完成                       → read → eval_rag（内部闭环）
  - eval_rag 判定:
    - need_refinement              → refine → 回到 rag
    - quality_ok                   → 写入 blackboard → supervisor
    - need_web                     → web → re-read → eval_rag
  ↑ 原有 Eval/Refine 闭环保留，不动

finance_data / business_compute:
  - 执行完成                       → supervisor (检查是否有剩余子任务)

supervisor (聚合后):
  - 还有未完成子任务               → plan (继续调度)
  - 全部完成                       → global_eval

global_eval (新增):
  - 数据完整 + 计算合理            → generate
  - 数据缺失 + 不可恢复            → generate (标注 fallback)
  - 计算异常 + 可重算              → business_compute (重新计算)
  - 图文不一致                     → generate (标注 attention)

generate:
  - 包含所有 blackboard 内容:
    local_results + web_results + market_data + computed_results
    + chart_paths + key_findings + citations
  - → END
```

---

## 十、依赖清单

```toml
# 金融智能体新增依赖（追加到 pyproject.toml）

[project.optional-dependencies]
finance = [
    "akshare>=1.14.0",          # A 股数据
    "yfinance>=0.2.40",         # 美股/港股数据
    "pandas>=2.0.0",            # 数据处理
    "matplotlib>=3.8.0",        # 基础绑图
    "plotly>=5.18.0",           # 交互式图表
    "mplfinance>=0.12.10b0",    # K线图
    "jinja2>=3.1.0",            # 报告模板渲染
    "numpy>=1.24.0",            # 数值计算
]
```

---

## 十一、实施顺序（Phase 计划）

### Phase 1：基础管线 ✅
```
1. 新增 query_market_data MCP Tool（akshare + yfinance）       ✅
2. 新增 FinancialCalculator（纯 Python 函数，12 公式）          ✅
3. 新增 FinanceDataAgent（封装 MCP Tool）                      ✅
4. 扩展 settings.yaml                                          ✅
```
5. 新增 DataProcessor（Pandas 清洗/对比/排名/异常检测）         ✅
6. 新增 DataVisualizer（Matplotlib K线/柱状/饼图）              ✅

### Phase 2：编排改造 ✅
```
7. 新增 Supervisor Agent（Plan-then-Execute + DAG 拓扑排序）    ✅
8. 改造 multi_agent_system.py 的 LangGraph 图（13 节点）        ✅
9. 扩展 intent 分类（financial_analysis/market/report）        ✅
10. 扩展 AgentState（6 个金融属性）                             ✅
11. 兼容现有 RAG / Web Agent 逻辑                              ✅
```

### Phase 3：业务计算 ✅
```
12. 新增 BusinessComputeAgent                                 ✅
13. 新增 ReportRenderer（Jinja2 模板渲染）                     ✅
14. Agent 间数据流调通                                        ✅
```

### Phase 4：端到端验证 ✅
```
15. E2E 测试 10 场景（无需真实 LLM）                           ✅
16. 异常处理 + NaN/None/空状态 + 错误边界                       ✅
17. MCP 回归测试 6/6 通过                                     ✅
```

### Phase 5：研报生成 🆕
```
18. Generate Node 金融意图路由 → _generate_financial_report   🆕 ✅
19. 全上下文注入（7 路数据源 → LLM analyst prompt）             🆕 ✅
20. 兜底报告策略（LLM 故障时直接输出数据）                       🆕 ✅
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
