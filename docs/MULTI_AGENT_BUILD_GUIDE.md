# 从零搭建 LangGraph 多智能体 RAG 系统 —— 逐步实战指南

> 本文档以 Modular RAG MCP Server 项目为例，按**搭建顺序**逐步讲解多 Agent 系统的设计思路、代码实现和踩坑经验。适合面试准备和系统复盘。

---

## 目录

- [全局架构总览](#全局架构总览)
- [Step 1: 设计共享状态 — AgentState（Blackboard Pattern）](#step-1-设计共享状态--agentstateblackboard-pattern)
- [Step 2: 构建 Router Agent — 意图识别与路由](#step-2-构建-router-agent--意图识别与路由)
- [Step 3: 构建 Search Agent — 本地知识库检索](#step-3-构建-search-agent--本地知识库检索)
- [Step 4: 构建 Web Agent — 联网搜索](#step-4-构建-web-agent--联网搜索)
- [Step 5: 构建 Eval Agent — 检索质量评估](#step-5-构建-eval-agent--检索质量评估)
- [Step 6: 构建 Refine Agent — 查询优化与重试](#step-6-构建-refine-agent--查询优化与重试)
- [Step 7: 构建 Citation 溯源模块](#step-7-构建-citation-溯源模块)
- [Step 8: 构建 Parallel Controller — 并行融合控制器](#step-8-构建-parallel-controller--并行融合控制器)
- [Step 9: 用 LangGraph 编排所有 Agent](#step-9-用-langgraph-编排所有-agent)
- [Step 10: 上下文控制 — 对话记忆与指代消解](#step-10-上下文控制--对话记忆与指代消解)
- [踩坑记录与调试经验](#踩坑记录与调试经验)
- [面试高频问题](#面试高频问题)

---

## 全局架构总览

```
用户输入
  │
  ▼
┌─────────────┐
│ Router Agent │ ← 三维分类（intent + needs_local + needs_web + complexity）
└─────┬───────┘
      │ 条件路由
      ├──────────────┬────────────────┐
      ▼              ▼                ▼
┌──────────┐   ┌──────────┐    ┌──────────┐
│   Plan   │   │ Retrieve │    │ Generate │ ← chat 意图直接生成
│  (复杂)  │   │ (统一节点)│    │  (LLM)   │
└────┬─────┘   └────┬─────┘    └──────────┘
     │              │
     └──────┬───────┘
            ▼
     ┌─────────────┐
     │ Read Node   │ ← 阅读检索结果，形成阶段性观察
     └──────┬──────┘
            ▼
     ┌────────────┐
     │ Eval Agent │ ← 评估检索质量（相关性/多样性/覆盖度）
     └──────┬─────┘
            │
     ┌──────┴──────┐
     │  需要优化？  │
     ├─── Refine ──→ 回到 Retrieve（重试循环）
     ├─── Web ────→ 联网升级 → 回到 Read
     ├─── Plan ───→ 推进下一步子问题
     └─── Generate → 最终回答（带引用 + 忠实度检查）
```

**核心设计模式：**

| 模式 | 说明 |
|------|------|
| **Blackboard Pattern** | 所有 Agent 共享一个 `AgentState`，通过 `blackboard` 字典读写数据 |
| **LangGraph StateGraph** | 用有向图定义 Agent 间的流转关系，支持条件分支和循环 |
| **渐进式升级** | Eval → Refine → Retrieve 闭环重试，耗尽后自动升级联网（CRAG 思想） |
| **Plan-Read-Eval 循环** | 复杂查询先拆子问题（Plan），逐步检索（Retrieve→Read），反思推进（Eval） |
| **溯源 + 忠实度** | Citation 模块追踪每条回答的来源，N-gram 检查是否存在幻觉 |

**文件结构：**

```
src/agent/multi_agent/
├── __init__.py              # 统一导出
├── state.py                 # AgentState（共享状态）
├── router_agent.py          # 意图识别 + 路由（三维分类）
├── search_agent.py          # 本地向量库检索
├── web_agent.py             # 联网搜索
├── eval_agent.py            # 检索质量评估
├── refine_agent.py          # 查询优化
├── citation.py              # 溯源 + 忠实度检查
├── parallel_controller.py   # 并行融合控制器
└── multi_agent_system.py    # LangGraph 主编排器（含 Plan/Read 节点）
```

---

## Step 1: 设计共享状态 — AgentState（Blackboard Pattern）

**文件：** `src/agent/multi_agent/state.py`

### 为什么第一步是设计状态？

多 Agent 系统的核心难题是 **Agent 间如何通信**。有两种经典方案：

| 方案 | 思路 | 缺点 |
|------|------|------|
| 消息传递 | Agent A 发消息给 Agent B | 耦合度高，A 要知道 B 的存在 |
| **黑板模式** | 所有 Agent 读写同一个共享空间 | Agent 解耦，只需要知道 key 名 |

我们选择黑板模式。所有 Agent 共享同一个 `AgentState` 实例。

### 核心数据结构

```python
@dataclass
class AgentState:
    # ===== 输入 =====
    user_input: str = ""
    conversation_history: List[Dict[str, str]] = field(default_factory=list)

    # ===== 黑板（核心！）=====
    blackboard: Dict[str, Any] = field(default_factory=dict)
    # blackboard 的 key 约定：
    #   "intent"              → Router 写入的意图
    #   "routing_decision"    → Router 写入的完整路由决策
    #   "needs_local"         → Router 写入的是否需要本地检索
    #   "needs_web"           → Router 写入的是否需要联网搜索
    #   "query_complexity"    → Router 写入的查询复杂度
    #   "router_confidence"   → Router 写入的置信度
    #   "retrieve_plan"       → Router 写入的检索计划 (none/local/web/both)
    #   "local_results"       → Search Agent 写入的本地检索结果
    #   "web_results"         → Web Agent 写入的联网搜索结果
    #   "evaluation"          → Eval Agent 写入的评估结果
    #   "refined_query"       → Refine Agent 写入的优化查询
    #   "citations"           → Generate Agent 写入的引用列表
    #   "faithfulness_check"  → Generate Agent 写入的忠实度检查
    #   "task_plan"           → Plan Node 写入的子问题计划
    #   "plan_current_step"   → Plan Node 写入的当前步骤索引
    #   "plan_observations"   → Read Node 写入的阶段性观察列表
    #   "reading_assessment"  → Read Node 写入的当前阅读评估
    #   "active_sub_question" → Plan Node 写入的当前子问题
    #   "active_retrieve_plan"→ Plan Node 写入的当前步骤检索计划
    #   "web_search_attempted"→ 标记是否已执行过联网搜索

    # ===== 重试控制 =====
    retry_count: int = 0
    max_retries: int = 2
    fallback_triggered: bool = False
    fallback_reason: Optional[FallbackReason] = None

    # ===== 执行追踪 =====
    execution_log: List[str] = field(default_factory=list)
    execution_trace: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    # ===== 输出 =====
    final_answer: str = ""
```

### 关键设计决策

**1. 用 `@property` 封装 blackboard 读取：**

```python
@property
def intent(self) -> Optional[str]:
    return self.blackboard.get("intent")

@property
def local_results(self) -> List:
    return self.blackboard.get("local_results", [])

@property
def web_results(self) -> List:
    return self.blackboard.get("web_results", [])

@property
def evaluation(self) -> Dict:
    return self.blackboard.get("evaluation", {})

@property
def refined_query(self) -> str:
    return self.blackboard.get("refined_query", self.user_input)

@property
def should_fallback(self) -> bool:
    return self.retry_count >= self.max_retries or self.fallback_triggered
```

好处：外部代码写 `state.local_results` 而不是 `state.blackboard.get("local_results", [])`，更简洁，也方便将来重构存储方式。

**2. 写入方法附带 agent 签名（可追溯）：**

```python
def add_to_blackboard(self, key: str, value: Any, agent: str):
    self.blackboard[key] = value
    self.execution_log.append(f"{agent}: wrote {key}")
```

每次写入都记录是哪个 Agent 写的，方便调试。

**3. 读取方法：**

```python
def read_from_blackboard(self, key: str) -> Any:
    return self.blackboard.get(key)
```

**4. 指标记录方法：**

```python
def add_metric(self, key: str, value: Any):
    self.metrics[key] = value
```

**5. `get_all_context()` 方法用于最终生成：**

```python
def get_all_context(self) -> Dict[str, Any]:
    return {
        "user_input": self.user_input,
        "intent": self.intent,
        "local_results": self.local_results,
        "web_results": self.web_results,
        "evaluation": self.evaluation,
        "refined_query": self.refined_query,
        "conversation_history": self.conversation_history,
        "metrics": self.metrics,
        "retry_count": self.retry_count,
        "fallback_triggered": self.fallback_triggered,
        "fallback_reason": self.fallback_reason.value if self.fallback_reason else None,
    }
```

**6. `reset()` 方法用于多轮对话：**

每轮对话开始时清空 blackboard 和 trace，但保留 `user_input` 和 `conversation_history`。这样 Agent 看到的是干净的新一轮状态，但仍然有对话记忆。

**7. 序列化支持：**

```python
def to_dict(self) -> Dict[str, Any]: ...
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "AgentState": ...
```

支持将状态序列化为字典和从字典恢复，方便持久化和跨进程传递。

---

## Step 2: 构建 Router Agent — 意图识别与路由

**文件：** `src/agent/multi_agent/router_agent.py`

### 职责

拿到用户输入，判断应该调用哪些 Agent。

### 三维分类体系

Router Agent 不再使用简单的五种意图分类，而是采用**三维分类**：

| 维度 | 含义 | 取值 |
|------|------|------|
| **intent** | 查询意图类型 | `chat` / `fact_query` / `document_qa` / `summarization` / `comparison` / `analysis` / `unknown` |
| **needs_local** | 是否需要本地知识库 | `true` / `false` |
| **needs_web** | 是否需要联网搜索 | `true` / `false` |
| **complexity** | 查询复杂度 | `simple` / `medium` / `complex` |

**七种意图详解：**

| 意图 | 路由目标 | 示例 |
|------|----------|------|
| `chat` | → Generate | "你好"、"谢谢" |
| `fact_query` | → Retrieve | "RAG 是什么" |
| `document_qa` | → Search | "这篇论文的结论是什么" |
| `summarization` | → Search | "总结一下这篇文档" |
| `comparison` | → Search + Web（可能并行） | "比较两种方法的优劣" |
| `analysis` | → Search + Web（可能并行） | "分析一下行业趋势" |
| `unknown` | → Search（兜底尝试） | 无法分类的查询 |

### RoutingDecision 数据类

```python
@dataclass
class RoutingDecision:
    intent: str
    agents_to_invoke: List[AgentType]
    needs_local: bool = False
    needs_web: bool = False
    complexity: str = "simple"
    parallel: bool = False
    confidence: float = 0.0
    reasoning: str = ""
    parameters: Dict[str, Any] = None
```

关键设计：`agents_to_invoke` 由 `needs_local` 和 `needs_web` 推导得出，而非 LLM 直接输出。这避免了 LLM 输出不一致的问题。

### 实现方式：LLM 分类 + 规则推导

```python
class RouterAgent:
    SYSTEM_PROMPT = """You are an intent classifier and router for a multi-agent RAG system.

Classify the user query into FOUR dimensions:

1. intent
   - chat: simple chitchat / general conversation
   - fact_query: direct factual question
   - document_qa: asks about uploaded/local documents or internal knowledge
   - summarization: asks to summarize a paper, file, report, or document
   - comparison: compare two methods / products / documents / viewpoints
   - analysis: asks for deeper judgment, evaluation, recommendation, tradeoff analysis
   - unknown: impossible or unclear request

2. needs_local
   - true if the answer should use local knowledge base / uploaded files / internal docs
   - signals: "根据文档", "上传的 PDF", "内部资料", "本地知识库", paper/file names

3. needs_web
   - true if the answer needs public/latest/external information from the internet
   - signals: "最新", "今天", "本周", "实时", "新闻", "2026", "官网", "公开资料", "竞品"

4. complexity
   - simple: single-hop, direct answer likely enough
   - medium: may need synthesis or comparison, but usually one retrieval round is enough
   - complex: multi-step reasoning, comparison + judgment, planning, or local + web synthesis

Respond ONLY in JSON:
{
    "intent": "analysis",
    "needs_local": true,
    "needs_web": true,
    "complexity": "complex",
    "agents_to_invoke": ["SearchAgent", "WebAgent"],
    "parallel": true,
    "confidence": 0.92,
    "reasoning": "The question requires both uploaded documents and latest public information.",
    "parameters": {}
}"""

    def classify(self, query, context=None) -> RoutingDecision:
        from datetime import datetime
        now = datetime.now()
        current_dt = f"{now.strftime('%Y-%m-%d')} {weekdays[now.weekday()]} {now.strftime('%H:%M')}"

        response = self.chain.invoke({
            "query": query,
            "context": context_str,
            "current_datetime": current_dt,
        })
        result = self._parse_response(response.content)
        return RoutingDecision(
            intent=result.get("intent", "unknown"),
            agents_to_invoke=self._build_agents_to_invoke(result),
            needs_local=bool(result.get("needs_local", False)),
            needs_web=bool(result.get("needs_web", False)),
            complexity=str(result.get("complexity", "simple")).lower(),
            parallel=result.get("parallel", False),
            confidence=float(result.get("confidence", 0.0)),
            reasoning=result.get("reasoning", ""),
        )
```

### 关键设计：`_build_agents_to_invoke` 规则推导

```python
def _build_agents_to_invoke(self, result: Dict[str, Any]) -> List[AgentType]:
    needs_local = bool(result.get("needs_local", False))
    needs_web = bool(result.get("needs_web", False))
    intent = str(result.get("intent", "unknown")).lower()

    agents: List[AgentType] = []
    if needs_local:
        agents.append(AgentType.SEARCH)
    if needs_web:
        agents.append(AgentType.WEB)

    if agents:
        return agents

    # Backward compatibility with older prompt outputs / fallback parsing
    for agent in result.get("agents_to_invoke", []):
        try:
            parsed = AgentType(agent.lower().replace("agent", ""))
            if parsed not in agents:
                agents.append(parsed)
        except ValueError:
            continue

    if agents:
        return agents

    # Final fallback based on intent
    if intent in ("document_qa", "summarization", "local_search"):
        return [AgentType.SEARCH]
    if intent in ("web_search",):
        return [AgentType.WEB]
    if intent in ("comparison", "analysis", "hybrid_search"):
        return [AgentType.SEARCH, AgentType.WEB]
    return []
```

**设计原则：** 优先使用 `needs_local`/`needs_web` 标志推导，而非直接依赖 LLM 输出的 `agents_to_invoke`。因为布尔标志比列表更稳定、更不容易出错。

### 简单分类（快速判断）

```python
def classify_simple(self, query: str) -> str:
    query_lower = query.lower()
    if any(kw in query_lower for kw in ["你好", "谢谢", "再见", "你是谁", "聊聊天"]):
        return "chat"
    if any(kw in query_lower for kw in ["最新", "今天", "本周", "实时", "新闻", "天气", "股票"]):
        return "web_search"
    if any(kw in query_lower for kw in ["结合", "和网上", "+ 网上", "内部文档和网上"]):
        return "hybrid_search"
    if any(kw in query_lower for kw in ["公司文档", "本地", "我们的", "内部"]):
        return "local_search"
    return "unknown"
```

用于不需要 LLM 调用的快速规则匹配场景。

### 重要：为什么 `unknown` 也路由到 Search？

早期版本中 `unknown` 直接走 fallback，导致很多正常查询（比如"论文结论是什么"）因为 LLM 分类不稳定而直接返回"未找到信息"。改成默认搜索后，即使分类错误也不会丢失结果。

**设计原则：宁可多搜一次，不可漏掉答案。**

---

## Step 3: 构建 Search Agent — 本地知识库检索

**文件：** `src/agent/multi_agent/search_agent.py`

### 职责

封装底层的 hybrid search（Dense + Sparse + RRF 融合），对外暴露简洁的 `search()` 接口。

### 核心流程

```
用户 query
  ↓
指代消解（如果有对话历史）
  ↓
QueryKnowledgeHubTool.execute()
  → Dense Search (embedding 相似度)
  → Sparse Search (BM25 关键词)
  → RRF 融合排序
  ↓
格式化结果 → 返回 List[Dict]
```

### 关键代码

```python
class SearchAgent:
    def __init__(self, settings):
        self.tool = QueryKnowledgeHubTool(settings)

    def search(self, query, top_k=5, collection=None, context=None) -> List[Dict]:
        if context:
            query = self._resolve_pronouns(query, context)
        result = self.tool.execute(query=query, top_k=top_k, collection=collection)
        if result.success:
            return self._format_results(result.data)
        else:
            raise Exception(f"Search failed: {result.error}")
```

### 结果格式

```python
{
    "content": "chunk 文本内容...",
    "score": 0.85,
    "source": "1-s2.0-S0925400503004453-main.pdf",
    "chunk_index": 3,
    "metadata": {
        "type": "local",
        "dense_sparse_fusion": True,
        "fusion_method": "RRF",
    }
}
```

统一的结果格式很重要——后续的 Eval Agent、Citation 模块都依赖这个结构。

### 扩展接口

**`search_with_metadata()`** — 返回包含详细元数据的检索结果：

```python
def search_with_metadata(self, query, top_k=5, collection=None, context=None) -> Dict:
    results = self.search(query, top_k, collection, context)
    return {
        "results": results,
        "total_results": len(results),
        "query": query,
        "top_k": top_k,
        "collection": collection or "default",
        "metadata": {
            "search_type": "hybrid",
            "dense_sparse_fusion": True,
            "fusion_method": "RRF",
            "parallel_execution": True,
        }
    }
```

**`batch_search()`** — 批量检索，单个查询失败不影响其他：

```python
def batch_search(self, queries, top_k=5, collection=None) -> List[List[Dict]]:
    all_results = []
    for query in queries:
        try:
            results = self.search(query, top_k, collection)
            all_results.append(results)
        except Exception:
            all_results.append([])
    return all_results
```

---

## Step 4: 构建 Web Agent — 联网搜索

**文件：** `src/agent/multi_agent/web_agent.py`

### 职责

搜索互联网获取实时信息，支持 Tavily（默认）/ Google / Bing。

### 核心逻辑

```python
class WebSearchAgent:
    def __init__(self, settings=None, search_engine="tavily"):
        self.search_tool = WebSearchTool(
            search_engine=engine,
            tavily_api_key=tavily_api_key,
            google_api_key=google_api_key,
            google_search_engine_id=google_search_engine_id,
            bing_api_key=bing_api_key,
        )

    def search(self, query, num_results=5, time_range="y", local_results=None) -> List[Dict]:
        if local_results:
            refined_query = f"{query} 2025 2026 最新进展"
        else:
            refined_query = query
        results = self.search_tool.search(query=refined_query, num_results=num_results, time_range=time_range)
        return self._format_results(results)
```

### 结果格式

```python
{
    "title": "文章标题",
    "url": "https://example.com/article",
    "snippet": "文章摘要片段...",
    "source": "example.com",
    "published_date": "2025-03-15",
    "metadata": {
        "type": "web",
        "search_engine": "tavily",
    }
}
```

### 设计亮点：条件查询优化

Web Agent 可以**读取 Search Agent 已经找到的结果**（通过 Blackboard），据此调整搜索策略。比如：
- 本地已有论文原文 → 联网搜索"最新进展、引用情况"
- 本地没有结果 → 直接搜原始 query

这就是 Blackboard 模式的优势：Agent 之间无需直接调用，通过共享空间协作。

### 便捷方法

```python
def search_news(self, query, num_results=5) -> List[Dict]:
    return self.search(query, num_results, time_range="d")

def search_recent(self, query, num_results=5) -> List[Dict]:
    return self.search(query, num_results, time_range="w")
```

---

## Step 5: 构建 Eval Agent — 检索质量评估

**文件：** `src/agent/multi_agent/eval_agent.py`

### 职责

评估 Search / Web Agent 的检索结果是否能回答用户的问题。

### 评估四维度

| 维度 | 含义 | 范围 |
|------|------|------|
| **Relevance** | 结果与查询的相关性 | 0.0 ~ 1.0 |
| **Diversity** | 结果是否覆盖不同角度 | 0.0 ~ 1.0 |
| **Coverage** | 查询的所有方面是否都被覆盖 | 0.0 ~ 1.0 |
| **Confidence** | 综合置信度 | 0.0 ~ 1.0 |

### 决策逻辑

```
LLM 评估检索结果 → 得到四个分数
            │
            ▼
    强制规则检查（_apply_rules）
    ├── relevance < 0.2  → fallback（结果完全不相关）
    ├── retry_count >= 2 → fallback（已达重试上限）
    ├── 不可能的查询     → fallback（"我昨天吃了什么"）
    ├── confidence < 0.7 → need_refinement = True
    └── 否则             → 正常生成
```

### 关键设计：规则覆盖 LLM

LLM 的评估不一定稳定，所以用 `_apply_rules()` 做硬性兜底：

```python
def _apply_rules(self, result, query, retry_count, max_retries):
    if relevance < 0.2:
        fallback_suggested = True  # 不管 LLM 怎么说
    if retry_count >= max_retries:
        fallback_suggested = True  # 强制停止重试
    if self._is_impossible_query(query):
        fallback_suggested = True  # 系统无法回答的问题
```

**面试话术：** "Eval Agent 采用 LLM 评估 + 规则兜底的双层机制。LLM 做软评估提供分数，规则做硬约束防止极端情况。"

### `_is_impossible_query` 精准匹配

```python
def _is_impossible_query(self, query: str) -> bool:
    q = query.strip()
    starts_with_patterns = [
        "我昨天", "我前天", "我上周", "我上个月",
        "我的隐私", "我的秘密", "我心里",
    ]
    contains_patterns = [
        "我家地址", "我的密码",
    ]
    return (
        any(q.startswith(p) for p in starts_with_patterns)
        or any(p in q for p in contains_patterns)
    )
```

注意使用 `startswith` 而非简单的 `in` 匹配，避免误判"我想了解..."、"我感觉..."等正常查询。

---

## Step 6: 构建 Refine Agent — 查询优化与重试

**文件：** `src/agent/multi_agent/refine_agent.py`

### 职责

当 Eval Agent 判定检索质量不够时，改写 query 让下一次检索更精准。

### 优化策略

```
原始 query: "RAG 技术"
Eval 反馈: "结果太泛泛"
     │
     ▼
Refine Agent 改写:
  → "RAG 检索增强生成 技术原理 2025 年最新进展 详细说明"
  → changes_made: ["添加完整术语", "添加时间限定", "添加详细程度要求"]
```

### 关键代码

```python
class RefineAgent:
    def refine(self, original_query, evaluation, retry_count) -> RefinementResult:
        context = self._build_context(original_query, evaluation, retry_count)
        response = self.chain.invoke({"context": context})
        result = self._parse_json_response(response.content)
        return RefinementResult(
            refined_query=result.get('refined_query'),
            changes_made=result.get('changes_made'),
            reasoning=result.get('reasoning'),
        )
```

### 重试循环

```
Retrieve → Read → Eval → [quality low] → Refine → Retrieve → Read → Eval → ... (max 2 rounds)
```

Refine Agent 每次执行后会 `state.increment_retry("refine")`，并把 `refined_query` 写入 Blackboard。下一轮 Retrieve 节点检测到 `retry_count > 0` 就会使用优化后的 query。

---

## Step 7: 构建 Citation 溯源模块

**文件：** `src/agent/multi_agent/citation.py`

### 四个核心组件

#### 1. Citation 数据类

```python
@dataclass
class Citation:
    type: CitationType     # LOCAL / WEB
    source: str            # 文档名或网站名
    content: str           # 引用的具体内容
    confidence: float      # 置信度
    relevance: float       # 相关性
    url: Optional[str]     # 仅 web 类型
    page: Optional[str]    # 页码或段落号
    metadata: Dict[str, Any]

    def format_citation(self) -> str:
        if self.type == CitationType.LOCAL:
            if self.page:
                return f"[Local: {self.source}，p.{self.page}]"
            return f"[Local: {self.source}]"
        else:
            domain = urlparse(self.url).netloc.replace('www.', '')
            return f"[Web: {domain}]"
```

#### 2. CitationManager（引用管理器）

负责从检索结果创建 Citation 对象，按排名自动赋予递减的置信度：

```python
@staticmethod
def create_citations_from_results(local_results, web_results, top_k=5):
    for i, result in enumerate(local_results[:top_k]):
        confidence = max(0.5, 1.0 - (i * 0.1))  # 第1名1.0，第2名0.9，...
        relevance = max(0.5, 1.0 - (i * 0.1))
        citation = create_citation_from_local_result(result, confidence, relevance)
    # ... 同理处理 web_results
```

#### 3. FaithfulnessCheck（忠实度检查结果）

```python
@dataclass
class FaithfulnessCheck:
    is_faithful: bool = True
    hallucination_detected: bool = False
    unsupported_claims: List[str] = field(default_factory=list)
    confidence: float = 1.0
    suggestions: List[str] = field(default_factory=list)
```

#### 4. CitationManager.check_faithfulness（忠实度检查逻辑）

生成回答后，检查回答是否忠实于检索结果（而非 LLM 自己编造）：

```
检查策略（无需额外 LLM 调用，纯规则）：

1. 引用标记检查：回答中是否有 [1]、[Local: ...]、[Web: ...] 等标记？
   → marker_score = 1.0（有标记）或 0.3（无标记）

2. 内容重叠检查：回答的 4-gram 与每条 citation 内容的 4-gram 重叠度
   → 对每条 citation 计算重叠比例，overlap > 0.15 视为"已使用"

3. 引用覆盖率：多少条 citation 的内容在回答中有体现？
   → coverage = used_citations / total_citations

综合得分 = 0.2 × marker_score + 0.5 × min(avg_overlap × 3, 1.0) + 0.3 × coverage
```

#### 5. format_answer_with_citations（格式化工具函数）

```python
def format_answer_with_citations(answer, citations, include_reference_list=True) -> str:
    if "[Local:" in answer or "[Web:" in answer:
        # 回答中已有引用标记，只追加引用列表
        ...
    else:
        # 简单地在末尾添加引用列表
        ...
```

---

## Step 8: 构建 Parallel Controller — 并行融合控制器

**文件：** `src/agent/multi_agent/parallel_controller.py`

### 为什么需要并行？

当用户说"结合内部文档和网上资料分析一下"时，需要**同时**调用 Search Agent 和 Web Agent，然后融合结果。串行等待会浪费时间。

### 实现方式：ThreadPoolExecutor

```python
class ParallelFusionController:
    def __init__(self, search_func, web_func):
        self.search_func = search_func
        self.web_func = web_func

    def execute_parallel_search(self, state, agents_to_invoke):
        with ThreadPoolExecutor(max_workers=5) as executor:
            for agent_type in agents_to_invoke:
                if agent_type == AgentType.SEARCH:
                    future = executor.submit(self._execute_search, state.user_input, state.conversation_history)
                    futures[future] = "search"
                elif agent_type == AgentType.WEB:
                    local_results = state.read_from_blackboard("local_results")
                    future = executor.submit(self._execute_web, state.user_input, local_results)
                    futures[future] = "web"

            for future in as_completed(futures):
                result = future.result()
                # 结果写入 Blackboard
                state.add_to_blackboard("local_results", result, "parallel_controller")
```

### 容错设计

单个 Agent 失败不会影响另一个：

```python
try:
    result = future.result()
    results[agent_name] = {"success": True, "data": result}
except Exception as e:
    results[agent_name] = {"success": False, "error": str(e)}
    # 继续处理其他 Agent 的结果
```

### 串行备用方案

```python
def execute_sequential(self, state, agents_to_invoke):
    for agent_type in agents_to_invoke:
        if agent_type == AgentType.SEARCH:
            results = self._execute_search(state.user_input, state.conversation_history)
            state.add_to_blackboard("local_results", results, "sequential_controller")
        elif agent_type == AgentType.WEB:
            local_results = state.read_from_blackboard("local_results")
            results = self._execute_web(state.user_input, local_results)
            state.add_to_blackboard("web_results", results, "sequential_controller")
```

---

## Step 9: 用 LangGraph 编排所有 Agent

**文件：** `src/agent/multi_agent/multi_agent_system.py`

这是整个系统的**大脑**——把前面所有零件组装成一个有向图。

### 9.1 初始化所有 Agent

```python
class MultiAgentRAG:
    def __init__(self, llm, settings=None, enable_logging=True, store=None):
        self.llm = llm
        self.settings = settings or {}
        self.store = store  # LangGraph BaseStore（长期记忆）

        self.router_agent = RouterAgent(self.llm)
        self.search_agent = SearchAgent(self.settings)
        self.web_agent = WebSearchAgent(self.settings)
        self.eval_agent = EvalAgent(self.llm)
        self.refine_agent = RefineAgent(self.llm)

        self.workflow = self._build_graph()
```

注意 `store` 参数：LangGraph 的 `BaseStore` 实例，用于长期记忆。通过 `store=` 传入 `workflow.compile()`，图内节点可通过 `get_store()` 访问。

### 9.2 构建 LangGraph 状态图

当前实现包含 **8 个节点**：`router` → `plan` → `retrieve` → `read` → `eval` → `refine`/`web`/`generate`。

**统一 `retrieve` 节点** 写入黑板，**hybrid / 低 Router 置信度** 在 `retrieve` 内用 `ThreadPoolExecutor` **真并行**跑本地检索与联网（再进 `read` → `eval` 汇合）。这样避免 LangGraph `Send` 并行分支对 `dataclass` 状态合并的限制。

```python
def _build_graph(self) -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("router", self._router_node)
    workflow.add_node("plan", self._plan_node)         # 复杂查询显式规划
    workflow.add_node("retrieve", self._retrieve_node)  # local / web / 并行 both
    workflow.add_node("web", self._web_node)            # Eval 升级联网专用
    workflow.add_node("read", self._read_node)          # 阅读检索结果
    workflow.add_node("eval", self._eval_node)
    workflow.add_node("refine", self._refine_node)
    workflow.add_node("generate", self._generate_node)

    workflow.set_entry_point("router")

    # Router → 闲聊直出 Generate；复杂问题先走 Plan；其余进入 Retrieve
    workflow.add_conditional_edges("router", self._route_after_router, {
        "generate": "generate",
        "plan": "plan",
        "retrieve": "retrieve",
    })

    # Plan → 步骤完成则生成，否则进入 Retrieve
    workflow.add_conditional_edges("plan", self._route_after_plan, {
        "generate": "generate",
        "retrieve": "retrieve",
    })

    # Retrieve → Read → Eval
    workflow.add_edge("retrieve", "read")
    workflow.add_edge("web", "read")
    workflow.add_edge("read", "eval")

    # Eval → Generate / Plan / Refine / Web
    workflow.add_conditional_edges("eval", self._eval_next_step, {
        "generate": "generate",
        "plan": "plan",
        "refine": "refine",
        "web": "web",
    })

    # Refine → 按原 retrieve_plan 重新检索
    workflow.add_edge("refine", "retrieve")

    # Generate → END
    workflow.add_edge("generate", END)

    compile_kwargs = {}
    if self.store is not None:
        compile_kwargs["store"] = self.store
    return workflow.compile(**compile_kwargs)
```

**黑板字段 `retrieve_plan`**：`none`（闲聊）| `local` | `web` | `both`。Router 在 **非 chat 且置信度 < 0.7** 时强制 `both`，与 `HYBRID_SEARCH` / `parallel: true` 一致。

### 9.3 Plan 节点 — 复杂查询显式规划

Plan 节点是复杂查询的**拆解器**，将一个大问题拆成多个子问题，逐步执行：

```python
def _plan_node(self, state: AgentState) -> AgentState:
    plan_data = state.blackboard.get("task_plan")
    replan_requested = bool(state.blackboard.get("replan_requested", False))

    if not plan_data or replan_requested:
        # 首次进入或需要重新规划：用 LLM 创建显式计划
        routing = state.blackboard.get("routing_decision", {})
        plan_data = self._create_explicit_plan(
            query=state.user_input,
            conversation_history=state.conversation_history,
            routing=routing,
        )
        state.add_to_blackboard("task_plan", plan_data, "plan")
        state.add_to_blackboard("plan_current_step", 0, "plan")
    elif state.blackboard.get("advance_plan_step", False):
        # 推进到下一步
        current_step = int(state.blackboard.get("plan_current_step", 0))
        state.add_to_blackboard("plan_current_step", current_step + 1, "plan")
        state.blackboard["advance_plan_step"] = False

    # 设置当前子问题的检索计划
    step = plan_steps[current_step]
    state.add_to_blackboard("active_sub_question", step.get("sub_question"), "plan")
    state.add_to_blackboard("active_retrieve_plan", step.get("preferred_source"), "plan")
    state.add_to_blackboard("active_plan_goal", step.get("goal"), "plan")
```

**`_create_explicit_plan` 用 LLM 拆解子问题：**

```python
def _create_explicit_plan(self, query, conversation_history, routing) -> Dict:
    prompt = """你是一个多步检索规划器。请根据用户问题生成一个显式计划。
    要求：
    1. 将复杂问题拆成 1-3 个子问题。
    2. 对每个子问题指定 preferred_source，只能是 local / web / both。
    3. 每步都给一个简短 goal。
    4. 只返回 JSON。"""
    response = self.llm.invoke([HumanMessage(content=prompt)])
    return json.loads(response.content)
```

### 9.4 Read 节点 — 阅读与反思

Read 节点对检索结果进行归纳，形成阶段性观察，再由 Eval 节点据此决定下一步：

```python
def _read_node(self, state: AgentState) -> AgentState:
    sub_question = state.blackboard.get("active_sub_question", state.user_input)
    goal = state.blackboard.get("active_plan_goal", "")
    reading = self._read_retrieved_evidence(
        query=sub_question,
        goal=goal,
        local_results=state.local_results,
        web_results=state.web_results,
    )

    state.add_to_blackboard("reading_assessment", reading, "read")

    # 追加到观察列表
    observations = state.blackboard.get("plan_observations", [])
    observations.append({
        "step_index": state.blackboard.get("plan_current_step", 0),
        "sub_question": sub_question,
        "summary": reading.get("summary", ""),
        "enough_to_answer_sub_question": reading.get("enough_to_answer_sub_question", False),
        "suggested_next_action": reading.get("suggested_next_action", ""),
        "missing_information": reading.get("missing_information", ""),
    })
    state.blackboard["plan_observations"] = observations
```

**`_read_retrieved_evidence` 用 LLM 做阅读反思：**

```python
def _read_retrieved_evidence(self, query, goal, local_results, web_results) -> Dict:
    prompt = """你是一个阅读与反思节点。请阅读检索结果，判断当前子问题是否已经有足够证据回答。
    只返回 JSON：
    {
      "summary": "对当前证据的简短总结",
      "enough_to_answer_sub_question": true,
      "suggested_next_action": "continue|search_web|refine|generate",
      "missing_information": "还缺什么信息"
    }"""
    response = self.llm.invoke([HumanMessage(content=prompt)])
    return json.loads(response.content)
```

### 9.5 Retrieve 节点 — 统一检索

Retrieve 节点按黑板上的 `retrieve_plan`（或 `active_retrieve_plan`）执行本地 / 联网 / 真并行：

```python
def _retrieve_node(self, state: AgentState) -> AgentState:
    plan = state.blackboard.get("active_retrieve_plan", state.blackboard.get("retrieve_plan", "local"))
    base_query = state.blackboard.get("active_sub_question", state.user_input)
    query = state.refined_query if state.retry_count > 0 else base_query

    if plan == "local":
        results = self.search_agent.search(query=query, top_k=10, context=context)
        state.add_to_blackboard("local_results", results, "retrieve")
        state.add_to_blackboard("web_results", [], "retrieve")
    elif plan == "web":
        web_results = self.web_agent.search(query=query, num_results=5, time_range="y", local_results=None)
        state.add_to_blackboard("local_results", [], "retrieve")
        state.add_to_blackboard("web_results", web_results, "retrieve")
    else:
        # both — 真并行
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_local = ex.submit(_run_local)
            fut_web = ex.submit(_run_web_no_local_ctx)
            local_results = fut_local.result()
            web_results = fut_web.result()
        state.add_to_blackboard("local_results", local_results, "retrieve")
        state.add_to_blackboard("web_results", web_results, "retrieve")
```

### 9.6 生成节点 — LLM 总结

Generate 节点不是简单拼接检索结果，而是根据不同模式生成回答：

```python
def _generate_node(self, state: AgentState) -> AgentState:
    # 判断生成模式
    if is_chat_direct:
        gen_mode = "general_knowledge"      # 闲聊：LLM 通用知识
    elif not has_local and not has_web:
        if web_attempted:
            gen_mode = "general_knowledge"  # 检索+联网都失败：LLM 通用知识
        else:
            gen_mode = "fallback"           # 未尝试联网：兜底回复
    elif local_all_noise and not has_web and web_attempted:
        gen_mode = "general_knowledge"      # 本地全是噪音且联网也失败
    else:
        gen_mode = "normal"                 # 正常：基于检索结果 + 引用 + 忠实度检查
```

**正常模式（带引用 + 忠实度检查）：**

```python
def _generate_normal_response_with_citations(self, context):
    # 1. CRAG 噪音过滤
    local_results = self._filter_local_results(local_results, eval_confidence)

    # 2. 创建 Citation
    citations = CitationManager.create_citations_from_results(local_results, web_results, top_k=5)

    # 3. 构建检索结果块
    for i, result in enumerate(local_results[:5], 1):
        context_parts.append(f"[{i}] (本地知识库: {source})\n{content}")
    for i, result in enumerate(web_results[:3], 1):
        context_parts.append(f"[{offset+i}] (互联网: {title}, {url})\n{snippet}")

    # 4. 注入计划观察（如果有）
    if plan_observations:
        observations_block = "\n\n## 计划观察\n" + ...

    # 5. 构建 LLM messages
    messages = [SystemMessage(content=self._build_rag_system_prompt())]
    for turn in conv_history:
        messages.append(HumanMessage/AIMessage)
    messages.append(HumanMessage(content=f"## 检索结果\n{context}\n## 用户问题\n{query}"))

    # 6. LLM 生成
    response = self.llm.invoke(messages)

    # 7. 附加引用 + 忠实度检查
    answer = format_answer_with_citations(answer, citations)
    faithfulness = citation_manager.check_faithfulness(answer)
    return answer, citations, faithfulness
```

**通用知识降级模式：**

当所有检索（本地 + 联网）均失败时，用 LLM 通用知识直接回答，并在回答开头注明：

```python
def _generate_general_knowledge_answer(self, context) -> str:
    messages = [SystemMessage(content=self._build_general_knowledge_prompt())]
    # ... 注入对话历史和用户问题
    response = self.llm.invoke(messages)
    return response.content
```

System Prompt 要求：在回答开头注明 "以下回答基于AI通用知识，非来自知识库检索。"

### 9.7 关键条件路由函数

```python
def _route_after_router(self, state) -> Literal["generate", "plan", "retrieve"]:
    plan = state.blackboard.get("retrieve_plan", "local")
    complexity = state.blackboard.get("query_complexity", "simple")
    if plan == "none":
        return "generate"     # 闲聊直接生成
    if complexity == "complex":
        return "plan"         # 复杂问题先规划
    return "retrieve"         # 简单问题直接检索

def _route_after_plan(self, state) -> Literal["generate", "retrieve"]:
    if state.blackboard.get("plan_complete", False):
        return "generate"     # 所有步骤完成
    return "retrieve"         # 执行当前步骤

def _eval_next_step(self, state) -> Literal["generate", "plan", "refine", "web"]:
    # 0. 复杂任务：当前子问题已足够回答且仍有后续步骤 → 推进计划
    if enough and has_more_plan_steps:
        state.blackboard["advance_plan_step"] = True
        return "plan"

    # 0.1 阅读阶段建议补充 web → 联网
    if suggested_next_action == "search_web" and not web_attempted:
        return "web"

    # 1. 仍有重试额度且需要改写查询 → Refine
    if need_refinement and state.retry_count < state.max_retries:
        return "refine"

    # 2. 联网效果差 → 回退到本地检索
    if web_attempted and confidence < 0.4 and len(local_results) > 0 and state.retry_count < state.max_retries:
        state.add_to_blackboard("retrieve_plan", "local", "eval")
        return "refine"

    # 3. 已执行过联网 → 不再升级联网
    if web_attempted:
        return "generate"

    # 4. 明确兜底/放弃 → 升级联网
    if state.fallback_triggered or fallback_suggested:
        return "web"

    # 5. 重试耗尽但还没联网 → 升级联网
    if need_refinement and state.retry_count >= state.max_retries:
        return "web"

    # 6. 置信度偏低 → 联网补充
    if confidence < 0.5:
        return "web"

    return "generate"
```

### 9.8 CRAG 噪音过滤 + 信息冲突处理

**优化 1：本地噪音过滤（CRAG 思想）**

当本地 RAG 效果不好触发了联网搜索，说明 `local_results` 里大概率混杂了噪音。与其无脑把前 5 条全塞进 LLM prompt，不如按相关性分数过滤：

```python
_LOCAL_RELEVANCE_THRESHOLD = 0.3

def _filter_local_results(self, local_results, eval_confidence):
    if eval_confidence >= 0.7 or not local_results:
        return local_results  # Eval 评分高，本地结果质量好，保留全部
    # 过滤低分 chunk，只保留 score >= 0.3 的
    filtered = [r for r in local_results if r.get("score", 0) >= self._LOCAL_RELEVANCE_THRESHOLD]
    return filtered if filtered else local_results[:1]  # 至少保留 1 条
```

**优化 2：信息冲突处理原则**

当 local + web 两路结果同时存在时，可能出现矛盾（比如公司报销标准 vs 网上的通用标准）。在 System Prompt 中加入冲突处理规则：

> - 公司政策、内部规范、组织架构 → 以本地知识库为准
> - 客观事实、行业趋势、最新新闻 → 以互联网信息为补充
> - 如果存在矛盾 → 标注信息来源并说明差异

### 9.9 公共 API

```python
def run(self, user_input, conversation_history=None) -> AgentState:
    initial_state = AgentState(
        user_input=user_input,
        conversation_history=conversation_history or []
    )
    final_state = self.workflow.invoke(initial_state)
    return final_state
```

调用方只需要 `agent.run("问题", history)`，所有内部编排自动完成。

---

## Step 10: 上下文控制 — 对话记忆与指代消解

### 10.1 对话历史传递链路

```
Dashboard (Streamlit)
  → 从 st.session_state["chat_history"] 提取最近 N 轮
  → agent.run(query, conversation_history=[...])
    → AgentState.conversation_history
      → Search Agent._resolve_pronouns()     # 检索端：指代消解
      → _generate_normal_response()          # 生成端：LLM 看到历史
```

### 10.2 检索端：指代消解

```python
def _resolve_pronouns(self, query, context):
    pronouns = [
        "它", "这个", "那个", "其", "该",
        "这篇", "那篇", "上面", "刚才",
        "还有呢", "继续", "接着说",
    ]
    needs_context = (
        any(p in query for p in pronouns)
        or len(query.strip()) < 6  # very short follow-up like "结论呢"
    )

    if needs_context:
        # 从历史中找到上一轮的主题
        prev_query = find_last_user_message(context)
        # 提取关键主题（去除疑问词）
        topic = remove_question_words(prev_query)
        # 保持主题简洁
        if len(topic) > 50:
            topic = topic[:50]
        return f"{topic} {query}"
```

**效果：**
- 用户第一轮："这篇文献的结论是啥 xxx.pdf" → 正常搜索
- 用户第二轮："它用了什么方法" → 改写为 "xxx.pdf 它用了什么方法"
- 用户第三轮："还有呢" → 改写为 "xxx.pdf 用了什么方法 还有呢"

### 10.3 生成端：滑动窗口

```python
conv_history = context.get("conversation_history", [])
for turn in conv_history:
    role = turn.get("role", "")
    text = turn.get("content", "")
    if role == "user":
        messages.append(HumanMessage(content=text))
    elif role == "assistant":
        messages.append(AIMessage(content=text))
```

### 10.4 Token 预算分配

```
LLM context window（假设 8K tokens）
├── System Prompt:       ~200 tokens（固定）
├── 对话历史（6 条）:     ~600 tokens（变量，窗口控制）
├── 检索结果（5+3 条）:   ~3000 tokens（变量，top_k 控制）
├── 计划观察（如有）:     ~500 tokens（变量，多步推理场景）
├── 用户问题:             ~50 tokens（变量）
└── 留给生成的空间:       ~3650 tokens
```

**核心原则：对话历史占得越少，留给检索结果的空间越大。而检索结果才是 RAG 回答质量的决定性因素。**

---

## 踩坑记录与调试经验

### 坑 1: LangGraph 的 StateGraph 不能有重复边

```python
# 错误：同时定义了固定边和条件边
workflow.add_edge("search", "eval")  # 固定边
workflow.add_conditional_edges("search", ...)  # 条件边 → 冲突！
```

**症状：** 图编译不报错，但运行时行为不可预测。
**解决：** 一个节点的出边只能用一种方式定义（固定边 **或** 条件边）。

### 坑 2: Router 分类不稳定导致"未找到信息"

**问题：** LLM 有时返回 `"LOCAL_SEARCH"`（大写），有时返回 `"local_search"`（小写），或者返回了无法匹配的意图。
**解决：**
1. 对 LLM 返回的 intent 做 `.lower()` 规范化
2. `unknown` 意图默认路由到 Search 而不是 Fallback
3. 用 `needs_local`/`needs_web` 布尔标志推导 `agents_to_invoke`，而非直接依赖 LLM 输出的列表

### 坑 3: Dashboard 从 AgentState 提取结果的路径错误

```python
# 错误：直接从顶层取
local_results = final_state.get('local_results', [])

# 正确：从 blackboard 中取
bb = final_state.get('blackboard', {})
local_results = bb.get('local_results', [])
```

**教训：** AgentState 的 `local_results` 是 `@property`，但序列化为 dict 后只有 `blackboard` 字典里有数据。

### 坑 4: Eval Agent 评分偏低触发无限重试

**问题：** 有检索结果（10 条），但 Eval Agent 给出低分 → 触发 Refine → 再搜 → 还是低分 → 再 Refine → 达到 max_retries → Fallback → "未找到信息"。
**解决：** Generate 节点加判断：只要有检索结果就尝试生成，不管 Eval 怎么说。同时引入通用知识降级模式——当所有检索均失败时，用 LLM 通用知识回答而非直接返回"未找到信息"。

```python
has_local = bool(state.local_results)
has_web = bool(state.web_results)
web_attempted = state.blackboard.get("web_search_attempted", False)

if not has_local and not has_web:
    if web_attempted:
        gen_mode = "general_knowledge"  # 检索+联网都失败，用 LLM 通用知识
    else:
        gen_mode = "fallback"
else:
    gen_mode = "normal"
```

### 坑 5: 指代消解拼接完整句子导致搜索质量下降

**问题：** 把 "这篇文献的结论是啥 xxx.pdf" 整句拼到 "它用了什么方法" 前面，搜索引擎被长 query 迷惑。
**解决：** 只提取上一句的主题实体（去掉疑问词），且限制主题长度不超过 50 字符，拼接更短更精准的 query。

### 坑 6: 中文引号导致 Python SyntaxError

```python
# 错误：中文 "" 被解释为 Python 字符串终止符
"例如用户说"它"、"这篇"时..."

# 正确：改用单引号
"例如用户说'它'、'这篇'时..."
```

### 坑 7: `_is_impossible_query` 误判正常查询

**问题：** 早期用 `in` 匹配，"我想了解..."、"我感觉..."等正常查询被误判为"不可能的查询"。
**解决：** 改用 `startswith` 匹配，只对以"我昨天"、"我前天"等开头的查询触发，同时单独处理"我家地址"、"我的密码"等包含模式。

---

## 面试高频问题

### Q1: 为什么选择 LangGraph 而不是 LangChain 的 AgentExecutor？

> AgentExecutor 是线性的 ReAct 循环（Thought → Action → Observation），适合简单的工具调用场景。但 RAG 系统需要**条件分支**（根据意图路由到不同 Agent）和**重试循环**（Eval → Refine → Search），这些在 AgentExecutor 中很难优雅实现。LangGraph 的 StateGraph 天然支持有向图的条件边和循环边，更适合多 Agent 编排。

### Q2: Blackboard Pattern 和消息传递有什么区别？

> 消息传递（如 Actor Model）要求 Agent A 知道 Agent B 的地址才能通信，耦合度高。Blackboard 模式下，所有 Agent 只和共享空间交互——Search Agent 不需要知道 Web Agent 的存在，它们只是在 blackboard 上写不同的 key。这样增加新 Agent 不需要修改已有 Agent 的代码。

### Q3: 如何控制上下文？

> 五个层面：检索层（top_k + RRF 排序截断）、生成层（检索结果取前 5+3 条，历史滑动窗口）、Agent 架构层（Blackboard 隔离信息流向）、对话层（指代消解只提取主题实体，不传完整历史）、质量层（Eval 重试 + Faithfulness 检查）。

### Q4: 如何检测幻觉？

> 用规则方法（不需要额外 LLM 调用）：(1) 检查回答中是否有引用标记 [1]、[Local:...]、[Web:...]；(2) 计算回答与 citation 内容的字符 4-gram 重叠度；(3) 计算有多少条 citation 的内容真正在回答中出现了。三者加权得到忠实度分数：0.2 × 标记分 + 0.5 × 重叠度 + 0.3 × 覆盖率。

### Q5: 如果 Eval Agent 自身判断不准怎么办？

> 两层防御：(1) Eval Agent 的 LLM 评分只是软信号，`_apply_rules()` 中有硬性规则覆盖（如 relevance < 0.2 强制 fallback）；(2) Generate 节点做了二次保护——只要有检索结果就尝试生成，不会因为 Eval 误判而直接放弃。此外还有通用知识降级模式，即使所有检索失败也不会返回空结果。

### Q6: 重试机制会不会导致延迟太高？

> 最多重试 2 次，每次重试增加一轮 Refine + Retrieve + Read + Eval。实际测试中，大部分查询在第一轮就通过评估，重试率约 10-20%。对于确实需要重试的查询（比如原始 query 太模糊），2 轮优化后质量提升明显，用户体验收益大于延迟成本。可以通过 `max_retries` 配置来平衡。

### Q7: 本地检索多次失败后怎么办？了解 CRAG 吗？

> 我们实现了类似 CRAG（Corrective RAG）的渐进式升级策略。当本地检索重试耗尽仍无法满足质量要求时，系统不会直接放弃，而是自动升级到联网搜索。同时在生成阶段会做噪音过滤——既然本地 RAG 效果不好，说明 local_results 里大概率有噪音，我们会根据 Eval 的 confidence 和 chunk 的相关性分数（阈值 0.3）过滤掉低质量的本地结果，只保留勉强相关的部分与 web 结果融合。此外，System Prompt 中还设定了信息冲突处理原则：公司内部信息以本地知识库为准，客观事实和最新动态以互联网为补充，矛盾时标注来源让用户判断。

### Q8: 本地知识库和联网搜索的信息冲突怎么处理？

> 在 Generate Agent 的 System Prompt 中内置了冲突处理原则：(1) 公司政策、内部规范等以本地知识库为权威来源；(2) 客观事实、行业趋势以互联网信息补充；(3) 存在矛盾时明确标注两方来源和差异，不替用户做判断。这种设计源于实际业务场景——比如公司报销标准和网上的通用标准可能不同，这时候应该信公司内部文档。

### Q9: 对话记忆是如何管理的？了解 Compaction 机制吗？

> 我们实现了受 OpenClaw 启发的双层记忆架构：
>
> **短期记忆（Compaction）：** 维护一个 token 预算（默认 4000）。当对话历史超出预算时，将较旧的消息通过 LLM 压缩为结构化摘要，强制保留任务状态、关键决策、TODO 和文件名/URL 等标识符。最终上下文 = [compaction 摘要] + [最近 4 条原始消息]，既保留了长程信息又不会超出窗口。
>
> **长期记忆（LangGraph Native Store）：** 使用 LangGraph 原生的 `InMemoryStore` 做长期记忆，每条记忆以 JSON 文档格式存储，按 `(user_id, "conversations")` 和 `(user_id, "facts")` 两级命名空间组织。每次对话结束后自动写入 Store；下次对话时用 `recall()` 检索相关记忆注入上下文。Store 通过 JSON 快照持久化到 `data/memory/langgraph_store.json`，进程重启后自动恢复。同时 Store 实例通过 `store=` 参数注入到 LangGraph 图编译中，图内节点可通过 `get_store()` 直接访问长期记忆。
>
> 这解决了传统滑动窗口的两个痛点：(1) 窗口外信息完全丢失；(2) 跨 session 没有记忆延续。

### Q10: Plan-Read-Eval 循环是什么？为什么需要它？

> 对于复杂查询（如"分析 RAG 技术的发展趋势并与传统检索对比"），单次检索往往不够。Plan 节点用 LLM 将复杂问题拆成多个子问题（1-3 步），每步指定检索来源（local/web/both）。Retrieve 执行当前步骤的检索后，Read 节点对结果进行归纳，判断当前子问题是否已有足够证据。Eval 节点综合 Read 的观察决定下一步：推进到下一个子问题（→ Plan）、改写查询重试（→ Refine）、联网补充（→ Web）、或直接生成最终答案（→ Generate）。
>
> 这种设计借鉴了 ReAct 和 Plan-and-Solve 的思想，让系统具备**多步推理**能力，而非一次性检索就下结论。

---

## 更新日志

### 2026-04-30：图文联动修复 & Agent 图片返回

#### 问题背景
系统虽然设计了完整的图文链路（ImageCaptioner + MultimodalAssembler），但实际端到端存在两个断点：
1. **ChromaDB metadata 损坏**：`ChromaStore._sanitize_metadata()` 将 `images: List[Dict]` 拍平为逗号分隔字符串，导致 MultimodalAssembler 无法解析
2. **图片路径找不到**：`resolve_image_path()` 未尝试 `data/images/{collection}/{doc_hash}/` 子目录
3. **Agent 不返回图片**：SimpleAgent / SearchAgent 格式化结果时丢弃了 `images` 字段，AgentResponse 不携带图片

#### 修改清单

| 文件 | 修改 |
|------|------|
| `src/ingestion/storage/vector_upserter.py` | 存储前将 `images`/`image_captions` JSON 序列化，避免 ChromaDB 损坏数据结构 |
| `src/core/response/multimodal_assembler.py` | 新增 `_deserialize_json_field()` 兼容 JSON 字符串；`resolve_image_path()` 增加 `doc_hash` 子目录搜索；`assemble_for_result()` 传递 metadata |
| `src/agent/tool_caller.py` | 格式化结果中保留 `images`/`image_refs`/`doc_hash`/`source_path` 字段 |
| `src/agent/multi_agent/search_agent.py` | 同上，保留图片字段 |
| `src/agent/simple_agent.py` | `AgentResponse` 新增 `image_data: List[Dict]`；新增 `_extract_images_from_results()` 加载并 base64 编码图片 |
| `src/agent/intent_classifier.py` | Fallback 关键词新增 `"图片"`/`"图"`/`"image"`/`"figure"` 等，确保纯图片查询不依赖 LLM 也能路由到 QUERY |
| `scripts/run_agent.py` | Verbose 模式打印图片数量和元数据 |
| `src/observability/dashboard/pages/agent_chat.py` | 新增 `_extract_images_from_local_results()` 提取图片；`process_query()` 返回 `image_data`；`display_chat_message()` 渲染图片为 inline `<img>` |
| `pyproject.toml` | 新增 `PyMuPDF>=1.23.0` 依赖（PdfLoader 图片提取所需） |

#### 验证结果
- MultimodalAssembler 54 个单元测试全部通过
- 端到端测试：Ingest 含图片 PDF → 搜索 "查询图片" → Agent 返回文字 + 图片 base64 ✅
- Intent 分类：`"图片" / "找图片" / "图"` 均正确路由到 QUERY（conf=0.70）
- Dashboard 图片正确渲染

### Q11: 联网搜索效果差时怎么办？

> `_eval_next_step` 中有一个"回退本地"的逻辑：如果已执行联网搜索但置信度仍然很低（< 0.4），且本地之前有检索结果，系统会切换 `retrieve_plan` 为 `local` 并进入 Refine 优化后重试本地检索。这避免了"联网也没用就彻底放弃"的情况——有时候本地知识库确实有答案，只是第一次检索的 query 不够精准。
