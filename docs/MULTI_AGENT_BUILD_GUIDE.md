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
│ Router Agent │ ← 意图识别（chat / local_search / web_search / hybrid_search）
└─────┬───────┘
      │ 条件路由
      ├────────────────┬──────────────┐
      ▼                ▼              ▼
┌──────────┐   ┌────────────┐   ┌──────────┐
│  Search  │   │    Web     │   │ Generate │ ← chat 意图直接生成
│  Agent   │   │   Agent    │   │  (LLM)   │
└────┬─────┘   └─────┬──────┘   └──────────┘
     │               │
     └───────┬───────┘
             ▼
      ┌────────────┐
      │ Eval Agent │ ← 评估检索质量（相关性/多样性/覆盖度）
      └──────┬─────┘
             │
      ┌──────┴──────┐
      │  需要优化？  │
      ├─── 是 ──→ Refine Agent → 回到 Search（重试循环）
      └─── 否 ──→ Generate Agent → 最终回答（带引用 + 忠实度检查）
```

**核心设计模式：**

| 模式 | 说明 |
|------|------|
| **Blackboard Pattern** | 所有 Agent 共享一个 `AgentState`，通过 `blackboard` 字典读写数据 |
| **LangGraph StateGraph** | 用有向图定义 Agent 间的流转关系，支持条件分支和循环 |
| **重试-兜底机制** | Eval → Refine → Search 构成闭环，最多重试 2 次后兜底 |
| **溯源 + 忠实度** | Citation 模块追踪每条回答的来源，N-gram 检查是否存在幻觉 |

**文件结构：**

```
src/agent/multi_agent/
├── __init__.py              # 统一导出
├── state.py                 # AgentState（共享状态）
├── router_agent.py          # 意图识别 + 路由
├── search_agent.py          # 本地向量库检索
├── web_agent.py             # 联网搜索
├── eval_agent.py            # 检索质量评估
├── refine_agent.py          # 查询优化
├── citation.py              # 溯源 + 忠实度检查
├── parallel_controller.py   # 并行融合控制器
└── multi_agent_system.py    # LangGraph 主编排器
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
    #   "intent"          → Router 写入的意图
    #   "local_results"   → Search Agent 写入的本地检索结果
    #   "web_results"     → Web Agent 写入的联网搜索结果
    #   "evaluation"      → Eval Agent 写入的评估结果
    #   "refined_query"   → Refine Agent 写入的优化查询
    #   "citations"       → Generate Agent 写入的引用列表
    #   "faithfulness_check" → Generate Agent 写入的忠实度检查

    # ===== 重试控制 =====
    retry_count: int = 0
    max_retries: int = 2
    fallback_triggered: bool = False
    fallback_reason: Optional[FallbackReason] = None

    # ===== 执行追踪 =====
    execution_trace: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    # ===== 输出 =====
    final_answer: str = ""
```

### 关键设计决策

**1. 用 `@property` 封装 blackboard 读取：**

```python
@property
def local_results(self) -> List:
    return self.blackboard.get("local_results", [])

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

**3. `reset()` 方法用于多轮对话：**

每轮对话开始时清空 blackboard 和 trace，但保留 `user_input` 和 `conversation_history`。这样 Agent 看到的是干净的新一轮状态，但仍然有对话记忆。

---

## Step 2: 构建 Router Agent — 意图识别与路由

**文件：** `src/agent/multi_agent/router_agent.py`

### 职责

拿到用户输入，判断应该调用哪些 Agent。

### 五种意图

| 意图 | 路由目标 | 示例 |
|------|----------|------|
| `chat` | → Generate | "你好"、"谢谢" |
| `local_search` | → Search | "这篇论文的结论是什么" |
| `web_search` | → Web | "今天 AI 领域的最新新闻" |
| `hybrid_search` | → Search + Web（并行） | "结合内部文档和网上资料" |
| `unknown` | → Search（兜底尝试） | 无法分类的查询 |

### 实现方式：LLM 分类

```python
class RouterAgent:
    SYSTEM_PROMPT = """You are an intent classifier...
    Respond in JSON format:
    {
        "intent": "local_search",
        "agents_to_invoke": ["SearchAgent"],
        "parallel": false,
        "confidence": 0.95,
        "reasoning": "..."
    }"""

    def classify(self, query, context=None) -> RoutingDecision:
        response = self.chain.invoke({"query": query, "context": context_str})
        result = self._parse_response(response.content)
        return RoutingDecision(
            intent=result.get("intent", "unknown"),
            agents_to_invoke=[...],
            parallel=result.get("parallel", False),
            ...
        )
```

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

    def search(self, query, top_k=5, context=None) -> List[Dict]:
        if context:
            query = self._resolve_pronouns(query, context)
        result = self.tool.execute(query=query, top_k=top_k)
        return self._format_results(result.data)
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

---

## Step 4: 构建 Web Agent — 联网搜索

**文件：** `src/agent/multi_agent/web_agent.py`

### 职责

搜索互联网获取实时信息，支持 DuckDuckGo（免费）/ Google / Bing。

### 核心逻辑

```python
class WebSearchAgent:
    def search(self, query, num_results=5, local_results=None) -> List[Dict]:
        # 如果本地已有基础信息，搜索互补内容
        if local_results:
            refined_query = f"{query} 2025 2026 最新进展"
        else:
            refined_query = query
        results = self.search_tool.search(query=refined_query, ...)
        return self._format_results(results)
```

### 设计亮点：条件查询优化

Web Agent 可以**读取 Search Agent 已经找到的结果**（通过 Blackboard），据此调整搜索策略。比如：
- 本地已有论文原文 → 联网搜索"最新进展、引用情况"
- 本地没有结果 → 直接搜原始 query

这就是 Blackboard 模式的优势：Agent 之间无需直接调用，通过共享空间协作。

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
        return RefinementResult(
            refined_query=result.get('refined_query'),
            changes_made=result.get('changes_made'),
            reasoning=result.get('reasoning'),
        )
```

### 重试循环

```
Search → Eval → [quality low] → Refine → Search → Eval → ... (max 2 rounds)
```

Refine Agent 每次执行后会 `state.increment_retry("refine")`，并把 `refined_query` 写入 Blackboard。下一轮 Search Agent 检测到 `retry_count > 0` 就会使用优化后的 query。

---

## Step 7: 构建 Citation 溯源模块

**文件：** `src/agent/multi_agent/citation.py`

### 三个核心组件

#### 1. Citation 数据类

```python
@dataclass
class Citation:
    type: CitationType     # LOCAL / WEB
    source: str            # 文档名或网站名
    content: str           # 引用的具体内容
    confidence: float      # 置信度
    url: Optional[str]     # 仅 web 类型
    ...
```

#### 2. CitationManager（引用管理器）

负责从检索结果创建 Citation 对象，按排名自动赋予递减的置信度：

```python
@staticmethod
def create_citations_from_results(local_results, web_results, top_k=5):
    for i, result in enumerate(local_results[:top_k]):
        confidence = max(0.5, 1.0 - (i * 0.1))  # 第1名1.0，第2名0.9，...
        citation = create_citation_from_local_result(result, confidence)
    # ... 同理处理 web_results
```

#### 3. FaithfulnessCheck（忠实度检查）

生成回答后，检查回答是否忠实于检索结果（而非 LLM 自己编造）：

```
检查策略（无需额外 LLM 调用，纯规则）：

1. 引用标记检查：回答中是否有 [1]、[Local: ...]、[Web: ...] 等标记？
2. 内容重叠检查：回答的 4-gram 与每条 citation 内容的 4-gram 重叠度
3. 引用覆盖率：多少条 citation 的内容在回答中有体现？

综合得分 = 0.2 × 标记分 + 0.5 × 重叠度 + 0.3 × 覆盖率
```

---

## Step 8: 构建 Parallel Controller — 并行融合控制器

**文件：** `src/agent/multi_agent/parallel_controller.py`

### 为什么需要并行？

当用户说"结合内部文档和网上资料分析一下"时，需要**同时**调用 Search Agent 和 Web Agent，然后融合结果。串行等待会浪费时间。

### 实现方式：ThreadPoolExecutor

```python
class ParallelFusionController:
    def execute_parallel_search(self, state, agents_to_invoke):
        with ThreadPoolExecutor(max_workers=5) as executor:
            for agent_type in agents_to_invoke:
                if agent_type == AgentType.SEARCH:
                    future = executor.submit(self._execute_search, ...)
                elif agent_type == AgentType.WEB:
                    future = executor.submit(self._execute_web, ...)

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

---

## Step 9: 用 LangGraph 编排所有 Agent

**文件：** `src/agent/multi_agent/multi_agent_system.py`

这是整个系统的**大脑**——把前面所有零件组装成一个有向图。

### 9.1 初始化所有 Agent

```python
class MultiAgentRAG:
    def __init__(self, llm, settings):
        self.router_agent = RouterAgent(llm)
        self.search_agent = SearchAgent(settings)
        self.web_agent = WebSearchAgent(settings)
        self.eval_agent = EvalAgent(llm)
        self.refine_agent = RefineAgent(llm)
        self.parallel_controller = ParallelFusionController(...)
        self.workflow = self._build_graph()
```

### 9.2 构建 LangGraph 状态图

```python
def _build_graph(self) -> StateGraph:
    workflow = StateGraph(AgentState)

    # 添加节点（每个节点对应一个 Agent）
    workflow.add_node("router", self._router_node)
    workflow.add_node("search", self._search_node)
    workflow.add_node("web", self._web_node)
    workflow.add_node("eval", self._eval_node)
    workflow.add_node("refine", self._refine_node)
    workflow.add_node("generate", self._generate_node)

    # 入口
    workflow.set_entry_point("router")

    # 条件边：Router 根据意图路由
    workflow.add_conditional_edges("router", self._route_by_intent, {
        "generate": "generate",   # chat → 直接生成
        "search": "search",       # local_search → 本地检索
        "web": "web",             # web_search → 联网搜索
    })

    # 条件边：Search 后判断是否需要 Web（hybrid_search 需要）
    workflow.add_conditional_edges("search", self._should_search_web, {
        "yes": "web",   # hybrid → 继续联网
        "no": "eval",   # 纯本地 → 直接评估
    })

    # 固定边
    workflow.add_edge("web", "eval")       # Web → Eval
    workflow.add_edge("refine", "search")  # Refine → 重新 Search（循环！）
    workflow.add_edge("generate", END)     # Generate → 结束

    # 条件边：Eval 决定是生成还是重试
    workflow.add_conditional_edges("eval", self._should_refine, {
        "generate": "generate",
        "refine": "refine",
    })

    return workflow.compile()
```

### 9.3 节点实现模板

每个节点的职责都是：**读取 State → 调用对应 Agent → 写入 State**。

以 Search 节点为例：

```python
def _search_node(self, state: AgentState) -> AgentState:
    # 1. 读取：用优化后的 query 还是原始 query？
    query = state.refined_query if state.retry_count > 0 else state.user_input

    # 2. 调用 Agent
    try:
        results = self.search_agent.search(query=query, top_k=10, context=context)
    except Exception:
        results = []  # 容错：搜索失败不崩溃

    # 3. 写入 Blackboard
    state.add_to_blackboard("local_results", results, "search")

    # 4. 记录追踪
    state.add_execution_trace({...})
    return state
```

### 9.4 生成节点 — LLM 总结

Generate 节点不是简单拼接检索结果，而是调用 LLM 做总结：

```python
def _generate_normal_response_with_citations(self, context):
    # 1. 构建检索结果块
    for i, result in enumerate(local_results[:5], 1):
        context_parts.append(f"[{i}] (来源: {source})\n{content}")

    # 2. 构建 LLM messages
    messages = [SystemMessage(content=RAG_SYSTEM_PROMPT)]

    # 3. 注入对话历史（滑动窗口，最近 6 条）
    for turn in conv_history[-6:]:
        messages.append(HumanMessage/AIMessage)

    # 4. 注入检索结果 + 用户问题
    messages.append(HumanMessage(content=f"## 检索结果\n{context}\n## 用户问题\n{query}"))

    # 5. LLM 生成
    response = self.llm.invoke(messages)

    # 6. 附加引用 + 忠实度检查
    answer = format_answer_with_citations(answer, citations)
    faithfulness = citation_manager.check_faithfulness(answer)
    return answer, citations, faithfulness
```

### 9.5 关键条件路由函数

```python
def _route_by_intent(self, state) -> Literal["generate", "search", "web"]:
    intent_to_node = {
        "chat": "generate",
        "local_search": "search",
        "web_search": "web",
        "hybrid_search": "search",
        "unknown": "search",  # 重要！未知意图也尝试搜索
    }
    return intent_to_node.get(state.intent, "search")

def _should_refine(self, state) -> Literal["generate", "refine"]:
    if state.fallback_triggered:
        return "generate"  # 已触发兜底，不再重试
    evaluation = state.evaluation
    if evaluation.get("fallback_suggested"):
        return "generate"
    if evaluation.get("need_refinement"):
        return "refine"
    return "generate"
```

### 9.6 公共 API

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
    # 检测是否有代词或省略
    pronouns = ["它", "这个", "那个", "这篇", "还有呢", "继续"]
    needs_context = any(p in query for p in pronouns) or len(query) < 6

    if needs_context:
        # 从历史中找到上一轮的主题
        prev_query = find_last_user_message(context)
        # 提取关键主题（去除疑问词）
        topic = remove_question_words(prev_query)  # "结论是啥" → 文件名
        return f"{topic} {query}"  # 拼接成自包含的 query
```

**效果：**
- 用户第一轮："这篇文献的结论是啥 xxx.pdf" → 正常搜索
- 用户第二轮："它用了什么方法" → 改写为 "xxx.pdf 它用了什么方法"
- 用户第三轮："还有呢" → 改写为 "xxx.pdf 用了什么方法 还有呢"

### 10.3 生成端：滑动窗口

```python
conv_history = context.get("conversation_history", [])
for turn in conv_history[-6:]:  # 只取最近 6 条 message（≈3 轮 Q&A）
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
├── 用户问题:             ~50 tokens（变量）
└── 留给生成的空间:       ~4150 tokens
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
**解决：** Generate 节点加判断：只要有检索结果就尝试生成，不管 Eval 怎么说。

```python
has_results = bool(state.local_results or state.web_results)
if state.should_fallback and not has_results:
    # 真正没结果才 fallback
    state.final_answer = self._generate_fallback_response(state)
else:
    # 有结果就尝试 LLM 生成
    answer, citations, faithfulness = self._generate_normal_response_with_citations(...)
```

### 坑 5: 指代消解拼接完整句子导致搜索质量下降

**问题：** 把 "这篇文献的结论是啥 xxx.pdf" 整句拼到 "它用了什么方法" 前面，搜索引擎被长 query 迷惑。
**解决：** 只提取上一句的主题实体（去掉疑问词），拼接更短更精准的 query。

### 坑 6: 中文引号导致 Python SyntaxError

```python
# 错误：中文 "" 被解释为 Python 字符串终止符
"例如用户说"它"、"这篇"时..."

# 正确：改用单引号
"例如用户说'它'、'这篇'时..."
```

---

## 面试高频问题

### Q1: 为什么选择 LangGraph 而不是 LangChain 的 AgentExecutor？

> AgentExecutor 是线性的 ReAct 循环（Thought → Action → Observation），适合简单的工具调用场景。但 RAG 系统需要**条件分支**（根据意图路由到不同 Agent）和**重试循环**（Eval → Refine → Search），这些在 AgentExecutor 中很难优雅实现。LangGraph 的 StateGraph 天然支持有向图的条件边和循环边，更适合多 Agent 编排。

### Q2: Blackboard Pattern 和消息传递有什么区别？

> 消息传递（如 Actor Model）要求 Agent A 知道 Agent B 的地址才能通信，耦合度高。Blackboard 模式下，所有 Agent 只和共享空间交互——Search Agent 不需要知道 Web Agent 的存在，它们只是在 blackboard 上写不同的 key。这样增加新 Agent 不需要修改已有 Agent 的代码。

### Q3: 如何控制上下文？

> 五个层面：检索层（top_k + RRF 排序截断）、生成层（检索结果取前 5+3 条，历史滑动窗口 6 条）、Agent 架构层（Blackboard 隔离信息流向）、对话层（指代消解只提取主题实体，不传完整历史）、质量层（Eval 重试 + Faithfulness 检查）。

### Q4: 如何检测幻觉？

> 用规则方法（不需要额外 LLM 调用）：(1) 检查回答中是否有引用标记 [1]、[Local:...]；(2) 计算回答与 citation 内容的字符 4-gram 重叠度；(3) 计算有多少条 citation 的内容真正在回答中出现了。三者加权得到忠实度分数。

### Q5: 如果 Eval Agent 自身判断不准怎么办？

> 两层防御：(1) Eval Agent 的 LLM 评分只是软信号，`_apply_rules()` 中有硬性规则覆盖（如 relevance < 0.2 强制 fallback）；(2) Generate 节点做了二次保护——只要有检索结果就尝试生成，不会因为 Eval 误判而直接放弃。

### Q6: 重试机制会不会导致延迟太高？

> 最多重试 2 次，每次重试增加一轮 Refine + Search + Eval。实际测试中，大部分查询在第一轮就通过评估，重试率约 10-20%。对于确实需要重试的查询（比如原始 query 太模糊），2 轮优化后质量提升明显，用户体验收益大于延迟成本。可以通过 `max_retries` 配置来平衡。
