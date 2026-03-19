# LangGraph Agent 实现说明

本文档详细说明 LangGraph 版本的 Agent 实现，以及与原 SimpleAgent 的对比。

## 📊 架构对比

### SimpleAgent（传统实现）

```
用户输入 → IntentClassifier → if-else 分支 → Tool 调用 → 响应生成
```

**特点**：
- ✅ 简单直接，易于理解
- ❌ 大量 if-else 分支（[`_act_and_summarize`](src/agent/simple_agent.py#L161-L236)）
- ❌ 工作流逻辑硬编码在方法中
- ❌ 难以可视化和调试复杂流程

### LangGraphAgent（状态机实现）

```
[START] → classify_intent → route → [条件路由] → 各分支节点 → generate_response → [END]
```

**特点**：
- ✅ 显式的状态图定义
- ✅ 条件路由替代 if-else
- ✅ 每个节点独立，易于测试和扩展
- ✅ 原生支持可视化
- ✅ 支持 checkpointing（断点续传）

## 🏗️ 工作流图

### 完整 LangGraphAgent 工作流

```
┌─────────────────────────────────────────────────────────────────┐
│                     LangGraphAgent Workflow                      │
└─────────────────────────────────────────────────────────────────┘

    [START]
       ↓
┌──────────────────┐
│ classify_intent  │  LLM 意图分类
└────────┬─────────┘
         ↓
┌──────────────────┐
│      route       │  条件路由决策
└────────┬─────────┘
         ↓
    ┌────┴────┬────────────┬──────────────┬────────────┐
    ↓         ↓            ↓              ↓            ↓
┌────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐
│  chat  │ │  search  │ │ summary  │ │collections│ │unknown │
└───┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └───┬────┘
    ↓          │            │            │           ↓
┌────────┐    │            │            │       ┌─────────┐
│  END   │    ↓            │            │       │generate │
└────────┘ ┌──────────┐   │            │       │response │
           │ rerank   │   │            │       └────┬────┘
           └────┬─────┘   │            │            │
                ↓         │            │            │
         ┌──────┴─────────┴────────────┴────────────┘
         ↓
    [END]
```

### search_knowledge 节点内部流程（并行检索）

```
search_knowledge 节点
         ↓
    ┌────┴────────────────────────────┐
    │                                 │
    ↓                                 ↓
┌──────────────┐              ┌──────────────┐
│ Dense Search │              │ Sparse Search│
│ (语义检索)    │              │ (BM25 关键词) │
└──────┬───────┘              └──────┬───────┘
       │                             │
       │  ThreadPoolExecutor         │
       │  (并行执行)                 │
       │                             │
       └──────────┬──────────────────┘
                  ↓
         ┌────────────────┐
         │  RRF Fusion    │  ← 倒数排名融合
         └────────┬───────┘
                  ↓
         ┌────────────────┐
         │ Post-Filtering │  ← 元数据过滤
         └────────┬───────┘
                  ↓
         [检索结果返回]
```

**关键点**：
- ✅ `HybridSearch._run_retrievals()` 使用 `ThreadPoolExecutor` 并行执行
- ✅ Dense 和 Sparse 检索同时进行，互不阻塞
- ✅ RRF Fusion 等待两个检索都完成后执行
- ✅ 优雅降级：一个失败时使用另一个的结果

## 📝 核心组件

### 1. AgentState

[`AgentState`](src/agent/langgraph_agent.py#L62-L113) 是 LangGraph 的状态容器，管理整个工作流的数据流转：

```python
class AgentState(dict):
    """状态容器"""
    messages: List[BaseMessage]           # 对话历史
    intent: IntentType                     # 分类的意图
    intent_confidence: float               # 置信度
    query: str                             # 当前查询
    search_results: List[Dict]             # 检索结果
    reranked_results: List[Dict]           # 重排序结果
    final_answer: str                      # 最终回复
    tool_called: Optional[str]             # 调用的工具
    tool_result: Optional[Dict]            # 工具结果
    execution_trace: List[AgentStep]       # 执行轨迹
    error: Optional[str]                   # 错误信息
```

### 2. Nodes（节点）

每个节点是一个独立的处理单元：

| 节点 | 功能 | 对应 SimpleAgent 方法 |
|------|------|---------------------|
| `classify_intent` | 意图分类 | [`_plan`](src/agent/simple_agent.py#L121-L159) 部分逻辑 |
| `route` | 路由决策 | `_act_and_summarize` 的 if-else |
| `handle_chat` | 处理对话 | `_handle_chat` |
| `search_knowledge` | 知识检索 | `_act_and_summarize` QUERY 分支 |
| `rerank_results` | 结果重排 | 无（新增功能） |
| `generate_response` | 响应生成 | `_format_*_response` 方法 |
| `get_summary` | 文档摘要 | `_act_and_summarize` GET_SUMMARY 分支 |
| `list_collections` | 集合列表 | `_act_and_summarize` LIST_COLLECTIONS 分支 |

### 3. 条件路由

[`_route_by_intent`](src/agent/langgraph_agent.py#L376-L390) 实现条件路由：

```python
def _route_by_intent(self, state: AgentState) -> Literal["chat", "query", ...]:
    intent = state.get("intent")
    
    if intent == IntentType.QUERY:
        return "query"
    elif intent == IntentType.GET_SUMMARY:
        return "summary"
    # ...
```

这比 SimpleAgent 的 if-else 更清晰：

```python
# SimpleAgent 的实现（复杂）
if intent == IntentType.QUERY:
    # ...
elif intent == IntentType.LIST_COLLECTIONS:
    # ...
elif intent == IntentType.GET_SUMMARY:
    # ...
```

## 🚀 使用示例

### 基本使用

```python
from src.agent.langgraph_agent import LangGraphAgent
from src.core.settings import load_settings

settings = load_settings()
agent = LangGraphAgent(settings)

# 单轮查询
response = agent.run("什么是 RAG？")
print(response.content)

# 多轮对话（自动指代消解）
response = agent.run("它有什么优势？")  # "它" 自动解析为 RAG
print(response.content)
```

### 查看执行轨迹

```python
response = agent.run("查询关于电子舌的论文")

# 打印详细执行轨迹
for step in response.steps:
    print(f"步骤 {step.step}:")
    print(f"  节点：{step.node}")
    print(f"  思考：{step.thought}")
    print(f"  动作：{step.action}")
    print(f"  观察：{step.observation}")
```

### 启用 Rerank

```python
# 启用 rerank 节点（当前为 pass-through，可集成真实 reranker）
agent = LangGraphAgent(settings, enable_rerank=True)
response = agent.run("查询 RAG 相关论文")
```

## 📊 新旧对比示例

### 运行对比 Demo

```bash
# 运行对比演示
python examples/langgraph_agent_demo.py
```

### 输出示例

```
================================================================================
📝 查询 1: 查询关于 RAG 的论文
================================================================================

【SimpleAgent】
----------------------------------------
意图：query
置信度：0.95
工具调用：query_knowledge_hub
执行步骤：2
  1. 分析用户问题，确定需要在知识库中检索相关内容。
     动作：准备调用 query_knowledge_hub 进行混合检索。
  2. 从检索结果中挑选最相关的片段，组织成回答。

回复预览：为你找到 5 条与「查询关于 RAG 的论文」相关的内容...


【LangGraphAgent】
----------------------------------------
意图：query
置信度：0.95
工具调用：query_knowledge_hub
执行步骤：4
  1. [classify_intent] 分析用户意图：查询关于 RAG 的论文...
     动作：IntentClassifier.classify()
     观察：Intent: query (confidence: 0.95)
  2. [route] 根据意图路由到对应处理节点
     动作：route_by_intent(query)
  3. [search_knowledge] 在知识库中检索：查询关于 RAG 的论文...
     动作：query_knowledge_hub(query='查询关于 RAG 的论文', top_k=10)
     观察：检索到 5 条结果
  4. [generate_response] 整理结果并生成最终回复
     动作：format_response
     观察：生成查询回复 (5 条结果)

回复预览：为你找到 5 条与「查询关于 RAG 的论文」相关的内容...


【对比分析】
----------------------------------------
意图一致性：✅
步骤数对比：Simple=2 vs LangGraph=4
LangGraph 路径：classify_intent → route → search_knowledge → generate_response
```

## 🎯 优势总结

### 代码结构

| 维度 | SimpleAgent | LangGraphAgent |
|------|-------------|----------------|
| **路由逻辑** | if-else 硬编码 | 显式条件边 |
| **可观测性** | 简单步骤列表 | 详细节点轨迹 |
| **可扩展性** | 修改方法内部 | 添加节点/边 |
| **可测试性** | 方法耦合 | 节点独立 |
| **可视化** | ❌ | ✅ 状态图 |
| **Checkpointing** | ❌ | ✅ 原生支持 |

### 执行轨迹对比

**SimpleAgent**（2 步）：
```
1. 分析用户问题，确定需要在知识库中检索相关内容。
2. 从检索结果中挑选最相关的片段，组织成回答。
```

**LangGraphAgent**（4 步，更详细）：
```
1. [classify_intent] 分析用户意图 → Intent: query
2. [route] 路由决策 → route_by_intent(query)
3. [search_knowledge] 执行检索 → 5 条结果
4. [generate_response] 生成回复 → 格式化输出
```

## 🔧 扩展指南

### 添加新节点

```python
# 1. 定义新节点
def _node_evaluate_results(self, state: AgentState) -> AgentState:
    """评估检索结果质量"""
    results = state.get("search_results", [])
    
    # 评估逻辑
    quality_score = self._evaluate(results)
    
    state.update(result_quality=quality_score)
    return state

# 2. 添加到图中
workflow.add_node("evaluate_results", self._node_evaluate_results)

# 3. 添加边
workflow.add_edge("search_knowledge", "evaluate_results")
workflow.add_edge("evaluate_results", "generate_response")
```

### 添加并行检索

```python
from langgraph.graph import StateGraph

# 并行检索 dense 和 sparse
workflow.add_node("dense_search", self._node_dense_search)
workflow.add_node("sparse_search", self._node_sparse_search)
workflow.add_node("fuse_results", self._node_fuse_results)

workflow.add_edge("dense_search", "fuse_results")
workflow.add_edge("sparse_search", "fuse_results")
```

### 添加自我反思循环

```python
def _should_reflect(self, state: AgentState) -> Literal["reflect", "continue"]:
    """判断是否需要反思"""
    quality = state.get("result_quality", 1.0)
    if quality < 0.7:
        return "reflect"
    return "continue"

workflow.add_conditional_edges(
    "evaluate_results",
    _should_reflect,
    {
        "reflect": "refine_query",  # 优化查询重新检索
        "continue": "generate_response"
    }
)
```

## 📚 下一步优化建议

1. **集成真实 Reranker**
   - 当前 `rerank_results` 节点是 pass-through
   - 可集成现有的 [`Reranker`](src/libs/reranker/reranker_factory.py)

2. **添加 LangChain Callbacks**
   - 自动记录 LLM 调用
   - 集成到现有 dashboard

3. **实现 Checkpointing**
   - 使用 LangGraph 的 MemorySaver
   - 支持长对话的断点续传

4. **多 Agent 协作**
   - 不同意图使用专门 sub-agent
   - Supervisor 模式协调

## 🔗 相关文件

- [`langgraph_agent.py`](src/agent/langgraph_agent.py) - LangGraph Agent 实现
- [`simple_agent.py`](src/agent/simple_agent.py) - 原 SimpleAgent 实现（保留）
- [`langgraph_agent_demo.py`](examples/langgraph_agent_demo.py) - 对比演示脚本
- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)

## 💡 使用建议

### 何时使用 LangGraphAgent

- ✅ 需要复杂工作流（多分支、循环、并行）
- ✅ 需要可视化调试
- ✅ 需要 checkpointing（长对话）
- ✅ 需要详细的执行观察

### 何时使用 SimpleAgent

- ✅ 简单场景（单轮查询）
- ✅ 追求最小依赖
- ✅ 教学/学习目的（对比两种实现）

## 🎓 学习价值

保留 SimpleAgent 的价值：

1. **教学对比**：展示传统实现 vs 现代化框架
2. **理解原理**：SimpleAgent 更直接展示底层逻辑
3. **简历素材**：展示技术演进能力（"我从 if-else 重构为状态机"）
4. **降级方案**：LangGraph 出问题时可用 SimpleAgent

---

**作者**: 你的团队  
**创建时间**: 2026-03-18  
**更新时间**: 2026-03-18
