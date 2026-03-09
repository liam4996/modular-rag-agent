# Agent 系统 - 面试 Q&A

> 基于 5 天开发完成的 Modular RAG Agent 系统

---

## 📋 项目概述

**项目名称**: Modular RAG MCP Server with Agent
**开发周期**: 5 天
**核心功能**: ReAct Agent + Conversation Memory + Tool Chain

---

## 🔥 高频面试问题

### 1. 项目背景与目标

**Q: 为什么要给 RAG 系统加 Agent？**

**A:**
- 原始 RAG 是**被动响应**，用户问什么就查什么
- Agent 让系统具备**主动规划**能力，可以分解复杂任务
- 支持**多轮对话**和**上下文理解**，提升用户体验
- 面试展示技术深度，体现对 LLM 应用架构的理解

---

### 2. 技术架构

**Q: 你的 Agent 架构是怎样的？**

**A:**
```
用户输入
    ↓
Intent Classifier (LLM-based) → 识别意图
    ↓
Conversation Memory → 管理多轮上下文
    ↓
ReAct Agent → Thought → Action → Observation 循环
    ↓
Tool Chain Executor → 多工具组合调用
    ↓
生成回答
```

**核心组件**:
1. **IntentClassifier** - LLM-based 意图识别，支持上下文
2. **ConversationMemory** - 对话历史管理，自动 query 改写
3. **ReActAgent** - 推理+行动循环，显式展示思考过程
4. **ToolChainExecutor** - 工具链执行，支持重试和 fallback

---

### 3. ReAct 模式

**Q: 什么是 ReAct？你怎么实现的？**

**A:**
ReAct = **Re**asoning + **Act**ing（推理 + 行动）

**实现方式**:
```python
# ReAct 循环
for step in range(max_iterations):
    # 1. Thought: LLM 思考下一步做什么
    thought = llm.generate("Thought: ...")
    
    # 2. Action: 选择工具
    action = parse_action(thought)  # e.g., "query_knowledge_hub"
    
    # 3. Observation: 执行工具获取结果
    observation = tool_registry.execute(action)
    
    # 4. 如果信息足够，生成最终答案
    if action == "Final Answer":
        break
```

**优势**:
- 决策过程**透明可解释**
- 支持**多步推理**
- 可以**自我纠错**

---

### 4. 意图识别

**Q: 意图识别怎么做的？为什么不用规则？**

**A:**

**之前（规则-based）**:
```python
if "查询" in query or "搜索" in query:
    return IntentType.QUERY
```
- 缺点：难以覆盖所有情况，维护困难

**现在（LLM-based）**:
```python
prompt = f"""
对话历史: {context}
用户输入: {query}

请判断意图：query(查询) / chat(闲聊) / summarize(总结) / unknown(未知)
"""
intent = llm.generate(prompt)
```

**优势**:
- 利用 LLM **语义理解**能力
- 支持**上下文感知**
- 更容易**扩展新意图**

---

### 5. 对话记忆

**Q: 多轮对话怎么实现的？query 改写是什么？**

**A:**

**对话记忆**:
```python
class ConversationMemory:
    def add_turn(self, role, content, intent, tool_called):
        self.turns.append(Turn(...))
    
    def get_context_for_prompt(self):
        # 返回格式化的对话历史
        return format_turns(self.turns)
```

**Query 改写（指代消解）**:
```python
# Turn 1
User: "什么是RAG？"

# Turn 2 - "它"需要解析为"RAG"
User: "它有什么优势？"

# 改写后
rewritten = "RAG有什么优势？"  # LLM 自动改写
```

---

### 6. 工具调用

**Q: 工具调用失败怎么办？**

**A:**

**1. 重试机制**:
```python
def execute(tool_name, max_retries=2):
    for attempt in range(max_retries + 1):
        try:
            return tool.execute()
        except Exception:
            if attempt < max_retries:
                time.sleep(0.5 * (attempt + 1))  # 指数退避
```

**2. Fallback 机制**:
```python
def execute_with_fallback(primary, fallback):
    result = execute(primary)
    if not result.success:
        result = execute(fallback)
    return result
```

**3. 错误处理策略**:
- `stop`: 出错停止
- `skip`: 跳过继续
- `fallback`: 切换到备用方案

---

### 7. 工具链

**Q: 什么是工具链？举个例子**

**A:**

工具链 = **多个工具串联执行**

**例子 - 搜索并分析**:
```python
chain = [
    # Step 1: 搜索
    ChainStep(
        name="search",
        tool_name="query_knowledge_hub",
        params={"query": "RAG", "top_k": 5}
    ),
    # Step 2: 转换数据
    ChainStep(
        name="analyze",
        step_type=ChainStepType.TRANSFORM,
        transform=lambda ctx: {
            "count": len(ctx["search"]["results"]),
            "has_results": len(ctx["search"]["results"]) > 0
        }
    )
]

result = executor.execute_chain(chain)
```

**应用场景**:
- 搜索 → 总结 → 回答
- 列出集合 → 选择 → 查询
- 条件判断 → 分支执行

---

### 8. 挑战与解决

**Q: 开发过程中遇到什么挑战？**

**A:**

| 挑战 | 解决方案 |
|------|----------|
| LLM 调用失败 | 添加 fallback 到规则匹配 |
| 多轮上下文丢失 | 实现 ConversationMemory |
| 工具调用超时 | 添加重试 + 指数退避 |
| ReAct 循环无限执行 | 设置 max_iterations 限制 |
| Query 指代不明 | LLM-based query 改写 |

---

### 9. 性能优化

**Q: 有没有做性能优化？**

**A:**

1. **Memory 截断**: 只保留最近 N 轮对话
2. **Token 限制**: 控制上下文长度，避免超出 LLM 限制
3. **并行检索**: Hybrid Search 中 Dense + Sparse 并行
4. **缓存**: 向量检索结果可缓存

---

### 10. 未来规划

**Q: 如果继续优化，你会做什么？**

**A:**

1. **Plan-and-Execute**: 更复杂的任务规划
2. **Self-Reflection**: Agent 自我评估和纠错
3. **Multi-Agent**: 多个 Agent 协作
4. **Function Calling**: 使用 OpenAI function calling 替代 prompt 解析
5. **Observability**: 添加更完善的日志和监控

---

## 🎯 演示场景

### Scene 1: 基础查询
```
User: 什么是RAG？
Agent: [查询知识库] 为你找到相关内容...
```

### Scene 2: 多轮对话
```
User: 什么是RAG？
Agent: RAG是检索增强生成...

User: 它有什么优势？  # "它"自动解析为"RAG"
Agent: 优势包括减少幻觉、知识更新...

User: 详细介绍下减少幻觉
Agent: [查询知识库] 减少幻觉的机制是...
```

### Scene 3: ReAct 推理
```
User: 总结这份文档的核心观点

Agent Thought: 用户想要总结，我需要先搜索相关文档
Agent Action: query_knowledge_hub
Agent Observation: [搜索结果]
Agent Thought: 已获得足够信息，可以生成总结
Agent Action: Final Answer

Agent: 核心观点是...
```

---

## 💡 技术亮点总结

1. **ReAct 架构** - 显式推理过程，可解释性强
2. **LLM-based 意图识别** - 比规则更智能
3. **Conversation Memory** - 支持多轮上下文
4. **Query 改写** - 自动指代消解
5. **Tool Chain** - 多工具组合
6. **错误处理** - 重试 + fallback + 策略化错误处理

---

## 📁 关键文件

```
src/agent/
├── intent_classifier.py   # Day 1: LLM-based 意图识别
├── memory.py              # Day 2: 对话记忆
├── simple_agent.py        # Day 2: 基础 Agent
├── react_agent.py         # Day 3: ReAct Agent
├── tool_caller.py         # Day 4: 工具调用 + 重试
└── tool_chain.py          # Day 4: 工具链执行器
```

---

## ✅ 面试准备清单

- [ ] 能画出架构图
- [ ] 能解释 ReAct 循环
- [ ] 能演示多轮对话
- [ ] 能说明错误处理策略
- [ ] 能讲清楚技术选型原因

**Good Luck! 🚀**
