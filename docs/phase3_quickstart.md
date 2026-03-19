# Phase 3 溯源与忠实度 - 快速开始指南

## 🎯 功能概述

Phase 3 实现了完整的溯源（Citation）与忠实度（Faithfulness）检查机制，确保生成的回答：
- ✅ **可追溯**: 每个陈述都有明确的来源
- ✅ **可验证**: 用户可以追溯和验证信息
- ✅ **防幻觉**: 自动检测没有依据的陈述

---

## 🚀 快速使用

### 1. 基本使用

```python
from src.agent.multi_agent import (
    Citation,
    CitationType,
    CitationManager,
    FaithfulnessCheck,
    format_answer_with_citations,
)

# 创建引用
citation = Citation(
    type=CitationType.LOCAL,
    source="RAG 原理文档",
    content="RAG 是一种检索增强生成技术",
    confidence=0.95,
    relevance=0.90,
    page=5
)

# 格式化引用标记
print(citation.format_citation())
# 输出：[Local: RAG 原理文档，p.5]
```

### 2. 引用管理

```python
# 创建引用管理器
manager = CitationManager()

# 添加引用
manager.add_citation(citation)

# 从检索结果批量创建
local_results = [
    {"content": "内容 1", "source": "文档 1", "page": 5},
    {"content": "内容 2", "source": "文档 2", "page": 10},
]
web_results = [
    {"title": "AI News", "snippet": "AI latest news", "url": "https://ainews.com/1"},
]

citations = CitationManager.create_citations_from_results(
    local_results=local_results,
    web_results=web_results,
    top_k=5
)

# 格式化所有引用
formatted = manager.format_all_citations()
print(formatted)
# 输出:
# 1. [Local: 文档 1, p.5]
# 2. [Local: 文档 2, p.10]
# 3. [Web: ainews.com]
```

### 3. 忠实度检查

```python
# 有引用支持
manager.add_citation(citation)
text_with_citation = "RAG 技术很重要 [Local: RAG 原理文档]。"
check = manager.check_faithfulness(text_with_citation)

print(f"是否忠实：{check.is_faithful}")  # True
print(f"检测到臆造：{check.hallucination_detected}")  # False
print(f"置信度：{check.confidence}")  # 0.8

# 无引用支持
manager.clear()
text_without_citation = "RAG 技术很重要，但没有引用。"
check = manager.check_faithfulness(text_without_citation)

print(f"是否忠实：{check.is_faithful}")  # False
print(f"检测到臆造：{check.hallucination_detected}")  # True
print(f"不支持的陈述：{check.unsupported_claims}")  # ["生成文本没有任何引用支持"]
```

### 4. 格式化带引用的回答

```python
from src.agent.multi_agent import format_answer_with_citations

answer = """
根据检索到的信息：

RAG 技术是一种检索增强生成技术 [Local: RAG 原理文档，p.5]。
它结合了检索和生成模型 [Web: towardsdatascience.com]。
"""

citations = [
    Citation(type=CitationType.LOCAL, source="RAG 原理文档", content="内容", page=5),
    Citation(type=CitationType.WEB, source="Towards Data Science", content="内容", url="https://towardsdatascience.com"),
]

formatted = format_answer_with_citations(
    answer=answer,
    citations=citations,
    include_reference_list=True
)

print(formatted)
# 输出:
# 根据检索到的信息：
# 
# RAG 技术是一种检索增强生成技术 [Local: RAG 原理文档，p.5]。
# 它结合了检索和生成模型 [Web: towardsdatascience.com]。
# 
# ## 引用来源
# 1. [Local: RAG 原理文档，p.5]
# 2. [Web: towardsdatascience.com]
```

---

## 📊 核心类说明

### Citation 数据类

**属性**:
- `type`: 引用类型（`CitationType.LOCAL` 或 `CitationType.WEB`）
- `source`: 来源名称（文档名或网站名）
- `content`: 引用的具体内容
- `confidence`: 置信度 (0-1)，默认 1.0
- `relevance`: 相关性 (0-1)，默认 1.0
- `url`: URL（仅 web 类型）
- `page`: 页码或段落号（可选）
- `metadata`: 额外元数据

**方法**:
- `format_citation()`: 格式化引用标记
- `to_dict()`: 转换为字典
- `from_dict()`: 从字典创建

### FaithfulnessCheck 数据类

**属性**:
- `is_faithful`: 是否忠实（布尔值）
- `hallucination_detected`: 是否检测到臆造（布尔值）
- `unsupported_claims`: 不支持的陈述列表
- `confidence`: 检查置信度 (0-1)
- `suggestions`: 改进建议列表

**方法**:
- `to_dict()`: 转换为字典

### CitationManager 类

**方法**:
- `add_citation(citation)`: 添加单个引用
- `add_citations(citations)`: 批量添加引用
- `clear()`: 清空引用列表
- `get_local_citations()`: 获取本地引用
- `get_web_citations()`: 获取联网引用
- `format_all_citations()`: 格式化所有引用
- `check_faithfulness(text)`: 检查忠实度

**静态方法**:
- `create_citations_from_results(local_results, web_results, top_k)`: 从检索结果创建引用

---

## 🔧 在多智能体系统中的使用

### 自动溯源

当使用 `MultiAgentRAG` 时，溯源是自动进行的：

```python
from src.agent.multi_agent import MultiAgentRAG

# 创建系统
rag = MultiAgentRAG(llm=llm)

# 运行查询
state = rag.run("什么是 RAG 技术？")

# 查看引用
citations = state.blackboard.get("citations", [])
print(f"引用数量：{len(citations)}")

# 查看忠实度检查
faithfulness = state.blackboard.get("faithfulness_check", {})
print(f"忠实度：{faithfulness}")

# 查看指标
print(f"生成模式：{state.metrics.get('generation_mode')}")
print(f"引用数量：{state.metrics.get('citation_count')}")
print(f"忠实度分数：{state.metrics.get('faithfulness_score')}")
print(f"检测到臆造：{state.metrics.get('hallucination_detected')}")
```

---

## 📝 引用格式说明

### 本地引用
- 有页码：`[Local: 文档名，p.5]`
- 无页码：`[Local: 文档名]`

### 联网引用
- 有 URL: `[Web: domain.com]` (自动简化 URL)
- 无 URL: `[Web: 网站名]`

---

## ⚠️ 注意事项

### 1. 置信度递减
引用按排名自动递减置信度和相关性：
```
排名 1: confidence=1.00, relevance=1.00
排名 2: confidence=0.90, relevance=0.90
排名 3: confidence=0.80, relevance=0.80
...
```

### 2. 页码处理
- 支持 `int` 和 `str` 类型
- 自动添加 `p.` 前缀（如果缺失）
- 避免重复前缀（如 `p.p.5` → `p.5`）

### 3. 忠实度检查逻辑
- **无引用** → 检测到臆造（置信度 0.0）
- **有引用但无标记** → 检测到臆造（置信度 0.3）
- **有引用有标记** → 通过检查（置信度 0.8）

---

## 🧪 测试

### 运行单元测试
```bash
python examples/test_phase3_citations.py
```

### 运行集成测试
```bash
python examples/test_phase3_integration.py
```

---

## 📚 示例代码

完整示例请参考：
- [`test_phase3_citations.py`](file://d:\projects\MODULAR-RAG-MCP-SERVER\examples\test_phase3_citations.py) - 单元测试
- [`test_phase3_integration.py`](file://d:\projects\MODULAR-RAG-MCP-SERVER\examples\test_phase3_integration.py) - 集成测试

---

## 🎓 最佳实践

### 1. 使用引用管理器
```python
# ✅ 推荐：使用 CitationManager
citations = CitationManager.create_citations_from_results(
    local_results=results,
    web_results=web_results
)

# ❌ 不推荐：手动创建每个引用
```

### 2. 检查忠实度
```python
# ✅ 推荐：生成后检查
answer = generate_answer()
faithfulness = citation_manager.check_faithfulness(answer)
if not faithfulness.is_faithful:
    # 处理低忠实度回答
```

### 3. 格式化引用
```python
# ✅ 推荐：使用 format_answer_with_citations
formatted = format_answer_with_citations(
    answer=answer,
    citations=citations,
    include_reference_list=True
)

# ❌ 不推荐：手动拼接引用
```

---

## 🔮 未来优化

### 短期
- [ ] 使用 LLM 进行更深入的忠实度检查
- [ ] 评估引用来源的权威性
- [ ] 支持一个陈述多个引用

### 长期
- [ ] 自动引用修正
- [ ] 引用图谱可视化
- [ ] 跨文档引用追踪

---

**相关文档**: [`PHASE3_COMPLETION_REPORT.md`](file://d:\projects\MODULAR-RAG-MCP-SERVER\docs\PHASE3_COMPLETION_REPORT.md)
