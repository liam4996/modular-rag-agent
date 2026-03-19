# Dashboard 多智能体对话集成指南

## 📋 概述

已成功将多智能体 RAG 系统集成到 Dashboard，提供以下功能：

1. **🤖 Agent Chat** - 交互式对话界面
2. **🧪 Test Panel** - 测试运行和结果查看
3. **📊 现有 Dashboard 功能** - 数据浏览、追踪、评估等

---

## 🚀 快速启动

### 方法 1: 使用启动脚本

```bash
python scripts/start_dashboard.py
```

### 方法 2: 直接使用 Streamlit

```bash
streamlit run src/observability/dashboard/app.py
```

### 方法 3: 指定端口

```bash
python scripts/start_dashboard.py --port 8502
```

Dashboard 将在浏览器中打开：`http://localhost:8501`

---

## 🎯 新增功能

### 1. Agent Chat (🤖)

**位置**: Dashboard 左侧菜单 → "Agent Chat"

**功能**:
- 💬 与多智能体 RAG 进行实时对话
- ⚙️ 配置路由模式（自动/本地/联网/混合）
- 📚 查看引用来源
- 📊 查看执行指标
- 🔍 查看执行轨迹

**使用步骤**:
1. 点击左侧菜单的 "Agent Chat"
2. 在侧边栏配置参数：
   - **路由模式**: 选择搜索类型
   - **检索数量**: 设置返回结果数 (1-20)
3. 在底部输入框输入问题
4. 查看 Agent 回复，包括：
   - 回答内容
   - 引用来源（可展开查看）
   - 执行指标（检索数、引用数、置信度）
   - 执行轨迹（各 Agent 的执行步骤）

**界面布局**:
```
┌─────────────────────────────────────┐
│  🤖 多智能体 RAG 对话                │
├─────────────────────────────────────┤
│  [侧边栏]          [聊天区域]       │
│  - 路由模式        👤 用户问题     │
│  - 检索数量        🤖 Agent 回答    │
│  - 清空对话         📚 引用来源    │
│  - 系统状态         📊 执行指标    │
│                    🔍 执行轨迹     │
└─────────────────────────────────────┘
```

---

### 2. Test Panel (🧪)

**位置**: Dashboard 左侧菜单 → "Test Panel"

**功能**:
- ▶️ 运行单元测试、集成测试、性能测试
- 📊 查看测试结果概览
- 📝 查看详细测试输出
- 📈 查看历史测试记录

**使用步骤**:
1. 点击左侧菜单的 "Test Panel"
2. 选择测试套件：
   - 单元测试
   - 集成测试
   - 性能测试
   - 全部
3. 点击 "▶️ 运行测试"
4. 查看测试结果：
   - 状态（成功/失败）
   - 测试数量
   - 通过/失败数量
   - 耗时
   - 详细输出

**测试结果展示**:
```
┌─────────────────────────────────────┐
│  测试结果概览                        │
├─────────────────────────────────────┤
│  ✅ 成功    测试：42    通过：42    │
│  失败：0     耗时：1.23 秒          │
├─────────────────────────────────────┤
│  📝 测试输出                         │
│  [查看详细输出 ▼]                   │
│  ...                                │
└─────────────────────────────────────┘
```

---

## 📊 现有 Dashboard 功能

### Overview (📊)
- 系统概览
- 关键指标展示

### Data Browser (🔍)
- 浏览数据
- 数据检索

### Ingestion Manager (📥)
- 数据导入管理
- 导入配置

### Ingestion Traces (🔬)
- 导入追踪
- 详细日志

### Query Traces (🔎)
- 查询历史
- 阶段瀑布图
- Dense vs Sparse 对比

### Evaluation Panel (📏)
- RAGAS 评估
- LLM-as-Judge 评分

---

## 🔧 技术实现

### 文件结构

```
src/observability/dashboard/
├── app.py                          # Dashboard 主应用
├── pages/
│   ├── overview.py                 # 概览页面
│   ├── data_browser.py             # 数据浏览
│   ├── ingestion_manager.py        # 导入管理
│   ├── ingestion_traces.py         # 导入追踪
│   ├── query_traces.py             # 查询追踪
│   ├── evaluation_panel.py         # 评估面板
│   ├── test_panel.py               # [新增] 测试面板
│   └── agent_chat.py               # [新增] Agent 对话
└── services/
    ├── trace_service.py            # 追踪服务
    ├── data_service.py             # 数据服务
    └── config_service.py           # 配置服务
```

### Agent Chat 架构

```python
# 核心组件
- MultiAgentRAG: 多智能体 RAG 系统
- AgentState: 共享状态容器
- CitationManager: 引用管理
- Citation: 引用数据类

# 工作流程
1. 用户输入问题
2. RouterAgent 分类意图
3. SearchAgent/WebAgent 检索
4. EvalAgent 评估结果
5. GenerateAgent 生成回答
6. 显示回答和引用
```

### Test Panel 架构

```python
# 测试文件
- test_phase4_comprehensive.py  # 单元测试
- test_phase4_integration.py    # 集成测试
- test_phase4_performance.py    # 性能测试

# 功能
- 运行测试并捕获输出
- 解析测试结果
- 存储历史记录
- 可视化展示
```

---

## 📝 使用示例

### Agent Chat 示例

**用户**: "RAG 是什么？"

**Agent 回答**:
```
根据检索到的信息，关于「RAG 是什么？」：

1. RAG 是检索增强生成 (Retrieval-Augmented Generation) 技术
   [Local: RAG 文档]

2. 它结合了检索和生成模型的优势
   [Local: 技术文档]

以上信息基于本地知识库。
```

**引用来源** (可展开):
```
1. [Local: RAG 文档，p.1]
   内容：RAG 是检索增强生成技术...

2. [Local: 技术文档，p.5]
   内容：结合了检索和生成模型...
```

**执行指标** (可展开):
```
检索文档数：2
引用数量：2
置信度：0.90
```

**执行轨迹** (可展开):
```
🎯 RouterAgent - classify_intent (14:30:15.123)
  {"intent": "local_search", "confidence": 0.95}

🔍 SearchAgent - retrieve (14:30:15.456)
  {"results_count": 2}

📏 EvalAgent - evaluate (14:30:15.789)
  {"relevance": 0.90, "confidence": 0.85}

✍️ GenerateAgent - generate_answer (14:30:16.012)
  {"answer_length": 256, "citations_count": 2}
```

---

## ⚙️ 配置选项

### Agent Chat 配置

| 选项 | 说明 | 可选值 | 默认值 |
|------|------|--------|--------|
| 路由模式 | 选择搜索类型 | auto, local_search, web_search, hybrid_search | auto |
| 检索数量 | 返回结果数 | 1-20 | 5 |

### Test Panel 配置

| 选项 | 说明 | 可选值 |
|------|------|--------|
| 测试套件 | 选择测试类型 | 单元测试，集成测试，性能测试，全部 |

---

## 🔍 故障排查

### Dashboard 无法启动

**问题**: `ModuleNotFoundError: No module named 'streamlit'`

**解决**:
```bash
pip install streamlit
```

### Agent 初始化失败

**问题**: `Agent not initialized`

**解决**:
1. 检查依赖是否安装：`pip install -r requirements.txt`
2. 检查模型配置是否正确
3. 查看详细错误日志

### 测试运行失败

**问题**: `Test file not found`

**解决**:
1. 确认在项目根目录运行
2. 检查测试文件是否存在：
   ```bash
   ls examples/test_phase4_*.py
   ```

---

## 📊 性能优化

### 响应时间优化

1. **缓存 Agent 实例**: 使用 `st.session_state` 缓存
2. **异步执行**: 长时间操作使用 `st.spinner`
3. **增量更新**: 使用 `st.rerun()` 而非全量刷新

### 内存优化

1. **限制历史记录**: 只保留最近 10 轮对话
2. **延迟加载**: 按需导入模块
3. **清理状态**: 提供清空对话功能

---

## 🎓 最佳实践

### 使用 Agent Chat

1. ✅ 明确表达问题
2. ✅ 利用配置选项调整行为
3. ✅ 查看引用验证答案
4. ✅ 检查执行轨迹了解内部流程

### 使用 Test Panel

1. ✅ 定期运行测试确保稳定性
2. ✅ 查看历史趋势
3. ✅ 失败时查看详细输出
4. ✅ 运行所有测试确保完整性

---

## 🚀 未来扩展

### 计划功能

1. **对话历史管理**
   - 保存/加载对话
   - 导出对话记录

2. **高级配置**
   - 模型选择
   - 温度参数
   - 超时设置

3. **可视化增强**
   - 执行流程图
   - 性能趋势图
   - 对比分析

4. **协作功能**
   - 共享对话
   - 团队协作
   - 评论标注

---

## 📚 相关文档

- [Phase 4 完成报告](PHASE4_COMPLETION_REPORT.md)
- [多智能体系统文档](MULTI_AGENT_SPEC.md)
- [Dashboard 文档](DASHBOARD.md)

---

## ✅ 总结

现在您可以通过 Dashboard 与多智能体 RAG 系统进行交互式对话，同时可以：

- 🤖 **实时对话**: 直接提问并获取带引用的答案
- 🧪 **运行测试**: 验证系统功能
- 📊 **监控性能**: 查看执行指标和轨迹
- 🔍 **调试分析**: 深入了解 Agent 工作流程

Dashboard 已成为一个完整的 RAG 系统管理和交互平台！
