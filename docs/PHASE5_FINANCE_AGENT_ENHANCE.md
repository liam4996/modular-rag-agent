# Phase 5: 智能投研增强 — 基于数据源粒度的多 Agent 研报平台

> **定位**：在现有 `MULTI_AGENT_SPEC` (v2.0，能力正交 Agent) 基础上，融合按数据源细粒度分工 + 溯源校验 + 研报模板 + 冲突检测 + 多数据源容灾 + 图表迭代优化 + 分析链推理，做一层增量增强。
>
> **核心原则**：不删现有代码，不做重构，只做**加法**。
>
> **设计哲学**：「Agent 粒度由粗到细，数据源类型决定路由精度，检索结果决定溯源力度，多数据源确保容灾，VLM 确保图表质量，CoA 确保分析深度」

---

## 更新日志

| 版本 | 日期 | 更新内容 |
|------|------|---------|
| v1.1 | 2026-05-03 | 🆕 新增 Phase 5-I 图表迭代优化、5-J 三层溯源校验、5-K 运行模式、5-L Chain-of-Analysis、5-M 章节级 Prompt 管理、5-N 流式输出+思考过程可视化 |
| v1.0 | 2026-05-03 | 初版：Phase 5-A 到 5-H 完成 |

---

## 一、现有能力盘点（全部已实现，直接复用）

### 1.1 现有 Agent 体系

| 现有 Agent | 文件 | 状态 | 对应设计中的角色 |
|-----------|------|------|----------------|
| **RouterAgent** | [router_agent.py](file:///d:/projects/MODULAR-RAG-MCP-SERVER/src/agent/multi_agent/router_agent.py) | ✅ 已识别 financial_* 系列意图 → Phase 5-H 合并入 Supervisor | 意图识别层（合并后移除节点） |
| **SupervisorAgent** | [supervisor_agent.py](file:///d:/projects/MODULAR-RAG-MCP-SERVER/src/agent/multi_agent/supervisor_agent.py) | ✅ Plan-then-Execute 编排 → Phase 5-H 接管 classify | 总控 Agent 核心 |
| **SearchAgent** (RAG) | [search_agent.py](file:///d:/projects/MODULAR-RAG-MCP-SERVER/src/agent/multi_agent/search_agent.py) | ✅ 本地知识库检索 | 财报/公告/行业数据检索入口 |
| **WebSearchAgent** | [web_agent.py](file:///d:/projects/MODULAR-RAG-MCP-SERVER/src/agent/multi_agent/web_agent.py) | ✅ 联网搜索 | 实时资讯/券商研报补充 |
| **FinanceDataAgent** | [finance_data_agent.py](file:///d:/projects/MODULAR-RAG-MCP-SERVER/src/agent/multi_agent/finance_data_agent.py) | ✅ MCP 行情查询 | 估值数据源 |
| **BusinessComputeAgent** | [business_compute_agent.py](file:///d:/projects/MODULAR-RAG-MCP-SERVER/src/agent/multi_agent/business_compute_agent.py) | ✅ 计算+可视化+报告 | 估值计算/图表/对标分析 |
| **EvalAgent** | [eval_agent.py](file:///d:/projects/MODULAR-RAG-MCP-SERVER/src/agent/multi_agent/eval_agent.py) | ✅ 质量评估 | 质量闭环 |
| **RefineAgent** | [refine_agent.py](file:///d:/projects/MODULAR-RAG-MCP-SERVER/src/agent/multi_agent/refine_agent.py) | ✅ 查询改写 | 重试机制 |

### 1.2 现有工具链

| 工具 | 文件 | 能力 |
|------|------|------|
| FinancialCalculator | [financial_calculator.py](file:///d:/projects/MODULAR-RAG-MCP-SERVER/src/agent/multi_agent/tools/financial_calculator.py) | 12 个财务指标公式 (ROE/PE/PB/同比增长等) |
| DataProcessor | [data_processor.py](file:///d:/projects/MODULAR-RAG-MCP-SERVER/src/agent/multi_agent/tools/data_processor.py) | Pandas 清洗/对比/排名/异常检测 |
| DataVisualizer | [data_visualizer.py](file:///d:/projects/MODULAR-RAG-MCP-SERVER/src/agent/multi_agent/tools/data_visualizer.py) | K线/趋势/柱状/饼图 Matplotlib 生成 |
| ReportRenderer | [report_renderer.py](file:///d:/projects/MODULAR-RAG-MCP-SERVER/src/agent/multi_agent/tools/report_renderer.py) | Jinja2 模板报告渲染 |
| CitationManager | [citation.py](file:///d:/projects/MODULAR-RAG-MCP-SERVER/src/agent/multi_agent/citation.py) | 引用管理 + 忠实度检查 |
| QueryMarketDataTool | [query_market_data.py](file:///d:/projects/MODULAR-RAG-MCP-SERVER/src/mcp_server/tools/query_market_data.py) | akshare + yfinance 行情接口 |

### 1.3 现有编排能力

- **LangGraph 13 节点**：router → supervisor → plan → retrieve → web → read → eval → refine → generate + finance_data → business_compute → aggregate → global_eval
- **并行执行**：无依赖子任务 ThreadPoolExecutor 并行
- **Blackboard 通信**：所有 Agent 读写共享状态
- **质量闭环**：Eval → Refine → Retry 自动循环
- **容错机制**：max_retries + fallback 兜底

---

## 二、设计亮点 & 现有 Gap 分析

| 设计点 | 现有实现 | Gap | 处理策略 |
|--------|---------|-----|---------|
| **财报解析 Agent** | SearchAgent 统一检索 + FinancialCalculator 计算 | 财报可路由到专用 collection | 新增 collection 路由，Supervisor 生成子任务时指定 collection=financial_reports |
| **公告事件 Agent** | 同上 | 无独立的情感分类 (利好/利空/中性) | 新增公告情感分析工具 |
| **行业对标 Agent** | BusinessComputeAgent._compare() | 无持久化的行业均值数据库 | 新增行业基准数据管理模块 |
| **估值分析 Agent** | FinancialCalculator 算 PE/PB | 无估值区间判断、无 DCF、无敏感性分析 | 新增估值分析工具函数 |
| **研报合成 Agent** | ReportRenderer + _generate_financial_report | 模板不够丰富、无段落级溯源脚注 | 扩展 ReportRenderer + 增强溯源 |
| **溯源校验 Agent** | CitationManager.check_faithfulness() | 无细粒度段落级来源校验 | 新增段落级溯源校验装饰器 |
| **按数据源分 7 类** | 无 collection 管理意识 | 数据摄入时未分类 | 新增 ingestion 阶段 collection 规划 |
| **跨 Agent 矛盾监测** | 无 | 各 Agent 结果可能相互矛盾无人发现 | 🆕 新增冲突检测节点 |
| **A 股数据源单一** | 仅 akshare | akshare 可能不稳定/限速 | 🆕 新增 Tushare 作为 A 股主数据源，akshare 为备选降级 |

---

## 三、改造后整体架构

```
                        ┌──────────────────────────────────────┐
                        │          Supervisor Agent            │
                        │  (Plan-then-Execute, 保持现有不变)    │
                        └──────────┬───────────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────────┐
          │                        │                            │
          ▼                        ▼                            ▼
┌──────────────────┐   ┌──────────────────┐   ┌────────────────────┐
│   RAG Agent      │   │  Finance Data    │   │  Web Agent         │
│  (SearchAgent)   │   │  Agent (增强)     │   │  (现有)             │
│                  │   │                  │   │                    │
│  ┌─ collection:  │   │  A股/港股/美股    │   │  实时新闻           │
│  │ financial_    │   │  行情+基本面      │   │  券商观点           │
│  │   reports     │   │                  │   │                    │
│  │ announcements │   │  🆕 Tushare 主    │   │                    │
│  │ industry_data │   │  + akshare 备      │   │                    │
│  │ research      │   │  (Tushare 限频时   │   │                    │
│  │               │   │   自动降级 akshare)  │   │                    │
│  └─ 自动检测路由   │   │                  │   │                    │
└────────┬─────────┘   └────────┬─────────┘   └────────┬───────────┘
         │                      │                       │
         └──────────────────────┼───────────────────────┘
                                ▼
          ┌──────────────────────────────────────────────┐
          │        Business Compute Agent (增强)          │
          │                                              │
          │  ┌────────────────┐  ┌──────────────────┐   │
          │  │  FinancialCalc  │  │  ValuationCalc   │ 🆕 │
          │  │   (现有 12 公式) │  │  DCF/敏感性分析   │   │
          │  └────────────────┘  └──────────────────┘   │
          │  ┌────────────────┐  ┌──────────────────┐   │
          │  │  DataProcessor  │  │  SentimentAnalyzer│ 🆕│
          │  │  (Pandas 现有)   │  │  公告情感分类      │   │
          │  └────────────────┘  └──────────────────┘   │
          │  ┌────────────────┐  ┌──────────────────┐   │
          │  │  DataVisualizer │  │  ReportRenderer  │   │
          │  │  (Matplotlib)   │  │  (Jinja2 增强)    │ 🆕│
          │  └────────────────┘  └──────────────────┘   │
          └──────────────────────┬───────────────────────┘
                                 │
                                 ▼
          ┌──────────────────────────────────────────────┐
          │       Conflict Detector 🆕                   │
          │  跨 Agent 输出矛盾检测                         │
          │                                              │
          │  检测维度:                                    │
          │  ① 数据冲突: RAG vs 行情 vs 计算值不一致       │
          │  ② 逻辑冲突: 营收增长但股价跌 → 需要解释        │
          │  ③ 情绪冲突: 新闻利好但基本面利空               │
          │  ④ 基准冲突: 行业PE均值 vs LLM判断不一致       │
          │                                              │
          │  输出: conflict_flags + warning_labels       │
          └──────────────────────┬───────────────────────┘
                                 │
                                 ▼
          ┌──────────────────────────────────────────────┐
          │            Generate Agent (增强)              │
          │                                              │
          │  1. 收集所有 blackboard 数据                  │
          │  2. 分析研报类型 → 选模板                      │
          │  3. 读取冲突标记 → 写入"风险提示"章节 🆕      │
          │  4. 生成时自动绑定来源 (段落级溯源) 🆕         │
          │  5. LLM 撰写 → 嵌入图表引用                    │
          │  6. 溯源校验 → 数据校验 → 输出最终研报        │
          └──────────────────────────────────────────────┘
```

---

## 四、分阶段实施计划

### Phase 5-A: 数据源层 Collection 规划（按数据源分工的基础）

#### 目标

让 7 类数据源在 RAG Server 中有对应的独立 collection，Supervisor 能路由到正确的数据集合。

#### 4A.1 Collection 命名规范

```yaml
# config/settings.yaml 新增 finance 配置段
finance:
  collections:
    financial_reports:    # 上市公司财报（PDF）
      description: "年度/季度财报、财务摘要"
      chunk_size: 1024
    announcements:        # 交易所公告
      description: "公司公告、重大事件、澄清说明"
      chunk_size: 512
    research_reports:     # 券商公开研报
      description: "卖方研报、行业深度报告"
      chunk_size: 1024
    industry_data:        # 行业景气度数据
      description: "行业平均估值、景气指数、产业链数据"
      chunk_size: 512
    peer_comparison:      # 同业对标数据
      description: "竞品公司财务摘要、市场份额"
      chunk_size: 512
    private_research:     # 私有投研文档
      description: "内部调研纪要、投研笔记"
      chunk_size: 1024
```

#### 4A.2 数据摄入扩展

```python
# scripts/ingest_finance.py (新增)
"""金融文档批量摄取脚本。

按文档类型自动路由到对应 collection。

用法:
  python scripts/ingest_finance.py --path data/finance/reports/宁德时代_2024Q3.pdf
  python scripts/ingest_finance.py --dir data/finance/announcements/ --collection announcements
"""
```

#### 4A.3 文件变更

| 操作 | 文件 | 说明 |
|------|------|------|
| 🆕 新增 | `scripts/ingest_finance.py` | 金融文档批量摄取脚本 |
| 🆕 新增 | `src/agent/multi_agent/tools/collection_router.py` | 根据 query 决定检索哪些 collection |
| ✏️ 修改 | `config/settings.yaml` | 新增 finance.collections 配置段 |

---

### Phase 5-B: 路由层细化（按数据源粒度分配子任务的核心）

#### 目标

Supervisor 在拆解子任务时，能给 RAG Agent 指定具体的 `collection`。

#### 4B.1 CollectionRouter 工具

```python
# src/agent/multi_agent/tools/collection_router.py


class CollectionRouter:
    """
    输入：用户 query + intent
    输出：要检索的 collection 列表
    """

    RULES = [
        (["财报", "利润", "营收", "净利润", "毛利率", "ROE", "Q1", "Q2", "Q3", "Q4", "半年报", "年报"], 0.7, "financial_reports"),
        (["公告", "重大", "提示", "停牌", "复牌", "减持", "增持", "分红", "股权"], 0.7, "announcements"),
        (["行业", "景气", "产业链", "上下游", "渗透率", "市场规模"], 0.6, "industry_data"),
        (["研报", "券商", "推荐", "评级", "目标价", "买入", "增持"], 0.6, "research_reports"),
        (["对比", "对标", "竞品", "行业平均", "同行", "同业", "排名"], 0.7, "peer_comparison"),
        (["估值", "PE", "PB", "PS", "DCF", "目标价", "估值区间", "贵", "便宜"], 0.7, "financial_reports"),
        (["内部", "调研", "纪要", "私有", "尽调"], 0.8, "private_research"),
    ]

    def route(self, query: str, intent: str) -> Dict[str, Any]:
        matched = []
        for keywords, weight, collection in self.RULES:
            score = sum(1 for kw in keywords if kw in query) / len(keywords)
            if score >= weight:
                matched.append((score, collection))

        intent_collection_map = {
            "financial_analysis": ["financial_reports", "industry_data", "peer_comparison"],
            "financial_market": ["research_reports"],
            "financial_report": ["financial_reports", "announcements", "industry_data"],
        }

        all_collections = list(set(
            [c for _, c in sorted(matched, reverse=True)] + intent_collection_map.get(intent, [])
        ))
        return {"collections": all_collections, "primary": all_collections[0] if all_collections else "financial_reports"}
```

#### 4B.2 文件变更

| 操作 | 文件 | 说明 |
|------|------|------|
| 🆕 新增 | `src/agent/multi_agent/tools/collection_router.py` | Collection 路由决策器 |
| ✏️ 修改 | `src/agent/multi_agent/supervisor_agent.py` | `_plan_financial()` 中调用 CollectionRouter |
| ✏️ 修改 | `src/agent/multi_agent/multi_agent_system.py` | `_retrieve_node` 透传 collection 参数 |

---

### Phase 5-C: 研报模板系统增强（研报合成 Agent 的实现）

#### 4C.1 多模板体系

```python
TEMPLATES = {
    "earnings_review": """# {{ symbol }} {{ period }}财报分析

**生成时间**: {{ generated_at }}

## 投资要点
{{ investment_highlights }}

## 核心财务指标
| 指标 | 本期 | 同比 | 行业均值 | 评价 |
|------|------|------|---------|------|
{% for row in financial_highlights_table %}
| {{ row.name }} | {{ row.value }} | {{ row.yoy }} | {{ row.industry_avg }} | {{ row.rating }} |
{% endfor %}

## 营收与利润分析
{{ revenue_analysis }}

## 估值分析
{{ valuation_analysis }}

{% if conflict_warnings %}
## ⚠️ 数据异常提示
{% for w in conflict_warnings %}
- {{ w }}
{% endfor %}
{% endif %}

## 风险提示
{% for risk in risk_factors %}
- {{ risk }}
{% endfor %}

## 图表
{% for chart in charts %}
![{{ chart.title }}]({{ chart.path }})
{% endfor %}
---
*报告由 AI 投研助手自动生成，数据来源见正文引用标记*
""",

    "industry_comparison": """# 行业对比分析: {{ industry_name }}
...
""",

    "event_impact": """# 重大事件影响分析
...
""",

    "valuation_report": """# 估值分析报告
...
""",
}
```

#### 4C.2 段落级溯源脚注 🆕

```python
# src/agent/multi_agent/tools/citation_footnoter.py (新增)


class CitationFootnoter:
    """
    在 LLM 生成研报正文后，逐段扫描，
    为每个数据点绑定对应的 RAG 检索原文引用。
    """

    def process(self, report_text: str, source_docs: List[Dict]) -> str:
        paragraphs = report_text.split("\n\n")
        footnoted = []
        for para in paragraphs:
            footnote = self._find_source_for_paragraph(para, source_docs)
            if footnote:
                footnoted.append(f"{para}\n\n> 📖 {footnote}")
            else:
                footnoted.append(para)
        return "\n\n".join(footnoted)

    def _find_source_for_paragraph(self, para: str, docs: List[Dict]) -> str:
        """提取段落中的关键数字/指标，与 doc content 匹配"""
        ...
```

#### 4C.3 文件变更

| 操作 | 文件 | 说明 |
|------|------|------|
| 🆕 新增 | `src/agent/multi_agent/tools/citation_footnoter.py` | 段落级溯源脚注生成 |
| ✏️ 修改 | `src/agent/multi_agent/tools/report_renderer.py` | 扩展多模板 + 集成 Footnoter + conflict_warnings 字段 |

---

### Phase 5-D: 增强溯源校验（溯源校验的实现）

#### 4D.1 Generate 阶段的溯源校验三步走

```python
def _generate_financial_report(self, state):
    context = self._collect_all_context(state)
    report_text = self._llm_invoke_with_context(context, state)

    # 🆕 三层校验
    verified_report = self._verify_and_annotate(
        report_text=report_text,
        source_docs=state.local_results,
        market_data=state.blackboard.get("market_data", {}),
        computed=state.blackboard.get("computed_results", {}),
    )

    state.final_answer = verified_report
    return state


def _verify_and_annotate(self, report_text, source_docs, market_data, computed):
    """
    Layer 1 — 忠实度检查 (CitationManager.check_faithfulness)
    Layer 2 — 段落溯源 (CitationFootnoter)
    Layer 3 — 数据校验 (DataVerifier)
    """
```

#### 4D.2 数据校验器

```python
# src/agent/multi_agent/tools/data_verifier.py


class DataVerifier:
    """
    检查 LLM 生成的财务数字与实际计算值是否一致。
    不一致时自动修正并用 ⚠️ 标记。

    例:
      LLM: "ROE 为 19.5%"
      实际: 18.2%
      → 修正: "ROE 为 18.2% ⚠️原述19.5%已校正"
    """

    def verify(self, report: str, computed: Dict) -> str:
        import re
        corrections = []
        for cat, metrics in computed.get("metrics", {}).items():
            if isinstance(metrics, dict):
                for metric_name, actual_value in metrics.items():
                    if actual_value is None:
                        continue
                    # 扫描报告中该指标的数字
                    pattern = rf"{metric_name}\s*[:：]?\s*[\d]+(?:\.[\d]+)?%?"
                    for match in re.finditer(pattern, report):
                        ...  # 解析并比对
        return report  # 应用修正
```

#### 4D.3 文件变更

| 操作 | 文件 | 说明 |
|------|------|------|
| ✏️ 修改 | `src/agent/multi_agent/multi_agent_system.py` | `_generate_financial_report` 增加校验调用 |
| 🆕 新增 | `src/agent/multi_agent/tools/data_verifier.py` | 数据校验器 |

---

### Phase 5-E: 公告情感分析 & 行业基准数据管理

#### 5E.1 公告情感分析 🆕

```python
# src/agent/multi_agent/tools/sentiment_analyzer.py


class SentimentAnalyzer:
    POSITIVE_KEYWORDS = ["回购", "增持", "分红", "业绩预增", "中标", "获批", "战略合作", "订单"]
    NEGATIVE_KEYWORDS = ["减持", "预亏", "违规", "处罚", "诉讼", "退市", "ST", "资产冻结"]

    def classify(self, title: str, content: str = "") -> Dict:
        text = f"{title} {content}"
        pos_score = sum(1 for kw in self.POSITIVE_KEYWORDS if kw in text)
        neg_score = sum(1 for kw in self.NEGATIVE_KEYWORDS if kw in text)
        if pos_score > neg_score:
            return {"sentiment": "利好", "confidence": round(pos_score / (pos_score + neg_score + 1), 2)}
        elif neg_score > pos_score:
            return {"sentiment": "利空", "confidence": round(neg_score / (pos_score + neg_score + 1), 2)}
        else:
            return {"sentiment": "中性", "confidence": 0.5}
```

#### 5E.2 行业基准数据管理 🆕

```python
# src/agent/multi_agent/tools/industry_benchmark.py


class IndustryBenchmark:
    DEFAULT_BENCHMARKS = {
        "新能源电池": {"pe_avg": 25.0, "pb_avg": 3.5, "roe_avg": 15.0, "gross_margin_avg": 25.0},
        "光伏":       {"pe_avg": 20.0, "pb_avg": 2.5, "roe_avg": 12.0, "gross_margin_avg": 20.0},
        "半导体":     {"pe_avg": 45.0, "pb_avg": 5.0, "roe_avg": 10.0, "gross_margin_avg": 35.0},
        "白酒":       {"pe_avg": 30.0, "pb_avg": 8.0, "roe_avg": 25.0, "gross_margin_avg": 75.0},
        "银行":       {"pe_avg": 6.0,  "pb_avg": 0.6, "roe_avg": 10.0, "gross_margin_avg": None},
        "医药":       {"pe_avg": 35.0, "pb_avg": 4.0, "roe_avg": 12.0, "gross_margin_avg": 60.0},
    }

    def get_benchmark(self, industry: str) -> Dict:
        return self.DEFAULT_BENCHMARKS.get(industry, {})

    def compare_to_benchmark(self, metrics: Dict, industry: str) -> Dict:
        """将公司指标与行业基准对比，返回评价"""
        ...
```

#### 5E.3 文件变更

| 操作 | 文件 | 说明 |
|------|------|------|
| 🆕 新增 | `src/agent/multi_agent/tools/sentiment_analyzer.py` | 公告情感分类器 |
| 🆕 新增 | `src/agent/multi_agent/tools/industry_benchmark.py` | 行业基准数据管理 |

---

### Phase 5-F: 估值分析计算模块（估值分析 Agent 的实现）

#### 5F.1 估值分析工具

```python
# financial_calculator.py 中新增

@dataclass
class ValuationResult:
    pe: Optional[float] = None
    pb: Optional[float] = None
    ps: Optional[float] = None
    ev_ebitda: Optional[float] = None
    peg: Optional[float] = None
    industry_pe_avg: Optional[float] = None
    pe_percentile: Optional[float] = None
    valuation_assessment: str = ""  # "低估"/"合理"/"高估"
    target_price: Optional[float] = None
    target_price_lower: Optional[float] = None
    target_price_upper: Optional[float] = None


def assess_valuation(pe, pb, industry_pe_avg, historical_pe_low=0, historical_pe_high=100, growth_rate=0, eps=0) -> ValuationResult:
    """
    PE < 行业均值×0.7 → 低估
    PE 行业均值×0.7~1.3 → 合理
    PE > 行业均值×1.3 → 高估
    """
    ...


def sensitivity_analysis(base_eps, growth_rates, discount_rates) -> Dict:
    """DCF 敏感性分析矩阵"""
    ...
```

#### 5F.2 文件变更

| 操作 | 文件 | 说明 |
|------|------|------|
| ✏️ 修改 | `src/agent/multi_agent/tools/financial_calculator.py` | 新增 `ValuationResult`、`assess_valuation()`、`sensitivity_analysis()` |
| ✏️ 修改 | `src/agent/multi_agent/business_compute_agent.py` | 增加估值分析调用分支 |

---

### Phase 5-G: 跨 Agent 冲突检测 + 多数据源容灾 🆕

#### 目标

参考 FinSight 的 Conflict Detection 机制，在 aggregate 节点之后增加跨 Agent 输出矛盾检测，以及 A 股数据源的双活容灾。

#### 5G.1 冲突检测器

```python
# src/agent/multi_agent/tools/conflict_detector.py (新增)
"""
跨 Agent 输出矛盾检测器。

在 aggregate 节点后运行，检测 4 类冲突:
1. 数据冲突: RAG 提取的财报数据 vs FinanceData 的行情数据不一致
2. 逻辑冲突: 营收增长但股价暴跌 → 需要 LLM 给出解释
3. 情绪冲突: 新闻分析师看多但基本面指标看空
4. 基准冲突: LLM 判断"估值合理"但 PE 远高于行业均值
"""

from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ConflictFlag:
    """一条冲突标记"""
    dimension: str           # "data" / "logic" / "sentiment" / "benchmark"
    severity: str            # "high" / "medium" / "low"
    source_a: str            # 冲突方 A (如 "RAG Agent")
    source_b: str            # 冲突方 B (如 "Finance Data Agent")
    claim_a: str             # A 的表述
    claim_b: str             # B 的表述
    explanation: str         # 可能的解释 / 建议 LLM 关注点


class ConflictDetector:
    """
    用法:
      detector = ConflictDetector()
      flags = detector.detect(state.blackboard)
      state.blackboard["conflict_flags"] = flags
    """

    def detect(self, blackboard: Dict[str, Any]) -> List[ConflictFlag]:
        flags = []
        flags.extend(self._check_revenue_vs_price(blackboard))
        flags.extend(self._check_pe_vs_judgment(blackboard))
        flags.extend(self._check_sentiment_vs_fundamental(blackboard))
        flags.extend(self._check_growth_vs_valuation(blackboard))
        return flags

    def _check_revenue_vs_price(self, blackboard: Dict) -> List[ConflictFlag]:
        """
        检测维度: 营收增长 vs 股价表现

        RAG 财报说营收 +20%，但股价最近跌了 30%。
        → 高优先级冲突，需要 LLM 解释(行业利空/估值消化/市场情绪等)
        """
        flags = []
        local = blackboard.get("local_results", [])
        market = blackboard.get("market_data", {})

        revenue_growth = self._extract_revenue_growth(local)
        price_change = self._extract_price_change(market)

        if revenue_growth is not None and price_change is not None:
            if revenue_growth > 10 and price_change < -10:
                flags.append(ConflictFlag(
                    dimension="logic",
                    severity="high",
                    source_a="RAG Agent (财报)",
                    source_b="Finance Data Agent (行情)",
                    claim_a=f"营收同比增长 {revenue_growth:+.1f}%",
                    claim_b=f"股价同期下跌 {price_change:+.1f}%",
                    explanation="营收增长但股价下跌，可能原因：行业估值中枢下移、市场提前消化预期、"
                                "财报存在一次性收益、市场关注非财务风险因素。建议研报中重点分析。"
                ))
        return flags

    def _check_pe_vs_judgment(self, blackboard: Dict) -> List[ConflictFlag]:
        """
        检测维度: 实际 PE 与行业均值 + LLM 判断

        行业 PE 均值 25x，公司 PE 为 45x，但 LLM 说"估值合理"。
        → 中优先级冲突，标记不一致
        """
        flags = []
        computed = blackboard.get("computed_results", {})
        metrics = computed.get("metrics", {})

        for symbol, m in metrics.items():
            pe = m.get("valuation", {}).get("pe")
            industry_pe = m.get("valuation", {}).get("industry_pe_avg")

            # 通过 IndustryBenchmark 获取行业基准
            if pe is not None and industry_pe is not None:
                if pe > industry_pe * 1.3:
                    flags.append(ConflictFlag(
                        dimension="data",
                        severity="medium",
                        source_a="Business Compute Agent (计算)",
                        source_b="IndustryBenchmark (行业基准)",
                        claim_a=f"{symbol} PE = {pe:.1f}x",
                        claim_b=f"行业均值 PE = {industry_pe:.1f}x",
                        explanation=f"PE 超出行业均值 {((pe/industry_pe)-1)*100:.0f}%，需要 LLM 判断是否合理。"
                    ))
        return flags

    def _check_sentiment_vs_fundamental(self, blackboard: Dict) -> List[ConflictFlag]:
        """
        检测维度: 新闻情绪 vs 基本面

        新闻全是利好(回购/中标)，但基本面指标(ROE/毛利率)在下降。
        """
        flags = []
        computed = blackboard.get("computed_results", {})
        sentiment = computed.get("sentiment", {})
        metrics = computed.get("metrics", {})

        for symbol, m in metrics.items():
            roe = m.get("profitability", {}).get("roe")
            sentiment_label = sentiment.get("label")

            if roe is not None and sentiment_label == "利好":
                if roe < 5:
                    flags.append(ConflictFlag(
                        dimension="sentiment",
                        severity="medium",
                        source_a="SentimentAnalyzer (新闻情绪)",
                        source_b="Business Compute Agent (基本面)",
                        claim_a=f"新闻情绪: {sentiment_label}",
                        claim_b=f"ROE = {roe:.1f}% (低于5%)",
                        explanation="新闻面利好但基本面弱势，建议关注是否短期炒作。"
                    ))
        return flags

    def _check_growth_vs_valuation(self, blackboard: Dict) -> List[ConflictFlag]:
        """
        检测维度: 成长性 vs 估值 (PEG 矛盾)

        PEG > 2 说明高估，但 LLM 可能说"成长性良好"。
        """
        flags = []
        computed = blackboard.get("computed_results", {})
        metrics = computed.get("metrics", {})

        for symbol, m in metrics.items():
            pe = m.get("valuation", {}).get("pe")
            growth = m.get("growth", {}).get("revenue_yoy")

            if pe is not None and growth is not None and growth > 0:
                peg = pe / growth
                if peg > 2:
                    flags.append(ConflictFlag(
                        dimension="benchmark",
                        severity="low",
                        source_a="PEG 指标",
                        source_b="估值合理性",
                        claim_a=f"PEG = {peg:.2f} (> 2.0)",
                        claim_b=f"PE = {pe:.1f}x, 增长率 = {growth:.1f}%",
                        explanation="PEG > 2 表明估值偏高，需 LLM 判断是否有特殊溢价逻辑。"
                    ))
        return flags

    @staticmethod
    def _extract_revenue_growth(local_results: List[Dict]) -> Optional[float]:
        """从 RAG 检索的财报原文中提取营收增长率"""
        for doc in local_results:
            content = doc.get("content", "")
            import re
            match = re.search(r"营收(?:同比)?(?:增长|下降|增加|减少)?[：:]?\s*([+-]?\d+\.?\d*)%", content)
            if match:
                val = float(match.group(1))
                if "下降" in content or "减少" in content:
                    val = -val
                return val
        return None

    @staticmethod
    def _extract_price_change(market_data: Dict) -> Optional[float]:
        """从行情数据中提取涨跌幅"""
        for q in market_data.get("quote", []):
            if q.get("change_pct") is not None:
                return q["change_pct"]
        return None
```

#### 5G.2 在 LangGraph 中的位置

```
business_compute 完成
        │
        ▼
┌──────────────────────┐
│  aggregate 节点       │  ← 现有，收集所有 Agent 输出
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  conflict_detect 🆕  │  ← 新增节点
│                      │
│  1. 运行所有检测规则   │
│  2. 写入 conflict_    │
│     flags 到 blackboa │
│     rd               │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  global_eval (原有)   │  ← 质量检查
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  generate             │
│                      │
│  读取 conflict_flags  │  ← 冲突标记写入"数据异常提示"章节
│  来源: blackboard     │
└──────────────────────┘
```

#### 5G.3 Tushare 集成：A 股双数据源容灾 🆕

**现状问题**：A 股行情目前仅依赖 akshare，这是一个免费接口，存在以下风险：
- 请求频率限制（QPS 约 1-2）
- 偶发超时/空数据返回
- 无稳定的基本面历史数据接口

**解决方案**：引入 Tushare 作为 **A 股主数据源**，akshare 作为备选降级。查询 A 股数据时，按优先级降级：

```
查询 A 股数据:
  1. Tushare (主) → 成功? → 返回
  2. Tushare 限频/失败? → akshare (备) → 返回
  3. 两者都失败? → 返回空 + log warning
```

##### settings.yaml 新增配置

```yaml
# config/settings.yaml finance 配置段扩展
finance:
  market_data:
    a_share:
      providers:
        primary: "tushare"
        fallback: "akshare"
      tushare_token: "3664fe220cb675ae1661e7ad96c51e2592a0ef72c93d29da3d65b692"  # 用户提供的 token
      enabled: true
    us_hk:
      provider: "yfinance"
      enabled: true
```

##### query_market_data.py Tushare 实现

```python
# src/mcp_server/tools/query_market_data.py 中新增

class TushareClient:
    """
    Tushare A 股数据客户端。
    
    作为 A 股主数据源，akshare 作为备选降级。
    
    API Token: 从 settings.yaml 或环境变量 TUSHARE_API_TOKEN 读取
    """

    def __init__(self, token: Optional[str] = None):
        self._token = token
        self._api = None

    @property
    def api(self):
        if self._api is None:
            try:
                import tushare as ts
                self._api = ts.pro_api(self._token)
            except ImportError:
                logger.warning("tushare not installed, A-share fallback unavailable")
                self._api = None
        return self._api

    @property
    def available(self) -> bool:
        return self.api is not None

    def fetch_quote(self, symbols: List[str]) -> List[MarketQuote]:
        """
        Tushare 实时行情。
        
        接口: ts.pro_api().daily(trade_date=...)
        """
        if not self.available:
            return []

        # Tushare 不支持批量实时行情，逐个查询
        results = []
        clean_symbols = {s.replace(".SZ", "").replace(".SH", "") for s in symbols}
        for code in clean_symbols:
            try:
                df = self.api.daily(ts_code=f"{code}.{'SZ' if code.startswith(('0','3')) else 'SH'}")
                if df is not None and not df.empty:
                    row = df.iloc[0]
                    suffix = ".SZ" if code.startswith(("0", "3")) else ".SH"
                    results.append(MarketQuote(
                        symbol=f"{code}{suffix}",
                        name="",
                        price=_safe_float(row.get("close")),
                        change_pct=_safe_float(row.get("pct_chg")),
                        volume=_safe_float(row.get("vol")),
                        currency="CNY",
                    ))
            except Exception as e:
                logger.warning(f"Tushare quote failed for {code}: {e}")
        return results

    def fetch_fundamentals(self, symbols: List[str]) -> List[MarketFundamentals]:
        """
        Tushare 基本面数据。
        
        接口: ts.pro_api().fina_indicator(ts_code=...)
        """
        if not self.available:
            return []

        results = []
        clean_symbols = {s.replace(".SZ", "").replace(".SH", "") for s in symbols}
        for code in clean_symbols:
            try:
                ts_code = f"{code}.{'SZ' if code.startswith(('0','3')) else 'SH'}"
                df = self.api.fina_indicator(ts_code=ts_code, limit=1)
                if df is not None and not df.empty:
                    row = df.iloc[0]
                    suffix = ".SZ" if code.startswith(("0", "3")) else ".SH"
                    results.append(MarketFundamentals(
                        symbol=f"{code}{suffix}",
                        pe=_safe_float(row.get("pe")),
                        pb=_safe_float(row.get("pb")),
                        roe=_safe_float(row.get("roe")),
                        eps=_safe_float(row.get("eps")),
                        revenue=_safe_float(row.get("revenue")),
                        net_profit=_safe_float(row.get("net_profit")),
                        gross_margin=_safe_float(row.get("gross_margin")),
                        debt_to_asset=_safe_float(row.get("debt_to_assets")),
                        currency="CNY",
                    ))
            except Exception as e:
                logger.warning(f"Tushare fundamentals failed for {code}: {e}")
        return results

    def fetch_history(self, symbols: List[str], period: str) -> List[MarketHistory]:
        """
        Tushare 历史 K 线数据。
        
        接口: ts.pro_api().daily(ts_code=..., start_date=..., end_date=...)
        """
        if not self.available:
            return []

        results = []
        clean_symbols = {s.replace(".SZ", "").replace(".SH", "") for s in symbols}
        for code in clean_symbols:
            try:
                ts_code = f"{code}.{'SZ' if code.startswith(('0','3')) else 'SH'}"

                # 根据 period 折算日期范围
                from datetime import datetime, timedelta
                end = datetime.now()
                period_days = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "3y": 1095, "5y": 1825}
                start = end - timedelta(days=period_days.get(period, 90))

                df = self.api.daily(
                    ts_code=ts_code,
                    start_date=start.strftime("%Y%m%d"),
                    end_date=end.strftime("%Y%m%d"),
                )
                if df is not None and not df.empty:
                    data = []
                    for _, row in df.iterrows():
                        data.append({
                            "date": str(row.get("trade_date", "")),
                            "open": _safe_float(row.get("open")),
                            "high": _safe_float(row.get("high")),
                            "low": _safe_float(row.get("low")),
                            "close": _safe_float(row.get("close")),
                            "volume": _safe_float(row.get("vol")),
                        })
                    suffix = ".SZ" if code.startswith(("0", "3")) else ".SH"
                    results.append(MarketHistory(
                        symbol=f"{code}{suffix}",
                        period=period,
                        data=data,
                        data_count=len(data),
                    ))
            except Exception as e:
                logger.warning(f"Tushare history failed for {code}: {e}")
        return results
```

##### QueryMarketDataTool 的双源降级逻辑

```python
# query_market_data.py — _fetch_a_share_quote 改造示例

def __init__(self, settings=None, config=None):
    self._settings = settings
    self._config = config or QueryMarketDataConfig()
    self._tushare = TushareClient(token=self._get_tushare_token())  # 🆕 主数据源

def _get_tushare_token(self) -> Optional[str]:
    """从 settings 或环境变量读取 Tushare token"""
    try:
        return self.settings.get("finance.market_data.a_share.tushare_token",
                                 os.getenv("TUSHARE_API_TOKEN"))
    except:
        return os.getenv("TUSHARE_API_TOKEN")

def _fetch_a_share_quote(self, symbols: List[str]) -> List[MarketQuote]:
    """A 股行情 — 双源降级: Tushare (主) → akshare (备) → 空"""
    # 1. 主数据源: Tushare
    tushare_results = self._tushare.fetch_quote(symbols)
    if tushare_results:
        return tushare_results
    logger.info("Tushare returned empty/limited, falling back to akshare")

    # 2. 备选降级: akshare
    try:
        import akshare as ak
        results = ...  # 原有 akshare 逻辑
        if results:
            return results
    except Exception as e:
        logger.warning(f"akshare also failed ({e})")

    return []

# 同理改造 _fetch_a_share_fundamentals, _fetch_a_share_history
```

#### 5G.4 Generate 节点中的冲突标记注入

```python
# multi_agent_system.py _generate_financial_report 中的新增逻辑

def _generate_financial_report(self, state):
    context = self._collect_all_context(state)

    # 🆕 读取冲突标记
    conflict_flags = state.blackboard.get("conflict_flags", [])
    if conflict_flags:
        conflict_summary = self._summarize_conflicts(conflict_flags)
        # 注入到研报上下文中，作为 "⚠️ 数据异常提示" 章节的数据源
        context["conflict_warnings"] = [
            f"[{f.severity.upper()}] {f.claim_a} vs {f.claim_b} — {f.explanation}"
            for f in conflict_flags
        ]
        state.add_metric("conflict_count", len(conflict_flags))

    report_text = self._llm_invoke_with_context(context, state)
    verified_report = self._verify_and_annotate(...)
    state.final_answer = verified_report
    return state
```

#### 5G.5 文件变更

| 操作 | 文件 | 说明 |
|------|------|------|
| 🆕 新增 | `src/agent/multi_agent/tools/conflict_detector.py` | 跨 Agent 冲突检测器 |
| 🆕 新增 | `src/agent/multi_agent/tools/tushare_client.py` | Tushare A 股数据客户端 |
| ✏️ 修改 | `src/mcp_server/tools/query_market_data.py` | 双源降级 + 集成 Tushare |
| ✏️ 修改 | `src/agent/multi_agent/multi_agent_system.py` | 新增 `_conflict_detect_node` + Generate 集成 |
| ✏️ 修改 | `config/settings.yaml` | 新增 Tushare token + 多 Provider 配置 |
| 🆕 新增 | `.env` 新增 `TUSHARE_API_TOKEN=3664fe220cb675ae1661e7ad96c51e2592a0ef72c93d29da3d65b692` | 环境变量（不提交 git） |

---

### Phase 5-H: RouterAgent 与 SupervisorAgent 合并 🆕

#### 目标

RouterAgent 的职责是**意图识别 + 路由决策**，SupervisorAgent 的职责是**任务拆解 + 执行编排**。当前流程中 Router 先跑一次 LLM 分类意图，Supervisor 再跑一次 LLM 拆解任务——两次 LLM 调用中有大量重复（Router 已识别 financial 意图，Supervisor 再读一次 query 做规划）。

**合并后**：SupervisorAgent 直接接管意图识别，一次 LLM 调用完成"分类 + 规划"。

#### 5H.1 合并前后的对比

```
合并前:                                   合并后:
                                          ┌──────────────────┐
用户 query → Router (LLM 1次)              │  User Query      │
  ├─ intent: financial_analysis            └────────┬─────────┘
  ├─ needs_local: true                              │
  └─ needs_web: false               Supervisor (LLM 1次，新增)
                                      ↓                    ┌──────────────────┐
用户 query → Supervisor (LLM 1次)     ┌──────────────────┐ │  Plan Prompt     │
  ├─ 读取 intent                     │  classify+plan    │ │  (原 Router 的    │
  ├─ 拆子任务                         │   (一次调用)       │ │  意图识别 +       │
  └─ 写入 task_plan                   └────────┬─────────┘ │  原 Supervisor 的 │
                                               │            │  任务拆解)        │
总共: 2 次 LLM 调用 (1 Router + 1 Supervisor)  │            └──────────────────┘
                                                │
                                     ┌──────────┴──────────┐
                                     │  subtasks + routing  │
                                     └─────────────────────┘
                                     
总共: 1 次 LLM 调用 (合并后的 Supervisor)
```

#### 5H.2 Plan Prompt 合并

将原 RouterAgent 的 `SYSTEM_PROMPT` 中的意图分类逻辑合并入 SupervisorAgent 的 `PLAN_PROMPT`：

```python
# supervisor_agent.py — 合并后的 PLAN_PROMPT

PLAN_PROMPT = """You are both an intent classifier and a task planner
for a multi-agent financial analysis system.

## Step 1: 意图分类
Classify the user query into one of:
- chat: simple chitchat / general conversation
- document_qa: asks about uploaded/local documents or internal knowledge
- web_search: needs latest/public information from the internet
- financial_analysis: analyze financial reports, stock performance, company fundamentals
- financial_market: real-time market data, stock prices
- financial_report: generate a structured financial report or visualization

## Step 2: 任务拆解
If financial intent, decompose into subtasks.
Available agent types: document_search, web_search, financial_market, financial_computation

Output JSON:
{
    "intent": "financial_analysis",
    "needs_local": true,
    "needs_web": false,
    "complexity": "complex",
    "reasoning": "why this classification",
    "task_plan": {
        "subtasks": [
            {
                "id": "task_1",
                "type": "document_search",
                "description": "...",
                "query": "...",
                "depends_on": []
            }
        ]
    }
}

For non-financial intents, task_plan can be null or a single retrieve task.
"""
```

#### 5H.3 SupervisorAgent 新增 classify 方法

```python
# supervisor_agent.py 中新增

class SupervisorAgent:
    """
    Merged with RouterAgent.
    Handles both intent classification and task planning in one LLM call.
    """

    def classify_and_plan(self, state: AgentState) -> AgentState:
        """
        一次 LLM 调用完成:
        1. 意图分类 (原 RouterAgent 职责)
        2. 路由决策 (原 RouterAgent 职责)
        3. 任务拆解 (原 SupervisorAgent 职责)
        """
        response = self._plan_chain.invoke({"query": state.user_input})

        parsed = json.loads(self._extract_json(response.content))

        # 写回意图和路由信息
        intent = parsed.get("intent", "document_qa")
        state.add_to_blackboard("intent", intent, "supervisor")
        state.add_to_blackboard("routing_decision", {
            "needs_local": parsed.get("needs_local", True),
            "needs_web": parsed.get("needs_web", False),
            "complexity": parsed.get("complexity", "simple"),
        }, "supervisor")

        # 写回任务计划
        task_plan = parsed.get("task_plan")
        if task_plan:
            state.add_to_blackboard("task_plan", task_plan, "supervisor")

        state.add_execution_trace({
            "agent": "supervisor",
            "action": "classify_and_plan",
            "intent": intent,
            "task_count": len(task_plan.get("subtasks", [])) if task_plan else 0,
        })

        return state
```

#### 5H.4 LangGraph 图结构调整

```python
# multi_agent_system.py 中

# 合并前:
workflow.add_node("router", self._router_node)      # RouterAgent 节点
workflow.add_node("supervisor", self._supervisor_node)  # Supervisor 节点
workflow.set_entry_point("router")

# 合并后:
# workflow.add_node("router", ...)  ← 删除
workflow.add_node("supervisor", self._supervisor_node)  # 合并后的节点，内置 classify
workflow.set_entry_point("supervisor")  # 直接从 Supervisor 开始
```

#### 5H.5 向后兼容性

| 改动 | 影响范围 | 非金融场景影响 |
|------|---------|--------------|
| 删除 `router` LangGraph 节点 | `multi_agent_system.py` | 无 — Supervisor 直接处理所有 intent |
| RouterAgent 类保留（不删） | `router_agent.py` | 无 — 保留不动，外部引用仍有效 |
| `_route_after_router` 条件边删除 | `multi_agent_system.py` | 无 — 逻辑移入 Supervisor |
| Supervisor PLAN_PROMPT 扩展 | `supervisor_agent.py` | 无 — 非金融意图直接返回 null task_plan |

**RouterAgent 类文件保留不删除**，确保已存在的测试用例和外部引用不报错。

#### 5H.6 文件变更

| 操作 | 文件 | 说明 |
|------|------|------|
| ✏️ 修改 | `src/agent/multi_agent/supervisor_agent.py` | PLAN_PROMPT 合并意图分类；新增 `classify_and_plan()` 方法 |
| ✏️ 修改 | `src/agent/multi_agent/multi_agent_system.py` | 删除 `router` 节点 + `_route_after_router`；`set_entry_point` 改为 `supervisor` |
| ❌ 不动 | `src/agent/multi_agent/router_agent.py` | 保留文件不删，兼容已有引用 |

---

### Phase 5-I: 图表迭代优化 — VLM Chart Critique Loop 🆕

#### 5I.2 VLM 评审 Prompt

```python
# src/agent/multi_agent/tools/chart_critic.py (新增)

CHART_CRITIC_PROMPT = """You are a professional financial chart critic.
Evaluate the given chart image and score it on four dimensions:

1. data_accuracy (0-1): Are the data points correctly represented?
   Check: axis labels match the data, scales are appropriate

2. label_clarity (0-1): Are all labels clear and readable?
   Check: title, axis labels, legend, data labels, font size

3. aesthetics (0-1): Is the chart visually professional?
   Check: color scheme, layout, consistency, cleanliness

4. information_density (0-1): Does the chart convey the right amount?
   Check: not too cluttered, not too sparse, appropriate chart type

Output JSON:
{
    "scores": {"data_accuracy": 0.9, "label_clarity": 0.7, "aesthetics": 0.6, "information_density": 0.8},
    "overall": 0.75,
    "issues": ["图例过小", "X轴标签旋转角度不合适"],
    "suggestions": ["增大图例字体", "将X轴标签改为45度旋转"],
    "verdict": "REFINE"  # "PASS" | "REFINE" | "REJECT"
}
"""


class ChartCritic:
    """
    VLM 图表评审器。

    用法:
      critic = ChartCritic(vlm_llm)
      verdict, feedback = critic.review(chart_path)
      if verdict == "REFINE":
          new_code = self._apply_feedback(old_code, feedback)
          new_path = execute(new_code)
          verdict, feedback = critic.review(new_path)  # 迭代
    """
    ...
```

#### 5I.3 与现有 DataVisualizer 的集成

```python
# business_compute_agent.py 中新增方法

def _visualize_with_critique(self, state, params, session_id):
    """
    带 VLM 评审的图表生成。

    比 _visualize 多了:
    1. VLM 评审 → 不满意则迭代
    2. 最多迭代 3 轮
    3. 最终图表附带评审评分元数据
    """
    chart_type = params.get("chart_type", "trend")
    enable_critique = params.get("enable_chart_critique",
                                  self.settings.get("finance.computation.visualization.enable_critique", False))

    # Step 1: 首次生成
    chart_path = self._visualize_single(state, chart_type, params, session_id)

    if not enable_critique or not chart_path:
        return chart_path

    # Step 2: VLM 评审循环
    max_iterations = self.settings.get("finance.computation.visualization.max_chart_iterations", 3)
    for i in range(max_iterations):
        verdict, feedback = self.chart_critic.review(chart_path)
        state.add_execution_trace({
            "agent": "chart_critic", "iteration": i + 1,
            "scores": feedback.get("scores", {}), "verdict": verdict,
        })
        if verdict == "PASS":
            break
        # 重写绘图代码
        old_code = self._get_chart_code(state, chart_type, session_id)
        new_code = self._apply_critic_feedback(old_code, feedback)
        chart_path = self._execute_chart_code(new_code, session_id)

    return chart_path
```

#### 5I.4 文件变更

| 操作 | 文件 | 说明 |
|------|------|------|
| 🆕 新增 | `src/agent/multi_agent/tools/chart_critic.py` | VLM 图表评审器 |
| ✏️ 修改 | `src/agent/multi_agent/business_compute_agent.py` | `_visualize` 增加可选的 VLM 评审迭代分支 |
| ✏️ 修改 | `config/settings.yaml` | 新增 `finance.computation.visualization.enable_critique` 和 `max_chart_iterations` |

---

### Phase 5-J: 三层溯源校验架构升级 🆕

#### 目标

受 stock-analyst 的三层推荐引擎启发，将你现有的 `_verify_and_annotate` 升级为完整的三层架构。

#### 5J.1 三层架构

```
你现有的:
  Layer 1: CitationFootnoter（段落溯源）
  Layer 2: DataVerifier（数字校正）

升级为:
  Layer 1: Calculator（纯数学计算引擎）
            → 计算目标价、预期收益、PEG 等
            → 完全不经过 LLM，保证数字绝对准确
            → 产出：FixedNumbers（不可变数据结构）

  Layer 2: EvidenceExtractor + LLM
            → 从 blackboard 构造证据包（E1, E2, ...）
            → LLM 只写叙事，不编数字
            → 产出：带引用的分析文本

  Layer 3: Validator
            → 解析 LLM 输出的数字
            → 比对 Layer 1 的计算结果
            → 偏差 > 1% 时自动修正并标记 ⚠️
            → 引用覆盖率检查：> 95% 段落必须有来源
```

#### 5J.2 Calculator 层

```python
# src/agent/multi_agent/tools/verification_calculator.py (新增)

@dataclass
class FixedNumbers:
    """不可变数字集合 — 所有计算值只在这里产生，LLM 无权修改"""
    target_price: Optional[float] = None
    target_price_3m: Optional[float] = None
    target_price_6m: Optional[float] = None
    target_price_12m: Optional[float] = None
    expected_return: Optional[float] = None
    pe_vs_industry: Optional[float] = None  # PE 偏离行业均值的百分比
    peg_ratio: Optional[float] = None
    revenue_growth: Optional[float] = None
    profit_growth: Optional[float] = None
    valuation_gap: Optional[float] = None  # (市价 - 内在价值) / 内在价值


class VerificationCalculator:
    """
    纯数学计算引擎。
    输入：market_data + computed_results
    输出：FixedNumbers（不可变）
    不调 LLM，所有计算是确定性的。
    """

    def compute(self, market_data: Dict, computed: Dict) -> FixedNumbers:
        """计算所有需要 LLM 引用的数字"""
        ...
```

#### 5J.3 Validator 层

```python
# src/agent/multi_agent/tools/verification_validator.py (新增)


class VerificationValidator:
    """
    校验 LLM 输出中的所有数字与 FixedNumbers 一致。

    策略:
    1. 正则提取 LLM 输出中的 "XX 元" / "XX%" / "PE=XX" 等模式
    2. 与 FixedNumbers 对比
    3. 偏差 > 1% → 自动修正 + ⚠️ 标记
    4. 统计引用覆盖率
    """

    def validate(self, report: str, fixed: FixedNumbers) -> Tuple[str, List[str]]:
        """
        Returns:
            (corrected_report, warnings)
        """
        warnings = []
        # 提取所有数字模式
        for pattern, field_name, expected in self._get_number_patterns(fixed):
            matches = re.findall(pattern, report)
            for match in matches:
                actual = float(match)
                if expected is not None and abs(actual - expected) / (abs(expected) + 1e-6) > 0.01:
                    report = report.replace(str(actual), f"{expected:.2f}⚠️")
                    warnings.append(f"{field_name}: {actual} → {expected:.2f}（自动修正）")
        return report, warnings

    def check_citation_coverage(self, report: str, source_docs: List[Dict]) -> float:
        """检查引用覆盖率：多少段落有来源标记"""
        paragraphs = [p for p in report.split("\n\n") if len(p.strip()) > 20]
        cited = sum(1 for p in paragraphs if "📖" in p or "[来源" in p or "[1]" in p or "[2]" in p)
        return cited / len(paragraphs) if paragraphs else 0
```

#### 5J.4 在 Generate 中的集成

```python
def _generate_financial_report(self, state):
    # Step 0: 计算所有数字（纯数学，不调 LLM）🆕
    calculator = VerificationCalculator()
    fixed_numbers = calculator.compute(
        market_data=state.blackboard.get("market_data", {}),
        computed=state.blackboard.get("computed_results", {}),
    )
    state.blackboard["fixed_numbers"] = asdict(fixed_numbers)

    # Step 1: 构造证据包
    evidence_pack = self._build_evidence_pack(state, fixed_numbers)

    # Step 2: LLM 撰写（约束：必须引用 evidence_pack 中的数字）
    context = self._collect_all_context(state)
    context["fixed_numbers"] = asdict(fixed_numbers)
    context["evidence_pack"] = evidence_pack
    report_text = self._llm_invoke_with_context(context, state)

    # Step 3: 三层校验
    validated_report = self._verify_and_annotate(
        report_text=report_text,
        source_docs=state.local_results,
        fixed_numbers=fixed_numbers,  # 🆕
    )

    state.final_answer = validated_report
    return state
```

#### 5J.5 文件变更

| 操作 | 文件 | 说明 |
|------|------|------|
| 🆕 新增 | `src/agent/multi_agent/tools/verification_calculator.py` | 纯数学计算引擎（Layer 1） |
| 🆕 新增 | `src/agent/multi_agent/tools/verification_validator.py` | 数字校验器 + 引用覆盖率检查（Layer 3） |
| ✏️ 修改 | `src/agent/multi_agent/multi_agent_system.py` | `_generate_financial_report` 集成三层架构 |

---

### Phase 5-K: 运行模式切换（Quick / Standard / Deep）🆕

#### 目标

参考 portfolio-research-team 的模式参数，让你的研报系统支持不同深度级别。

#### 5K.1 模式定义

```yaml
# config/settings.yaml — finance 配置段新增
finance:
  analysis_modes:
    quick:
      description: "快速分析 — 仅行情+基础指标，60秒内完成"
      agents: ["finance_data", "business_compute"]
      tasks: ["quote", "fundamentals", "calculate_metrics"]
      enable_rag: false
      enable_charts: false
      enable_critique: false
      enable_conflict_detect: false
    standard:
      description: "标准研报 — 文档+行情+计算+图表，3-5分钟"
      agents: ["rag", "finance_data", "business_compute", "generate"]
      tasks: ["document_search", "quote", "fundamentals", "calculate_metrics", "compare", "visualize", "generate_report"]
      enable_rag: true
      enable_charts: true
      enable_critique: false
      enable_conflict_detect: true
    deep:
      description: "深度研报 — 全链路+DCF估值+VLM评审+冲突检测，8-15分钟"
      agents: ["rag", "finance_data", "business_compute", "generate"]
      tasks: ["document_search", "quote", "fundamentals", "history", "calculate_metrics", "compare", "valuation", "visualize", "generate_report"]
      enable_rag: true
      enable_charts: true
      enable_critique: true
      enable_conflict_detect: true
      enable_dcf: true
      enable_chain_of_analysis: true
```

#### 5K.2 在 Supervisor 中的集成

```python
# supervisor_agent.py classify_and_plan() 中新增

def _detect_analysis_depth(self, query: str) -> str:
    """
    从 query 中检测用户期望的分析深度。

    关键词匹配:
    - "快速" / "简单" / "看一眼" → quick
    - "深度" / "详细" / "全面" / "完整" → deep
    - 默认 → standard
    """
    quick_keywords = ["快速", "简单", "看一眼", "brief", "quick", "simple"]
    deep_keywords = ["深度", "详细", "全面", "完整", "deep", "detailed", "comprehensive"]

    for kw in quick_keywords:
        if kw in query.lower():
            return "quick"
    for kw in deep_keywords:
        if kw in query.lower():
            return "deep"
    return "standard"


def classify_and_plan(self, state: AgentState) -> AgentState:
    # ... 原有意图分类逻辑 ...

    # 🆕 检测分析深度
    depth = self._detect_analysis_depth(state.user_input)
    mode_config = self._get_mode_config(depth)
    state.add_to_blackboard("analysis_depth", depth, "supervisor")
    state.add_to_blackboard("analysis_mode_config", mode_config, "supervisor")

    return state
```

#### 5K.3 在 LangGraph 中的效果

```
用户说 "快速看看宁德时代" → depth=quick
  Supervisor 只发出 2 个节点:
  ├── finance_data → business_compute(仅指标) → generate
  └── 不调 RAG，不画图，不冲突检测

用户说 "深度分析宁德时代" → depth=deep
  Supervisor 发出全链路:
  ├── rag + finance_data(并行) → business_compute(全流程)
  ├── chart_critic(VLM评审) → conflict_detector
  └── generate(含 CoA + 三层校验)
```

#### 5K.4 文件变更

| 操作 | 文件 | 说明 |
|------|------|------|
| ✏️ 修改 | `src/agent/multi_agent/supervisor_agent.py` | 新增 `_detect_analysis_depth()` + mode_config 注入 |
| ✏️ 修改 | `src/agent/multi_agent/multi_agent_system.py` | `_dispatch_from_supervisor` 根据 depth 过滤子任务 |
| ✏️ 修改 | `config/settings.yaml` | 新增 `finance.analysis_modes` 配置段 |

---

### Phase 5-L: Chain-of-Analysis（先分析后写作）🆕

#### 目标

受 FinSight 的 CoA 机制启发。当前的 Generate 节点是 LLM 直接基于原始数据写报告，缺少中间推理层。CoA 的做法是：**先产出分析链（结构化洞察），再基于分析链写报告。**

#### 5L.1 CoA 是什么

```json
// Chain-of-Analysis 是一组结构化的分析结论，每一条都是
// "数据观察 → 业务解释 → 投资含义" 的三段式推理

{
    "chains": [
        {
            "observation": "营收连续3年增长20%+，但Q3增速从22%放缓至15%",
            "interpretation": "行业渗透率趋于饱和，公司进入平稳增长期",
            "implication": "估值中枢可能从成长股向价值股切换",
            "confidence": 0.85,
            "evidence": ["financial_reports: p.12", "market_data: PE从35x降至28x"]
        },
        {
            "observation": "毛利率从35%降至32%，费用率上升3个百分点",
            "interpretation": "竞争加剧导致价格战，同时研发投入增加",
            "implication": "短期利润承压，但研发投入可能产出新产品",
            "confidence": 0.75,
            "evidence": ["financial_reports: p.15", "research_reports: 行业分析"]
        },
        {
            "observation": "PE 45x vs 行业均值25x，ROE 18% vs 行业15%",
            "interpretation": "高估值有基本面支撑，溢价合理",
            "implication": "当前估值偏高但不极端，考虑等待回调",
            "confidence": 0.80,
            "evidence": ["computed_results: valuation", "industry_benchmark: 新能源电池"]
        }
    ]
}
```

#### 5L.2 两阶段写作框架

```
当前流程:
  blackboard(原始数据) → LLM → 研报全文

CoA 流程:
  blackboard(原始数据)
       ↓
  Stage 1: CoA 生成器
       ↓  LLM 分析数据 → 产出结构化的分析链（每一条含观察+解释+含义）
       ↓
  Stage 2: 报告写作器
       ↓  LLM 基于 CoA 写报告（每段对应一条 CoA，展开详细论述）
       ↓
  最终研报
```

好处：
- **逻辑链可追溯**：每条结论都能追溯到数据来源
- **避免跳跃**：LLM 不能跳过 CoA 直接写结论
- **可修正**：如果 CoA 某条错了，只需修正该条，不用重写全文

#### 5L.3 实现

```python
# multi_agent_system.py _generate_financial_report 中新增


class ChainOfAnalysis:
    """
    CoA 生成器。

    输入: blackboard 中的所有结构化数据
    输出: 3-7 条分析链，每条含 observation + interpretation + implication + confidence + evidence
    """

    PROMPT = """You are a senior financial analyst.
Based on the following data, generate 3-7 structured analysis chains.

Each chain must follow:
- observation: specific data point from the provided data
- interpretation: what this data means in business context
- implication: what this means for investment decision

Rules:
- EVERY observation must have a corresponding data source
- confidence must be based on data completeness
- DO NOT repeat the same data point in multiple chains

Output JSON:
{"chains": [{"observation": "...", "interpretation": "...", "implication": "...", "confidence": 0.85, "evidence": ["src1", "src2"]}]}
"""


def _generate_with_coa(self, state, context):
    """两阶段写作"""

    # Stage 1: CoA 生成
    coa_prompt = ChainOfAnalysis.PROMPT + "\n\n## 数据\n" + json.dumps(context, ensure_ascii=False)
    coa_response = self.llm.invoke([HumanMessage(content=coa_prompt)])
    coa = json.loads(self._extract_json(coa_response.content))
    state.add_to_blackboard("chain_of_analysis", coa, "generate")
    state.add_metric("coa_count", len(coa.get("chains", [])))

    # Stage 2: 基于 CoA 写报告
    report_prompt = self._build_financial_analyst_prompt()
    report_prompt += "\n\n## 分析链（基于此撰写报告，每段对应一条分析链）\n"
    for i, chain in enumerate(coa.get("chains", []), 1):
        report_prompt += f"\n[{i}] {chain['observation']}\n    解释: {chain['interpretation']}\n    含义: {chain['implication']}\n"

    report = self.llm.invoke([HumanMessage(content=report_prompt)])
    return report.content
```

#### 5L.4 文件变更

| 操作 | 文件 | 说明 |
|------|------|------|
| 🆕 新增 | `src/agent/multi_agent/tools/chain_of_analysis.py` | CoA 生成器 + Prompt |
| ✏️ 修改 | `src/agent/multi_agent/multi_agent_system.py` | `_generate_financial_report` 增加两阶段写作分支 |

---

### Phase 5-M: 研报章节级 Prompt 管理 + PDF 导出 🆕

#### 目标

参考 FinRobot 的分章节管理和 stock-analyst 的 PDF 导出，将报告生成从单 Prompt 升级为章节级 Prompt 管理。

#### 5M.1 章节 Prompt 目录结构

```
templates/finance/
├── earnings_review/
│   ├── 00_system.md                  # 系统角色定义（CFA 分析师身份）
│   ├── 01_investment_highlights.md   # Prompt: 写投资要点
│   ├── 02_financial_analysis.md      # Prompt: 写财务分析
│   ├── 03_industry_comparison.md     # Prompt: 写行业对比
│   ├── 04_valuation.md               # Prompt: 写估值分析
│   ├── 05_risk_factors.md            # Prompt: 写风险提示
│   ├── 06_conflict_warnings.md       # Prompt: 处理数据异常提示
│   └── 07_conclusion.md              # Prompt: 写结论
├── industry_comparison/
│   └── ...
└── event_impact/
    └── ...
```

每个章节 Prompt 示例：

```markdown
# templates/finance/earnings_review/04_valuation.md

## 估值分析

### 可用数据
- {{ fixed_numbers | tojson }}
- {{ market_data.fundamentals | tojson }}
- {{ industry_benchmark | tojson }}
- {{ computed_results.valuation | tojson }}

### 写作要求
1. 先列 PE/PB/PS 当前值
2. 与行业均值对比
3. 与历史区间对比
4. 如果 DCF 数据可用，给出目标价
5. 最终给出估值判断：低估/合理/高估

### 数字约束
- PE、PB 等指标必须来自 fixed_numbers，不要自己计算
- 目标价格式: XX.XX 元
- 所有百分比保留两位小数
```

#### 5M.2 章节拼接器

```python
# src/agent/multi_agent/tools/section_assembler.py (新增)


class SectionAssembler:
    """
    章节级 Prompt 管理。

    用法:
      assembler = SectionAssembler(template_dir="templates/finance")
      report = assembler.assemble(
          report_type="earnings_review",
          data={"fixed_numbers": ..., "market_data": ..., ...},
      )
      # 内部流程:
      # 1. 读取 00_system.md 到 07_conclusion.md
      # 2. 对每个章节，用 LLM 生成内容
      # 3. 拼接完整报告
    """

    def __init__(self, template_dir: str):
        self.template_dir = Path(template_dir)

    def assemble(self, report_type: str, data: Dict) -> str:
        """逐章节生成，返回完整研报 Markdown"""
        sections = self._load_sections(report_type)
        answers = []
        for section_file in sections:
            prompt = self._render_prompt(section_file, data)
            section_content = self._llm_generate(prompt)  # 可选：单 LLM 调用完成全部章节
            answers.append(section_content)
        return "\n\n---\n\n".join(answers)

    def assemble_single_call(self, report_type: str, data: Dict) -> str:
        """
        单次 LLM 调用版本（更高效）。
        将所有章节 Prompt 合并为一个大 Prompt，
        让 LLM 一次性输出完整报告。
        """
        all_prompts = self._load_all_prompts(report_type)
        combined = "\n\n".join(all_prompts)
        return self._llm_generate({
            "system": combined,
            "data": data,
        })
```

#### 5M.3 PDF 导出

```python
# src/agent/multi_agent/tools/pdf_exporter.py (新增)


class PDFExporter:
    """
    将 Markdown 研报导出为 PDF。

    依赖:
      pip install weasyprint  # 推荐
      或 pandoc (备用)

    用法:
      exporter = PDFExporter()
      pdf_path = exporter.export(
          markdown=report_text,
          output_path="data/reports/宁德时代_2024Q3.pdf",
          styling="professional",  # "professional" | "minimal"
      )
    """

    def export(self, markdown: str, output_path: str, styling: str = "professional") -> str:
        """
        WeasyPrint 方式：
        1. markdown → HTML (通过 markdown 库)
        2. HTML → PDF (通过 weasyprint)
        3. 应用专业级 CSS 样式
        """
        ...

    def export_via_pandoc(self, markdown: str, output_path: str) -> str:
        """
        备用方式：
        pandoc report.md -o report.pdf --pdf-engine=xelatex
        """
        ...
```

#### 5M.4 PDF 样式

```css
/* templates/finance/pdf_professional.css */
@page {
  size: A4;
  margin: 2.5cm 2cm 2cm 2cm;
  @top-center {
    content: element(header);
  }
  @bottom-center {
    content: counter(page);
    font-size: 9pt;
    color: #666;
  }
}
body {
  font-family: "Source Han Serif SC", "Noto Serif CJK SC", serif;
  font-size: 10.5pt;
  line-height: 1.8;
  color: #333;
}
h1 { font-size: 18pt; color: #1a1a1a; border-bottom: 2px solid #1a1a1a; }
h2 { font-size: 14pt; color: #333; }
table { width: 100%; border-collapse: collapse; margin: 1em 0; }
th { background: #f5f5f5; font-weight: 600; }
td, th { border: 1px solid #ddd; padding: 6pt 10pt; }
blockquote { border-left: 3px solid #c00; padding-left: 1em; color: #555; }
```

#### 5M.5 文件变更

| 操作 | 文件 | 说明 |
|------|------|------|
| 🆕 新增 | `src/agent/multi_agent/tools/section_assembler.py` | 章节级 Prompt 拼接器 |
| 🆕 新增 | `src/agent/multi_agent/tools/pdf_exporter.py` | PDF 导出器 |
| 🆕 新增 | `templates/finance/earnings_review/*.md` | 章节 Prompt 模板（7 个文件） |
| 🆕 新增 | `templates/finance/pdf_professional.css` | PDF 样式表 |

---

### Phase 5-N: Agent 流式输出 + 思考过程可视化 🆕

#### 目标

当前系统的执行模式是**全链路运行完一次性返回结果**，用户看到的是最终答案，看不到：
- 各 Agent 正在执行什么（如"正在检索财报数据…"）
- 思考过程（如"为什么选择这些指标"）
- 中间发现（如"营收增速放缓，需要关注"）

参考 ChatGPT 的"展示推理过程"按钮和 Wealth Hub Agent 的 SSE 推流，实现**实时流式 Agent 输出 + 可展开的思考过程面板**。

#### 5N.1 最终效果

```
用户: "分析宁德时代"

  ┌─ 消息气泡 ──────────────────────────────────────┐
  │                                                   │
  │  🤖 宁德时代 2024 Q3 财报分析                    │
  │                                                   │
  │  ⚡ 投资评级: 增持                               │
  │  ... (完整研报内容)                               │
  │                                                   │
  │  ── ── ── ── ── ── ── ── ── ── ──              │
  │                                                   │
  │  🧠 [展示推理过程 ▼]  ← 可折叠，默认收起         │
  │                                                   │
  │  ┌─ 思考过程 ─────────────────────────────────┐  │
  │  │                                              │  │
  │  │  ✅ Router → "financial_report"             │  │
  │  │  │  intent: financial_analysis              │  │
  │  │  │  depth: deep                             │  │
  │  │  │                                           │  │
  │  │  ✅ Supervisor → 拆解 7 个子任务             │  │
  │  │  │  task_1: 检索财报文档 (financial_reports) │  │
  │  │  │  task_2: 查询行情数据 (300750.SZ)        │  │
  │  │  │  task_3: 计算财务指标                     │  │
  │  │  │  ...                                      │  │
  │  │  │                                           │  │
  │  │  ⏳ FinanceDataAgent → 查询中...             │  │
  │  │  │  Tushare → 成功，返回 PE=23.4, ROE=18.2% │  │
  │  │  │                                           │  │
  │  │  ✅ ConflictDetector → 发现 1 个冲突         │  │
  │  │  │  [HIGH] 营收+18.5% vs 股价-12.3%         │  │
  │  │  │                                           │  │
  │  │  ✅ Generate → 研报生成完成                  │  │
  │  │  │  分 析 链: 3 条  ·  字 数: 1,234         │  │
  │  │  └──────────────────────────────────────────┘  │
  │                                                   │
  │  📚 [查看引用来源 ▼]                              │
  └───────────────────────────────────────────────────┘
```

#### 5N.2 架构设计

```
传统模式:
  run() → 全链路执行 → 返回最终结果 → Dashboard 渲染

流式模式:
  run_stream() → 每步 yield 事件 → SSE 推流 → Dashboard 逐步渲染
                            ↓
                    Event 类型:
                    ├── agent_start   → "Agent X 开始执行"
                    ├── agent_step    → "Agent X 正在做 Y"
                    ├── agent_result  → "Agent X 完成，产出 Z"
                    ├── thinking      → "思考过程片段"
                    ├── conflict      → "发现 1 个冲突"
                    ├── partial_text  → LLM 生成了部分文本
                    └── done          → "全部完成"
```

#### 5N.3 事件流协议

```python
# src/agent/multi_agent/stream_events.py (新增)

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
from datetime import datetime


class EventType(Enum):
    AGENT_START = "agent_start"          # Agent 开始执行
    AGENT_STEP = "agent_step"            # Agent 执行中的一步
    AGENT_RESULT = "agent_result"        # Agent 完成
    THINKING = "thinking"                # 思考过程片段
    CONFLICT = "conflict"               # 检测到冲突
    PARTIAL_TEXT = "partial_text"        # LLM 生成的部分文本
    CHART_GENERATED = "chart_generated" # 图表生成完成
    DONE = "done"                       # 全部完成
    ERROR = "error"                     # 错误


@dataclass
class AgentEvent:
    """Agent 流式事件"""
    type: EventType
    agent_name: str
    timestamp: str = ""
    content: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    trace_id: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class StreamingSession:
    """流式会话状态"""
    events: List[AgentEvent] = field(default_factory=list)
    partial_answer: str = ""  # 逐步累积的答案
    current_agent: str = ""
    finished: bool = False

    def add_event(self, event: AgentEvent):
        self.events.append(event)
```

#### 5N.4 Agent 运行器改造

```python
# src/agent/multi_agent/multi_agent_system.py 中新增

class MultiAgentRAG:
    """
    新增流式执行方法 run_stream()。
    原有 run() 保持不变，完全向后兼容。
    """

    def run_stream(self, user_input: str, conversation_history=None):
        """
        流式执行全链路，每次 yield 一个 AgentEvent。

        用法:
          for event in agent.run_stream("分析宁德时代"):
              if event.type == EventType.AGENT_STEP:
                  print(f"⏳ {event.content}")
              elif event.type == EventType.AGENT_RESULT:
                  print(f"✅ {event.content}")
              elif event.type == EventType.PARTIAL_TEXT:
                  print(f"✍️ {event.content}", end="")
        """
        state = AgentState(
            user_input=user_input,
            conversation_history=conversation_history or [],
        )

        # 阶段 1: Router / Supervisor
        yield AgentEvent(
            type=EventType.AGENT_START,
            agent_name="Router",
            content="正在识别意图…",
        )
        routing_decision = self.router_agent.classify(user_input, conversation_history)
        yield AgentEvent(
            type=EventType.THINKING,
            agent_name="Router",
            content=f"意图识别为「{routing_decision.intent}」，置信度 {routing_decision.confidence:.2f}",
            details={"intent": routing_decision.intent, "confidence": routing_decision.confidence},
        )

        # 阶段 2: Supervisor 规划
        if routing_decision.intent.startswith("financial_"):
            yield AgentEvent(
                type=EventType.AGENT_START,
                agent_name="Supervisor",
                content="正在拆解分析任务…",
            )
            state = self.supervisor_agent.classify_and_plan(state)
            subtasks = state.task_plan.get("subtasks", [])
            yield AgentEvent(
                type=EventType.THINKING,
                agent_name="Supervisor",
                content=f"拆解为 {len(subtasks)} 个子任务: " +
                        ", ".join(t["type"] for t in subtasks),
                details={"subtask_count": len(subtasks)},
            )

        # 阶段 3: 依次执行子任务（流式产出）
        for subtask in self._get_ordered_subtasks(state):
            yield AgentEvent(
                type=EventType.AGENT_START,
                agent_name=subtask["type"],
                content=f"正在执行 {subtask['description']}…",
            )
            # 执行子任务，逐步骤 yield
            for step_event in self._execute_subtask_stream(subtask, state):
                yield step_event
            yield AgentEvent(
                type=EventType.AGENT_RESULT,
                agent_name=subtask["type"],
                content=f"完成: {subtask['description']}",
            )

        # 阶段 4: Generate (流式输出文本)
        yield AgentEvent(
            type=EventType.AGENT_START,
            agent_name="Generate",
            content="正在撰写研报…",
        )
        for text_chunk in self._generate_stream(state):
            yield AgentEvent(
                type=EventType.PARTIAL_TEXT,
                agent_name="Generate",
                content=text_chunk,
            )

        # 阶段 5: 冲突检测
        conflict_flags = state.blackboard.get("conflict_flags", [])
        if conflict_flags:
            for flag in conflict_flags:
                yield AgentEvent(
                    type=EventType.CONFLICT,
                    agent_name="ConflictDetector",
                    content=f"[{flag.severity}] {flag.claim_a} vs {flag.claim_b}",
                    details=flag.__dict__,
                )

        yield AgentEvent(
            type=EventType.DONE,
            agent_name="System",
            content="研报生成完成",
            details={"total_events": len(state.execution_trace)},
        )
```

#### 5N.5 Dashboard 流式渲染

```python
# src/observability/dashboard/pages/agent_chat.py 中修改

import queue
import threading


def process_query_stream(query, config, event_queue: queue.Queue):
    """
    在后台线程中运行流式 Agent，通过 queue 传递事件到主线程。

    Streamlit 的 rerun 机制无法直接处理 SSE，
    因此使用 queue + st.empty() 占位符逐步更新。
    """
    agent = get_agent()
    if not agent:
        event_queue.put(AgentEvent(type=EventType.ERROR, agent_name="System",
                                   content="Agent 未初始化"))
        return

    try:
        for event in agent.run_stream(query, conversation_history=[]):
            event_queue.put(event)
    except Exception as e:
        event_queue.put(AgentEvent(type=EventType.ERROR, agent_name="System",
                                   content=str(e)))


def render_streaming_chat():
    """
    流式聊天的渲染入口。

    与传统渲染的区别:
    1. 用户发送消息后，立即创建一个"流式消息占位符"
    2. 后台线程开始执行，通过 queue 发送事件
    3. 主线程每秒轮询 queue，更新占位符
       - thinking 事件 → 更新"思考过程"面板
       - partial_text → 追加到消息内容
       - agent_start/result → 更新 Agent 状态指示器
       - done → 停止轮询，完成渲染
    """
    ...

    # 思考过程面板 (可折叠，默认收起)
    with st.expander("🧠 展示推理过程", expanded=False):
        thinking_container = st.container()
        # 逐步填充 Agent 执行步骤
        for event in events:
            if event.type == EventType.AGENT_START:
                thinking_container.markdown(
                    f"**{event.agent_name}** 开始: {event.content}"
                )
            elif event.type == EventType.THINKING:
                thinking_container.markdown(f"> {event.content}")
                if event.details:
                    thinking_container.json(event.details)
            elif event.type == EventType.AGENT_RESULT:
                thinking_container.markdown(
                    f"✅ **{event.agent_name}** 完成: {event.content}"
                )
            elif event.type == EventType.CONFLICT:
                thinking_container.markdown(
                    f"⚠️ 冲突检测: {event.content}"
                )
```

#### 5N.6 思考过程面板的 Toggle 设计

```python
# src/observability/dashboard/pages/agent_chat.py — display_chat_message 中新增

def display_chat_message(message):
    role = message["role"]
    content = message["content"]

    if role == "assistant":
        with st.chat_message("assistant"):
            # 完整研报内容
            st.markdown(content)

            # 🆕 思考过程 toggle (默认收起)
            thinking_process = message.get("thinking_process", [])
            if thinking_process:
                with st.expander("🧠 展示推理过程", expanded=False):
                    for step in thinking_process:
                        agent = step.get("agent", "")
                        action = step.get("action", "")
                        icon = _AGENT_ICONS.get(agent, "🤖")
                        status = step.get("status", "done")  # running / done / error

                        if status == "running":
                            st.markdown(f"{icon} **{agent}** → {action} ⏳")
                        elif status == "done":
                            st.markdown(f"{icon} ✅ **{agent}** → {action}")

                        detail = {k: v for k, v in step.items()
                                  if k not in ("agent", "action", "status")}
                        if detail:
                            with st.expander("详情", expanded=False):
                                st.json(detail)

            # 原有引用和指标
            if "citations" in message and message["citations"]:
                with st.expander("📚 查看引用来源", expanded=False):
                    ...

            if "metrics" in message and message["metrics"]:
                with st.expander("📊 执行指标", expanded=False):
                    ...
```

#### 5N.7 后端支持：execution_trace 增强

当前 `execution_trace` 只记录了 agent + action + details，缺少**过程描述和状态标记**。扩展 schema：

```python
# state.py AgentState 中的 execution_trace 条目增强

{
    "agent": "finance_data",
    "action": "query_market",
    "status": "done",           # "running" | "done" | "error"  🆕
    "description": "正在查询宁德时代行情数据…",  # 🆕 友好的过程描述
    "timestamp": "2026-05-03T10:30:00",         # 🆕 时间戳
    "elapsed_ms": 2340,                          # 🆕 该步骤耗时
    "details": {                                 # 原有
        "symbols": ["300750.SZ"],
        "quote_count": 1,
    }
}
```

```python
# state.py 中新增 add_thinking_step 方法

def add_thinking_step(self, agent: str, action: str,
                       description: str = "", status: str = "running",
                       details: Optional[Dict] = None):
    """
    添加一个思考过程步骤，用于流式展示。

    Args:
        agent: Agent 名称
        action: 执行的动作
        description: 用户友好的过程描述
        status: "running" | "done" | "error"
        details: 额外数据
    """
    import time
    step = {
        "agent": agent,
        "action": action,
        "status": status,
        "description": description,
        "timestamp": datetime.now().isoformat(),
        "details": details or {},
    }
    self.execution_trace.append(step)
```

#### 5N.8 Streamlit 前端轮询方案

```python
# agent_chat.py — 轮询渲染核心逻辑

def render_streaming_response(response_container, event_queue, thinking_container):
    """
    轮询 event_queue，逐步更新 Streamlit 占位符。

    策略:
    - 因为有 st.rerun() 限制，采用「批量更新」模式
    - 每 0.5 秒检查一次 queue
    - 累积事件后一次性重绘
    - 收到 DONE 事件后停止
    """
    accumulated_text = ""
    thinking_steps = []
    import time

    while True:
        try:
            event = event_queue.get(timeout=0.5)
        except queue.Empty:
            # 超时后重绘一次
            with response_container.container():
                st.markdown(accumulated_text or "⏳ 正在生成…")
                if thinking_steps:
                    with st.expander("🧠 展示推理过程", expanded=False):
                        for step in thinking_steps:
                            st.markdown(step)
            continue

        if event.type == EventType.PARTIAL_TEXT:
            accumulated_text += event.content
        elif event.type == EventType.THINKING:
            thinking_steps.append(f"> {event.content}")
        elif event.type == EventType.AGENT_START:
            thinking_steps.append(f"⏳ **{event.agent_name}**: {event.content}")
        elif event.type == EventType.AGENT_RESULT:
            thinking_steps.append(f"✅ **{event.agent_name}**: {event.content}")
        elif event.type == EventType.CONFLICT:
            thinking_steps.append(f"⚠️ {event.content}")
        elif event.type == EventType.DONE:
            with response_container.container():
                st.markdown(accumulated_text)
                if thinking_steps:
                    with st.expander("🧠 展示推理过程", expanded=False):
                        for step in thinking_steps:
                            st.markdown(step)
            break
        elif event.type == EventType.ERROR:
            with response_container.container():
                st.error(event.content)
            break

        # 每次事件后重绘
        with response_container.container():
            st.markdown(accumulated_text or "⏳ 正在生成…")
            if thinking_steps:
                with st.expander("🧠 展示推理过程", expanded=False):
                    for step in thinking_steps:
                        st.markdown(step)
```

#### 5N.9 文件变更

| 操作 | 文件 | 说明 |
|------|------|------|
| 🆕 新增 | `src/agent/multi_agent/stream_events.py` | 事件类型定义 + StreamingSession |
| ✏️ 修改 | `src/agent/multi_agent/multi_agent_system.py` | 新增 `run_stream()` 流式执行方法 |
| ✏️ 修改 | `src/agent/multi_agent/state.py` | execution_trace 增加 status/description/timestamp 字段；新增 `add_thinking_step()` |
| ✏️ 修改 | `src/observability/dashboard/pages/agent_chat.py` | 新增流式渲染支持 + 思考过程 toggle + 后台线程轮询 |

#### 5N.10 向后兼容性

| 场景 | 影响 |
|------|------|
| 现有 `run()` 方法 | **完全不变**，`run_stream()` 是新增方法 |
| 现有 execution_trace 格式 | **向前兼容**，新增字段是 optional 的，旧 trace 仍可正常显示 |
| 非流式 Dashboard | **不变**，流式渲染是新增的渲染路径，不影响现有 display_execution_trace() |
| CLI/API 调用 | **不变**，仅 Streamlit Dashboard 触发流式路径 |

---
## 五、完整交互流程（改造后）

```
用户: "分析宁德时代2024年Q3财报，对比行业，出估值分析和图表"

Step 1: Router Agent
  intent = "financial_report"
  → 路由到 Supervisor

Step 2: Supervisor (Plan + CollectionRouter)
  collections: ["financial_reports", "industry_data", "peer_comparison"]

  拆解子任务:
  ├── task_1: document_search (collections=["financial_reports"])
  ├── task_2: financial_market (symbols=["300750.SZ"])
  │     └── FinanceDataAgent: Tushare → 失败? → akshare 🆕
  ├── task_3: calculate_metrics (依赖 task_1, task_2)
  ├── task_4: compare (依赖 task_3)
  ├── task_5: valuation (industry="新能源电池") 🆕
  ├── task_6: visualize (依赖 task_5)
  └── task_7: generate_report (template="earnings_review")

Step 3: 并行+顺序执行
  task_1 ──┐
  task_2 ──┤(并行)──→ task_3→task_4→task_5→task_6
           │          ↑计算   ↑对比  ↑估值
           │          metrics 行业  DCF敏感性
           └──────────────────────────────┘
                                │
                                ▼
                        ┌───────────────┐
                        │  aggregate    │
                        │  + conflict   │ 🆕 检测矛盾
                        │  _detect      │
                        └───────┬───────┘
                                │
                                ▼
                        ┌───────────────┐
                        │  Generate     │
                        │  ├─ LLM 撰写  │
                        │  ├─ Footnoter │ 溯源
                        │  ├─ Verifier  │ 校验
                        │  └─ 冲突标记   │ 🆕 数据异常提示
                        └───────────────┘

Step 4: 最终输出
  # 宁德时代 2024 Q3 财报分析

  ## 投资要点
  **评级**: 增持  |  营收同比增长... [1]

  ## 核心财务指标
  | ROE | 18.2% | +2.1% | 15.0% | 领先 ✅ |
  > 📖 数据来源: 宁德时代2024年三季报, p.12

  ## 估值分析
  DCF 敏感性分析矩阵:
  | 增长率\折现率 | 8% | 10% | 12% |
  |--------------|-----|-----|-----|
  | 10% | 280 | 220 | 180 |

  ## ⚠️ 数据异常提示  🆕
  - [HIGH] 营收同比增长 +18.5% vs 股价同期下跌 -12.3%
    → 可能原因：行业估值中枢下移、市场提前消化预期

  ## 风险提示
  - 行业竞争加剧导致毛利率承压

  ## 图表
  ![K线图](data/charts/sess_001/kline_300750.png)
```

---

## 六、完整文件变更清单

### 新增文件（8 个）

| 文件 | 阶段 | 说明 |
|------|------|------|
| `scripts/ingest_finance.py` | A | 金融文档批量摄取脚本 |
| `src/agent/multi_agent/tools/collection_router.py` | B | Collection 路由决策器 |
| `src/agent/multi_agent/tools/citation_footnoter.py` | C | 段落级溯源脚注生成 |
| `src/agent/multi_agent/tools/data_verifier.py` | D | 数据校验器（数字一致性检查） |
| `src/agent/multi_agent/tools/sentiment_analyzer.py` | E | 公告情感分类器 |
| `src/agent/multi_agent/tools/industry_benchmark.py` | E | 行业基准数据管理 |
| `src/agent/multi_agent/tools/conflict_detector.py` | G 🆕 | 跨 Agent 冲突检测器 |
| `src/agent/multi_agent/tools/tushare_client.py` | G 🆕 | Tushare A 股数据客户端 |

### 修改文件（9 个）

| 文件 | 阶段 | 改动内容 |
|------|------|---------|
| `config/settings.yaml` | A/G | 新增 finance.collections + Tushare token |
| `src/agent/multi_agent/supervisor_agent.py` | B/H 🆕 | `_plan_financial()` 调用 CollectionRouter；PLAN_PROMPT 合并意图分类；新增 `classify_and_plan()` |
| `src/agent/multi_agent/multi_agent_system.py` | B/C/D/G/H 🆕 | 删除 `router` 节点；删除 `_route_after_router`；set_entry_point→supervisor；透传 collection；集成溯源校验+冲突检测+Generate 注入冲突标记 |
| `src/agent/multi_agent/tools/report_renderer.py` | C | 扩展多模板 + conflict_warnings 字段 |
| `src/agent/multi_agent/tools/financial_calculator.py` | F | 新增估值分析函数 |
| `src/agent/multi_agent/business_compute_agent.py` | F | 增加估值分析分支 |
| `src/mcp_server/tools/query_market_data.py` | G 🆕 | 双源降级逻辑 + Tushare 主数据源集成 |
| `.env` (或 .env.example) | G | 新增 `TUSHARE_API_TOKEN` 环境变量 |

### 无需改动（直接复用）

| 文件 | 原因 |
|------|------|
| `src/agent/multi_agent/state.py` | Blackboard 已支持任意 key |
| `src/agent/multi_agent/eval_agent.py` | 检索评估逻辑不变 |
| `src/agent/multi_agent/refine_agent.py` | 查询改写逻辑不变 |
| `src/agent/multi_agent/search_agent.py` | 已支持 collection 参数 |
| `src/agent/multi_agent/finance_data_agent.py` | 行情查询逻辑不变 |
| `src/agent/multi_agent/citation.py` | Citation 类 + FaithfulnessCheck 沿用 |
| `src/agent/multi_agent/router_agent.py` | 保留不动，兼容已有引用 |
| `src/ingestion/` 全目录 | 摄取 pipeline 不变 |

---

## 七、实施顺序建议

```
Phase 5-A: 数据源 Collection 规划
  ├── config/settings.yaml 加配置
  ├── scripts/ingest_finance.py
  └── 验收: 能用不同 collection 摄入和检索
          ↓
Phase 5-B: 路由层细化
  ├── collection_router.py
  ├── supervisor_agent.py 扩展
  └── 验收: "宁德时代" 自动路由到 financial_reports
          ↓
Phase 5-C: 研报模板系统
  ├── report_renderer.py 多模板
  ├── citation_footnoter.py
  └── 验收: 生成研报每段有来源脚注
          ↓
Phase 5-D: 溯源校验
  ├── data_verifier.py
  ├── multi_agent_system.py 集成
  └── 验收: LLM 写的数字与实际值不一致时自动修正
          ↓
Phase 5-E: 公告情感 + 行业基准
  ├── sentiment_analyzer.py
  ├── industry_benchmark.py
  └── 验收: "回购"公告 → 利好
          ↓
Phase 5-F: 估值分析
  ├── financial_calculator.py 扩展
  ├── business_compute_agent.py 扩展
  └── 验收: 输出含 DCF 敏感性分析矩阵的估值报告
          ↓
Phase 5-G: 冲突检测 + Tushare 🆕
  ├── conflict_detector.py
  ├── tushare_client.py
  ├── query_market_data.py 双源降级 (Tushare 主)
  ├── multi_agent_system.py 冲突节点
  └── 验收:
      ├── Tushare 正常时返回 Tushare 数据
      ├── Tushare 不可用时自动切 akshare
      └── 营收+20% 但股价跌-10% → 研报出 "⚠️ 数据异常提示"
          ↓
Phase 5-H: Router + Supervisor 合并 🆕
  ├── supervisor_agent.py PLAN_PROMPT 合并
  ├── multi_agent_system.py 删除 router 节点
  └── 验收: 金融 query 仍正常走 Supervisor 编排，少一次 LLM 调用
```

---

## 八、向后兼容性保证

| 场景 | 改前 | 改后 | 变化 |
|------|------|------|------|
| "最近的人事制度" (文档检索) | 原路径 | 完全一致 | ❌ 无变化 |
| "今天天气" (闲聊) | 原路径 | 完全一致 | ❌ 无变化 |
| "对比 RAG 和微调" (分析) | 原路径 | 完全一致 | ❌ 无变化 |
| "宁德时代 Q3 分析" (金融) | 原路径 | +collection 路由+溯源+冲突检测 | ✅ 增强 |
| "生成研报" (金融报告) | 原路径 | +多模板+估值分析+Tushare 容灾 | ✅ 增强 |

**非金融场景完全不受影响，金融场景只在现有能力上叠加增强。**

---

## 九、验收标准

### 9.1 E2E 测试场景

```python
# examples/test_phase5_e2e.py (新增)

SCENARIOS = [
    {
        "name": "完整研报生成",
        "query": "分析宁德时代2024年Q3财报并对比行业平均，出K线图和估值分析",
        "check": [
            "final_answer 包含 '投资要点'",
            "final_answer 包含 '财务分析'",
            "final_answer 包含 '估值分析'",
            "final_answer 包含 '风险提示'",
            "final_answer 包含来源脚注标记 📖",
            "blackboard.chart_paths 非空",
            "blackboard.computed_results.valuation 非空",
        ]
    },
    {
        "name": "冲突检测",
        "query": "分析宁德时代Q3财报（假设营收增20%但股价跌12%）",
        "check": [
            "final_answer 包含 '数据异常提示'",
            "blackboard.conflict_flags 非空",
            "存在 HIGH 级别冲突标记",
        ]
    },
    {
        "name": "公告情感分类",
        "query": "查询宁德时代最新公告并判断利好利空",
        "check": ["blackboard.computed_results 包含 sentiment 字段"],
    },
    {
        "name": "行业对标",
        "query": "对比宁德时代和比亚迪的ROE和毛利率",
        "check": [
            "final_answer 包含对比表格",
            "blackboard.chart_paths 包含对比图",
        ]
    },
    {
        "name": "Tushare 容灾 🆕",
        "query": "查询宁德时代实时行情 (Tushare 主数据源)",
        "check": [
            "market_data.quote 非空（Tushare 返回数据）",
        ]
    },
]
```

### 9.2 核心指标

| 指标 | 目标值 |
|------|--------|
| 研报溯源覆盖率 | 每段至少 1 个来源标记 |
| 数值校验准确率 | LLM 数字与实际值偏差 < 1% |
| 数据源路由准确率 | query 匹配 collection > 85% |
| 非金融场景零侵入 | 非金融 query 走原路径，输出完全一致 |
| **A 股数据可用率** 🆕 | Tushare 主数据源 + akshare 备选，双源保障 > 95% |
| **冲突检测召回率** 🆕 | 明显的营收/股价背离场景 100% 检出 |

---

## 十、总结：设计点 vs 现有架构的融合关系

```
设计点                    →   现有架构中的对应实现
──────────────────────────────────────────────────
① 6 Agent 按数据源分工    →   1 个 SearchAgent + CollectionRouter
② 研报合成 Agent          →   ReportRenderer(多模板) + LLM Generate
③ 溯源校验 Agent          →   CitationFootnoter + DataVerifier(Generate 内置)
④ 估值分析 Agent          →   FinancialCalculator 扩展 (ValuationResult)
⑤ 行业对标 Agent          →   IndustryBenchmark + BusinessComputeAgent._compare()
⑥ 公告事件 Agent          →   SentimentAnalyzer (情感分类工具)
⑦ 7 类数据源              →   6 个 collection + 1 个私有文档 collection
⑧ 冲突检测 🆕            →   ConflictDetector (aggregate 后新增节点)
⑨ A 股容灾 🆕            →   TushareClient (Tushare 主 + akshare 备降级)
⑩ Router合并 🆕           →   SupervisorAgent.classify_and_plan() (1次LLM替代2次)
⑪ 流式输出+思考过程 🆕      →   run_stream() + AgentEvent + SSE轮询 + Toggle面板

结果: 所有设计点都在现有架构中有对应的轻量化实现，
      **无需新增独立 Agent 节点**，只需扩展现有模块的能力边界。
      新增 9 个工具文件 + 修改 12 个现有文件 = 21 个文件的轻量改造。
```

---

## 附录 A: Tushare 安装与配置

### A.1 安装

```bash
pip install tushare
```

### A.2 Token 配置（二选一）

**方式一：环境变量（推荐，不提交 git）**

```bash
# .env 文件
TUSHARE_API_TOKEN=3664fe220cb675ae1661e7ad96c51e2592a0ef72c93d29da3d65b692
```

**方式二：settings.yaml**

```yaml
finance:
  market_data:
    a_share:
      tushare_token: "3664fe220cb675ae1661e7ad96c51e2592a0ef72c93d29da3d65b692"
```

### A.3 Tushare 接口说明

Tushare Pro 是 Tushare 的升级版，需要 token 访问。免费用户有积分限制（基础接口 200 次/分钟，基础积分 200 分）。

本项目使用的 Tushare 接口：

| 接口 | 功能 | 积分要求 |
|------|------|---------|
| `ts.pro_api().daily()` | 日线行情 | 基础 |
| `ts.pro_api().fina_indicator()` | 财务指标 | 基础 |
| `ts.pro_api().forcast()` | 业绩预告 | 基础 |
| `ts.pro_api().news()` | 新闻快讯 | 基础 |

---

*本文档编写于 2026-05-03，基于 Modular RAG MCP Server 现有代码库 (main branch)*
