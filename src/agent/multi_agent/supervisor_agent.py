"""
Supervisor Agent — Plan-then-Execute orchestrator (Merged with RouterAgent)

Implements intent classification + task planning in a single LLM call,
then Plan → Dispatch → Execute → Aggregate → Evaluate.

Phase 5-H: Merged RouterAgent's intent classification into PLAN_PROMPT.
RouterAgent class retained for backward compatibility.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.language_models import BaseLLM
from langchain_core.prompts import ChatPromptTemplate

from .state import AgentState, FallbackReason
from .tools.json_parser import safe_parse_json_with_default


class TaskType:
    DOCUMENT_SEARCH = "document_search"
    WEB_SEARCH = "web_search"
    FINANCIAL_MARKET = "financial_market"
    FINANCIAL_COMPUTATION = "financial_computation"


class SupervisorAgent:
    """
    Plan-then-Execute Supervisor (with merged intent classification).

    Phase 5-H: 一次 LLM 调用完成:
    1. 意图分类 (原 RouterAgent 职责)
    2. 路由决策 (原 RouterAgent 职责)
    3. 任务拆解 (原 Supervisor 职责)

    RouterAgent 类保留不动，用于测试兼容。
    """

    PLAN_PROMPT = """You are both an intent classifier and a task planner for a multi-agent system.

## Step 1: Intent Classification
Classify the user query into one of:
- chat: simple chitchat / general conversation (no retrieval needed)
- document_qa: asks about uploaded/local documents or internal knowledge
- web_search: needs latest/public/external information from the internet
- financial_analysis: analyze financial reports, stock performance, company fundamentals
  (signals: "分析财报", "PE", "ROE", "毛利率", "估值", "营收", "利润")
- financial_market: real-time market data, stock prices, market indices
  (signals: "股价", "涨跌", "市值", "市盈率", "行情")
- financial_report: generate a structured financial report or visualization
  (signals: "生成报告", "画图", "可视化", "K线图", "图表", "出具报告")

Also determine:
- needs_local (bool): should use local knowledge base?
  true for: document_qa, financial_analysis, financial_report
- needs_web (bool): should search the internet?
  true for: web_search, financial_market
- complexity: "simple" | "medium" | "complex"
  financial_* intents are almost always "complex"
- reasoning: brief explanation of your classification

## Step 2: Task Decomposition (only for non-chat intents)
If the intent requires work, decompose into subtasks.

Available agent types:
- document_search: search local knowledge base
- web_search: search the internet
- financial_market: query real-time stock quotes, fundamentals
- financial_computation: calculate financial ratios, compare, generate charts

Rules:
- Tasks with empty depends_on can run in parallel
- For financial queries, always include document_search + financial_market
- For "generate a report" or "create a chart", add a financial_computation task

Output JSON:
{{
    "intent": "financial_analysis",
    "needs_local": true,
    "needs_web": false,
    "complexity": "complex",
    "reasoning": "why this classification",
    "confidence": 0.95,
    "retrieve_plan": "financial",
    "task_plan": {{
        "subtasks": [
            {{
                "id": "task_1",
                "type": "document_search",
                "description": "Search quarterly reports for revenue data",
                "query": "Company Q3 2024 revenue profit gross margin",
                "depends_on": []
            }},
            {{
                "id": "task_2",
                "type": "financial_market",
                "description": "Get real-time quotes",
                "query": "Get PE PB ROE",
                "symbols": ["300750.SZ"],
                "depends_on": []
            }}
        ]
    }}
}}

For chat intent, task_plan should be null.
For simple document_qa or web_search, task_plan can be a single retrieve task.
For financial intents, always produce a full task plan with at least document_search + financial_market.
"""

    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self._plan_chain = (
            ChatPromptTemplate.from_messages([
                ("system", self.PLAN_PROMPT + "\n\nCurrent datetime: {current_datetime}"),
                ("user", "{query}"),
            ])
            | self.llm
        )
        self._collection_router = None

    @property
    def collection_router(self):
        if self._collection_router is None:
            try:
                from .tools.collection_router import CollectionRouter
                self._collection_router = CollectionRouter()
            except ImportError:
                self._collection_router = None
        return self._collection_router

    def classify_and_plan(self, state: AgentState) -> AgentState:
        """
        一次 LLM 调用完成意图分类 + 任务拆解。
        """
        from datetime import datetime
        now = datetime.now()
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        current_dt = f"{now.strftime('%Y-%m-%d')} {weekdays[now.weekday()]} {now.strftime('%H:%M')}"

        response = self._plan_chain.invoke({
            "query": state.user_input,
            "current_datetime": current_dt,
        })

        content = response.content if hasattr(response, "content") else str(response)
        parsed = safe_parse_json_with_default(content, {
            "intent": "document_qa",
            "needs_local": True,
            "needs_web": False,
            "complexity": "simple",
            "confidence": 0.5,
            "reasoning": "parse error, using defaults",
            "task_plan": None,
        })

        intent = parsed.get("intent", "document_qa")
        needs_local = bool(parsed.get("needs_local", True))
        needs_web = bool(parsed.get("needs_web", False))
        complexity = str(parsed.get("complexity", "simple")).lower()
        confidence = float(parsed.get("confidence", 0.5))
        reasoning = parsed.get("reasoning", "")
        retrieve_plan = parsed.get("retrieve_plan", "local")

        state.add_to_blackboard("intent", intent.lower(), "supervisor")
        state.add_to_blackboard("routing_decision", {
            "needs_local": needs_local,
            "needs_web": needs_web,
            "complexity": complexity,
            "reasoning": reasoning,
        }, "supervisor")
        state.add_to_blackboard("needs_local", needs_local, "supervisor")
        state.add_to_blackboard("needs_web", needs_web, "supervisor")
        state.add_to_blackboard("query_complexity", complexity, "supervisor")
        state.add_to_blackboard("router_confidence", confidence, "supervisor")

        # 🆕 检测分析深度
        depth = self._detect_analysis_depth(state.user_input)
        state.add_to_blackboard("analysis_depth", depth, "supervisor")

        # 决定 retrieve_plan
        if intent == "chat":
            plan = "none"
        elif intent.startswith("financial_"):
            plan = "financial"
        elif needs_local and needs_web:
            plan = "both"
        elif needs_web and not needs_local:
            plan = "web"
        elif needs_local and not needs_web:
            plan = "local"
        elif complexity == "complex":
            plan = "both"
        else:
            plan = "local"
        state.add_to_blackboard("retrieve_plan", plan, "supervisor")

        # 任务拆解
        task_plan = parsed.get("task_plan")
        if task_plan and task_plan.get("subtasks"):
            state.add_to_blackboard("task_plan", task_plan, "supervisor")
        else:
            state.add_to_blackboard("task_plan", None, "supervisor")

        state.add_execution_trace({
            "agent": "supervisor",
            "action": "classify_and_plan",
            "intent": intent,
            "confidence": confidence,
            "complexity": complexity,
            "retrieve_plan": plan,
            "task_count": len(task_plan.get("subtasks", [])) if task_plan and task_plan.get("subtasks") else 0,
        })
        return state

    @staticmethod
    def _detect_analysis_depth(query: str) -> str:
        q = query.lower()
        quick_keywords = ["快速", "简单", "看一眼", "brief", "quick", "simple", "粗略"]
        deep_keywords = ["深度", "详细", "全面", "完整", "deep", "detailed", "comprehensive", "深入"]
        for kw in quick_keywords:
            if kw in q:
                return "quick"
        for kw in deep_keywords:
            if kw in q:
                return "deep"
        return "standard"

    def _plan_financial(self, state: AgentState) -> AgentState:
        # 保留原逻辑作为 fallback，但 classify_and_plan 已合并此功能
        routing = state.blackboard.get("routing_decision", {})
        needs_local = state.blackboard.get("needs_local", True)
        needs_web = state.blackboard.get("needs_web", False)
        intent = state.intent or "financial_analysis"

        subtasks = []
        tid = 0

        # 通过 CollectionRouter 确定要检索的 collections
        collections = ["financial_reports"]
        if self.collection_router:
            route_result = self.collection_router.route(state.user_input, intent)
            collections = route_result.get("collections", ["financial_reports"])

        if needs_local:
            tid += 1
            subtasks.append({
                "id": f"task_{tid}",
                "type": TaskType.DOCUMENT_SEARCH,
                "description": "检索本地知识库中的相关文档",
                "query": state.user_input,
                "collections": collections,
                "depends_on": [],
            })

        tid += 1
        subtasks.append({
            "id": f"task_{tid}",
            "type": TaskType.FINANCIAL_MARKET,
            "description": "获取实时行情与基本面数据",
            "query": state.user_input,
            "symbols": self._extract_symbols(state.user_input),
            "depends_on": [],
        })

        tid += 1
        subtasks.append({
            "id": f"task_{tid}",
            "type": TaskType.FINANCIAL_COMPUTATION,
            "description": "计算财务指标并生成分析结果",
            "query": state.user_input,
            "depends_on": [f"task_{tid - 2}"] if needs_local else [f"task_{tid - 1}"],
        })

        plan = {
            "original_query": state.user_input,
            "reasoning": "financial intent decomposition",
            "subtasks": subtasks,
        }
        state.add_to_blackboard("task_plan", plan, "supervisor")
        state.add_to_blackboard("finance_collections", collections, "supervisor")
        return state

    def _plan_document(self, state: AgentState) -> AgentState:
        subtasks = [{
            "id": "task_1",
            "type": TaskType.DOCUMENT_SEARCH,
            "description": "检索本地知识库",
            "query": state.user_input,
            "depends_on": [],
        }]
        plan = {
            "original_query": state.user_input,
            "reasoning": "document intent — single retrieval task",
            "subtasks": subtasks,
        }
        state.add_to_blackboard("task_plan", plan, "supervisor")
        return state

    def get_next_subtasks(self, state: AgentState) -> List[Dict[str, Any]]:
        plan = state.task_plan
        if not plan:
            return []
        subtasks = plan.get("subtasks", [])
        completed_ids = set(state.blackboard.get("completed_tasks", []))
        ready = []
        for task in subtasks:
            if task["id"] in completed_ids:
                continue
            deps = task.get("depends_on", [])
            if all(d in completed_ids for d in deps):
                ready.append(task)
        return ready

    def mark_completed(self, state: AgentState, task_id: str) -> AgentState:
        completed = state.blackboard.get("completed_tasks", [])
        if task_id not in completed:
            completed.append(task_id)
        state.add_to_blackboard("completed_tasks", completed, "supervisor")
        return state

    def all_completed(self, state: AgentState) -> bool:
        plan = state.task_plan
        if not plan:
            return True
        completed = state.blackboard.get("completed_tasks", [])
        return len(completed) >= len(plan.get("subtasks", []))

    def aggregate(self, state: AgentState) -> AgentState:
        aggregated = {
            "local_docs": state.blackboard.get("local_results", []),
            "web_info": state.blackboard.get("web_results", []),
            "market": state.blackboard.get("market_data", {}),
            "computation": state.blackboard.get("computed_results", {}),
            "charts": state.blackboard.get("chart_paths", []),
            "report": state.blackboard.get("generated_report", None),
        }
        state.add_to_blackboard("aggregated_context", aggregated, "supervisor")
        return state

    def _extract_symbols(self, query: str) -> List[str]:
        patterns = [
            r'\b\d{6}\.S[ZH]\b',
            r'\b[A-Z]{1,5}\b',
            r'\b\d{4}\.HK\b',
        ]
        symbols = []
        for p in patterns:
            found = re.findall(p, query.upper())
            symbols.extend(f for f in found if f not in symbols)
        return symbols[:5]
