"""
Supervisor Agent — Plan-then-Execute orchestrator

Implements Plan → Dispatch → Execute → Aggregate → Evaluate
for the multi-agent financial RAG system.

Replaces the simple Router-based flow with structured task decomposition.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.language_models import BaseLLM
from langchain_core.prompts import ChatPromptTemplate

from .state import AgentState, FallbackReason


class TaskType:
    DOCUMENT_SEARCH = "document_search"
    WEB_SEARCH = "web_search"
    FINANCIAL_MARKET = "financial_market"
    FINANCIAL_COMPUTATION = "financial_computation"


class SupervisorAgent:
    """
    Plan-then-Execute Supervisor.

    四个阶段：
    1. Plan: LLM 一次性拆出所有子任务，写入 task_plan
    2. Dispatch: 按 depends_on 拓扑排序，无依赖并行派发
    3. Aggregate: 收集所有 sub-agent 输出，合并为统一上下文
    4. Evaluate: 全局审核 → generate / replan / retry
    """

    PLAN_PROMPT = """You are a task planner for a multi-agent financial analysis system.

Given a user query, decompose it into subtasks that can be executed by specialized agents.

Available agent types:
- document_search: search local knowledge base for reports, filings, transcripts
- web_search: search the internet for public/latest information
- financial_market: query real-time stock quotes, fundamentals, history
- financial_computation: calculate financial ratios, compare metrics, generate charts

Output a JSON with:
- subtasks: list of task objects
- Each task has:
  - id: unique task id (e.g. "task_1")
  - type: one of the agent types above
  - description: what this task should accomplish
  - query: the specific search query or operation description
  - symbols: (if financial_market) list of stock symbols
  - depends_on: list of task IDs this task must wait for

Rules:
- Tasks with empty depends_on can run in parallel
- Add a "synthesize" task at the end if complex analysis is needed
- For financial queries, always include document_search for report details
- For "generate a report" or "create a chart", add a financial_computation task

Respond ONLY in JSON:
{{
    "original_query": "user query",
    "reasoning": "why we decomposed it this way",
    "subtasks": [
        {{
            "id": "task_1",
            "type": "document_search",
            "description": "Search quarterly reports for revenue and profit data",
            "query": "Company Q3 2024 revenue profit gross margin",
            "depends_on": []
        }},
        {{
            "id": "task_2",
            "type": "financial_market",
            "description": "Get real-time quotes and fundamentals",
            "query": "Get PE PB ROE market cap",
            "symbols": ["300750.SZ"],
            "depends_on": []
        }}
    ]
}}"""

    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self._plan_chain = (
            ChatPromptTemplate.from_messages([
                ("system", self.PLAN_PROMPT),
                ("user", "{query}"),
            ])
            | self.llm
        )

    def classify_and_plan(self, state: AgentState) -> AgentState:
        intent = state.intent or "fact_query"

        if intent.startswith("financial_"):
            return self._plan_financial(state)
        elif intent in ("document_qa", "summarization", "comparison", "analysis"):
            return self._plan_document(state)
        else:
            state.add_to_blackboard("task_plan", None, "supervisor")
            return state

    def _plan_financial(self, state: AgentState) -> AgentState:
        subtasks = []
        tid = 0

        needs_local = state.blackboard.get("needs_local", True)

        if needs_local:
            tid += 1
            subtasks.append({
                "id": f"task_{tid}",
                "type": TaskType.DOCUMENT_SEARCH,
                "description": "检索本地知识库中的相关文档",
                "query": state.user_input,
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
        state.add_execution_trace({
            "agent": "supervisor",
            "action": "plan_financial",
            "task_count": len(subtasks),
        })
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
