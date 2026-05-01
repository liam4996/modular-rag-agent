"""
多智能体 RAG 系统 - 主编排器

使用 LangGraph 编排多个专门智能体：
- Router Agent: 意图识别和路由
- Supervisor Agent: Plan-then-Execute 编排
- Search Agent: 本地知识库检索
- Web Agent: 联网搜索
- Finance Data Agent: 金融行情查询
- Read Node: 阅读检索结果并提炼阶段性观察
- Eval Agent: 质量评估 / 反思下一步动作
- Refine Agent: 查询优化
- Aggregate Node: 多源结果聚合
- Global Eval Node: 最终质量审核
- Generate Agent: 最终回答生成
"""

from concurrent.futures import ThreadPoolExecutor
import json
from typing import Optional, List, Dict, Literal, Tuple, Any

from langgraph.graph import StateGraph, END

from .state import AgentState, FallbackReason
from .router_agent import RouterAgent, AgentType, RoutingDecision
from .search_agent import SearchAgent
from .web_agent import WebSearchAgent
from .eval_agent import EvalAgent, EvaluationResult
from .refine_agent import RefineAgent
from .supervisor_agent import SupervisorAgent
from .finance_data_agent import FinanceDataAgent
from .citation import (
    Citation,
    CitationType,
    CitationManager,
    FaithfulnessCheck,
    format_answer_with_citations,
)


class MultiAgentRAG:
    """
    多智能体 RAG 系统主类
    
    使用 LangGraph 编排多个专门智能体，支持：
    - 并行融合检索
    - 共享状态（Blackboard Pattern）
    - 容错机制（重试 + 兜底）
    - 溯源与忠实度保证
    - Plan-then-Execute 金融 Agent 编排
    
    工作流程：
    1. Router Agent 识别意图
    2. 非金融意图走原有闭环（RAG → Eval → Refine）
    3. 金融意图走 Supervisor → Plan → Dispatch → Execute → Aggregate → Global Eval
    4. Generate Agent 生成最终回答
    """
    
    def __init__(
        self,
        llm,
        settings: Optional[Dict] = None,
        enable_logging: bool = True,
        store=None,
    ):
        self.llm = llm
        self.settings = settings or {}
        self.enable_logging = enable_logging
        self.store = store
        
        # 初始化所有智能体
        self.router_agent = RouterAgent(self.llm)
        self.search_agent = SearchAgent(self.settings)
        self.web_agent = WebSearchAgent(self.settings)
        self.eval_agent = EvalAgent(self.llm)
        self.refine_agent = RefineAgent(self.llm)
        self.supervisor_agent = SupervisorAgent(self.llm)
        self.finance_data_agent = FinanceDataAgent(self.settings)
        
        # 构建工作流图
        self.workflow = self._build_graph()

    _ROUTER_PARALLEL_CONFIDENCE_THRESHOLD = 0.7
    _LOCAL_RELEVANCE_THRESHOLD = 0.3
    
    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 状态机（包含容错机制 + 金融 Agent 节点）"""
        
        workflow = StateGraph(AgentState)
        
        # ========== 添加节点 ==========
        workflow.add_node("router", self._router_node)
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("web", self._web_node)
        workflow.add_node("read", self._read_node)
        workflow.add_node("eval", self._eval_node)
        workflow.add_node("refine", self._refine_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("finance_data", self._finance_data_node)
        workflow.add_node("aggregate", self._aggregate_node)
        workflow.add_node("global_eval", self._global_eval_node)
        
        workflow.set_entry_point("router")
        
        workflow.add_conditional_edges(
            "router",
            self._route_after_router,
            {
                "generate": "generate",
                "plan": "plan",
                "retrieve": "retrieve",
                "supervisor": "supervisor",
            },
        )

        workflow.add_conditional_edges(
            "plan",
            self._route_after_plan,
            {
                "generate": "generate",
                "retrieve": "retrieve",
            },
        )

        workflow.add_conditional_edges(
            "supervisor",
            self._dispatch_from_supervisor,
            {
                "retrieve": "retrieve",
                "web": "web",
                "finance_data": "finance_data",
                "aggregate": "aggregate",
                "generate": "generate",
            },
        )

        workflow.add_edge("retrieve", "read")
        workflow.add_edge("web", "read")
        workflow.add_edge("read", "eval")
        workflow.add_edge("finance_data", "aggregate")
        
        workflow.add_conditional_edges(
            "eval",
            self._eval_next_step,
            {
                "generate": "generate",
                "plan": "plan",
                "refine": "refine",
                "web": "web",
                "aggregate": "aggregate",
            }
        )
        
        workflow.add_edge("refine", "retrieve")
        workflow.add_edge("aggregate", "global_eval")
        workflow.add_edge("global_eval", "generate")
        workflow.add_edge("generate", END)
        
        compile_kwargs = {}
        if self.store is not None:
            compile_kwargs["store"] = self.store
        return workflow.compile(**compile_kwargs)
    
    # ========== 节点实现 ==========
    
    def _router_node(self, state: AgentState) -> AgentState:
        """Router Agent 节点"""
        decision = self.router_agent.classify(
            query=state.user_input,
            context=state.conversation_history
        )
        
        state.add_to_blackboard("intent", decision.intent.lower(), "router")
        state.add_to_blackboard(
            "routing_decision",
            {
                "agents_to_invoke": [a.value for a in decision.agents_to_invoke],
                "needs_local": decision.needs_local,
                "needs_web": decision.needs_web,
                "complexity": decision.complexity,
                "parallel": decision.parallel,
                "reasoning": decision.reasoning,
            },
            "router"
        )

        intent = decision.intent.lower()
        conf = float(decision.confidence)
        needs_local = bool(decision.needs_local)
        needs_web = bool(decision.needs_web)
        complexity = (decision.complexity or "simple").lower()

        state.add_to_blackboard("needs_local", needs_local, "router")
        state.add_to_blackboard("needs_web", needs_web, "router")
        state.add_to_blackboard("query_complexity", complexity, "router")
        state.add_to_blackboard("router_confidence", conf, "router")

        if intent == "chat":
            retrieve_plan = "none"
        elif intent.startswith("financial_"):
            retrieve_plan = "financial"
        elif intent != "chat" and conf < self._ROUTER_PARALLEL_CONFIDENCE_THRESHOLD:
            retrieve_plan = "both"
        elif needs_local and needs_web:
            retrieve_plan = "both"
        elif needs_web and not needs_local:
            retrieve_plan = "web"
        elif needs_local and not needs_web:
            retrieve_plan = "local"
        elif complexity == "complex" and intent != "chat":
            retrieve_plan = "both"
        elif decision.parallel or intent == "hybrid_search":
            retrieve_plan = "both"
        elif intent == "web_search":
            retrieve_plan = "web"
        else:
            retrieve_plan = "local"

        state.add_to_blackboard("retrieve_plan", retrieve_plan, "router")
        
        state.add_execution_trace({
            "agent": "router",
            "action": "classify_intent",
            "result": decision.intent,
            "confidence": decision.confidence,
            "agents_to_invoke": [a.value for a in decision.agents_to_invoke],
            "needs_local": needs_local,
            "needs_web": needs_web,
            "complexity": complexity,
            "parallel": decision.parallel,
            "retrieve_plan": retrieve_plan,
        })
        
        return state

    def _plan_node(self, state: AgentState) -> AgentState:
        """显式规划节点"""
        plan_data = state.blackboard.get("task_plan")
        replan_requested = bool(state.blackboard.get("replan_requested", False))

        if not plan_data or replan_requested:
            routing = state.blackboard.get("routing_decision", {})
            plan_data = self._create_explicit_plan(
                query=state.user_input,
                conversation_history=state.conversation_history,
                routing=routing,
            )
            state.add_to_blackboard("task_plan", plan_data, "plan")
            state.add_to_blackboard("plan_current_step", 0, "plan")
            state.blackboard["replan_requested"] = False
        elif state.blackboard.get("advance_plan_step", False):
            current_step = int(state.blackboard.get("plan_current_step", 0))
            state.add_to_blackboard("plan_current_step", current_step + 1, "plan")
            state.blackboard["advance_plan_step"] = False

        plan_steps = plan_data.get("steps", [])
        current_step = int(state.blackboard.get("plan_current_step", 0))

        if current_step >= len(plan_steps):
            state.blackboard["plan_complete"] = True
            state.add_execution_trace({"agent": "plan", "action": "plan_complete", "step_count": len(plan_steps)})
            return state

        step = plan_steps[current_step]
        active_sub_question = step.get("sub_question", state.user_input)
        preferred_source = step.get("preferred_source", state.blackboard.get("retrieve_plan", "local"))

        if preferred_source not in ("local", "web", "both"):
            preferred_source = state.blackboard.get("retrieve_plan", "local")

        state.blackboard["plan_complete"] = False
        state.add_to_blackboard("active_sub_question", active_sub_question, "plan")
        state.add_to_blackboard("active_retrieve_plan", preferred_source, "plan")
        state.add_to_blackboard("active_plan_goal", step.get("goal", ""), "plan")

        state.add_execution_trace({
            "agent": "plan", "action": "select_plan_step",
            "step_index": current_step, "step_count": len(plan_steps),
            "sub_question": active_sub_question, "preferred_source": preferred_source,
            "goal": step.get("goal", ""),
        })

        return state
    
    def _retrieve_node(self, state: AgentState) -> AgentState:
        """统一检索节点"""
        plan = state.blackboard.get("active_retrieve_plan", state.blackboard.get("retrieve_plan", "local"))
        base_query = state.blackboard.get("active_sub_question", state.user_input)
        query = state.refined_query if state.retry_count > 0 else base_query
        context = state.conversation_history

        state.blackboard["retrieval_executed"] = True

        local_error: Optional[str] = None

        def _run_local() -> List[Dict]:
            try:
                return self.search_agent.search(query=query, top_k=10, context=context)
            except Exception as e:
                nonlocal local_error
                local_error = str(e)
                return []

        def _run_web_no_local_ctx() -> List[Dict]:
            try:
                return self.web_agent.search(query=query, num_results=5, time_range="y", local_results=None)
            except Exception:
                return []

        if plan == "local":
            results = _run_local()
            state.add_to_blackboard("local_results", results, "retrieve")
            state.add_to_blackboard("web_results", [], "retrieve")
            state.add_metric("local_result_count", len(results))
            state.add_metric("retrieve_mode", "local")
        elif plan == "web":
            web_results = _run_web_no_local_ctx()
            state.add_to_blackboard("local_results", [], "retrieve")
            state.add_to_blackboard("web_results", web_results, "retrieve")
            state.blackboard["web_search_attempted"] = True
            state.add_metric("web_result_count", len(web_results))
            state.add_metric("retrieve_mode", "web")
        else:
            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_local = ex.submit(_run_local)
                fut_web = ex.submit(_run_web_no_local_ctx)
                local_results = fut_local.result()
                web_results = fut_web.result()

            state.add_to_blackboard("local_results", local_results, "retrieve")
            state.add_to_blackboard("web_results", web_results, "retrieve")
            state.blackboard["web_search_attempted"] = True
            state.add_metric("local_result_count", len(local_results))
            state.add_metric("web_result_count", len(web_results))
            state.add_metric("retrieve_mode", "parallel_both")

        return state
    
    def _web_node(self, state: AgentState) -> AgentState:
        """Web Agent 节点（联网搜索）"""
        local_results = state.read_from_blackboard("local_results")
        active_query = state.blackboard.get("active_sub_question", state.user_input)
        web_results = self.web_agent.search(query=active_query, num_results=5, time_range="y", local_results=local_results)
        state.blackboard["web_search_attempted"] = True
        state.add_to_blackboard("web_results", web_results, "web")
        state.add_metric("web_result_count", len(web_results))
        return state

    def _read_node(self, state: AgentState) -> AgentState:
        """阅读节点"""
        sub_question = state.blackboard.get("active_sub_question", state.user_input)
        goal = state.blackboard.get("active_plan_goal", "")
        reading = self._read_retrieved_evidence(
            query=sub_question, goal=goal,
            local_results=state.local_results, web_results=state.web_results,
        )
        state.add_to_blackboard("reading_assessment", reading, "read")
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
        return state
    
    def _eval_node(self, state: AgentState) -> AgentState:
        """Eval Agent 节点（质量评估）"""
        local_results = state.local_results
        web_results = state.web_results
        eval_query = state.blackboard.get("active_sub_question", state.user_input)
        evaluation = self.eval_agent.evaluate(
            local_results=local_results, web_results=web_results,
            query=eval_query, retry_count=state.retry_count, max_retries=state.max_retries
        )
        state.add_to_blackboard("evaluation", {
            "relevance": evaluation.relevance, "diversity": evaluation.diversity,
            "coverage": evaluation.coverage, "confidence": evaluation.confidence,
            "need_refinement": evaluation.need_refinement,
            "fallback_suggested": evaluation.fallback_suggested,
            "reason": evaluation.reason, "step_query": eval_query,
        }, "eval")
        state.add_metric("evaluation_relevance", evaluation.relevance)
        state.add_metric("evaluation_confidence", evaluation.confidence)
        if evaluation.fallback_suggested:
            if state.retry_count >= state.max_retries:
                state.trigger_fallback(FallbackReason.MAX_RETRIES_EXCEEDED, "eval")
            elif evaluation.relevance < 0.2:
                state.trigger_fallback(FallbackReason.NO_RESULTS_FOUND, "eval")
            elif evaluation.confidence < 0.3:
                state.trigger_fallback(FallbackReason.LOW_CONFIDENCE, "eval")
        return state
    
    def _refine_node(self, state: AgentState) -> AgentState:
        """Refine Agent 节点（查询优化）"""
        evaluation_data = state.evaluation
        evaluation = EvaluationResult(
            relevance=evaluation_data.get("relevance", 0.5),
            diversity=evaluation_data.get("diversity", 0.5),
            coverage=evaluation_data.get("coverage", 0.5),
            confidence=evaluation_data.get("confidence", 0.5),
            need_refinement=evaluation_data.get("need_refinement", True),
            fallback_suggested=evaluation_data.get("fallback_suggested", False),
            reason=evaluation_data.get("reason", ""),
        )
        target_query = state.blackboard.get("active_sub_question", state.user_input)
        refinement = self.refine_agent.refine(
            original_query=target_query, evaluation=evaluation, retry_count=state.retry_count
        )
        state.increment_retry("refine")
        state.add_to_blackboard("refined_query", refinement.refined_query, "refine")
        state.add_metric("refinement_changes", refinement.changes_made)
        return state
    
    def _generate_node(self, state: AgentState) -> AgentState:
        """Generate Agent 节点（最终生成）"""
        all_context = state.get_all_context()
        has_local = bool(state.local_results)
        has_web = bool(state.web_results)
        web_attempted = state.blackboard.get("web_search_attempted", False)
        eval_confidence = state.blackboard.get("evaluation", {}).get("confidence", 1.0)

        local_all_noise = False
        if has_local and eval_confidence < 0.3:
            useful = [r for r in state.local_results if r.get("score", 0) >= self._LOCAL_RELEVANCE_THRESHOLD]
            local_all_noise = len(useful) == 0

        intent = (state.intent or "").lower()
        retrieval_ran = bool(state.blackboard.get("retrieval_executed", False))
        search_was_attempted = bool(state.retry_count > 0 or web_attempted or retrieval_ran)
        is_chat_direct = intent == "chat" and not search_was_attempted

        if is_chat_direct:
            gen_mode = "general_knowledge"
        elif not has_local and not has_web:
            gen_mode = "general_knowledge" if web_attempted else "fallback"
        elif local_all_noise and not has_web and web_attempted:
            gen_mode = "general_knowledge"
        else:
            gen_mode = "normal"

        if gen_mode == "fallback":
            state.final_answer = self._generate_fallback_response(state)
        elif gen_mode == "general_knowledge":
            state.final_answer = self._generate_general_knowledge_answer(all_context)
        else:
            answer, citations, faithfulness = self._generate_normal_response_with_citations(all_context)
            state.final_answer = answer
            state.add_to_blackboard("citations", [c.to_dict() for c in citations], "generate")
            state.add_to_blackboard("faithfulness_check", faithfulness.to_dict(), "generate")

        state.add_metric("generation_mode", gen_mode)
        return state
    
    # ========== 金融 Agent 节点 ==========
    
    def _supervisor_node(self, state: AgentState) -> AgentState:
        """Supervisor 节点：Plan 阶段"""
        self.supervisor_agent.classify_and_plan(state)
        state.add_execution_trace({
            "agent": "supervisor", "action": "plan_complete",
            "task_count": len(state.blackboard.get("task_plan", {}).get("subtasks", [])),
        })
        return state
    
    def _dispatch_from_supervisor(
        self, state: AgentState
    ) -> Literal["retrieve", "web", "finance_data", "aggregate", "generate"]:
        """按 task_plan 拓扑排序派发"""
        next_tasks = self.supervisor_agent.get_next_subtasks(state)
        if not next_tasks:
            return "aggregate" if self.supervisor_agent.all_completed(state) else "generate"
        
        task = next_tasks[0]
        task_type = task.get("type", "")
        self.supervisor_agent.mark_completed(state, task["id"])
        
        query = task.get("query", state.user_input)
        state.add_to_blackboard("active_sub_question", query, "supervisor")
        state.add_to_blackboard("current_task", task, "supervisor")
        
        state.add_execution_trace({"agent": "supervisor", "action": "dispatch", "task_id": task["id"], "task_type": task_type})
        
        if task_type == "document_search":
            return "retrieve"
        elif task_type == "web_search":
            return "web"
        elif task_type == "financial_market":
            return "finance_data"
        else:
            return "aggregate"
    
    def _finance_data_node(self, state: AgentState) -> AgentState:
        """Finance Data Agent 节点"""
        task = state.blackboard.get("current_task", {})
        symbols = task.get("symbols", [])
        if not symbols:
            symbols = self.supervisor_agent._extract_symbols(state.user_input)
        try:
            self.finance_data_agent.query_and_store(
                state=state, symbols=symbols or ["000001.SZ"], data_types=["quote", "fundamentals"],
            )
        except Exception as e:
            state.add_execution_trace({"agent": "finance_data", "action": "error", "error": str(e)})
        return state
    
    def _aggregate_node(self, state: AgentState) -> AgentState:
        """聚合节点"""
        self.supervisor_agent.aggregate(state)
        return state
    
    def _global_eval_node(self, state: AgentState) -> AgentState:
        """Global Eval 节点：最终质量审核"""
        aggregated = state.blackboard.get("aggregated_context", {})
        market_data = aggregated.get("market", {})
        issues = []
        if not market_data and state.is_financial_intent:
            issues.append("market_data_missing")
        state.add_to_blackboard("global_eval_issues", issues, "global_eval")
        state.add_to_blackboard("global_eval_passed", len(issues) == 0, "global_eval")
        return state
    
    # ========== 条件路由函数 ==========
    
    def _route_after_router(self, state: AgentState) -> Literal["generate", "plan", "retrieve", "supervisor"]:
        """闲聊跳过检索；金融意图走 Supervisor；复杂问题先走 plan，其余进入 retrieve。"""
        intent = (state.intent or "").lower()
        plan = state.blackboard.get("retrieve_plan", "local")
        complexity = state.blackboard.get("query_complexity", "simple")
        
        if intent.startswith("financial_"):
            return "supervisor"
        if plan == "none":
            return "generate"
        if complexity == "complex":
            return "plan"
        return "retrieve"

    def _route_after_plan(self, state: AgentState) -> Literal["generate", "retrieve"]:
        """规划完成后若步骤已完成直接生成，否则进入检索。"""
        if state.blackboard.get("plan_complete", False):
            return "generate"
        return "retrieve"
    
    def _eval_next_step(self, state: AgentState) -> Literal["generate", "plan", "refine", "web", "aggregate"]:
        """Eval 之后的多路决策（含自动联网升级 + 回退本地检索 + 金融聚合）。"""
        intent = (state.intent or "").lower()
        
        if intent.startswith("financial_"):
            state.blackboard["advance_plan_step"] = False
            return "aggregate"
        
        web_attempted = state.blackboard.get("web_search_attempted", False)
        evaluation = state.evaluation
        reading = state.blackboard.get("reading_assessment", {})
        need_refinement = evaluation.get("need_refinement", False)
        fallback_suggested = evaluation.get("fallback_suggested", False)
        confidence = evaluation.get("confidence", 0.5)
        enough = bool(reading.get("enough_to_answer_sub_question", False))
        suggested_next_action = reading.get("suggested_next_action", "")
        task_plan = state.blackboard.get("task_plan", {})
        plan_steps = task_plan.get("steps", [])
        current_step = int(state.blackboard.get("plan_current_step", 0))
        has_more_plan_steps = current_step < max(len(plan_steps) - 1, 0)
        retrieve_plan = state.blackboard.get("retrieve_plan", "local")
        web_results = state.web_results
        local_results = state.local_results

        def _escalate_to_web(reason: str) -> Literal["web"]:
            state.add_execution_trace({"agent": "eval", "action": "escalate_to_web", "reason": reason})
            return "web"

        def _fallback_to_local(reason: str) -> Literal["refine"]:
            state.add_to_blackboard("retrieve_plan", "local", "eval")
            state.add_execution_trace({"agent": "eval", "action": "fallback_to_local", "reason": reason})
            return "refine"

        if enough and has_more_plan_steps:
            state.blackboard["advance_plan_step"] = True
            return "plan"

        if suggested_next_action == "search_web" and not web_attempted:
            return _escalate_to_web("reader suggested web search for missing information")

        if need_refinement and state.retry_count < state.max_retries:
            return "refine"

        if web_attempted and confidence < 0.4 and len(local_results) > 0 and state.retry_count < state.max_retries:
            return _fallback_to_local("web search confidence too low")

        if web_attempted:
            return "generate"

        if state.fallback_triggered or fallback_suggested:
            return _escalate_to_web("local search fallback triggered")

        if need_refinement and state.retry_count >= state.max_retries:
            return _escalate_to_web("retries exhausted, trying web as last resort")

        if confidence < 0.5:
            return _escalate_to_web(f"low confidence ({confidence}), trying web")

        if enough and not has_more_plan_steps:
            return "generate"

        return "generate"
    