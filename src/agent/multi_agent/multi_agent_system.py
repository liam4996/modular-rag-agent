"""
多智能体 RAG 系统 - 主编排器

使用 LangGraph 编排多个专门智能体：
- Router Agent: 意图识别和路由
- Plan Node: 显式规划（拆解子问题、决定先本地还是先联网）
- Search Agent: 本地知识库检索
- Web Agent: 联网搜索
- Read Node: 阅读检索结果并提炼阶段性观察
- Eval Agent: 质量评估 / 反思下一步动作
- Refine Agent: 查询优化
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
    
    工作流程：
    1. Router Agent 识别意图
    2. 根据意图路由到不同的 Agent
    3. 复杂查询触发并行检索（Search + Web）
    4. Eval Agent 评估质量
    5. Refine Agent 优化查询（如果需要）
    6. Generate Agent 生成最终回答
    """
    
    def __init__(
        self,
        llm,
        settings: Optional[Dict] = None,
        enable_logging: bool = True,
        store=None,
    ):
        """
        初始化多智能体 RAG 系统
        
        Args:
            llm: 语言模型实例
            settings: 系统配置
            enable_logging: 是否启用日志
            store: LangGraph BaseStore (长期记忆), 可选
        """
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
        
        # 构建工作流图
        self.workflow = self._build_graph()

    # Router 置信度低于此阈值且非闲聊 → 强制本地+联网并行检索（黑板汇合后再 Eval）
    _ROUTER_PARALLEL_CONFIDENCE_THRESHOLD = 0.7
    
    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 状态机（包含容错机制）"""
        
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
        
        # ========== 设置入口点 ==========
        workflow.set_entry_point("router")
        
        # Router → 闲聊直出 Generate；复杂问题先显式规划；其余进入 retrieve
        workflow.add_conditional_edges(
            "router",
            self._route_after_router,
            {
                "generate": "generate",
                "plan": "plan",
                "retrieve": "retrieve",
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

        # retrieve（含真·并行）→ read → eval
        workflow.add_edge("retrieve", "read")

        # Web（Eval / Read 升级联网）→ 再回到 read
        workflow.add_edge("web", "read")
        workflow.add_edge("read", "eval")
        
        # Eval → Generate / Refine / Web / Plan（继续计划）
        workflow.add_conditional_edges(
            "eval",
            self._eval_next_step,
            {
                "generate": "generate",
                "plan": "plan",
                "refine": "refine",
                "web": "web",
            }
        )
        
        # Refine → 按原 retrieve_plan 重新检索（hybrid 仍为并行）
        workflow.add_edge("refine", "retrieve")
        
        # Generate → END
        workflow.add_edge("generate", END)
        
        compile_kwargs = {}
        if self.store is not None:
            compile_kwargs["store"] = self.store
        return workflow.compile(**compile_kwargs)
    
    # ========== 节点实现 ==========
    
    def _router_node(self, state: AgentState) -> AgentState:
        """
        Router Agent 节点
        
        职责：意图识别 + 写入黑板
        """
        # 意图分类
        decision = self.router_agent.classify(
            query=state.user_input,
            context=state.conversation_history
        )
        
        # 写入黑板（共享！）— 统一小写，防止大小写不一致导致路由失败
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

        # retrieve_plan: none | local | web | both
        if intent == "chat":
            retrieve_plan = "none"
        elif intent != "chat" and conf < self._ROUTER_PARALLEL_CONFIDENCE_THRESHOLD:
            retrieve_plan = "both"
            state.add_execution_trace({
                "agent": "router",
                "action": "force_parallel_low_confidence",
                "confidence": conf,
                "reason": f"intent confidence < {self._ROUTER_PARALLEL_CONFIDENCE_THRESHOLD}, run local+web in parallel",
            })
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
        
        # 记录执行轨迹
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
        """
        显式规划节点：
        1. 将复杂问题拆成多个子问题
        2. 为每个子问题决定优先检索来源（local / web / both）
        3. 在多步循环中推进到当前应执行的步骤
        """
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
            state.add_execution_trace({
                "agent": "plan",
                "action": "plan_complete",
                "step_count": len(plan_steps),
            })
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
            "agent": "plan",
            "action": "select_plan_step",
            "step_index": current_step,
            "step_count": len(plan_steps),
            "sub_question": active_sub_question,
            "preferred_source": preferred_source,
            "goal": step.get("goal", ""),
        })

        return state
    
    def _retrieve_node(self, state: AgentState) -> AgentState:
        """
        统一检索节点：按黑板上的 retrieve_plan 执行本地 / 联网 / 真并行。

        hybrid 与「低置信度强制双路」在 plan==both 时用线程池同时跑 Search 与 Web，
        再写入黑板，最后在 Eval 汇合（避免 LangGraph 对 dataclass 并行写同一字段的合并限制）。
        """
        plan = state.blackboard.get(
            "active_retrieve_plan",
            state.blackboard.get("retrieve_plan", "local")
        )
        base_query = state.blackboard.get("active_sub_question", state.user_input)
        query = state.refined_query if state.retry_count > 0 else base_query
        context = state.conversation_history

        state.blackboard["retrieval_executed"] = True

        local_error: Optional[str] = None

        def _run_local() -> List[Dict]:
            try:
                return self.search_agent.search(
                    query=query, top_k=10, context=context
                )
            except Exception as e:
                nonlocal local_error
                local_error = str(e)
                return []

        def _run_web_no_local_ctx() -> List[Dict]:
            try:
                return self.web_agent.search(
                    query=query,
                    num_results=5,
                    time_range="y",
                    local_results=None,
                )
            except Exception:
                return []

        if plan == "local":
            results = _run_local()
            if local_error:
                state.add_execution_trace({
                    "agent": "retrieve",
                    "action": "local_search_error",
                    "error": local_error,
                })
            state.add_to_blackboard("local_results", results, "retrieve")
            state.add_to_blackboard("web_results", [], "retrieve")
            state.add_metric("local_result_count", len(results))
            state.add_metric("retrieve_mode", "local")
            state.add_execution_trace({
                "agent": "retrieve",
                "action": "local_search",
                "query_used": query,
                "result_count": len(results),
                "parallel": False,
            })
        elif plan == "web":
            web_results = _run_web_no_local_ctx()
            state.add_to_blackboard("local_results", [], "retrieve")
            state.add_to_blackboard("web_results", web_results, "retrieve")
            state.blackboard["web_search_attempted"] = True
            state.add_metric("web_result_count", len(web_results))
            state.add_metric("retrieve_mode", "web")
            state.add_execution_trace({
                "agent": "retrieve",
                "action": "web_search",
                "query": query,
                "result_count": len(web_results),
                "parallel": False,
            })
        else:
            # both — 真并行（仅在主线程写 state / trace）
            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_local = ex.submit(_run_local)
                fut_web = ex.submit(_run_web_no_local_ctx)
                local_results = fut_local.result()
                web_results = fut_web.result()
            if local_error:
                state.add_execution_trace({
                    "agent": "retrieve",
                    "action": "local_search_error",
                    "error": local_error,
                })

            state.add_to_blackboard("local_results", local_results, "retrieve")
            state.add_to_blackboard("web_results", web_results, "retrieve")
            state.blackboard["web_search_attempted"] = True
            state.add_metric("local_result_count", len(local_results))
            state.add_metric("web_result_count", len(web_results))
            state.add_metric("retrieve_mode", "parallel_both")
            state.add_execution_trace({
                "agent": "retrieve",
                "action": "parallel_retrieve",
                "query_used": query,
                "local_count": len(local_results),
                "web_count": len(web_results),
                "parallel": True,
            })

        if state.retry_count > 0:
            state.add_metric("used_refined_query", True)
            state.add_metric("refined_query", query)

        return state
    
    def _web_node(self, state: AgentState) -> AgentState:
        """
        Web Agent 节点（联网搜索）
        
        职责：搜索互联网 → 写入黑板
        
        关键点：可以读取 Search Agent 的结果！
        """
        local_results = state.read_from_blackboard("local_results")
        
        # WebSearchAgent.search() handles query refinement internally,
        # so pass the original query to avoid double-appending keywords.
        active_query = state.blackboard.get("active_sub_question", state.user_input)
        web_results = self.web_agent.search(
            query=active_query,
            num_results=5,
            time_range="y",
            local_results=local_results
        )
        
        # 标记联网搜索已执行（防止重复升级）
        state.blackboard["web_search_attempted"] = True
        
        # 写入黑板（Eval Agent 可以看到！）
        state.add_to_blackboard("web_results", web_results, "web")
        
        # 记录指标
        state.add_metric("web_result_count", len(web_results))
        
        # 执行轨迹
        state.add_execution_trace({
            "agent": "web",
            "action": "web_search",
            "query": active_query,
            "result_count": len(web_results),
            "used_local_context": local_results is not None,
        })
        
        return state

    def _read_node(self, state: AgentState) -> AgentState:
        """
        阅读节点：对当前一步检索到的结果进行归纳，形成阶段性观察，
        再由 Eval 节点据此决定继续执行哪个动作。
        """
        sub_question = state.blackboard.get("active_sub_question", state.user_input)
        goal = state.blackboard.get("active_plan_goal", "")
        reading = self._read_retrieved_evidence(
            query=sub_question,
            goal=goal,
            local_results=state.local_results,
            web_results=state.web_results,
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

        state.add_execution_trace({
            "agent": "read",
            "action": "synthesize_evidence",
            "sub_question": sub_question,
            "summary": reading.get("summary", ""),
            "enough_to_answer_sub_question": reading.get("enough_to_answer_sub_question", False),
            "suggested_next_action": reading.get("suggested_next_action", ""),
        })

        return state
    
    def _eval_node(self, state: AgentState) -> AgentState:
        """
        Eval Agent 节点（质量评估）
        
        职责：
        - 评估检索结果质量
        - 判断是否需要优化（Refine）
        - 判断是否应该兜底（Fallback）
        
        关键点：
        - 读取 Search Agent 和 Web Agent 的结果
        - 根据评估结果决定下一步走向
        """
        # 获取所有检索结果
        local_results = state.local_results
        web_results = state.web_results
        
        # 执行评估
        eval_query = state.blackboard.get("active_sub_question", state.user_input)
        evaluation = self.eval_agent.evaluate(
            local_results=local_results,
            web_results=web_results,
            query=eval_query,
            retry_count=state.retry_count,
            max_retries=state.max_retries
        )
        
        # 写入黑板
        state.add_to_blackboard("evaluation", {
            "relevance": evaluation.relevance,
            "diversity": evaluation.diversity,
            "coverage": evaluation.coverage,
            "confidence": evaluation.confidence,
            "need_refinement": evaluation.need_refinement,
            "fallback_suggested": evaluation.fallback_suggested,
            "reason": evaluation.reason,
            "step_query": eval_query,
        }, "eval")
        
        # 记录指标
        state.add_metric("evaluation_relevance", evaluation.relevance)
        state.add_metric("evaluation_confidence", evaluation.confidence)
        
        # 判断是否需要触发兜底
        if evaluation.fallback_suggested:
            if state.retry_count >= state.max_retries:
                state.trigger_fallback(FallbackReason.MAX_RETRIES_EXCEEDED, "eval")
            elif evaluation.relevance < 0.2:
                state.trigger_fallback(FallbackReason.NO_RESULTS_FOUND, "eval")
            elif evaluation.confidence < 0.3:
                state.trigger_fallback(FallbackReason.LOW_CONFIDENCE, "eval")
        
        # 执行轨迹
        state.add_execution_trace({
            "agent": "eval",
            "action": "evaluate_results",
            "relevance": evaluation.relevance,
            "confidence": evaluation.confidence,
            "need_refinement": evaluation.need_refinement,
            "fallback_suggested": evaluation.fallback_suggested,
            "reason": evaluation.reason,
        })
        
        return state
    
    def _refine_node(self, state: AgentState) -> AgentState:
        """
        Refine Agent 节点（查询优化）
        
        职责：
        - 分析 Eval Agent 的反馈
        - 改写查询
        - 增加重试计数
        
        关键点：
        - 基于评估结果优化查询
        - 优化后重新进入 retrieve 节点（按 retrieve_plan 再搜）
        """
        # 获取评估结果
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
        
        # 执行优化
        target_query = state.blackboard.get("active_sub_question", state.user_input)
        refinement = self.refine_agent.refine(
            original_query=target_query,
            evaluation=evaluation,
            retry_count=state.retry_count
        )
        
        # 增加重试计数
        state.increment_retry("refine")
        
        # 写入优化后的查询
        state.add_to_blackboard("refined_query", refinement.refined_query, "refine")
        
        # 记录指标
        state.add_metric("refinement_changes", refinement.changes_made)
        
        # 执行轨迹
        state.add_execution_trace({
            "agent": "refine",
            "action": "refine_query",
            "original_query": target_query,
            "refined_query": refinement.refined_query,
            "changes_made": refinement.changes_made,
            "reasoning": refinement.reasoning,
            "new_retry_count": state.retry_count,
        })
        
        return state
    
    def _generate_node(self, state: AgentState) -> AgentState:
        """
        Generate Agent 节点（最终生成）
        
        职责：汇总所有 Agent 的结果 → 生成最终回答
        
        关键点：
        - 可以读取黑板上的所有数据！
        - 生成带溯源的回答
        - 进行忠实度检查
        - 当所有检索均失败时，降级为 LLM 通用知识回答
        """
        all_context = state.get_all_context()
        
        has_local = bool(state.local_results)
        has_web = bool(state.web_results)
        web_attempted = state.blackboard.get("web_search_attempted", False)
        eval_confidence = state.blackboard.get("evaluation", {}).get("confidence", 1.0)

        # Determine if local results are all noise (CRAG pre-check)
        local_all_noise = False
        if has_local and eval_confidence < 0.3:
            useful = [
                r for r in state.local_results
                if r.get("score", 0) >= self._LOCAL_RELEVANCE_THRESHOLD
            ]
            local_all_noise = len(useful) == 0

        intent = (state.intent or "").lower()
        retrieval_ran = bool(state.blackboard.get("retrieval_executed", False))
        search_was_attempted = bool(
            state.retry_count > 0 or web_attempted or retrieval_ran
        )
        is_chat_direct = intent == "chat" and not search_was_attempted

        if is_chat_direct:
            gen_mode = "general_knowledge"
        elif not has_local and not has_web:
            if web_attempted:
                gen_mode = "general_knowledge"
            else:
                gen_mode = "fallback"
        elif local_all_noise and not has_web and web_attempted:
            gen_mode = "general_knowledge"
        else:
            gen_mode = "normal"

        if gen_mode == "fallback":
            state.final_answer = self._generate_fallback_response(state)
            state.add_metric("generation_mode", "fallback")
        elif gen_mode == "general_knowledge":
            answer = self._generate_general_knowledge_answer(all_context)
            state.final_answer = answer
            state.add_metric("generation_mode", "general_knowledge")
        else:
            answer, citations, faithfulness = self._generate_normal_response_with_citations(
                all_context
            )
            state.final_answer = answer
            state.add_to_blackboard("citations", [c.to_dict() for c in citations], "generate")
            state.add_to_blackboard("faithfulness_check", faithfulness.to_dict(), "generate")
            state.add_metric("generation_mode", "normal_with_citations")
            state.add_metric("citation_count", len(citations))
            state.add_metric("faithfulness_score", faithfulness.confidence)
            state.add_metric("hallucination_detected", faithfulness.hallucination_detected)
        
        state.add_execution_trace({
            "agent": "generate",
            "action": "generate_final_answer",
            "answer_length": len(state.final_answer),
            "generation_mode": gen_mode,
            "citation_count": len(state.blackboard.get("citations", [])),
            "faithfulness_check": state.blackboard.get("faithfulness_check", {}),
        })
        
        return state
    
    # ========== 条件路由函数 ==========
    
    def _route_after_router(self, state: AgentState) -> Literal["generate", "plan", "retrieve"]:
        """闲聊跳过检索；复杂问题先走 plan，其余进入 retrieve。"""
        plan = state.blackboard.get("retrieve_plan", "local")
        complexity = state.blackboard.get("query_complexity", "simple")
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
    
    def _eval_next_step(self, state: AgentState) -> Literal["generate", "plan", "refine", "web"]:
        """
        Eval 之后的三路决策（含自动联网升级）。

        注意：先判断 Refine，再判断「是否还能升级联网」——
        避免 hybrid/并行已联网后无法再走 Refine。
        """
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

        def _escalate_to_web(reason: str) -> Literal["web"]:
            state.add_execution_trace({
                "agent": "eval",
                "action": "escalate_to_web",
                "reason": reason,
                "retry_count": state.retry_count,
                "confidence": confidence,
            })
            return "web"

        # 0. 对复杂任务：如果当前子问题已经足够回答且仍有后续步骤，推进计划
        if enough and has_more_plan_steps:
            state.blackboard["advance_plan_step"] = True
            state.add_execution_trace({
                "agent": "eval",
                "action": "advance_plan",
                "current_step": current_step,
                "next_step": current_step + 1,
            })
            return "plan"

        # 0.1 阅读阶段已经明确建议补充 web，则优先联网
        if suggested_next_action == "search_web" and not web_attempted:
            return _escalate_to_web("reader suggested web search for missing information")

        # 1. 仍有重试额度且需要改写查询 → Refine（与是否已联网无关）
        if need_refinement and state.retry_count < state.max_retries:
            return "refine"

        # 2. 已经执行过联网（含并行 retrieve 里的 web）→ 不再「升级联网」
        if web_attempted:
            return "generate"

        # 3. 明确兜底/放弃 → 升级联网
        if state.fallback_triggered or fallback_suggested:
            return _escalate_to_web("local search fallback triggered")

        # 4. 重试耗尽但还没联网 → 升级联网
        if need_refinement and state.retry_count >= state.max_retries:
            return _escalate_to_web("retries exhausted, trying web as last resort")

        # 5. 置信度偏低 → 联网补充
        if confidence < 0.5:
            return _escalate_to_web(f"low confidence ({confidence}), trying web")

        # 6. 如果是复杂计划，当前步骤已基本完成且没有更多步骤，则生成最终答案
        if enough and not has_more_plan_steps:
            return "generate"

        return "generate"
    
    # ========== 生成逻辑 ==========
    
    @staticmethod
    def _get_date_context() -> str:
        from datetime import datetime
        now = datetime.now()
        weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        return f"当前时间：{now.strftime('%Y年%m月%d日')} {weekdays[now.weekday()]} {now.strftime('%H:%M')}"

    def _build_rag_system_prompt(self) -> str:
        return (
            f"系统信息：{self._get_date_context()}\n\n"
            "你是一个专业的知识库问答助手。根据下方提供的检索结果和对话历史回答用户的问题。\n"
            "要求：\n"
            "1. 仅基于检索结果中的信息作答，不要编造内容。\n"
            "2. 用清晰、结构化的中文回答。如果原文是英文，请翻译为中文后作答。\n"
            "3. 在回答末尾用 [1][2] 等标注引用了哪些检索结果。\n"
            "4. 如果检索结果不足以回答问题，如实说明。\n"
            "5. 结合对话历史理解用户意图，例如用户说'它'、'这篇'时，根据上文确定指代对象。\n"
            "6. 信息冲突处理原则：\n"
            "   - 公司政策、内部规范、组织架构等问题，以 [本地知识库] 的信息为准。\n"
            "   - 客观事实、行业趋势、最新新闻等问题，以 [互联网信息] 作为补充。\n"
            "   - 如果两者存在矛盾，请明确标注信息来源并说明差异，让用户自行判断。\n"
            "7. 如果上下文中提供了 [计划观察]，请把它们作为多步分析过程中的中间结论，整合进最终答案。\n"
        )

    def _build_general_knowledge_prompt(self) -> str:
        return (
            f"系统信息：{self._get_date_context()}\n\n"
            "你是一个知识渊博的AI助手。\n"
            "本地知识库和联网搜索均未能找到与用户问题相关的信息。\n"
            "请根据你自身的知识储备，用清晰、结构化的中文回答用户的问题。\n"
            "要求：\n"
            "1. 在回答开头注明：'以下回答基于AI通用知识，非来自知识库检索。'\n"
            "2. 尽量准确、客观地回答。\n"
            "3. 如果你不确定，请如实说明。\n"
            "4. 结合对话历史理解用户意图。\n"
        )

    # CRAG-inspired relevance threshold: local chunks below this score
    # are considered noise and filtered out before generation.
    _LOCAL_RELEVANCE_THRESHOLD = 0.3

    def _create_explicit_plan(
        self,
        query: str,
        conversation_history: List[Dict],
        routing: Dict,
    ) -> Dict:
        """Use LLM to create an explicit multi-step plan for complex queries."""
        from langchain_core.messages import HumanMessage

        prompt = f"""你是一个多步检索规划器。请根据用户问题生成一个显式计划。

要求：
1. 将复杂问题拆成 1-3 个子问题。
2. 对每个子问题指定 preferred_source，只能是 local / web / both。
3. 每步都给一个简短 goal。
4. 如果问题并不复杂，也至少生成 1 步。
5. 只返回 JSON，不要输出解释。

用户问题：
{query}

对话历史：
{conversation_history}

路由信息：
{routing}

JSON 格式：
{{
  "strategy": "先读本地文档，再用公开资料补充",
  "steps": [
    {{
      "sub_question": "子问题1",
      "preferred_source": "local",
      "goal": "先确认文档里的核心观点"
    }}
  ]
}}"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content if hasattr(response, "content") else str(response)
            match = json.loads(content)
            if isinstance(match, dict) and match.get("steps"):
                return match
        except Exception:
            pass

        preferred = "both" if routing.get("needs_local") and routing.get("needs_web") else (
            "web" if routing.get("needs_web") else "local"
        )
        return {
            "strategy": "默认单步计划",
            "steps": [
                {
                    "sub_question": query,
                    "preferred_source": preferred,
                    "goal": "回答用户当前问题",
                }
            ],
        }

    def _read_retrieved_evidence(
        self,
        query: str,
        goal: str,
        local_results: List[Dict],
        web_results: List[Dict],
    ) -> Dict[str, Any]:
        """Read retrieved evidence and produce a reflective observation."""
        from langchain_core.messages import HumanMessage

        local_block = "\n\n".join(
            f"[Local {i}] {r.get('source', 'unknown')}\n{r.get('content', '')[:400]}"
            for i, r in enumerate(local_results[:3], 1)
        ) or "No local results"
        web_block = "\n\n".join(
            f"[Web {i}] {r.get('title', r.get('source', 'web'))}\n{r.get('snippet', r.get('content', ''))[:400]}"
            for i, r in enumerate(web_results[:3], 1)
        ) or "No web results"

        prompt = f"""你是一个阅读与反思节点。请阅读检索结果，判断当前子问题是否已经有足够证据回答。

当前子问题：{query}
当前目标：{goal}

本地结果：
{local_block}

联网结果：
{web_block}

只返回 JSON：
{{
  "summary": "对当前证据的简短总结",
  "enough_to_answer_sub_question": true,
  "suggested_next_action": "continue|search_web|refine|generate",
  "missing_information": "还缺什么信息"
}}"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content if hasattr(response, "content") else str(response)
            return json.loads(content)
        except Exception:
            enough = bool(local_results or web_results)
            return {
                "summary": "已基于当前检索结果完成初步阅读。",
                "enough_to_answer_sub_question": enough,
                "suggested_next_action": "continue" if enough else "refine",
                "missing_information": "" if enough else "当前证据不足，需要更多结果。",
            }

    def _filter_local_results(
        self, local_results: List[Dict], eval_confidence: float
    ) -> List[Dict]:
        """CRAG-inspired noise filtering: remove low-relevance local chunks.

        When Eval confidence is low (meaning local results were mostly noise),
        aggressively filter by score threshold so that only genuinely useful
        chunks survive into the LLM prompt.  When confidence is high, keep
        everything — the results are already good.
        """
        if eval_confidence >= 0.7 or not local_results:
            return local_results

        filtered = [
            r for r in local_results
            if r.get("score", 0) >= self._LOCAL_RELEVANCE_THRESHOLD
        ]
        # Always keep at least 1 result to avoid an empty context
        return filtered if filtered else local_results[:1]

    def _generate_normal_response_with_citations(
        self,
        context: Dict
    ) -> Tuple[str, List[Citation], FaithfulnessCheck]:
        """用 LLM 基于检索结果生成回答，并附带溯源引用。"""
        local_results = context.get("local_results", [])
        web_results = context.get("web_results", [])
        query = context.get("user_input", "")
        plan_observations = context.get("blackboard", {}).get("plan_observations", [])

        # CRAG: filter noisy local chunks when eval confidence was low
        eval_confidence = context.get("evaluation", {}).get("confidence", 1.0)
        local_results = self._filter_local_results(local_results, eval_confidence)

        citation_manager = CitationManager()
        citations = CitationManager.create_citations_from_results(
            local_results=local_results,
            web_results=web_results,
            top_k=5
        )
        citation_manager.add_citations(citations)

        if not local_results and not web_results:
            answer = "抱歉，没有找到相关信息。"
            return answer, [], citation_manager.check_faithfulness(answer)

        # Build context block for LLM
        context_parts: list[str] = []
        for i, result in enumerate(local_results[:5], 1):
            content = result.get("content", "")
            source = result.get("source", "unknown")
            context_parts.append(f"[{i}] (本地知识库: {source})\n{content}")
        offset = len(local_results[:5])
        for i, result in enumerate(web_results[:3], 1):
            title = result.get("title", "")
            snippet = result.get("snippet", result.get("content", ""))
            url = result.get("url", "")
            context_parts.append(f"[{offset + i}] (互联网: {title}, {url})\n{snippet}")

        retrieval_context = "\n\n".join(context_parts)
        observations_block = ""
        if plan_observations:
            observations_lines = []
            for obs in plan_observations[-5:]:
                observations_lines.append(
                    f"- 子问题: {obs.get('sub_question', '')}\n"
                    f"  观察: {obs.get('summary', '')}\n"
                    f"  缺失: {obs.get('missing_information', '')}"
                )
            observations_block = "\n\n## 计划观察\n" + "\n".join(observations_lines)

        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        messages: list = [SystemMessage(content=self._build_rag_system_prompt())]

        conv_history = context.get("conversation_history", [])
        for turn in conv_history:
            role = turn.get("role", "")
            text = turn.get("content", "")
            if role == "system":
                messages.append(SystemMessage(content=text))
            elif role == "user":
                messages.append(HumanMessage(content=text))
            elif role == "assistant":
                messages.append(AIMessage(content=text))

        messages.append(HumanMessage(content=(
            f"## 检索结果\n\n{retrieval_context}{observations_block}\n\n"
            f"## 用户问题\n{query}"
        )))

        try:
            response = self.llm.invoke(messages)
            answer = response.content if hasattr(response, "content") else str(response)
        except Exception:
            # LLM 调用失败时降级为拼接模式
            parts = []
            for i, r in enumerate(local_results[:3], 1):
                parts.append(f"{i}. {r.get('content', '')[:300]}...")
            answer = "根据检索结果：\n" + "\n".join(parts) if parts else "抱歉，没有找到相关信息。"

        answer = format_answer_with_citations(
            answer=answer,
            citations=citations,
            include_reference_list=True
        )

        faithfulness = citation_manager.check_faithfulness(answer)
        return answer, citations, faithfulness
    
    def _generate_normal_response(self, context: Dict) -> str:
        """
        生成正常回答（简化版，向后兼容）
        
        Args:
            context: 所有上下文信息
        
        Returns:
            生成的回答
        """
        answer, _, _ = self._generate_normal_response_with_citations(context)
        return answer
    
    def _generate_general_knowledge_answer(self, context: Dict) -> str:
        """当所有检索（本地+联网）均失败时，用 LLM 通用知识直接回答。"""
        query = context.get("user_input", "")

        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        messages: list = [SystemMessage(content=self._build_general_knowledge_prompt())]

        conv_history = context.get("conversation_history", [])
        for turn in conv_history[-6:]:
            role = turn.get("role", "")
            text = turn.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=text))
            elif role == "assistant":
                messages.append(AIMessage(content=text))

        messages.append(HumanMessage(content=query))

        try:
            response = self.llm.invoke(messages)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            return f"抱歉，检索和联网搜索均未找到结果，LLM 直接回答也遇到了问题：{exc}"

    def _generate_fallback_response(self, state: AgentState) -> str:
        """
        生成兜底回复
        
        Args:
            state: 当前状态
        
        Returns:
            兜底回复
        """
        reason_messages = {
            FallbackReason.MAX_RETRIES_EXCEEDED: 
                "经过多次检索和优化，我依然无法找到确切答案",
            FallbackReason.NO_RESULTS_FOUND:
                "本地知识库和互联网上都没有相关信息",
            FallbackReason.LOW_CONFIDENCE:
                "检索到的信息相关性较低，无法提供可靠答案",
            FallbackReason.USER_ASKED_UNKNOWN:
                "该问题涉及系统无法获取的信息",
        }
        
        reason = state.fallback_reason or FallbackReason.NO_RESULTS_FOUND
        reason_text = reason_messages.get(reason, "经过检索，我无法找到确切答案")
        
        return f"""抱歉，{reason_text}。

检索详情：
- 检索次数：{state.retry_count} 次
- 本地知识库结果：{len(state.local_results)} 条
- 互联网搜索结果：{len(state.web_results)} 条

建议您：
1. 重新描述问题，提供更多上下文
2. 尝试使用不同的表述方式
3. 或者询问其他我可能帮助的问题

如果您认为这个问题应该有答案，请联系管理员确认知识库配置。"""
    
    # ========== 公共 API ==========
    
    def run(
        self,
        user_input: str,
        conversation_history: Optional[List[Dict]] = None,
    ) -> AgentState:
        """运行多智能体系统。

        Args:
            user_input: 用户输入
            conversation_history: 对话历史（可选）

        Returns:
            完整的 AgentState（包含所有中间结果和最终回答）
        """
        initial_state = AgentState(
            user_input=user_input,
            conversation_history=conversation_history or []
        )
        
        final_state = self.workflow.invoke(initial_state)
        return final_state
