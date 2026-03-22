"""
多智能体 RAG 系统 - 主编排器

使用 LangGraph 编排多个专门智能体：
- Router Agent: 意图识别和路由
- Search Agent: 本地知识库检索
- Web Agent: 联网搜索
- Eval Agent: 质量评估
- Refine Agent: 查询优化
- Generate Agent: 最终回答生成
"""

from typing import Optional, List, Dict, Literal, Tuple
from langgraph.graph import StateGraph, END

from .state import AgentState, FallbackReason
from .router_agent import RouterAgent, AgentType, RoutingDecision
from .search_agent import SearchAgent
from .web_agent import WebSearchAgent
from .parallel_controller import ParallelFusionController
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
        enable_logging: bool = True
    ):
        """
        初始化多智能体 RAG 系统
        
        Args:
            llm: 语言模型实例
            settings: 系统配置
            enable_logging: 是否启用日志
        """
        self.llm = llm
        self.settings = settings or {}
        self.enable_logging = enable_logging
        
        # 初始化所有智能体
        self.router_agent = RouterAgent(self.llm)
        self.search_agent = SearchAgent(self.settings)
        self.web_agent = WebSearchAgent(self.settings)
        self.eval_agent = EvalAgent(self.llm)
        self.refine_agent = RefineAgent(self.llm)
        
        # 初始化并行融合控制器
        self.parallel_controller = ParallelFusionController(
            search_func=self.search_agent.search,
            web_func=self.web_agent.search
        )
        
        # 构建工作流图
        self.workflow = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 状态机（包含容错机制）"""
        
        workflow = StateGraph(AgentState)
        
        # ========== 添加节点 ==========
        workflow.add_node("router", self._router_node)
        workflow.add_node("search", self._search_node)
        workflow.add_node("web", self._web_node)
        workflow.add_node("eval", self._eval_node)
        workflow.add_node("refine", self._refine_node)
        workflow.add_node("generate", self._generate_node)
        
        # ========== 设置入口点 ==========
        workflow.set_entry_point("router")
        
        # ========== 添加边 ==========
        # Router → 条件路由
        workflow.add_conditional_edges(
            "router",
            self._route_by_intent,
            {
                "generate": "generate",
                "search": "search",
                "web": "web",
            }
        )
        
        # Search → conditional: hybrid needs web first, otherwise straight to eval
        workflow.add_conditional_edges(
            "search",
            self._should_search_web,
            {
                "yes": "web",
                "no": "eval",
            }
        )
        
        # Web → Eval
        workflow.add_edge("web", "eval")
        
        # Eval → Generate / Refine / Web（自动升级联网）
        workflow.add_conditional_edges(
            "eval",
            self._eval_next_step,
            {
                "generate": "generate",
                "refine": "refine",
                "web": "web",
            }
        )
        
        # Refine → Search（重新检索）
        workflow.add_edge("refine", "search")
        
        # Generate → END
        workflow.add_edge("generate", END)
        
        return workflow.compile()
    
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
                "parallel": decision.parallel,
            },
            "router"
        )
        
        # 记录执行轨迹
        state.add_execution_trace({
            "agent": "router",
            "action": "classify_intent",
            "result": decision.intent,
            "confidence": decision.confidence,
            "agents_to_invoke": [a.value for a in decision.agents_to_invoke],
            "parallel": decision.parallel,
        })
        
        return state
    
    def _search_node(self, state: AgentState) -> AgentState:
        """
        Search Agent 节点（本地知识库）
        
        职责：检索本地向量库 → 写入黑板
        
        关键点：如果有 refined_query（来自 Refine Agent），则使用优化后的查询
        """
        query = state.refined_query if state.retry_count > 0 else state.user_input
        context = state.conversation_history
        
        try:
            results = self.search_agent.search(
                query=query,
                top_k=10,
                context=context
            )
        except Exception as e:
            results = []
            state.add_execution_trace({
                "agent": "search",
                "action": "hybrid_search_error",
                "error": str(e),
            })
        
        state.add_to_blackboard("local_results", results, "search")
        
        state.add_metric("local_result_count", len(results))
        if state.retry_count > 0:
            state.add_metric("used_refined_query", True)
            state.add_metric("refined_query", query)
        
        state.add_execution_trace({
            "agent": "search",
            "action": "hybrid_search",
            "query_used": query,
            "result_count": len(results),
            "is_retried_search": state.retry_count > 0,
        })
        
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
        web_results = self.web_agent.search(
            query=state.user_input,
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
            "query": state.user_input,
            "result_count": len(web_results),
            "used_local_context": local_results is not None,
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
        evaluation = self.eval_agent.evaluate(
            local_results=local_results,
            web_results=web_results,
            query=state.user_input,
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
        - 优化后重新进入 Search 节点
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
        refinement = self.refine_agent.refine(
            original_query=state.user_input,
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
            "original_query": state.user_input,
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
        search_was_attempted = bool(state.retry_count > 0 or web_attempted or state.execution_trace)
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
    
    def _route_by_intent(self, state: AgentState) -> Literal["generate", "search", "web"]:
        """根据意图路由到对应的节点"""
        intent = (state.intent or "unknown").lower()
        
        intent_to_node = {
            "chat": "generate",
            "local_search": "search",
            "web_search": "web",
            "hybrid_search": "search",
            "unknown": "search",
        }
        
        return intent_to_node.get(intent, "search")
    
    def _should_search_web(self, state: AgentState) -> Literal["yes", "no"]:
        """判断是否需要联网搜索"""
        intent = (state.intent or "").lower()
        return "yes" if intent == "hybrid_search" else "no"
    
    def _eval_next_step(self, state: AgentState) -> Literal["generate", "refine", "web"]:
        """
        Eval 之后的三路决策（含自动联网升级）：
        
        优先级从高到低：
        1. 已联网且已兜底         → generate（所有手段用尽）
        2. 兜底/放弃 + 没联网过   → web（自动升级联网）
        3. 需要优化 + 有重试次数   → refine（改写 query 重搜）
        4. 重试耗尽 + 没联网过     → web（兜底前最后一搏）
        5. 置信度低 + 没联网过     → web（Eval 没明确说差，但分数不高）
        6. 其他                   → generate（正常生成）
        """
        web_attempted = state.blackboard.get("web_search_attempted", False)
        evaluation = state.evaluation
        need_refinement = evaluation.get("need_refinement", False)
        fallback_suggested = evaluation.get("fallback_suggested", False)
        confidence = evaluation.get("confidence", 0.5)

        def _escalate_to_web(reason: str) -> Literal["web"]:
            state.add_execution_trace({
                "agent": "eval",
                "action": "escalate_to_web",
                "reason": reason,
                "retry_count": state.retry_count,
                "confidence": confidence,
            })
            state.blackboard["web_search_attempted"] = True
            return "web"

        # 1. 联网也试过了 → 不再升级，直接生成
        if web_attempted:
            return "generate"

        # 2. 明确兜底/放弃 → 升级联网
        if state.fallback_triggered or fallback_suggested:
            return _escalate_to_web("local search fallback triggered")

        # 3. 需要优化且有重试次数 → refine
        if need_refinement and state.retry_count < state.max_retries:
            return "refine"

        # 4. 重试耗尽但还没联网 → 升级联网（别直接放弃）
        if need_refinement and state.retry_count >= state.max_retries:
            return _escalate_to_web("retries exhausted, trying web as last resort")

        # 5. Eval 没明确说差，但置信度偏低 → 联网补充
        if confidence < 0.5:
            return _escalate_to_web(f"low confidence ({confidence}), trying web")

        # 6. 质量足够 → 正常生成
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

        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        messages: list = [SystemMessage(content=self._build_rag_system_prompt())]

        # Inject conversation history so LLM understands multi-turn context
        conv_history = context.get("conversation_history", [])
        for turn in conv_history[-6:]:
            role = turn.get("role", "")
            text = turn.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=text))
            elif role == "assistant":
                messages.append(AIMessage(content=text))

        messages.append(HumanMessage(content=(
            f"## 检索结果\n\n{retrieval_context}\n\n"
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
