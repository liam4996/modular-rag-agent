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
        
        # Eval → Generate 或 Refine
        workflow.add_conditional_edges(
            "eval",
            self._should_refine,
            {
                "generate": "generate",
                "refine": "refine",
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
        
        # 写入黑板（共享！）
        state.add_to_blackboard("intent", decision.intent, "router")
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
        
        # 写入黑板（Eval Agent 可以看到！）
        state.add_to_blackboard("web_results", web_results, "web")
        
        # 记录指标
        state.add_metric("web_result_count", len(web_results))
        
        # 执行轨迹
        state.add_execution_trace({
            "agent": "web",
            "action": "web_search",
            "query": refined_query,
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
        """
        # ⭐ 获取所有上下文
        all_context = state.get_all_context()
        
        # Only fall back when there are truly NO results.
        # If we have retrieved documents, always attempt LLM generation.
        has_results = bool(state.local_results or state.web_results)
        if state.should_fallback and not has_results:
            state.final_answer = self._generate_fallback_response(state)
            state.add_metric("generation_mode", "fallback")
        else:
            # 正常生成模式（带溯源）
            answer, citations, faithfulness = self._generate_normal_response_with_citations(
                all_context
            )
            state.final_answer = answer
            
            # 写入引用信息
            state.add_to_blackboard("citations", [c.to_dict() for c in citations], "generate")
            state.add_to_blackboard("faithfulness_check", faithfulness.to_dict(), "generate")
            
            # 记录指标
            state.add_metric("generation_mode", "normal_with_citations")
            state.add_metric("citation_count", len(citations))
            state.add_metric("faithfulness_score", faithfulness.confidence)
            state.add_metric("hallucination_detected", faithfulness.hallucination_detected)
        
        # 执行轨迹
        state.add_execution_trace({
            "agent": "generate",
            "action": "generate_final_answer",
            "answer_length": len(state.final_answer),
            "fallback_triggered": state.should_fallback,
            "citation_count": len(state.blackboard.get("citations", [])),
            "faithfulness_check": state.blackboard.get("faithfulness_check", {}),
        })
        
        return state
    
    # ========== 条件路由函数 ==========
    
    def _route_by_intent(self, state: AgentState) -> Literal["generate", "search", "web"]:
        """根据意图路由到对应的节点"""
        intent = state.intent
        
        # 映射意图到节点名称
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
        intent = state.intent
        return "yes" if intent == "hybrid_search" else "no"
    
    def _should_refine(self, state: AgentState) -> Literal["generate", "refine"]:
        """
        判断是否需要优化查询
        
        决策逻辑：
        1. 如果已经触发兜底 → Generate（兜底回复）
        2. 如果 Eval 建议兜底 → Generate（兜底回复）
        3. 如果 Eval 认为需要优化 → Refine
        4. 否则 → Generate（正常回复）
        """
        # 已经触发兜底
        if state.fallback_triggered:
            return "generate"
        
        # 读取评估结果
        evaluation = state.evaluation
        need_refinement = evaluation.get("need_refinement", False)
        fallback_suggested = evaluation.get("fallback_suggested", False)
        
        # Eval 建议兜底
        if fallback_suggested:
            return "generate"
        
        # Eval 认为需要优化
        if need_refinement:
            return "refine"
        
        # 正常生成
        return "generate"
    
    # ========== 生成逻辑 ==========
    
    _RAG_SYSTEM_PROMPT = (
        "你是一个专业的知识库问答助手。根据下方提供的检索结果和对话历史回答用户的问题。\n"
        "要求：\n"
        "1. 仅基于检索结果中的信息作答，不要编造内容。\n"
        "2. 用清晰、结构化的中文回答。如果原文是英文，请翻译为中文后作答。\n"
        "3. 在回答末尾用 [1][2] 等标注引用了哪些检索结果。\n"
        "4. 如果检索结果不足以回答问题，如实说明。\n"
        "5. 结合对话历史理解用户意图，例如用户说'它'、'这篇'时，根据上文确定指代对象。\n"
    )

    def _generate_normal_response_with_citations(
        self,
        context: Dict
    ) -> Tuple[str, List[Citation], FaithfulnessCheck]:
        """用 LLM 基于检索结果生成回答，并附带溯源引用。"""
        local_results = context.get("local_results", [])
        web_results = context.get("web_results", [])
        query = context.get("user_input", "")

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
            context_parts.append(f"[{i}] (来源: {source})\n{content}")
        offset = len(local_results[:5])
        for i, result in enumerate(web_results[:3], 1):
            title = result.get("title", "")
            snippet = result.get("snippet", result.get("content", ""))
            url = result.get("url", "")
            context_parts.append(f"[{offset + i}] (网络来源: {title}, {url})\n{snippet}")

        retrieval_context = "\n\n".join(context_parts)

        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        messages: list = [SystemMessage(content=self._RAG_SYSTEM_PROMPT)]

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
