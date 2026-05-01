"""多智能体 RAG 系统 - 主编排器

使用 LangGraph 编排多个专门智能体。
Phase 3: 集成 BusinessComputeAgent（计算 + 图表 + 报告）。
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
from .business_compute_agent import BusinessComputeAgent
from .citation import (
    Citation, CitationType, CitationManager,
    FaithfulnessCheck, format_answer_with_citations,
)


class MultiAgentRAG:
    def __init__(self, llm, settings=None, enable_logging=True, store=None):
        self.llm = llm
        self.settings = settings or {}
        self.enable_logging = enable_logging
        self.store = store
        self.router_agent = RouterAgent(self.llm)
        self.search_agent = SearchAgent(self.settings)
        self.web_agent = WebSearchAgent(self.settings)
        self.eval_agent = EvalAgent(self.llm)
        self.refine_agent = RefineAgent(self.llm)
        self.supervisor_agent = SupervisorAgent(self.llm)
        self.finance_data_agent = FinanceDataAgent(self.settings)
        self.business_compute_agent = BusinessComputeAgent()
        self.workflow = self._build_graph()

    _ROUTER_PARALLEL_CONFIDENCE_THRESHOLD = 0.7
    _LOCAL_RELEVANCE_THRESHOLD = 0.3

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        for name, fn in [
            ("router", self._router_node), ("plan", self._plan_node),
            ("retrieve", self._retrieve_node), ("web", self._web_node),
            ("read", self._read_node), ("eval", self._eval_node),
            ("refine", self._refine_node), ("generate", self._generate_node),
            ("supervisor", self._supervisor_node), ("finance_data", self._finance_data_node),
            ("aggregate", self._aggregate_node), ("global_eval", self._global_eval_node),
            ("business_compute", self._business_compute_node),
        ]:
            workflow.add_node(name, fn)
        workflow.set_entry_point("router")
        workflow.add_conditional_edges("router", self._route_after_router, {
            "generate": "generate", "plan": "plan", "retrieve": "retrieve", "supervisor": "supervisor"})
        workflow.add_conditional_edges("plan", self._route_after_plan, {
            "generate": "generate", "retrieve": "retrieve"})
        workflow.add_conditional_edges("supervisor", self._dispatch_from_supervisor, {
            "retrieve": "retrieve", "web": "web", "finance_data": "finance_data",
            "aggregate": "aggregate", "generate": "generate",
            "business_compute": "business_compute"})
        workflow.add_edge("retrieve", "read")
        workflow.add_edge("web", "read")
        workflow.add_edge("read", "eval")
        workflow.add_edge("finance_data", "aggregate")
        workflow.add_edge("business_compute", "aggregate")
        workflow.add_conditional_edges("eval", self._eval_next_step, {
            "generate": "generate", "plan": "plan", "refine": "refine",
            "web": "web", "aggregate": "aggregate"})
        workflow.add_edge("refine", "retrieve")
        workflow.add_edge("aggregate", "global_eval")
        workflow.add_edge("global_eval", "generate")
        workflow.add_edge("generate", END)
        kw = {}
        if self.store is not None:
            kw["store"] = self.store
        return workflow.compile(**kw)

    def _router_node(self, state):
        decision = self.router_agent.classify(query=state.user_input, context=state.conversation_history)
        state.add_to_blackboard("intent", decision.intent.lower(), "router")
        state.add_to_blackboard("routing_decision", {
            "agents_to_invoke": [a.value for a in decision.agents_to_invoke],
            "needs_local": decision.needs_local, "needs_web": decision.needs_web,
            "complexity": decision.complexity, "parallel": decision.parallel,
            "reasoning": decision.reasoning}, "router")
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
            plan = "none"
        elif intent.startswith("financial_"):
            plan = "financial"
        elif intent != "chat" and conf < self._ROUTER_PARALLEL_CONFIDENCE_THRESHOLD:
            plan = "both"
        elif needs_local and needs_web:
            plan = "both"
        elif needs_web and not needs_local:
            plan = "web"
        elif needs_local and not needs_web:
            plan = "local"
        elif complexity == "complex" and intent != "chat":
            plan = "both"
        elif decision.parallel or intent == "hybrid_search":
            plan = "both"
        elif intent == "web_search":
            plan = "web"
        else:
            plan = "local"
        state.add_to_blackboard("retrieve_plan", plan, "router")
        return state

    def _plan_node(self, state):
        plan_data = state.blackboard.get("task_plan")
        replan_requested = bool(state.blackboard.get("replan_requested", False))
        if not plan_data or replan_requested:
            routing = state.blackboard.get("routing_decision", {})
            plan_data = self._create_explicit_plan(
                query=state.user_input, conversation_history=state.conversation_history, routing=routing)
            state.add_to_blackboard("task_plan", plan_data, "plan")
            state.add_to_blackboard("plan_current_step", 0, "plan")
            state.blackboard["replan_requested"] = False
        elif state.blackboard.get("advance_plan_step", False):
            cs = int(state.blackboard.get("plan_current_step", 0))
            state.add_to_blackboard("plan_current_step", cs + 1, "plan")
            state.blackboard["advance_plan_step"] = False
        steps = plan_data.get("steps", [])
        current = int(state.blackboard.get("plan_current_step", 0))
        if current >= len(steps):
            state.blackboard["plan_complete"] = True
            return state
        step = steps[current]
        sub = step.get("sub_question", state.user_input)
        src = step.get("preferred_source", state.blackboard.get("retrieve_plan", "local"))
        if src not in ("local", "web", "both"):
            src = state.blackboard.get("retrieve_plan", "local")
        state.blackboard["plan_complete"] = False
        state.add_to_blackboard("active_sub_question", sub, "plan")
        state.add_to_blackboard("active_retrieve_plan", src, "plan")
        state.add_to_blackboard("active_plan_goal", step.get("goal", ""), "plan")
        return state

    def _retrieve_node(self, state):
        plan = state.blackboard.get("active_retrieve_plan", state.blackboard.get("retrieve_plan", "local"))
        base = state.blackboard.get("active_sub_question", state.user_input)
        query = state.refined_query if state.retry_count > 0 else base
        ctx = state.conversation_history
        state.blackboard["retrieval_executed"] = True
        le: Optional[str] = None
        def _run_local():
            try:
                return self.search_agent.search(query=query, top_k=10, context=ctx)
            except Exception as e:
                nonlocal le
                le = str(e)
                return []
        def _run_web():
            try:
                return self.web_agent.search(query=query, num_results=5, time_range="y", local_results=None)
            except Exception:
                return []
        if plan == "local":
            r = _run_local()
            state.add_to_blackboard("local_results", r, "retrieve")
            state.add_to_blackboard("web_results", [], "retrieve")
        elif plan == "web":
            w = _run_web()
            state.add_to_blackboard("local_results", [], "retrieve")
            state.add_to_blackboard("web_results", w, "retrieve")
            state.blackboard["web_search_attempted"] = True
        else:
            with ThreadPoolExecutor(max_workers=2) as ex:
                fl = ex.submit(_run_local)
                fw = ex.submit(_run_web)
                lr = fl.result()
                wr = fw.result()
            state.add_to_blackboard("local_results", lr, "retrieve")
            state.add_to_blackboard("web_results", wr, "retrieve")
            state.blackboard["web_search_attempted"] = True
        return state

    def _web_node(self, state):
        lr = state.read_from_blackboard("local_results")
        q = state.blackboard.get("active_sub_question", state.user_input)
        wr = self.web_agent.search(query=q, num_results=5, time_range="y", local_results=lr)
        state.blackboard["web_search_attempted"] = True
        state.add_to_blackboard("web_results", wr, "web")
        return state

    def _read_node(self, state):
        sub = state.blackboard.get("active_sub_question", state.user_input)
        goal = state.blackboard.get("active_plan_goal", "")
        reading = self._read_retrieved_evidence(
            query=sub, goal=goal, local_results=state.local_results, web_results=state.web_results)
        state.add_to_blackboard("reading_assessment", reading, "read")
        obs = state.blackboard.get("plan_observations", [])
        obs.append({"step_index": state.blackboard.get("plan_current_step", 0), "sub_question": sub,
            "summary": reading.get("summary", ""),
            "enough_to_answer_sub_question": reading.get("enough_to_answer_sub_question", False),
            "suggested_next_action": reading.get("suggested_next_action", ""),
            "missing_information": reading.get("missing_information", "")})
        state.blackboard["plan_observations"] = obs
        return state

    def _eval_node(self, state):
        lr = state.local_results
        wr = state.web_results
        q = state.blackboard.get("active_sub_question", state.user_input)
        ev = self.eval_agent.evaluate(local_results=lr, web_results=wr, query=q,
            retry_count=state.retry_count, max_retries=state.max_retries)
        state.add_to_blackboard("evaluation", {
            "relevance": ev.relevance, "diversity": ev.diversity,
            "coverage": ev.coverage, "confidence": ev.confidence,
            "need_refinement": ev.need_refinement, "fallback_suggested": ev.fallback_suggested,
            "reason": ev.reason}, "eval")
        if ev.fallback_suggested:
            if state.retry_count >= state.max_retries:
                state.trigger_fallback(FallbackReason.MAX_RETRIES_EXCEEDED, "eval")
            elif ev.relevance < 0.2:
                state.trigger_fallback(FallbackReason.NO_RESULTS_FOUND, "eval")
            elif ev.confidence < 0.3:
                state.trigger_fallback(FallbackReason.LOW_CONFIDENCE, "eval")
        return state

    def _refine_node(self, state):
        ed = state.evaluation
        ev = EvaluationResult(
            relevance=ed.get("relevance", 0.5), diversity=ed.get("diversity", 0.5),
            coverage=ed.get("coverage", 0.5), confidence=ed.get("confidence", 0.5),
            need_refinement=ed.get("need_refinement", True),
            fallback_suggested=ed.get("fallback_suggested", False), reason=ed.get("reason", ""))
        q = state.blackboard.get("active_sub_question", state.user_input)
        ref = self.refine_agent.refine(original_query=q, evaluation=ev, retry_count=state.retry_count)
        state.increment_retry("refine")
        state.add_to_blackboard("refined_query", ref.refined_query, "refine")
        return state

    def _generate_node(self, state):
        ctx = state.get_all_context()
        has_l = bool(state.local_results)
        has_w = bool(state.web_results)
        wa = state.blackboard.get("web_search_attempted", False)
        ec = state.blackboard.get("evaluation", {}).get("confidence", 1.0)
        local_noise = False
        if has_l and ec < 0.3:
            useful = [r for r in state.local_results if r.get("score", 0) >= self._LOCAL_RELEVANCE_THRESHOLD]
            local_noise = len(useful) == 0
        intent = (state.intent or "").lower()
        ret_ran = bool(state.blackboard.get("retrieval_executed", False))
        searched = bool(state.retry_count > 0 or wa or ret_ran)
        chat_direct = intent == "chat" and not searched
        if chat_direct:
            mode = "general_knowledge"
        elif not has_l and not has_w:
            mode = "general_knowledge" if wa else "fallback"
        elif local_noise and not has_w and wa:
            mode = "general_knowledge"
        else:
            mode = "normal"
        if mode == "fallback":
            state.final_answer = self._generate_fallback_response(state)
        elif mode == "general_knowledge":
            state.final_answer = self._generate_general_knowledge_answer(ctx)
        else:
            ans, cits, fth = self._generate_normal_response_with_citations(ctx)
            state.final_answer = ans
            state.add_to_blackboard("citations", [c.to_dict() for c in cits], "generate")
            state.add_to_blackboard("faithfulness_check", fth.to_dict(), "generate")
        state.add_metric("generation_mode", mode)
        return state

    def _supervisor_node(self, state):
        self.supervisor_agent.classify_and_plan(state)
        return state

    def _dispatch_from_supervisor(self, state):
        next_t = self.supervisor_agent.get_next_subtasks(state)
        if not next_t:
            return "aggregate" if self.supervisor_agent.all_completed(state) else "generate"
        task = next_t[0]
        tt = task.get("type", "")
        self.supervisor_agent.mark_completed(state, task["id"])
        q = task.get("query", state.user_input)
        state.add_to_blackboard("active_sub_question", q, "supervisor")
        state.add_to_blackboard("current_task", task, "supervisor")
        if tt == "document_search": return "retrieve"
        elif tt == "web_search": return "web"
        elif tt == "financial_market": return "finance_data"
        elif tt == "financial_computation": return "business_compute"
        else: return "aggregate"

    def _finance_data_node(self, state):
        task = state.blackboard.get("current_task", {})
        sym = task.get("symbols", [])
        if not sym:
            sym = self.supervisor_agent._extract_symbols(state.user_input)
        try:
            self.finance_data_agent.query_and_store(
                state=state, symbols=sym or ["000001.SZ"], data_types=["quote", "fundamentals"])
        except Exception as e:
            state.add_execution_trace({"agent": "finance_data", "action": "error", "error": str(e)})
        return state

    def _business_compute_node(self, state):
        task = state.blackboard.get("current_task", {})
        operation = task.get("operation", "calculate_metrics")
        try:
            self.business_compute_agent.compute(state=state, operation=operation, task_params=task)
        except Exception as e:
            state.add_execution_trace({"agent": "business_compute", "action": "error", "error": str(e)})
        return state

    def _aggregate_node(self, state):
        self.supervisor_agent.aggregate(state)
        return state

    def _global_eval_node(self, state):
        agg = state.blackboard.get("aggregated_context", {})
        md = agg.get("market", {})
        issues = []
        if not md and state.is_financial_intent:
            issues.append("market_data_missing")
        state.add_to_blackboard("global_eval_issues", issues, "global_eval")
        state.add_to_blackboard("global_eval_passed", len(issues) == 0, "global_eval")
        return state

    def _route_after_router(self, state):
        intent = (state.intent or "").lower()
        plan = state.blackboard.get("retrieve_plan", "local")
        complexity = state.blackboard.get("query_complexity", "simple")
        if intent.startswith("financial_"): return "supervisor"
        if plan == "none": return "generate"
        if complexity == "complex": return "plan"
        return "retrieve"

    def _route_after_plan(self, state):
        return "generate" if state.blackboard.get("plan_complete", False) else "retrieve"

    def _eval_next_step(self, state):
        intent = (state.intent or "").lower()
        if intent.startswith("financial_"):
            state.blackboard["advance_plan_step"] = False
            return "aggregate"
        wa = state.blackboard.get("web_search_attempted", False)
        ev = state.evaluation
        rd = state.blackboard.get("reading_assessment", {})
        nf = ev.get("need_refinement", False)
        fs = ev.get("fallback_suggested", False)
        cf = ev.get("confidence", 0.5)
        en = bool(rd.get("enough_to_answer_sub_question", False))
        na = rd.get("suggested_next_action", "")
        tp = state.blackboard.get("task_plan", {})
        ps = tp.get("steps", [])
        cs = int(state.blackboard.get("plan_current_step", 0))
        hm = cs < max(len(ps) - 1, 0)
        wr = state.web_results
        lr = state.local_results
        def _web(r): state.add_execution_trace({"agent": "eval", "action": "escalate_to_web", "reason": r}); return "web"
        def _local(r): state.add_to_blackboard("retrieve_plan", "local", "eval"); return "refine"
        if en and hm: state.blackboard["advance_plan_step"] = True; return "plan"
        if na == "search_web" and not wa: return _web("reader suggested web")
        if nf and state.retry_count < state.max_retries: return "refine"
        if wa and cf < 0.4 and len(lr) > 0 and state.retry_count < state.max_retries: return _local("web confidence too low")
        if wa: return "generate"
        if state.fallback_triggered or fs: return _web("fallback triggered")
        if nf and state.retry_count >= state.max_retries: return _web("retries exhausted")
        if cf < 0.5: return _web(f"low confidence ({cf})")
        if en and not hm: return "generate"
        return "generate"

    @staticmethod
    def _get_date_context():
        from datetime import datetime
        now = datetime.now()
        wd = ["星期一","星期二","星期三","星期四","星期五","星期六","星期日"]
        return f"当前时间：{now.strftime('%Y年%m月%d日')} {wd[now.weekday()]} {now.strftime('%H:%M')}"

    def _build_rag_system_prompt(self):
        return (f"系统信息：{self._get_date_context()}\n\n"
            "你是一个专业的知识库问答助手。根据下方提供的检索结果和对话历史回答用户的问题。\n"
            "要求：\n1. 仅基于检索结果中的信息作答，不要编造内容。\n"
            "2. 用清晰、结构化的中文回答。\n3. 在回答末尾用 [1][2] 等标注引用。\n"
            "4. 如果检索结果不足以回答问题，如实说明。\n5. 结合对话历史理解用户意图。\n")

    def _build_general_knowledge_prompt(self):
        return (f"系统信息：{self._get_date_context()}\n\n"
            "你是一个知识渊博的AI助手。本地知识库和联网搜索均未能找到相关信息。\n"
            "请根据你自身的知识储备，用清晰、结构化的中文回答。\n"
            "在回答开头注明：'以下回答基于AI通用知识，非来自知识库检索。'\n")

    def _create_explicit_plan(self, query, conversation_history, routing):
        from langchain_core.messages import HumanMessage
        prompt = f"""你是一个多步检索规划器。将复杂问题拆成1-3个子问题。
每子问题指定preferred_source(local/web/both)。只返回JSON。
用户问题：{query}\n路由信息：{routing}
JSON格式：{{"strategy":"...","steps":[{{"sub_question":"...","preferred_source":"local","goal":"..."}}]}}"""
        try:
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            content = resp.content if hasattr(resp, "content") else str(resp)
            match = json.loads(content)
            if isinstance(match, dict) and match.get("steps"):
                return match
        except Exception:
            pass
        pref = "both" if routing.get("needs_local") and routing.get("needs_web") else ("web" if routing.get("needs_web") else "local")
        return {"strategy":"默认单步计划","steps":[{"sub_question":query,"preferred_source":pref,"goal":"回答用户当前问题"}]}

    def _read_retrieved_evidence(self, query, goal, local_results, web_results):
        from langchain_core.messages import HumanMessage
        lb = "\n\n".join(f"[Local {i}] {r.get('source','unknown')}\n{r.get('content','')[:400]}"
            for i, r in enumerate(local_results[:3], 1)) or "No local results"
        wb = "\n\n".join(f"[Web {i}] {r.get('title',r.get('source','web'))}\n{r.get('snippet',r.get('content',''))[:400]}"
            for i, r in enumerate(web_results[:3], 1)) or "No web results"
        prompt = f"""阅读检索结果判断是否已有足够证据。当前子问题：{query} 当前目标：{goal}
本地结果：{lb}\n联网结果：{wb}
只返回JSON：{{"summary":"...","enough_to_answer_sub_question":true,"suggested_next_action":"continue|search_web|refine|generate","missing_information":"..."}}"""
        try:
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            content = resp.content if hasattr(resp, "content") else str(resp)
            return json.loads(content)
        except Exception:
            enough = bool(local_results or web_results)
            return {"summary":"已基于当前检索结果完成初步阅读。","enough_to_answer_sub_question":enough,
                "suggested_next_action":"continue" if enough else "refine","missing_information":"" if enough else "当前证据不足。"}

    def _filter_local_results(self, local_results, eval_confidence):
        if eval_confidence >= 0.7 or not local_results:
            return local_results
        filtered = [r for r in local_results if r.get("score", 0) >= self._LOCAL_RELEVANCE_THRESHOLD]
        return filtered if filtered else local_results[:1]

    def _generate_normal_response_with_citations(self, context):
        local_results = context.get("local_results", [])
        web_results = context.get("web_results", [])
        query = context.get("user_input", "")
        plan_obs = context.get("blackboard", {}).get("plan_observations", [])
        ec = context.get("evaluation", {}).get("confidence", 1.0)
        local_results = self._filter_local_results(local_results, ec)
        cits = CitationManager.create_citations_from_results(local_results=local_results, web_results=web_results, top_k=5)
        cm = CitationManager()
        cm.add_citations(cits)
        if not local_results and not web_results:
            return "抱歉，没有找到相关信息。", [], cm.check_faithfulness("抱歉，没有找到相关信息。")
        parts = []
        for i, r in enumerate(local_results[:5], 1):
            parts.append(f"[{i}] (本地知识库: {r.get('source','unknown')})\n{r.get('content','')}")
        off = len(local_results[:5])
        for i, r in enumerate(web_results[:3], 1):
            parts.append(f"[{off+i}] (互联网: {r.get('title','')}, {r.get('url','')})\n{r.get('snippet',r.get('content',''))}")
        ctx = "\n\n".join(parts)
        ob = ""
        if plan_obs:
            ob = "\n\n## 计划观察\n" + "\n".join(
                f"- 子问题: {o.get('sub_question','')}\n  观察: {o.get('summary','')}" for o in plan_obs[-5:])
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        msgs = [SystemMessage(content=self._build_rag_system_prompt())]
        for t in context.get("conversation_history", []):
            r, c = t.get("role",""), t.get("content","")
            if r == "user": msgs.append(HumanMessage(content=c))
            elif r == "assistant": msgs.append(AIMessage(content=c))
        msgs.append(HumanMessage(content=f"## 检索结果\n\n{ctx}{ob}\n\n## 用户问题\n{query}"))
        try:
            resp = self.llm.invoke(msgs)
            ans = resp.content if hasattr(resp, "content") else str(resp)
        except Exception:
            parts2 = [f"{i}. {r.get('content','')[:300]}..." for i, r in enumerate(local_results[:3], 1)]
            ans = "根据检索结果：\n" + "\n".join(parts2) if parts2 else "抱歉，没有找到相关信息。"
        ans = format_answer_with_citations(answer=ans, citations=cits, include_reference_list=True)
        fth = cm.check_faithfulness(ans)
        return ans, cits, fth

    def _generate_normal_response(self, context):
        a, _, _ = self._generate_normal_response_with_citations(context)
        return a

    def _generate_general_knowledge_answer(self, context):
        query = context.get("user_input", "")
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        msgs = [SystemMessage(content=self._build_general_knowledge_prompt())]
        for t in context.get("conversation_history", [])[-6:]:
            if t.get("role") == "user": msgs.append(HumanMessage(content=t.get("content","")))
            elif t.get("role") == "assistant": msgs.append(AIMessage(content=t.get("content","")))
        msgs.append(HumanMessage(content=query))
        try:
            resp = self.llm.invoke(msgs)
            return resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            return f"抱歉，检索和联网搜索均未找到结果，LLM 直接回答也遇到了问题：{e}"

    def _generate_fallback_response(self, state):
        rm = {FallbackReason.MAX_RETRIES_EXCEEDED: "经过多次检索和优化，我依然无法找到确切答案",
            FallbackReason.NO_RESULTS_FOUND: "本地知识库和互联网上都没有相关信息",
            FallbackReason.LOW_CONFIDENCE: "检索到的信息相关性较低，无法提供可靠答案",
            FallbackReason.USER_ASKED_UNKNOWN: "该问题涉及系统无法获取的信息"}
        reason = state.fallback_reason or FallbackReason.NO_RESULTS_FOUND
        rt = rm.get(reason, "经过检索，我无法找到确切答案")
        return f"""抱歉，{rt}。\n检索详情：检索次数 {state.retry_count} 次，本地知识库 {len(state.local_results)} 条，互联网 {len(state.web_results)} 条。\n建议您：1. 重新描述问题，提供更多上下文 2. 尝试使用不同的表述方式 3. 或者询问其他我可能帮助的问题"""

    def run(self, user_input, conversation_history=None):
        s = AgentState(user_input=user_input, conversation_history=conversation_history or [])
        return self.workflow.invoke(s)
