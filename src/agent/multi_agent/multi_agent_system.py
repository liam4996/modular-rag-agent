"""多智能体 RAG 系统 - 主编排器

使用 LangGraph 编排多个专门智能体。
Phase 2: 集成 Supervisor + FinanceDataAgent，保持向后兼容。
"""

from concurrent.futures import ThreadPoolExecutor
import json
import logging
from typing import Optional, List, Dict, Literal, Tuple, Any

logger = logging.getLogger(__name__)

from langgraph.graph import StateGraph, END

from .state import AgentState, FallbackReason
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
from .tools.json_parser import safe_parse_json_with_default


class MultiAgentRAG:
    def __init__(self, llm, settings=None, enable_logging=True, store=None):
        self.llm = llm
        self.settings = settings or {}
        self.enable_logging = enable_logging
        self.store = store
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
            ("plan", self._plan_node),
            ("retrieve", self._retrieve_node), ("web", self._web_node),
            ("read", self._read_node), ("eval", self._eval_node),
            ("refine", self._refine_node), ("generate", self._generate_node),
            ("supervisor", self._supervisor_node), ("finance_data", self._finance_data_node),
            ("aggregate", self._aggregate_node), ("global_eval", self._global_eval_node),
            ("business_compute", self._business_compute_node),
        ]:
            workflow.add_node(name, fn)
        workflow.set_entry_point("supervisor")
        workflow.add_conditional_edges("supervisor", self._route_from_supervisor, {
            "generate": "generate", "plan": "plan", "retrieve": "retrieve",
            "web": "web", "finance_data": "finance_data",
            "aggregate": "aggregate", "business_compute": "business_compute"})
        workflow.add_conditional_edges("plan", self._route_after_plan, {
            "generate": "generate", "retrieve": "retrieve"})
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

    # ── Plan ──
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

    # ── Retrieve ──
    def _retrieve_node(self, state):
        plan = state.blackboard.get("active_retrieve_plan", state.blackboard.get("retrieve_plan", "local"))
        base = state.blackboard.get("active_sub_question", state.user_input)
        query = state.refined_query if state.retry_count > 0 else base
        ctx = state.conversation_history
        state.blackboard["retrieval_executed"] = True
        le: Optional[str] = None

        # 🆕 从当前子任务中获取 collections 参数
        task = state.blackboard.get("current_task", {})
        collections = task.get("collections") if isinstance(task, dict) else None

        def _run_local():
            try:
                kwargs = {"query": query, "top_k": 10, "context": ctx}
                if collections:
                    kwargs["collections"] = collections
                return self.search_agent.search(**kwargs)
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
            # 🆕 金融意图 + 本地检索不足 → 自动触发联网财报抓取
            if (state.is_financial_intent and len(r) < 2):
                self._auto_fetch_financial_docs(state)
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
            # 🆕 金融意图 + 本地检索不足 → 自动触发
            if state.is_financial_intent and len(lr) < 2:
                self._auto_fetch_financial_docs(state)
        return state

    def _auto_fetch_financial_docs(self, state):
        """当本地 RAG 检索不到财报时，自动通过 MCP Tool + WebSearch 抓取并入库。"""
        try:
            from .tools.financial_document_fetcher import FinancialDocumentFetcher
            fetcher = FinancialDocumentFetcher(
                settings=self.settings,
                finance_data_agent=self.finance_data_agent,
            )

            task = state.blackboard.get("current_task", {})
            query = state.blackboard.get("active_sub_question", state.user_input)
            symbols = self.supervisor_agent._extract_symbols(query)

            company = task.get("query", state.user_input)

            state.add_execution_trace({
                "agent": "retrieve",
                "action": "auto_fetch_triggered",
                "reason": "insufficient_local_results",
                "local_count": len(state.local_results),
                "symbols": symbols,
                "status": "running",
            })

            result = fetcher.fetch_and_ingest(
                company=company,
                symbols=symbols or [],
                state=state,
            )

            if result["success"] and result.get("documents"):
                lr = list(state.local_results)
                for doc in result["documents"]:
                    lr.append({
                        "content": doc.get("content", ""),
                        "source": doc.get("source", "financial_fetcher"),
                        "score": 0.8,
                        "source_type": "online_fetched",
                    })
                state.add_to_blackboard("local_results", lr, "retrieve")

            state.add_execution_trace({
                "agent": "retrieve",
                "action": "auto_fetch_done",
                "source": result["source"],
                "market": result.get("market", "unknown"),
                "doc_count": len(result.get("documents", [])),
                "ingested": result["ingested"],
                "chunk_count": result.get("chunk_count", 0),
                "status": "done",
            })
        except Exception as e:
            logger.warning(f"Auto fetch financial docs failed: {e}")
            state.add_execution_trace({
                "agent": "retrieve",
                "action": "auto_fetch_error",
                "error": str(e),
                "status": "error",
            })

    # ── Web ──
    def _web_node(self, state):
        lr = state.read_from_blackboard("local_results")
        q = state.blackboard.get("active_sub_question", state.user_input)
        wr = self.web_agent.search(query=q, num_results=5, time_range="y", local_results=lr)
        state.blackboard["web_search_attempted"] = True
        state.add_to_blackboard("web_results", wr, "web")
        return state

    # ── Read ──
    def _read_node(self, state):
        sub = state.blackboard.get("active_sub_question", state.user_input)
        goal = state.blackboard.get("active_plan_goal", "")
        reading = self._read_retrieved_evidence(
            query=sub, goal=goal,
            local_results=state.local_results, web_results=state.web_results)
        state.add_to_blackboard("reading_assessment", reading, "read")
        obs = state.blackboard.get("plan_observations", [])
        obs.append({
            "step_index": state.blackboard.get("plan_current_step", 0),
            "sub_question": sub,
            "summary": reading.get("summary", ""),
            "enough_to_answer_sub_question": reading.get("enough_to_answer_sub_question", False),
            "suggested_next_action": reading.get("suggested_next_action", ""),
            "missing_information": reading.get("missing_information", "")})
        state.blackboard["plan_observations"] = obs
        return state

    # ── Eval ──
    def _eval_node(self, state):
        lr = state.local_results
        wr = state.web_results
        q = state.blackboard.get("active_sub_question", state.user_input)
        ev = self.eval_agent.evaluate(
            local_results=lr, web_results=wr, query=q,
            retry_count=state.retry_count, max_retries=state.max_retries)
        state.add_to_blackboard("evaluation", {
            "relevance": ev.relevance, "diversity": ev.diversity,
            "coverage": ev.coverage, "confidence": ev.confidence,
            "need_refinement": ev.need_refinement,
            "fallback_suggested": ev.fallback_suggested,
            "reason": ev.reason}, "eval")
        if ev.fallback_suggested:
            if state.retry_count >= state.max_retries:
                state.trigger_fallback(FallbackReason.MAX_RETRIES_EXCEEDED, "eval")
            elif ev.relevance < 0.2:
                state.trigger_fallback(FallbackReason.NO_RESULTS_FOUND, "eval")
            elif ev.confidence < 0.3:
                state.trigger_fallback(FallbackReason.LOW_CONFIDENCE, "eval")
        return state

    # ── Refine ──
    def _refine_node(self, state):
        ed = state.evaluation
        ev = EvaluationResult(
            relevance=ed.get("relevance", 0.5), diversity=ed.get("diversity", 0.5),
            coverage=ed.get("coverage", 0.5), confidence=ed.get("confidence", 0.5),
            need_refinement=ed.get("need_refinement", True),
            fallback_suggested=ed.get("fallback_suggested", False),
            reason=ed.get("reason", ""))
        q = state.blackboard.get("active_sub_question", state.user_input)
        ref = self.refine_agent.refine(
            original_query=q, evaluation=ev, retry_count=state.retry_count)
        state.increment_retry("refine")
        state.add_to_blackboard("refined_query", ref.refined_query, "refine")
        return state

    # ── Generate ──
    def _generate_node(self, state):
        ctx = state.get_all_context()
        intent = (state.intent or "").lower()

        if intent.startswith("financial_"):
            return self._generate_financial_report(state)

        has_l = bool(state.local_results)
        has_w = bool(state.web_results)
        wa = state.blackboard.get("web_search_attempted", False)
        ec = state.blackboard.get("evaluation", {}).get("confidence", 1.0)
        local_noise = False
        if has_l and ec < 0.3:
            useful = [r for r in state.local_results if r.get("score", 0) >= self._LOCAL_RELEVANCE_THRESHOLD]
            local_noise = len(useful) == 0
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

    def _generate_financial_report(self, state):
        market = state.blackboard.get("market_data", {})
        computed = state.blackboard.get("computed_results", {})
        charts = state.chart_paths
        report_draft = state.blackboard.get("generated_report", "")
        local_docs = state.local_results
        query = state.user_input

        # 🆕 Layer 1: 纯数学计算引擎（VerificationCalculator）
        from .tools.verification_calculator import VerificationCalculator
        calc = VerificationCalculator()
        fixed_numbers = calc.compute(market, computed)
        state.blackboard["fixed_numbers"] = fixed_numbers.to_dict()

        # 🆕 Layer 2: EvidenceExtractor + LLM 撰写
        context_blocks = []

        if local_docs:
            context_blocks.append("## 文档原文引用\n")
            for i, r in enumerate(local_docs[:5], 1):
                src = r.get("source", r.get("source_path", "unknown"))
                content = r.get("content", "")[:800]
                context_blocks.append(f"[{i}] (来源: {src})\n{content}\n")

        fundamentals = market.get("fundamentals", [])
        quotes = market.get("quote", [])
        if fundamentals or quotes:
            context_blocks.append("## 行情与基本面数据\n")
            for f in fundamentals:
                if not isinstance(f, dict):
                    continue
                sym = f.get("symbol", "")
                items = []
                for k in ["pe", "pb", "roe", "gross_margin", "net_margin", "eps", "bvps"]:
                    if f.get(k) is not None:
                        items.append(f"{k.upper()}={f[k]}")
                if items:
                    context_blocks.append(f"**{sym}**: {' | '.join(items)}\n")
            for q in quotes:
                if not isinstance(q, dict):
                    continue
                sym = q.get("symbol", "")
                items = []
                for k in ["price", "change_pct", "market_cap"]:
                    if q.get(k) is not None:
                        items.append(f"{k}={q[k]}")
                if items:
                    context_blocks.append(f"**{sym}** (行情): {' | '.join(items)}\n")

        metrics = computed.get("metrics", {})
        if metrics:
            context_blocks.append("## 计算后的财务指标\n")
            import json as _json
            context_blocks.append(f"```json\n{_json.dumps(metrics, ensure_ascii=False, indent=2)}\n```\n")

        comparison = computed.get("comparison_table", "")
        if comparison:
            context_blocks.append("## 行业对比\n")
            context_blocks.append(comparison + "\n")

        if charts:
            context_blocks.append("## 已生成图表\n")
            for c in charts:
                if isinstance(c, dict):
                    context_blocks.append(f"- {c.get('title', '图表')}: {c.get('path', '')}\n")

        if report_draft:
            context_blocks.append("## 模板化分析初稿\n")
            context_blocks.append(report_draft[:2000] + "\n")

        # 🆕 将计算出的数字注入上下文（约束 LLM 引用）
        fixed_dict = fixed_numbers.to_dict()
        if fixed_dict:
            context_blocks.append("## 关键数字（必须引用）\n")
            for k, v in fixed_dict.items():
                context_blocks.append(f"- {k}: {v}\n")

        financial_context = "\n".join(context_blocks)

        # 🆕 Stage 1.5: Chain-of-Analysis（深度模式）
        depth = state.blackboard.get("analysis_depth", "standard")
        coa_chains = None
        if depth == "deep":
            try:
                from .tools.chain_of_analysis import ChainOfAnalysis
                coa_gen = ChainOfAnalysis()
                coa_result = coa_gen.generate({
                    "metrics": metrics,
                    "market_data": market,
                    "fixed_numbers": fixed_dict,
                }, self.llm)
                coa_chains = coa_result.get("chains", [])
                if coa_chains:
                    coa_text = "\n\n## 分析链（基于此撰写报告，每段对应一条分析链）\n"
                    for i, c in enumerate(coa_chains, 1):
                        coa_text += f"\n[{i}] {c['observation']}\n    解释: {c.get('interpretation', '')}\n    含义: {c.get('implication', '')}\n"
                    financial_context += coa_text
                    state.add_to_blackboard("chain_of_analysis", coa_chains, "generate")
                    state.add_metric("coa_count", len(coa_chains))
            except Exception as e:
                state.add_metric("coa_error", str(e))

        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        analysis_date = getattr(state, "analysis_date", None)
        msgs = [SystemMessage(content=self._build_financial_analyst_prompt(analysis_date=analysis_date))]
        for t in state.conversation_history or []:
            r, c = t.get("role", ""), t.get("content", "")
            if r == "user":
                msgs.append(HumanMessage(content=c))
            elif r == "assistant":
                msgs.append(AIMessage(content=c))
        msgs.append(HumanMessage(content=f"{financial_context}\n\n## 用户请求\n{query}"))

        try:
            resp = self.llm.invoke(msgs)
            ans = resp.content if hasattr(resp, "content") else str(resp)
        except Exception:
            ans = self._generate_fallback_report(state)

        # 🆕 Layer 3: CitationFootnoter + VerificationValidator
        from .tools.citation_footnoter import CitationFootnoter
        footnoter = CitationFootnoter()
        ans = footnoter.process(ans, local_docs)

        from .tools.verification_validator import VerificationValidator
        validator = VerificationValidator()
        ans, verify_warnings = validator.validate(ans, fixed_numbers)
        coverage = validator.check_citation_coverage(ans, local_docs)
        state.add_to_blackboard("citation_coverage", coverage, "generate")
        state.add_metric("verification_warnings", len(verify_warnings))
        if verify_warnings:
            state.add_to_blackboard("verification_warnings", verify_warnings, "generate")

        state.final_answer = ans
        state.add_metric("generation_mode", "financial_report")
        state.add_metric("report_length", len(ans))
        return state

    def _build_financial_analyst_prompt(self, analysis_date=None):
        from datetime import datetime
        effective_date = analysis_date or datetime.now().strftime("%Y-%m-%d")
        return (
            f"当前日期: {effective_date}\n\n"
            "你是一位资深证券分析师（CFA持证人），就职于一家头部券商研究所。\n"
            "请基于下方提供的行情数据、财务指标、行业对比和图表，撰写一份专业研报。\n\n"
            "## 推理框架（Financial CoT）\n"
            "请在实际写作前，先完成以下推理链（每条用一句概括，写在报告开篇的投资要点中）：\n"
            "1. Observation（观察到什么数据变化？）\n"
            "2. Interpretation（数据变化意味着什么？归因分析）\n"
            "3. Implication（对公司/行业/估值的影响是什么？）\n\n"
            "## 报告结构要求\n\n"
            "### 1. 投资要点（开篇 3-5 句话，包含推理链摘要）\n"
            "- 给出明确投资评级：买入/增持/中性/减持/卖出\n"
            "- 给出核心逻辑（不超过 3 条，每条对应一条推理链）\n"
            "- **必须给出目标价区间**，格式: `目标价区间: XX.XX - YY.YY 元`，其中下限对应保守估值，上限对应乐观估值\n"
            "- 目标价区间需综合基本面估值(PE/PB/PEG)、行业对比、成长性等多维度判断\n"
            "- 标注估值方法（如：PE估值法、PEG估值法、行业对比法）\n\n"
            "### 2. 财务分析\n"
            "- 对营收、利润、毛利率等关键指标的变动做归因分析\n"
            "- 引用提供的财务数据原文\n"
            "- 用 [1][2] 标注来源\n"
            "- 包含同比/环比分析，标明增速变化趋势\n"
            "- PE、PB、ROE、EPS 等指标必须基于获取到的真实数据分析\n\n"
            "### 3. 行业对比\n"
            "- 将公司与行业均值/竞品做对比\n"
            "- 标注公司的领先/落后指标\n"
            "- 使用标准化排名：超越/持平/不及\n\n"
            "### 4. 估值分析\n"
            "- PE/PB 水平与历史区间、行业均值对比\n"
            "- 判断当前估值是否合理（高估/合理/低估）\n"
            "- 结合成长性判断 PEG\n"
            "- 给出估值方法说明和目标价区间推导过程\n"
            "- 格式: `目标价: XX.XX - YY.YY 元（保守-乐观），基于 PE估值/PEG估值/行业对比`\n\n"
            "### 5. 风险提示\n"
            "- 列出至少 3 条具体风险\n"
            "- 从提供的数据中提取风险因素\n"
            "- 区分：财务风险/行业风险/市场风险/公司治理风险\n\n"
            "### 6. 图表引用\n"
            "- 在正文中自然引用已生成的图表（标注文件名）\n"
            "- 每张图表配一句解读文字\n\n"
            "## 写作要求\n"
            "- 专业、客观、有理有据\n"
            "- 数据引用标注来源 [1][2]\n"
            "- 中文撰写，数字保留两位小数\n"
            "- 如果某些数据缺失，如实标注\"数据未获取\"\n"
            "- 表格用 Markdown 格式\n"
            "- 报告总字数：2000-5000 字\n"
            "- 目标价必须输出为区间格式，不要输出单点值\n"
        )

    def _generate_fallback_report(self, state):
        market = state.blackboard.get("market_data", {})
        computed = state.blackboard.get("computed_results", {})
        charts = state.chart_paths
        lines = ["# 金融分析报告\n"]
        metrics = computed.get("metrics", {})
        if metrics:
            lines.append("## 财务指标\n")
            for sym, m in metrics.items():
                lines.append(f"### {sym}\n")
                for cat, vals in m.items():
                    if isinstance(vals, dict) and vals:
                        lines.append(f"**{cat}**: " + " | ".join(
                            f"{k}={v:.2f}" for k, v in vals.items() if v is not None) + "\n")
        comparison = computed.get("comparison_table", "")
        if comparison:
            lines.append("## 行业对比\n")
            lines.append(comparison + "\n")
        if charts:
            lines.append("## 图表\n")
            for c in charts:
                if isinstance(c, dict):
                    lines.append(f"- [{c.get('title', '图表')}]({c.get('path', '')})\n")
        lines.append(f"\n---\n*数据驱动分析，Business Compute Agent 自动生成*\n")
        return "\n".join(lines)

    # ── 金融节点 ──
    def _supervisor_node(self, state):
        """Merged Router + Supervisor: 一次调用完成意图分类 + 任务拆解"""
        self.supervisor_agent.classify_and_plan(state)
        state.add_execution_trace({"agent": "supervisor", "action": "classify_and_plan_complete",
            "intent": state.intent,
            "task_count": len(state.blackboard.get("task_plan", {}).get("subtasks", []))})
        return state

    def _route_from_supervisor(self, state):
        """
        合并路由：根据 Supervisor 分类 + analysis_depth 路由到对应节点。
        """
        intent = (state.intent or "").lower()
        plan = state.blackboard.get("retrieve_plan", "local")
        depth = state.blackboard.get("analysis_depth", "standard")

        # chat → 直接生成
        if plan == "none" or intent == "chat":
            return "generate"

        # 金融意图 → 走子任务派发（根据 depth 过滤）
        if intent.startswith("financial_"):
            # quick 模式：跳过文档检索，直奔 finance_data
            if depth == "quick":
                state.add_to_blackboard("active_sub_question", state.user_input, "supervisor")
                state.add_to_blackboard("current_task", {"type": "financial_market"}, "supervisor")
                return "finance_data"
            return self._dispatch_financial_subtask(state)

        # 复杂查询 → plan 节点
        complexity = state.blackboard.get("query_complexity", "simple")
        if complexity == "complex" or plan == "both":
            return "plan"

        # 联网搜索
        if plan == "web":
            return "web"

        # 默认：本地检索
        return "retrieve"

    def _dispatch_financial_subtask(self, state):
        """金融子任务派发（原 _dispatch_from_supervisor）"""
        next_t = self.supervisor_agent.get_next_subtasks(state)
        if not next_t:
            return "aggregate" if self.supervisor_agent.all_completed(state) else "generate"
        task = next_t[0]
        tt = task.get("type", "")
        self.supervisor_agent.mark_completed(state, task["id"])
        q = task.get("query", state.user_input)
        state.add_to_blackboard("active_sub_question", q, "supervisor")
        state.add_to_blackboard("current_task", task, "supervisor")
        state.add_execution_trace({"agent": "supervisor", "action": "dispatch", "task_id": task["id"], "task_type": tt})
        if tt == "document_search":
            return "retrieve"
        elif tt == "web_search":
            return "web"
        elif tt == "financial_market":
            return "finance_data"
        elif tt == "financial_computation":
            return "business_compute"
        else:
            return "aggregate"

    def _finance_data_node(self, state):
        task = state.blackboard.get("current_task", {})
        task_id = task.get("id", "unknown")
        task_desc = task.get("description", "")
        sym = task.get("symbols", [])
        if not sym:
            sym = self.supervisor_agent._extract_symbols(state.user_input)
        state.add_execution_trace({
            "agent": "finance_data", "action": "query_market_start",
            "task_id": task_id, "description": task_desc,
            "symbols": sym, "status": "running",
        })
        try:
            self.finance_data_agent.query_and_store(
                state=state, symbols=sym or ["000001.SZ"],
                data_types=["quote", "fundamentals"])
            state.add_execution_trace({
                "agent": "finance_data", "action": "query_market_done",
                "task_id": task_id, "status": "done",
            })
        except Exception as e:
            state.add_execution_trace({
                "agent": "finance_data", "action": "error",
                "task_id": task_id, "error": str(e), "status": "error",
            })
        return state

    def _business_compute_node(self, state):
        """Business Compute Agent 节点：计算 + 可视化 + 报告"""
        task = state.blackboard.get("current_task", {})
        task_id = task.get("id", "unknown")
        task_desc = task.get("description", "")
        operation = task.get("operation", "calculate_metrics")
        state.add_execution_trace({
            "agent": "business_compute", "action": f"{operation}_start",
            "task_id": task_id, "description": task_desc,
            "operation": operation, "status": "running",
        })
        try:
            self.business_compute_agent.compute(
                state=state, operation=operation, task_params=task)
            state.add_execution_trace({
                "agent": "business_compute", "action": f"{operation}_done",
                "task_id": task_id, "status": "done",
            })
        except Exception as e:
            state.add_execution_trace({
                "agent": "business_compute", "action": "error",
                "task_id": task_id, "error": str(e), "status": "error",
            })
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

    # ── 路由 ──
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

        if en and hm:
            state.blackboard["advance_plan_step"] = True
            return "plan"
        if na == "search_web" and not wa:
            return _web("reader suggested web")
        if nf and state.retry_count < state.max_retries:
            return "refine"
        if wa and cf < 0.4 and len(lr) > 0 and state.retry_count < state.max_retries:
            return _local("web confidence too low")
        if wa:
            return "generate"
        if state.fallback_triggered or fs:
            return _web("fallback triggered")
        if nf and state.retry_count >= state.max_retries:
            return _web("retries exhausted")
        if cf < 0.5:
            return _web(f"low confidence ({cf})")
        if en and not hm:
            return "generate"
        return "generate"

    # ── Helpers ──
    @staticmethod
    def _get_date_context():
        from datetime import datetime
        now = datetime.now()
        wd = ["星期一","星期二","星期三","星期四","星期五","星期六","星期日"]
        return f"当前时间：{now.strftime('%Y年%m月%d日')} {wd[now.weekday()]} {now.strftime('%H:%M')}"

    def _build_rag_system_prompt(self):
        return (
            f"系统信息：{self._get_date_context()}\n\n"
            "你是一个专业的知识库问答助手。根据下方提供的检索结果和对话历史回答用户的问题。\n"
            "要求：\n"
            "1. 仅基于检索结果中的信息作答，不要编造内容。\n"
            "2. 用清晰、结构化的中文回答。如果原文是英文，请翻译为中文后作答。\n"
            "3. 在回答末尾用 [1][2] 等标注引用了哪些检索结果。\n"
            "4. 如果检索结果不足以回答问题，如实说明。\n"
            "5. 结合对话历史理解用户意图。\n"
            "6. 信息冲突处理原则：公司政策以本地知识库为准，客观事实以互联网信息为补充。\n"
            "7. 如果上下文中提供了计划观察，请把它们作为多步分析过程中的中间结论，整合进最终答案。\n")

    def _build_general_knowledge_prompt(self):
        return (
            f"系统信息：{self._get_date_context()}\n\n"
            "你是一个知识渊博的AI助手。本地知识库和联网搜索均未能找到与用户问题相关的信息。\n"
            "请根据你自身的知识储备，用清晰、结构化的中文回答用户的问题。\n"
            "要求：\n"
            "1. 在回答开头注明：'以下回答基于AI通用知识，非来自知识库检索。'\n"
            "2. 尽量准确、客观地回答。\n"
            "3. 如果你不确定，请如实说明。\n"
            "4. 结合对话历史理解用户意图。\n")

    def _create_explicit_plan(self, query, conversation_history, routing):
        from langchain_core.messages import HumanMessage
        prompt = f"""你是一个多步检索规划器。请根据用户问题生成一个显式计划。

要求：将复杂问题拆成 1-3 个子问题。每个子问题指定 preferred_source（local/web/both）。每步给一个简短 goal。只返回 JSON。

用户问题：{query}
对话历史：{conversation_history}
路由信息：{routing}

JSON 格式：
{{"strategy": "...", "steps": [{{"sub_question": "...", "preferred_source": "local", "goal": "..."}}]}}"""
        try:
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            content = resp.content if hasattr(resp, "content") else str(resp)
            match = safe_parse_json_with_default(content, {"steps": []})
            if isinstance(match, dict) and match.get("steps"):
                return match
        except Exception:
            pass
        pref = "both" if routing.get("needs_local") and routing.get("needs_web") else (
            "web" if routing.get("needs_web") else "local")
        return {"strategy": "默认单步计划", "steps": [{"sub_question": query, "preferred_source": pref, "goal": "回答用户当前问题"}]}

    def _read_retrieved_evidence(self, query, goal, local_results, web_results):
        from langchain_core.messages import HumanMessage
        lb = "\n\n".join(f"[Local {i}] {r.get('source','unknown')}\n{r.get('content','')[:400]}"
            for i, r in enumerate(local_results[:3], 1)) or "No local results"
        wb = "\n\n".join(f"[Web {i}] {r.get('title',r.get('source','web'))}\n{r.get('snippet',r.get('content',''))[:400]}"
            for i, r in enumerate(web_results[:3], 1)) or "No web results"
        prompt = f"""你是一个阅读与反思节点。请阅读检索结果，判断当前子问题是否已经有足够证据回答。

当前子问题：{query}
当前目标：{goal}

本地结果：{lb}

联网结果：{wb}

只返回 JSON：{{"summary": "...", "enough_to_answer_sub_question": true, "suggested_next_action": "continue|search_web|refine|generate", "missing_information": "..."}}"""
        try:
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            content = resp.content if hasattr(resp, "content") else str(resp)
            return safe_parse_json_with_default(content, {
                "summary": "已基于当前检索结果完成初步阅读。",
                "enough_to_answer_sub_question": bool(local_results or web_results),
                "suggested_next_action": "continue" if (local_results or web_results) else "refine",
                "missing_information": "" if (local_results or web_results) else "当前证据不足。",
            })
        except Exception:
            enough = bool(local_results or web_results)
            return {"summary": "已基于当前检索结果完成初步阅读。",
                "enough_to_answer_sub_question": enough,
                "suggested_next_action": "continue" if enough else "refine",
                "missing_information": "" if enough else "当前证据不足。"}

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
        cits = CitationManager.create_citations_from_results(
            local_results=local_results, web_results=web_results, top_k=5)
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
                f"- 子问题: {o.get('sub_question','')}\n  观察: {o.get('summary','')}"
                for o in plan_obs[-5:])
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
        return f"""抱歉，{rt}。

检索详情：检索次数 {state.retry_count} 次，本地知识库 {len(state.local_results)} 条，互联网 {len(state.web_results)} 条。
建议您：1. 重新描述问题，提供更多上下文 2. 尝试使用不同的表述方式 3. 或者询问其他我可能帮助的问题"""

    def run(self, user_input, conversation_history=None):
        s = AgentState(user_input=user_input, conversation_history=conversation_history or [])
        return self.workflow.invoke(s)

    # ── 流式执行 ──
    def run_stream(self, user_input, conversation_history=None, analysis_date=None):
        """
        流式执行全链路，每次 yield 一个 AgentEvent。
        支持 chat / document / web / financial 全意图的实时进度推送。

        用法:
          for event in agent.run_stream("分析宁德时代"):
              if event.type == EventType.AGENT_STEP:
                  print(f"⏳ {event.content}")
        """
        import time as _time
        from .stream_events import AgentEvent, EventType
        from .tools.conflict_detector import ConflictDetector

        state = AgentState(
            user_input=user_input,
            conversation_history=conversation_history or [],
        )
        if analysis_date:
            state.analysis_date = analysis_date

        # ═══ Step 1: Supervisor 意图识别 + 任务拆解 ═══
        _now = _time.time()
        yield AgentEvent(type=EventType.AGENT_START, agent_name="supervisor",
                         content="正在识别意图并拆解任务…",
                         started_at=_now)
        state = self._supervisor_node(state)

        intent = (state.intent or "").lower()
        depth = state.blackboard.get("analysis_depth", "standard")
        task_plan = state.blackboard.get("task_plan", {})
        subtasks = task_plan.get("subtasks", []) if task_plan else []
        yield AgentEvent(type=EventType.THINKING, agent_name="supervisor",
                         content=f"意图: {intent}，拆解出 {len(subtasks)} 个子任务",
                         details={
                             "intent": intent,
                             "depth": depth,
                             "task_count": len(subtasks),
                             "subtasks": [{"id": t["id"], "type": t["type"], "description": t.get("description", "")} for t in subtasks],
                         })
        yield AgentEvent(type=EventType.AGENT_RESULT, agent_name="supervisor",
                         content="意图识别与任务拆解完成",
                         completed_at=_time.time())

        # 金融意图 — 独立分支，不调用 _route_from_supervisor
        if intent.startswith("financial_"):
            state = yield from self._run_stream_financial(state, intent, depth, subtasks)
            return state

        # ═══ 非金融意图 — 使用 LangGraph 路由 ═══
        route = self._route_from_supervisor(state)

        if route == "generate":
            _now = _time.time()
            yield AgentEvent(type=EventType.AGENT_START, agent_name="generate",
                             content="正在生成回答…", started_at=_now)
            state = self._generate_node(state)
            yield AgentEvent(type=EventType.DONE, agent_name="System",
                             content="完成", details={"answer_length": len(state.final_answer)},
                             completed_at=_time.time())
            return state

        if route == "web":
            _now = _time.time()
            yield AgentEvent(type=EventType.AGENT_START, agent_name="web",
                             content="正在联网搜索…", started_at=_now)
            state = self._web_node(state)
            state = self._read_node(state)
            state = self._eval_node(state)
            yield AgentEvent(type=EventType.AGENT_RESULT, agent_name="web",
                             content=f"搜索完成，获取 {len(state.web_results)} 条结果",
                             completed_at=_time.time())
            _now = _time.time()
            yield AgentEvent(type=EventType.AGENT_START, agent_name="generate",
                             content="正在生成回答…", started_at=_now)
            state = self._generate_node(state)
            yield AgentEvent(type=EventType.DONE, agent_name="System",
                             content="完成", details={"answer_length": len(state.final_answer)},
                             completed_at=_time.time())
            return state

        # route == "retrieve" or default
        yield from self._stream_single_retrieve(state)
        _now = _time.time()
        yield AgentEvent(type=EventType.AGENT_START, agent_name="generate",
                         content="正在生成回答…", started_at=_now)
        state = self._generate_node(state)
        yield AgentEvent(type=EventType.DONE, agent_name="System",
                         content="完成", details={"answer_length": len(state.final_answer)},
                         completed_at=_time.time())
        return state

    def run_stream_financial_direct(self, user_input, conversation_history=None, analysis_date=None, symbol=None):
        """
        研报生成专用快速通道：跳过 supervisor LLM 意图分类，直接构造金融分析任务列表并执行。
        """
        import time as _time
        from .stream_events import AgentEvent, EventType
        from .tools.conflict_detector import ConflictDetector

        state = AgentState(
            user_input=user_input,
            conversation_history=conversation_history or [],
        )
        if analysis_date:
            state.analysis_date = analysis_date
        if symbol:
            state.add_to_blackboard("symbol", symbol, "system")

        # 直接构造金融分析 subtasks
        subtasks = [
            {"id": "market_data", "type": "financial_market",
             "description": "行情数据查询",
             "query": f"查询 {symbol or user_input} 的实时行情和基本面数据"},
            {"id": "fundamental_data", "type": "financial_market",
             "description": "财务基本面数据查询",
             "query": f"查询 {symbol or user_input} 的 PE/PB/ROE/EPS 等基本面指标"},
            {"id": "compute_metrics", "type": "financial_computation",
             "description": "财务指标计算与图表生成",
             "query": f"计算 {symbol or user_input} 的估值指标并生成图表"},
        ]
        depth = state.blackboard.get("analysis_depth", "standard")
        intent = "financial_analysis"

        # 写入 state
        state.blackboard["intent"] = intent
        state.blackboard["task_plan"] = {"subtasks": subtasks}
        state.blackboard["analysis_depth"] = depth

        yield AgentEvent(type=EventType.AGENT_START, agent_name="supervisor",
                         content="研报模式启动，直接进入分析流程…",
                         started_at=_time.time())
        yield AgentEvent(type=EventType.THINKING, agent_name="supervisor",
                         content=f"意图: {intent}，拆解出 {len(subtasks)} 个子任务",
                         details={
                             "intent": intent, "depth": depth,
                             "task_count": len(subtasks),
                             "subtasks": [{"id": t["id"], "type": t["type"],
                                           "description": t.get("description", "")} for t in subtasks],
                         })
        yield AgentEvent(type=EventType.AGENT_RESULT, agent_name="supervisor",
                         content="任务构建完成（快速通道）",
                         completed_at=_time.time())

        state = yield from self._run_stream_financial(state, intent, depth, subtasks)
        return state

    def _run_stream_financial(self, state, intent, depth, subtasks):
        """金融意图流式执行：逐个子任务派发 + 聚合 + 生成研报。"""
        import time as _time
        from .stream_events import AgentEvent, EventType
        from .tools.conflict_detector import ConflictDetector

        total_tasks = len(subtasks)

        # ═══ quick 模式：跳过文档检索，直奔行情 + 计算 ═══
        if depth == "quick":
            _now = _time.time()
            yield AgentEvent(type=EventType.AGENT_START, agent_name="finance_data",
                             content="正在查询行情数据…", started_at=_now)
            state.add_to_blackboard("active_sub_question", state.user_input, "supervisor")
            state.add_to_blackboard("current_task", {"type": "financial_market", "operation": "calculate_metrics"}, "supervisor")
            state = self._finance_data_node(state)
            yield AgentEvent(type=EventType.AGENT_RESULT, agent_name="finance_data",
                             content="行情数据获取完成", completed_at=_time.time())
            _now = _time.time()
            yield AgentEvent(type=EventType.AGENT_START, agent_name="business_compute",
                             content="正在计算财务指标…", started_at=_now)
            self.business_compute_agent.compute(state=state, operation="calculate_metrics",
                                                 task_params=state.blackboard.get("current_task", {}))
            yield AgentEvent(type=EventType.AGENT_RESULT, agent_name="business_compute",
                             content="财务指标计算完成", completed_at=_time.time())
        else:
            # ═══ 标准/深度模式：逐步派发所有子任务 ═══
            completed_count = 0
            while not self.supervisor_agent.all_completed(state):
                next_tasks = self.supervisor_agent.get_next_subtasks(state)
                if not next_tasks:
                    break

                for task in next_tasks:
                    task_id = task.get("id", "unknown")
                    task_type = task.get("type", "")
                    task_desc = task.get("description", task_id)
                    task_query = task.get("query", state.user_input)

                    self.supervisor_agent.mark_completed(state, task_id)
                    state.add_to_blackboard("active_sub_question", task_query, "supervisor")
                    state.add_to_blackboard("current_task", task, "supervisor")
                    completed_count += 1

                    yield AgentEvent(type=EventType.AGENT_STEP, agent_name="supervisor",
                                     content=f"执行子任务 [{completed_count}/{total_tasks}]: {task_desc}",
                                     details={"task_id": task_id, "task_type": task_type,
                                              "task_index": completed_count, "total_tasks": total_tasks})

                    if task_type == "document_search":
                        yield from self._stream_single_retrieve(state)
                    elif task_type == "web_search":
                        _now = _time.time()
                        yield AgentEvent(type=EventType.AGENT_START, agent_name="web",
                                         content=f"正在联网搜索: {task_desc}", started_at=_now)
                        state = self._web_node(state)
                        state = self._read_node(state)
                        state = self._eval_node(state)
                        yield AgentEvent(type=EventType.AGENT_RESULT, agent_name="web",
                                         content=f"搜索完成，获取 {len(state.web_results)} 条结果",
                                         completed_at=_time.time())
                    elif task_type == "financial_market":
                        _now = _time.time()
                        yield AgentEvent(type=EventType.AGENT_START, agent_name="finance_data",
                                         content=f"正在查询行情: {task_desc}", started_at=_now)
                        state = self._finance_data_node(state)
                        yield AgentEvent(type=EventType.AGENT_RESULT, agent_name="finance_data",
                                         content="行情数据获取完成", completed_at=_time.time())
                    elif task_type == "financial_computation":
                        _now = _time.time()
                        yield AgentEvent(type=EventType.AGENT_START, agent_name="business_compute",
                                         content=f"正在计算: {task_desc}", started_at=_now)
                        state = self._business_compute_node(state)
                        yield AgentEvent(type=EventType.AGENT_RESULT, agent_name="business_compute",
                                         content="计算完成", completed_at=_time.time())

        # ═══ 数据汇总 + 全局评估 ═══
        _now = _time.time()
        yield AgentEvent(type=EventType.AGENT_START, agent_name="aggregate",
                         content="正在汇总各 Agent 数据…", started_at=_now)
        state = self._aggregate_node(state)
        state = self._global_eval_node(state)
        yield AgentEvent(type=EventType.AGENT_RESULT, agent_name="aggregate",
                         content="数据汇总完成", completed_at=_time.time())

        # ═══ 冲突检测 ═══
        detector = ConflictDetector()
        conflict_flags = detector.detect(state.blackboard)
        if conflict_flags:
            state.blackboard["conflict_flags"] = conflict_flags
            for flag in conflict_flags:
                yield AgentEvent(type=EventType.CONFLICT, agent_name="conflict",
                                 content=f"[{flag.severity}] {flag.claim_a} vs {flag.claim_b}",
                                 details=flag.__dict__)

        # ═══ 研报生成 ═══
        _now = _time.time()
        yield AgentEvent(type=EventType.AGENT_START, agent_name="generate",
                         content="正在撰写研报…", started_at=_now)
        state = self._generate_node(state)

        answer_len = len(state.final_answer) if state.final_answer else 0
        preview = state.final_answer[:300] if state.final_answer else ""
        yield AgentEvent(type=EventType.PARTIAL_TEXT, agent_name="generate",
                         content=preview)
        yield AgentEvent(type=EventType.AGENT_RESULT, agent_name="generate",
                         content=f"研报生成完成 ({answer_len} 字)",
                         completed_at=_time.time())

        yield AgentEvent(type=EventType.DONE, agent_name="System",
                         content="全流程执行完成",
                         details={
                             "answer_length": answer_len,
                             "task_count": total_tasks,
                         })

        return state

    def _stream_single_retrieve(self, state):
        """流式执行单次检索 + 阅读 + 评估，yield 事件。"""
        from .stream_events import AgentEvent, EventType
        yield AgentEvent(type=EventType.AGENT_START, agent_name="rag",
                         content="正在检索知识库…")
        state = self._retrieve_node(state)
        yield AgentEvent(type=EventType.AGENT_STEP, agent_name="rag",
                         content=f"检索到 {len(state.local_results)} 条本地结果, {len(state.web_results)} 条网络结果")
        state = self._read_node(state)
        state = self._eval_node(state)
        ev = state.evaluation
        yield AgentEvent(type=EventType.AGENT_RESULT, agent_name="rag",
                         content=f"检索评估: 相关度 {ev.get('relevance', 0):.2f}, 置信度 {ev.get('confidence', 0):.2f}",
                         details={"relevance": ev.get("relevance", 0), "confidence": ev.get("confidence", 0)})
