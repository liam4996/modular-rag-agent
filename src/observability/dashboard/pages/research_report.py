"""智能研报页面 — 一键生成专业级金融分析报告。

Layout:
1. Sidebar: 公司/周期/深度/选项配置 + 历史记录
2. 主区域: 实时进度 → 管道可视化 → 思考过程 → 研报正文 → 图表 → PDF导出
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from src.agent.multi_agent import (
    AgentState,
    MultiAgentRAG,
)
from src.agent.multi_agent.stream_events import AgentEvent, EventType

logger = logging.getLogger(__name__)

_AGENT_ICONS = {
    "supervisor": "🎯", "router": "🎯",
    "rag": "🔍", "retrieve": "🔍",
    "web": "🌐",
    "eval": "📏",
    "refine": "✨",
    "generate": "✍️",
    "finance_data": "💰",
    "business_compute": "📊",
    "conflict": "⚠️",
    "chart_critic": "🎨",
    "aggregate": "📦",
    "global_eval": "📏",
    "plan": "📋",
    "system": "⚙️", "System": "⚙️",
}

_STAGE_LABELS = {
    "supervisor": "🎯 意图识别与任务规划",
    "router": "🎯 意图识别",
    "rag": "🔍 知识库检索",
    "retrieve": "🔍 知识库检索",
    "web": "🌐 联网搜索",
    "finance_data": "💰 行情数据查询",
    "business_compute": "📊 财务指标计算",
    "aggregate": "📦 数据汇总",
    "global_eval": "📏 全局评估",
    "conflict": "⚠️ 数据冲突检测",
    "generate": "✍️ 研报撰写",
    "eval": "📏 检索质量评估",
    "refine": "✨ 查询优化",
    "plan": "📋 任务规划",
    "system": "⚙️ 系统",
}

_PIPELINE_STAGES = [
    {"key": "supervisor", "label": "意图识别", "icon": "🎯"},
    {"key": "finance_data", "label": "行情查询", "icon": "💰"},
    {"key": "business_compute", "label": "指标计算", "icon": "📊"},
    {"key": "aggregate", "label": "数据汇总", "icon": "📦"},
    {"key": "generate", "label": "研报撰写", "icon": "✍️"},
]

_PERIOD_OPTIONS = [
    "2025 Q1", "2024 Q4", "2024 Q3", "2024 Q2", "2024 Q1",
    "2023年报", "2023 Q4", "2023 Q3", "2023 Q2", "2023 Q1",
    "2022年报",
]


def initialize_state() -> None:
    defaults = {
        "report_history": [],
        "generating": False,
        "agent": None,
        "last_error": None,
        "agent_init_error": None,
        "agent_diag": None,
        "current_report_index": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_agent() -> Optional[MultiAgentRAG]:
    if st.session_state.agent is not None:
        return st.session_state.agent

    diag_parts = []
    try:
        from src.core.settings import load_settings
        from langchain_openai import ChatOpenAI

        settings = load_settings()
        api_key = settings.llm.api_key or ""
        is_placeholder = api_key.startswith("__") and api_key.endswith("__")

        diag_parts.append(f"模型: {settings.llm.model}")
        diag_parts.append(f"接口: {settings.llm.base_url or '(默认)'}")

        if is_placeholder:
            msg = "API Key 为占位符，请检查 .env 文件或环境变量 OPENAI_API_KEY"
            diag_parts.append(msg)
            st.session_state.agent_diag = diag_parts
            st.session_state.agent_init_error = msg
            return None

        diag_parts.append(f"Key: {api_key[:8]}...")
        st.session_state.agent_diag = diag_parts

        llm = ChatOpenAI(
            model=settings.llm.model,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
            api_key=api_key,
            base_url=settings.llm.base_url if settings.llm.base_url else None,
        )
        agent = MultiAgentRAG(llm=llm, settings=settings, enable_logging=True)
        st.session_state.agent = agent
        st.session_state.agent_init_error = None
        st.session_state.agent_diag = diag_parts
        logger.info("Multi-Agent RAG initialized for research report")
        return agent

    except Exception as e:
        err_msg = str(e)
        diag_parts.append(f"异常: {err_msg[:120]}")

        if "api_key" in err_msg.lower() or "401" in err_msg or "403" in err_msg:
            hint = "API Key 无效或权限不足，请检查 OPENAI_API_KEY"
        elif "connection" in err_msg.lower() or "timeout" in err_msg.lower():
            hint = "无法连接到 LLM 服务，请检查网络和 base_url 配置"
        elif "model" in err_msg.lower():
            hint = f"模型 '{settings.llm.model}' 不可用，请检查 model 配置"
        else:
            hint = f"Agent 初始化失败: {err_msg[:80]}"

        st.session_state.agent_diag = diag_parts
        st.session_state.agent_init_error = hint
        logger.error(f"Failed to initialize agent: {e}")
        return None


def render_pipeline(stage_statuses: Dict[str, str], stage_times: Dict[str, float] = None) -> None:
    cols = st.columns(len(_PIPELINE_STAGES))
    for i, stage in enumerate(_PIPELINE_STAGES):
        key = stage["key"]
        status = stage_statuses.get(key, "pending")
        icon = stage["icon"]
        label = stage["label"]

        if status == "done":
            display = f"✅ {icon}"
            elapsed = ""
            if stage_times and key in stage_times:
                elapsed = f" {stage_times[key]:.1f}s"
            caption = f"~~{label}~~{elapsed}"
        elif status == "running":
            display = f"🔄 {icon}"
            caption = f"**{label}**"
        elif status == "error":
            display = f"❌ {icon}"
            caption = f"~~{label}~~"
        else:
            display = f"⏳ {icon}"
            caption = label

        with cols[i]:
            st.markdown(f"<div style='text-align:center;font-size:1.4em;'>{display}</div>",
                        unsafe_allow_html=True)
            st.caption(caption)


def generate_report_with_progress(params: Dict) -> Optional[Dict[str, Any]]:
    agent = get_agent()
    if not agent:
        err = st.session_state.agent_init_error or "Agent 未初始化"
        st.session_state.last_error = err
        return None

    company = params.get("company", "")
    period = params.get("period", "2024 Q3")
    depth = params.get("depth", "standard")
    enable_charts = params.get("enable_charts", True)
    enable_comparison = params.get("enable_comparison", True)
    enable_valuation = params.get("enable_valuation", False)
    analysis_date = params.get("analysis_date", datetime.now().strftime("%Y-%m-%d"))

    query_parts = [f"分析{company}{period}财报"]
    if enable_comparison:
        query_parts.append("对比行业平均")
    if enable_valuation:
        query_parts.append("出估值分析")
    if enable_charts:
        query_parts.append("出图表")
    if depth == "deep":
        query_parts.insert(0, "深度")
    query = "，".join(query_parts)

    total_start = time.time()
    progress_steps: List[Dict] = []
    stage_statuses: Dict[str, str] = {s["key"]: "pending" for s in _PIPELINE_STAGES}
    stage_statuses["supervisor"] = "running"
    stage_times: Dict[str, float] = {}
    stage_start: Dict[str, float] = {}
    step_times: Dict[str, float] = {}
    final_state = None
    streamed_text = ""

    text_container = st.empty()
    pipeline_placeholder = st.empty()
    detail_placeholder = st.empty()

    with st.status("🚀 正在初始化...", expanded=True) as status:
        with pipeline_placeholder.container():
            render_pipeline(stage_statuses, stage_times)

        gen = agent.run_stream_financial_direct(
            user_input=query,
            conversation_history=[],
            analysis_date=analysis_date,
            symbol=company,
        )
        step_start = 0.0

        try:
            while True:
                try:
                    event = next(gen)
                except StopIteration as e:
                    final_state = e.value
                    break

                if event is None:
                    continue

                et = event.type
                agent_name = event.agent_name or ""
                content = event.content or ""
                icon = _AGENT_ICONS.get(agent_name, "🤖")

                if et == EventType.AGENT_START:
                    step_label = _STAGE_LABELS.get(agent_name, content)
                    step_start = event.started_at or time.time()
                    progress_steps.append({
                        "name": agent_name, "label": step_label,
                        "icon": icon, "status": "running",
                    })
                    status.update(label=f"⏳ {step_label}...", state="running")

                    for stage in _PIPELINE_STAGES:
                        if stage["key"] == agent_name and stage_statuses.get(stage["key"]) != "done":
                            stage_statuses[stage["key"]] = "running"
                            stage_start[stage["key"]] = step_start
                            break

                    with pipeline_placeholder.container():
                        render_pipeline(stage_statuses, stage_times)

                    with detail_placeholder.container():
                        st.write(f"{icon} ⏳ **{step_label}** — {content}")

                elif et == EventType.AGENT_STEP:
                    step_label = _STAGE_LABELS.get(agent_name, content)
                    status.update(label=f"⏳ {step_label}...", state="running")
                    with detail_placeholder.container():
                        st.write(f"{icon} ⏳ **{step_label}** — {content}")
                        step_info = event.details or {}
                        if step_info.get("task_index") and step_info.get("total_tasks"):
                            st.write(f"   ↳ 子任务 {step_info['task_index']}/{step_info['total_tasks']}")

                elif et == EventType.AGENT_RESULT:
                    completed_at = event.completed_at or time.time()
                    elapsed = round(completed_at - step_start, 1) if step_start else 0.0
                    if progress_steps:
                        progress_steps[-1]["status"] = "done"
                        progress_steps[-1]["elapsed"] = elapsed
                    step_times[agent_name] = elapsed

                    for stage in _PIPELINE_STAGES:
                        if stage["key"] == agent_name and stage_statuses.get(stage["key"]) == "running":
                            stage_statuses[stage["key"]] = "done"
                            if stage["key"] in stage_start and stage_start[stage["key"]]:
                                stage_times[stage["key"]] = round(completed_at - stage_start[stage["key"]], 1)
                            break

                    done_count = sum(1 for s in progress_steps if s.get("status") == "done")
                    status.update(
                        label=f"✅ 已完成 {done_count}/{len(progress_steps)} 步骤",
                        state="running",
                    )

                    with pipeline_placeholder.container():
                        render_pipeline(stage_statuses, stage_times)

                    with detail_placeholder.container():
                        st.write(f"{icon} ✅ {content} ({elapsed:.1f}s)")

                elif et == EventType.PARTIAL_TEXT:
                    streamed_text += content
                    text_container.markdown(streamed_text[:2000] + "\n\n*...正在生成中...*")

                elif et == EventType.THINKING:
                    with detail_placeholder.container():
                        with st.expander("🧠 推理详情", expanded=False):
                            st.caption(content)
                            if event.details:
                                st.json(event.details)

                elif et == EventType.CONFLICT:
                    with detail_placeholder.container():
                        st.warning(f"⚠️ 数据冲突: {content}")

                elif et == EventType.ERROR:
                    for stage in _PIPELINE_STAGES:
                        if stage["key"] == agent_name:
                            stage_statuses[stage["key"]] = "error"
                            break
                    if progress_steps:
                        progress_steps[-1]["status"] = "error"
                        progress_steps[-1]["error"] = content
                    with pipeline_placeholder.container():
                        render_pipeline(stage_statuses, stage_times)
                    with detail_placeholder.container():
                        st.error(f"❌ {content}")

                elif et == EventType.DONE:
                    pass

        except Exception as e:
            err_msg = f"执行出错: {e}"
            logger.exception("run_stream failed")
            st.session_state.last_error = err_msg
            return None

        total_elapsed = round(time.time() - total_start, 1)

        if final_state is None:
            st.session_state.last_error = "Agent 返回空状态，请重试"
            return None

        text_container.empty()

        if progress_steps:
            total_steps = len(progress_steps)
            done_steps = sum(1 for s in progress_steps if s.get("status") == "done")
            error_steps = sum(1 for s in progress_steps if s.get("status") == "error")
            st.progress(done_steps / max(total_steps, 1),
                        text=f"总进度: {done_steps}/{total_steps}"
                             + (f" · {error_steps} 个异常" if error_steps else ""))

            if done_steps > 1:
                avg_time = total_elapsed / done_steps
                remaining_steps = total_steps - done_steps
                estimated_remaining = avg_time * remaining_steps
                if estimated_remaining > 1:
                    st.caption(f"⏱ 已完成步骤平均耗时 {avg_time:.0f}s")

            with st.expander("📋 执行步骤详情", expanded=bool(error_steps)):
                for step in progress_steps:
                    s_icon = step.get("icon", "🤖")
                    s_label = step.get("label", "")
                    s_status = step.get("status", "pending")
                    s_elapsed = step.get("elapsed", 0)
                    elapsed_str = f" ({s_elapsed:.1f}s)" if s_elapsed else ""

                    if s_status == "done":
                        st.write(f"✅ {s_icon} {s_label}{elapsed_str}")
                    elif s_status == "error":
                        st.write(f"❌ {s_icon} {s_label} — {step.get('error', '未知错误')}")
                    elif s_status == "running":
                        st.write(f"🔄 {s_icon} {s_label}{elapsed_str}")
                    else:
                        st.write(f"⏳ {s_icon} {s_label}")

        status.update(
            label=f"✅ 研报生成完成 · 耗时 {total_elapsed:.1f}s · 共 {len(progress_steps)} 步骤"
                  + (f" · {error_steps} 异常" if error_steps else ""),
            state="complete",
        )

    if isinstance(final_state, dict):
        execution_trace = final_state.get("execution_trace", [])
        answer = final_state.get("final_answer", "")
        bb = final_state.get("blackboard", {})
    else:
        execution_trace = final_state.execution_trace
        answer = final_state.final_answer
        bb = final_state.blackboard

    local_results = final_state.local_results if not isinstance(final_state, dict) else bb.get("local_results", [])
    web_results = final_state.web_results if not isinstance(final_state, dict) else bb.get("web_results", [])
    citations = bb.get("citations", [])
    evaluation = final_state.evaluation if not isinstance(final_state, dict) else bb.get("evaluation", {})
    chart_paths = bb.get("chart_paths", []) or []
    conflict_flags = bb.get("conflict_flags", [])
    verification_warnings = bb.get("verification_warnings", [])

    error_count = sum(1 for s in progress_steps if s.get("status") == "error")
    if error_count:
        st.session_state.last_error = (
            f"生成完成但有 {error_count} 个步骤异常，已标注在报告中"
        )

    return {
        "success": bool(answer),
        "answer": answer or "抱歉，研报生成失败。",
        "citations": citations or [],
        "execution_trace": execution_trace,
        "chart_paths": chart_paths,
        "conflict_flags": conflict_flags,
        "step_times": step_times,
        "progress_steps": progress_steps,
        "error_count": error_count,
        "metrics": {
            "retrieval_count": len(local_results) + len(web_results),
            "citation_count": len(citations) if citations else 0,
            "confidence": evaluation.get("confidence", 0) if evaluation else 0,
            "total_time": total_elapsed,
            "step_count": len(progress_steps),
            "verification_warnings": len(verification_warnings),
            "error_count": error_count,
        },
    }


def render_thinking_process(execution_trace: List[Dict], step_times: Dict[str, float] = None) -> None:
    with st.expander("🧠 展示推理过程", expanded=False):
        for step in execution_trace:
            agent = step.get("agent", "")
            action = step.get("action", "")
            icon = _AGENT_ICONS.get(agent, "🤖")
            status = step.get("status", "done")
            description = step.get("description", action)

            elapsed = ""
            if step_times and agent in step_times:
                elapsed = f" — {step_times[agent]:.1f}s"

            if status == "running":
                st.markdown(f"{icon} **{agent}** → {description} ⏳")
            elif status == "error":
                st.markdown(f"{icon} ❌ **{agent}** → {description}")
            else:
                st.markdown(f"{icon} ✅ **{agent}** → {description}{elapsed}")

            detail = {k: v for k, v in step.items()
                      if k not in ("agent", "action", "status", "description", "timestamp")}
            if detail and any(v for v in detail.values()):
                with st.expander("详情", expanded=False):
                    st.json(detail)


def render_report_sections(report_text: str) -> None:
    sections = report_text.split("\n## ")
    for i, section in enumerate(sections):
        if i == 0:
            st.markdown(section)
        else:
            st.markdown(f"## {section}")


def render_charts(chart_paths: List[Dict]) -> None:
    if not chart_paths:
        return
    with st.expander("📊 图表展示", expanded=True):
        cols = st.columns(min(len(chart_paths), 3))
        for i, c in enumerate(chart_paths):
            path = c.get("path", "") if isinstance(c, dict) else str(c)
            title = c.get("title", f"图表{i+1}") if isinstance(c, dict) else f"图表{i+1}"
            if path and Path(path).exists():
                with cols[i % 3]:
                    st.image(path, caption=title, use_container_width=True)
            else:
                with cols[i % 3]:
                    st.caption(f"{title} (文件未找到)")


def render_citations(citations: List) -> None:
    if not citations:
        return
    from src.agent.multi_agent import Citation
    with st.expander("📚 查看引用来源", expanded=False):
        for idx, citation in enumerate(citations, 1):
            c = Citation.from_dict(citation) if isinstance(citation, dict) else citation
            st.markdown(f"**{idx}.** {c.format_citation()}")
            with st.expander(f"查看内容 #{idx}", expanded=False):
                st.markdown(c.content[:500] + "..." if len(c.content) > 500 else c.content)


def render_conflict_warnings(conflict_flags: List) -> None:
    if not conflict_flags:
        return
    with st.expander("⚠️ 数据异常提示", expanded=True):
        for f in conflict_flags:
            flag = f if isinstance(f, dict) else f.__dict__
            severity = flag.get("severity", "low")
            emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(severity, "⚪")
            st.markdown(f"{emoji} **[{severity.upper()}]** {flag.get('claim_a', '')} vs {flag.get('claim_b', '')}")
            st.caption(flag.get("explanation", ""))


def render_metrics(metrics: Dict) -> None:
    if not metrics:
        return
    with st.expander("📊 执行指标", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("检索文档数", metrics.get("retrieval_count", 0))
        with c2:
            st.metric("引用数量", metrics.get("citation_count", 0))
        with c3:
            st.metric("置信度", f"{metrics.get('confidence', 0):.2f}")
        with c4:
            st.metric("耗时", f"{metrics.get('total_time', 0):.1f}s")
        verify_warnings = metrics.get("verification_warnings", 0)
        if verify_warnings:
            st.warning(f"数据校验修正 {verify_warnings} 处")
        step_count = metrics.get("step_count", 0)
        error_count = metrics.get("error_count", 0)
        if step_count:
            parts = [f"共 {step_count} 个执行步骤"]
            if error_count:
                parts.append(f"{error_count} 个异常")
            st.caption(" · ".join(parts))


def export_pdf(report_text: str, filename: str) -> Optional[str]:
    try:
        from src.agent.multi_agent.tools.pdf_exporter import PDFExporter
        output_dir = Path("data/reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"{filename}.pdf")
        exporter = PDFExporter(css_path="templates/finance/pdf_professional.css")
        return exporter.export(report_text, output_path)
    except Exception as e:
        logger.warning(f"PDF export failed: {e}")
        return None


def render() -> None:
    st.header("📈 智能研报")
    initialize_state()

    # 显示持久化错误
    if st.session_state.last_error:
        st.error(f"⚠️ {st.session_state.last_error}")
        if st.button("清除错误", key="clear_error"):
            st.session_state.last_error = None
            st.rerun()

    with st.sidebar:
        st.subheader("📋 研报配置")

        company = st.text_input("公司名称", value="宁德时代",
                                help="输入上市公司名称，如 宁德时代、比亚迪、腾讯")
        period = st.selectbox("报告周期", _PERIOD_OPTIONS, index=2)
        analysis_date = st.date_input("分析基准日期", value=datetime.now().date(),
                                      help="所有数据查询以该日期为截止日").strftime("%Y-%m-%d")

        st.divider()
        st.subheader("⚙️ 分析选项")

        depth = st.select_slider("分析深度",
                                 options=["quick", "standard", "deep"],
                                 value="standard",
                                 format_func=lambda x: {"quick": "⚡ 快速", "standard": "📋 标准", "deep": "🧠 深度"}[x],
                                 help="快速=仅行情+指标，标准=文档+行情+图表，深度=全链路+DCF+VLM评审")

        enable_charts = st.checkbox("📊 包含图表", value=True)
        enable_comparison = st.checkbox("📋 行业对比", value=True)
        enable_valuation = st.checkbox("💰 DCF 估值分析", value=False)

        st.divider()
        st.subheader("📤 导出")

        if st.button("📥 导出 PDF", use_container_width=True, disabled=not st.session_state.report_history):
            last = st.session_state.report_history[-1]
            pdf_path = export_pdf(last["answer"], f"{company}_{period}")
            if pdf_path:
                with open(pdf_path, "rb") as f:
                    st.download_button("下载 PDF", f, file_name=f"{company}_{period}.pdf", mime="application/pdf")
            else:
                st.error("PDF 导出失败（请安装 weasyprint: pip install weasyprint）")

        st.divider()

        agent = get_agent()
        if agent:
            st.success("✅ Agent 就绪")
            if st.session_state.agent_diag:
                with st.expander("📋 诊断信息", expanded=False):
                    for line in st.session_state.agent_diag:
                        st.caption(line)
        else:
            err_msg = st.session_state.agent_init_error or "Agent 未初始化"
            st.error(f"❌ {err_msg}")
            if st.session_state.agent_diag:
                with st.expander("📋 诊断信息", expanded=True):
                    for line in st.session_state.agent_diag:
                        st.caption(line)
            if st.button("🔄 重试初始化", use_container_width=True, key="retry_agent"):
                st.session_state.agent = None
                st.session_state.agent_init_error = None
                st.session_state.agent_diag = None
                st.rerun()

        # 历史记录
        if st.session_state.report_history:
            st.divider()
            st.subheader("📚 历史记录")
            history = st.session_state.report_history
            for i, entry in enumerate(reversed(history)):
                idx = len(history) - 1 - i
                label = f"{entry.get('company', '?')} · {entry.get('period', '?')} · {entry.get('generated_at', '?')}"
                if st.button(label, key=f"hist_{idx}", use_container_width=True,
                             type="secondary" if idx != len(history) - 1 else "primary"):
                    st.session_state.current_report_index = idx
                    st.rerun()

    st.caption("一键生成专业级投资研究报告，数据来源可追溯")

    # 生成按钮
    disabled = st.session_state.generating or agent is None
    if st.button("🚀 生成研报", type="primary", use_container_width=True, disabled=disabled, key="generate_report"):
        st.session_state.last_error = None
        st.session_state.generating = True
        params = {
            "company": company,
            "period": period,
            "depth": depth,
            "enable_charts": enable_charts,
            "enable_comparison": enable_comparison,
            "enable_valuation": enable_valuation,
            "analysis_date": analysis_date,
        }

        result = generate_report_with_progress(params)

        if result and result["success"]:
            entry = {
                "company": company,
                "period": period,
                "depth": depth,
                "answer": result["answer"],
                "citations": result["citations"],
                "execution_trace": result["execution_trace"],
                "chart_paths": result["chart_paths"],
                "conflict_flags": result["conflict_flags"],
                "step_times": result.get("step_times", {}),
                "error_count": result.get("error_count", 0),
                "metrics": result["metrics"],
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
            st.session_state.report_history.append(entry)
            st.session_state.current_report_index = len(st.session_state.report_history) - 1
        elif result:
            st.session_state.last_error = f"研报生成失败: {result.get('error', '未知错误')}"

        st.session_state.generating = False
        st.rerun()

    # 显示最新研报
    history = st.session_state.report_history
    if history:
        idx = st.session_state.current_report_index
        idx = max(0, min(idx, len(history) - 1))
        report = history[idx]
        company_name = report.get("company", "")
        period_name = report.get("period", "")
        gen_time = report.get("generated_at", "")
        is_latest = (idx == len(history) - 1)

        st.markdown(f"### 📄 {company_name} {period_name} 深度研报"
                    + (" " if is_latest else f" (历史 #{idx + 1})"))
        st.caption(f"生成时间: {gen_time} · 分析深度: {report.get('depth', 'standard')}")

        if not is_latest and history:
            if st.button("📌 查看最新", key="goto_latest"):
                st.session_state.current_report_index = len(history) - 1
                st.rerun()

        st.divider()

        metrics = report.get("metrics", {})
        step_times = report.get("step_times", {})
        if metrics.get("total_time"):
            error_count = report.get("error_count", 0)
            msg = f"✅ 研报生成完成，耗时 {metrics['total_time']}s · 共 {metrics.get('step_count', 0)} 步骤"
            if error_count:
                st.warning(f"⚠️ {msg} · {error_count} 个步骤异常")
            else:
                st.success(msg)

        thinking_process = report.get("execution_trace", [])
        if thinking_process:
            render_thinking_process(thinking_process, step_times)

        diagnostics = [t for t in thinking_process if t.get("action") == "data_diagnostic"]
        if diagnostics:
            with st.expander("🔍 数据诊断", expanded=True):
                for d in diagnostics:
                    st.info(d.get("diagnostic", ""))

        conflict_flags = report.get("conflict_flags", [])
        if conflict_flags:
            render_conflict_warnings(conflict_flags)

        st.divider()
        report_text = report.get("answer", "")
        if report_text:
            render_report_sections(report_text)

        chart_paths = report.get("chart_paths", [])
        if chart_paths:
            st.divider()
            render_charts(chart_paths)

        citations = report.get("citations", [])
        if citations:
            st.divider()
            render_citations(citations)

        if metrics:
            st.divider()
            render_metrics(metrics)

        st.divider()
        st.subheader("📤 操作")
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            pdf_path = export_pdf(report_text, f"{company_name}_{period_name}")
            if pdf_path:
                with open(pdf_path, "rb") as f:
                    st.download_button("📥 导出 PDF", f,
                                       file_name=f"{company_name}_{period_name}.pdf",
                                       mime="application/pdf",
                                       use_container_width=True)
            else:
                st.button("📥 导出 PDF", disabled=True, help="需安装 weasyprint",
                          use_container_width=True)
        with col2:
            if st.button("📋 复制报告", use_container_width=True, key="copy_report"):
                st.code(report_text, language="markdown")
                st.success("👆 请手动选中上方文本复制（Streamlit 限制）")

    else:
        st.info("👈 在左侧配置研报参数，然后点击「生成研报」按钮")
        with st.expander("💡 示例查询", expanded=False):
            st.markdown("""
            | 查询示例 | 说明 |
            |---------|------|
            | 快速看看宁德时代 | ⚡ 快速模式，仅行情+指标 |
            | 分析宁德时代2024Q3财报 | 📋 标准模式，文档+行情+图表 |
            | 深度分析宁德时代，对比行业出估值 | 🧠 深度模式，全链路+DCF |
            """)
