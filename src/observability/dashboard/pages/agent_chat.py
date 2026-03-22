"""Multi-Agent Chat page – Interactive chat with the RAG Agent.

Layout:
1. Sidebar: config, file upload with auto-ingest, system status
2. Chat interface with message history
3. Real-time agent responses with citations
4. Execution trace visualization
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional

import streamlit as st

from src.agent.multi_agent import (
    AgentState,
    MultiAgentRAG,
    Citation,
    CitationManager,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Supported file types (matches ingestion pipeline)
_SUPPORTED_TYPES = ["pdf", "txt", "md", "docx"]
_FILE_ICONS = {
    ".pdf": "\U0001F4D1",   # 📑
    ".txt": "\U0001F4C4",   # 📄
    ".md": "\U0001F4DD",    # 📝
    ".docx": "\U0001F4C3",  # 📃
}


# ═══════════════════════════════════════════════════════════════
# Session State
# ═══════════════════════════════════════════════════════════════

def initialize_session_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "agent_state" not in st.session_state:
        st.session_state.agent_state = None
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "execution_traces" not in st.session_state:
        st.session_state.execution_traces = []
    # Track uploaded & ingested files: {filename: {status, collection, chunk_count, ...}}
    if "uploaded_files_info" not in st.session_state:
        st.session_state.uploaded_files_info = {}
    # Session-level collection name (so uploads go to one place)
    if "session_collection" not in st.session_state:
        st.session_state.session_collection = "default"


# ═══════════════════════════════════════════════════════════════
# Agent
# ═══════════════════════════════════════════════════════════════

def get_agent() -> Optional[MultiAgentRAG]:
    if st.session_state.agent is None:
        try:
            from src.core.settings import load_settings
            from langchain_openai import ChatOpenAI

            settings = load_settings()
            llm = ChatOpenAI(
                model=settings.llm.model,
                temperature=settings.llm.temperature,
                max_tokens=settings.llm.max_tokens,
                api_key=settings.llm.api_key,
                base_url=settings.llm.base_url if settings.llm.base_url else None,
            )
            st.session_state.agent = MultiAgentRAG(
                llm=llm, settings=settings, enable_logging=True
            )
            logger.info("Multi-Agent RAG initialized")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            st.error(f"初始化 Agent 失败：{e}")
            import traceback
            st.code(traceback.format_exc(), language="text")
            return None
    return st.session_state.agent


def _build_conversation_history() -> List[Dict[str, str]]:
    max_turns = 10
    history: List[Dict[str, str]] = []
    for msg in st.session_state.get("chat_history", []):
        role = msg.get("role")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            history.append({"role": role, "content": content[:500]})
    return history[-max_turns * 2 :]


# ═══════════════════════════════════════════════════════════════
# File Ingestion (inline, for chat uploads)
# ═══════════════════════════════════════════════════════════════

def _ingest_uploaded_file(
    uploaded_file, collection: str, status_placeholder
) -> bool:
    """Ingest a single uploaded file into the vector store."""
    from src.core.settings import load_settings
    from src.core.trace import TraceContext
    from src.ingestion.pipeline import IngestionPipeline

    settings = load_settings()
    suffix = Path(uploaded_file.name).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    _STAGE_LABELS = {
        "integrity": "🔍 校验文件…",
        "load": "📄 解析文档…",
        "split": "✂️ 切分段落…",
        "transform": "🔄 向量化处理…",
        "embed": "🔢 生成嵌入…",
        "upsert": "💾 存入向量库…",
    }

    progress_bar = status_placeholder.progress(0, text="准备中…")

    def on_progress(stage: str, current: int, total: int) -> None:
        frac = max(0.0, min((current - 1) / total, 1.0))
        label = _STAGE_LABELS.get(stage, stage)
        progress_bar.progress(frac, text=f"[{current}/{total}] {label}")

    trace = TraceContext(trace_type="ingestion")
    trace.metadata["source_path"] = uploaded_file.name
    trace.metadata["collection"] = collection
    trace.metadata["source"] = "chat_upload"

    try:
        pipeline = IngestionPipeline(settings, collection=collection, force=False)
        result = pipeline.run(file_path=tmp_path, trace=trace, on_progress=on_progress)

        if result.success and result.chunk_count > 0:
            progress_bar.progress(1.0, text="✅ 导入完成!")
            st.session_state.uploaded_files_info[uploaded_file.name] = {
                "status": "success",
                "chunks": result.chunk_count,
                "collection": collection,
                "doc_id": result.doc_id,
                "time": datetime.now().strftime("%H:%M:%S"),
            }
            return True
        elif result.success and result.chunk_count == 0:
            progress_bar.progress(1.0, text="⏭️ 文件已存在，跳过")
            st.session_state.uploaded_files_info[uploaded_file.name] = {
                "status": "skipped",
                "chunks": 0,
                "collection": collection,
                "time": datetime.now().strftime("%H:%M:%S"),
            }
            return True
        else:
            progress_bar.progress(0.0, text=f"❌ 失败: {result.error}")
            st.session_state.uploaded_files_info[uploaded_file.name] = {
                "status": "failed",
                "error": str(result.error),
                "time": datetime.now().strftime("%H:%M:%S"),
            }
            return False
    except Exception as e:
        progress_bar.progress(0.0, text=f"❌ 异常: {e}")
        st.session_state.uploaded_files_info[uploaded_file.name] = {
            "status": "failed",
            "error": str(e),
            "time": datetime.now().strftime("%H:%M:%S"),
        }
        return False


# ═══════════════════════════════════════════════════════════════
# Query Processing
# ═══════════════════════════════════════════════════════════════

def process_query(query: str, config: Dict[str, Any]) -> Dict[str, Any]:
    agent = get_agent()
    if not agent:
        return {"success": False, "error": "Agent not initialized"}

    try:
        start_time = time.time()
        conversation_history = _build_conversation_history()
        final_state = agent.run(
            user_input=query, conversation_history=conversation_history
        )

        if isinstance(final_state, dict):
            answer = final_state.get("final_answer", "")
            bb = final_state.get("blackboard", {})
            local_results = bb.get("local_results", [])
            web_results = bb.get("web_results", [])
            citations = bb.get("citations", [])
            evaluation = bb.get("evaluation", {})
            execution_trace = final_state.get("execution_trace", [])
        else:
            answer = final_state.final_answer
            local_results = final_state.local_results
            web_results = final_state.web_results
            citations = final_state.blackboard.get("citations", [])
            evaluation = final_state.evaluation
            execution_trace = final_state.execution_trace

        elapsed = time.time() - start_time
        metrics = {
            "retrieval_count": len(local_results) + len(web_results),
            "citation_count": len(citations) if citations else 0,
            "confidence": evaluation.get("confidence", 0) if evaluation else 0,
            "total_time": elapsed,
        }

        return {
            "success": bool(answer),
            "answer": answer or "抱歉，我暂时无法回答这个问题。",
            "citations": citations or [],
            "execution_trace": execution_trace,
            "metrics": metrics,
            "state": final_state,
        }
    except Exception as e:
        logger.exception(f"Error processing query: {e}")
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════
# Display Helpers
# ═══════════════════════════════════════════════════════════════

def display_chat_message(message: Dict[str, Any]) -> None:
    role = message["role"]
    content = message["content"]

    if role == "user":
        with st.chat_message("user"):
            # Show attached file names if present
            if message.get("attached_files"):
                file_tags = " ".join(
                    f"`📎 {f}`" for f in message["attached_files"]
                )
                st.markdown(f"{file_tags}")
            st.markdown(content)
    elif role == "assistant":
        with st.chat_message("assistant"):
            st.markdown(content)

            if "citations" in message and message["citations"]:
                with st.expander("📚 查看引用来源", expanded=False):
                    for idx, citation in enumerate(message["citations"], 1):
                        c = (
                            Citation.from_dict(citation)
                            if isinstance(citation, dict)
                            else citation
                        )
                        st.markdown(f"**{idx}.** {c.format_citation()}")
                        with st.expander(f"查看内容 #{idx}", expanded=False):
                            st.markdown(
                                c.content[:300] + "..."
                                if len(c.content) > 300
                                else c.content
                            )

            if "metrics" in message and message["metrics"]:
                with st.expander("📊 执行指标", expanded=False):
                    metrics = message["metrics"]
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("检索文档数", metrics.get("retrieval_count", 0))
                    with col2:
                        st.metric("引用数量", metrics.get("citation_count", 0))
                    with col3:
                        st.metric("置信度", f"{metrics.get('confidence', 0):.2f}")
                    with col4:
                        st.metric("耗时", f"{metrics.get('total_time', 0):.1f}s")


def display_execution_trace(trace: List[Dict[str, Any]]) -> None:
    st.subheader("🔍 执行轨迹")
    agent_icons = {
        "router": "🎯", "RouterAgent": "🎯",
        "search": "🔍", "SearchAgent": "🔍",
        "web": "🌐", "WebAgent": "🌐",
        "eval": "📏", "EvalAgent": "📏",
        "refine": "✨", "RefineAgent": "✨",
        "generate": "✍️", "GenerateAgent": "✍️",
        "parallel_controller": "⚡",
    }
    for idx, step in enumerate(trace, 1):
        agent = step.get("agent", "Unknown")
        action = step.get("action", "unknown")
        icon = agent_icons.get(agent, "🤖")
        detail = {k: v for k, v in step.items() if k not in ("agent", "action")}
        with st.expander(f"{icon} **{agent}** → {action}", expanded=False):
            if detail:
                st.json(detail)
            else:
                st.caption("无详细数据")


# ═══════════════════════════════════════════════════════════════
# Main Render
# ═══════════════════════════════════════════════════════════════

def _inject_css() -> None:
    st.markdown("""
    <style>
    .block-container { padding-bottom: 70px !important; }
    .file-chips {
        display: flex; flex-wrap: wrap; gap: 6px; padding: 4px 0;
    }
    .file-chip {
        background: rgba(128,128,128,0.12);
        border-radius: 12px; padding: 2px 10px;
        font-size: 0.8rem; white-space: nowrap;
    }
    </style>
    """, unsafe_allow_html=True)


def render() -> None:
    st.header("🤖 多智能体 RAG 对话")

    initialize_session_state()
    _inject_css()

    # ── Sidebar ────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("⚙️ 配置")
        routing_mode = st.selectbox(
            "路由模式",
            options=["auto", "local_search", "web_search", "hybrid_search"],
            index=0,
            help="auto = Router Agent 自动判断",
        )
        top_k = st.slider("检索数量", 1, 20, 5)
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.execution_traces = []
            st.session_state.uploaded_files_info = {}
            st.rerun()

        st.divider()
        st.subheader("📊 状态")
        agent = get_agent()
        st.success("✅ Agent 就绪") if agent else st.error("❌ Agent 未初始化")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("对话轮数", len(st.session_state.chat_history) // 2)
        with c2:
            n_files = sum(
                1 for v in st.session_state.uploaded_files_info.values()
                if v.get("status") in ("success", "skipped")
            )
            st.metric("已导入文件", n_files)

    # ── Chat history ───────────────────────────────────────────
    for message in st.session_state.chat_history:
        display_chat_message(message)

    # ── Uploaded file chips ─────────────────────────────────────
    chips = []
    for fname, info in st.session_state.uploaded_files_info.items():
        s = info.get("status", "")
        suffix = Path(fname).suffix.lower()
        icon = _FILE_ICONS.get(suffix, "📄")
        if s == "success":
            chips.append(f"{icon} {fname} ({info.get('chunks', 0)}段)")
        elif s == "skipped":
            chips.append(f"{icon} {fname} (已存在)")
        elif s == "failed":
            chips.append(f"❌ {fname}")
    if chips:
        st.caption(" · ".join(chips))

    # ── Chat input (native upload icon at the left) ─────────────
    chat_value = st.chat_input(
        "输入问题…",
        accept_file="multiple",
        file_type=_SUPPORTED_TYPES,
    )

    if chat_value:
        # chat_input returns either str or ChatInputValue
        if isinstance(chat_value, str):
            prompt = chat_value
            input_files = []
        else:
            prompt = getattr(chat_value, "text", "") or getattr(chat_value, "message", "")
            input_files = list(getattr(chat_value, "files", []) or [])

        # Auto-ingest files attached in this turn
        if input_files:
            with st.spinner("正在导入附件..."):
                new_files = [
                    f for f in input_files
                    if f.name not in st.session_state.uploaded_files_info
                ]
                for uf in new_files:
                    status_area = st.container()
                    _ingest_uploaded_file(
                        uf,
                        st.session_state.session_collection,
                        status_area,
                    )

        if not (prompt and prompt.strip()):
            st.rerun()

        active_files = [
            fname
            for fname, info in st.session_state.uploaded_files_info.items()
            if info.get("status") in ("success", "skipped")
        ]

        st.chat_message("user").markdown(prompt)
        user_msg: Dict[str, Any] = {"role": "user", "content": prompt}
        if active_files:
            user_msg["attached_files"] = active_files
        st.session_state.chat_history.append(user_msg)

        config = {"routing_mode": routing_mode, "top_k": top_k}

        with st.chat_message("assistant"):
            with st.spinner("正在思考..."):
                response = process_query(prompt, config)

        if response["success"]:
            assistant_message = {
                "role": "assistant",
                "content": response["answer"],
                "citations": response["citations"],
                "metrics": response["metrics"],
            }
            st.session_state.chat_history.append(assistant_message)
            st.session_state.execution_traces.extend(response["execution_trace"])
            display_chat_message(assistant_message)
            if response["execution_trace"]:
                display_execution_trace(response["execution_trace"])
            st.rerun()
        else:
            error_message = {
                "role": "assistant",
                "content": f"❌ 处理失败：{response.get('error', '未知错误')}",
            }
            st.session_state.chat_history.append(error_message)
            display_chat_message(error_message)
            st.rerun()


def main() -> None:
    render()


if __name__ == "__main__":
    main()
