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
from src.agent.compaction_memory import CompactionMemory, LangGraphMemoryStore

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
    if "uploaded_files_info" not in st.session_state:
        st.session_state.uploaded_files_info = {}
    if "session_collection" not in st.session_state:
        st.session_state.session_collection = "default"
    if "compaction_memory" not in st.session_state:
        st.session_state.compaction_memory = None  # lazily created after agent init
    if "long_term_memory" not in st.session_state:
        st.session_state.long_term_memory = LangGraphMemoryStore(user_id="default")


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
            ltm: LangGraphMemoryStore = st.session_state.long_term_memory
            st.session_state.agent = MultiAgentRAG(
                llm=llm, settings=settings, enable_logging=True,
                store=ltm.store,
            )
            logger.info("Multi-Agent RAG initialized")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            st.error(f"初始化 Agent 失败：{e}")
            import traceback
            st.code(traceback.format_exc(), language="text")
            return None
    return st.session_state.agent


def _get_compaction_memory() -> CompactionMemory:
    """Lazily init CompactionMemory with the agent's LLM for compaction."""
    if st.session_state.compaction_memory is None:
        agent = get_agent()
        llm = agent.llm if agent else None
        st.session_state.compaction_memory = CompactionMemory(
            llm=llm, token_budget=4000,
        )
    return st.session_state.compaction_memory


def _build_conversation_history() -> List[Dict[str, str]]:
    """Build history from CompactionMemory (compaction summary + recent raw).

    Also injects relevant long-term memories recalled via LangGraph Store.
    """
    mem = _get_compaction_memory()
    history = mem.build_history_dicts()

    ltm: LangGraphMemoryStore = st.session_state.long_term_memory
    last_user = ""
    for m in reversed(history):
        if m["role"] == "user":
            last_user = m["content"]
            break

    if last_user:
        recalled = ltm.recall(last_user, top_k=3)
        if recalled:
            recall_text = "\n".join(
                f"- [{r['date']}] {r['content'][:300]}" for r in recalled
            )
            history.insert(0, {
                "role": "user",
                "content": f"[长期记忆回忆]\n{recall_text}",
            })
            history.insert(1, {
                "role": "assistant",
                "content": "好的，我已参考了过去的对话记录。",
            })

    return history


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

def _extract_images_from_local_results(
    local_results: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """从检索结果中提取图片数据，编码为 base64。

    Args:
        local_results: 格式化后的检索结果列表（含 images 字段）

    Returns:
        List[Dict] 每个元素含 data/base64、mime_type、caption、image_id
    """
    import json
    from pathlib import Path
    from src.core.response.multimodal_assembler import MultimodalAssembler

    assembler = MultimodalAssembler()
    image_data: List[Dict[str, str]] = []
    seen_paths: set = set()

    for item in local_results:
        raw_images = item.get("images")
        if not raw_images:
            continue
        try:
            images_list = (
                json.loads(raw_images) if isinstance(raw_images, str) else raw_images
            )
        except (json.JSONDecodeError, TypeError):
            continue
        for img_info in images_list:
            file_path = img_info.get("path")
            if not file_path or file_path in seen_paths:
                continue
            seen_paths.add(file_path)
            p = Path(file_path)
            resolved = str(p.resolve()) if p.exists() else None
            if resolved:
                img = assembler.load_image(resolved)
                if img:
                    image_data.append({
                        "data": img.data,
                        "mime_type": img.mime_type,
                        "caption": img.caption or "",
                        "image_id": img.image_id,
                    })
    return image_data


def process_query(query: str, config: Dict[str, Any]) -> Dict[str, Any]:
    agent = get_agent()
    if not agent:
        return {"success": False, "error": "Agent not initialized"}

    try:
        start_time = time.time()
        conversation_history = _build_conversation_history()
        answer_mode = config.get("answer_mode", "agentic")

        if answer_mode == "baseline_rag":
            # Baseline RAG: single local retrieve + single-shot generation (no plan/eval/refine/web loop)
            local_results = agent.search_agent.search(
                query=query,
                top_k=int(config.get("top_k", 5)),
                context=conversation_history,
            )
            baseline_context = {
                "user_input": query,
                "conversation_history": conversation_history,
                "local_results": local_results,
                "web_results": [],
                "evaluation": {},
                "blackboard": {"plan_observations": []},
            }
            answer, citations, _faithfulness = agent._generate_normal_response_with_citations(  # noqa: SLF001
                baseline_context
            )
            final_state = {
                "final_answer": answer,
                "blackboard": {
                    "local_results": local_results,
                    "web_results": [],
                    "citations": [c.to_dict() for c in citations],
                    "evaluation": {"confidence": 0.0, "mode": "baseline_rag"},
                },
                "execution_trace": [
                    {"agent": "baseline_rag", "action": "local_retrieve", "result_count": len(local_results)},
                    {"agent": "baseline_rag", "action": "generate_once", "citation_count": len(citations)},
                ],
            }
        elif answer_mode == "retrieval_only":
            # Agentic retrieval-only: router/plan/retrieve/read/eval, but NO LLM generation.
            state = AgentState(user_input=query, conversation_history=conversation_history)

            # Use the same internal nodes as agentic flow, but stop after eval.
            state = agent._router_node(state)  # noqa: SLF001
            if state.blackboard.get("query_complexity") == "complex":
                state = agent._plan_node(state)  # noqa: SLF001
            if not state.blackboard.get("plan_complete", False):
                state = agent._retrieve_node(state)  # noqa: SLF001
                state = agent._read_node(state)  # noqa: SLF001
                state = agent._eval_node(state)  # noqa: SLF001

            local_results = state.blackboard.get("local_results", [])
            web_results = state.blackboard.get("web_results", [])

            citations = CitationManager.create_citations_from_results(
                local_results=local_results,
                web_results=web_results,
                top_k=5,
            )

            parts: list[str] = []
            parts.append("## Retrieval only（不生成回答）")
            eval_data = state.blackboard.get("evaluation", {})
            if eval_data:
                parts.append(
                    f"- **Eval confidence**: {eval_data.get('confidence', 0):.2f}\n"
                    f"- **Relevance**: {eval_data.get('relevance', 0):.2f}\n"
                    f"- **Need refinement**: {eval_data.get('need_refinement', False)}\n"
                    f"- **Reason**: {eval_data.get('reason', '')}"
                )

            if local_results:
                parts.append("\n## 本地知识库片段（Top）")
                for i, r in enumerate(local_results[: min(len(local_results), int(config.get('top_k', 5)))], 1):
                    src = r.get("source", "unknown")
                    score = r.get("score", 0)
                    content = (r.get("content", "") or "").strip()
                    parts.append(f"**Local {i}** · `{src}` · score={score:.3f}\n\n{content[:800]}")

            if web_results:
                parts.append("\n## 联网片段（Top）")
                for i, r in enumerate(web_results[:3], 1):
                    title = r.get("title", "")
                    url = r.get("url", "")
                    snippet = (r.get("snippet", r.get("content", "")) or "").strip()
                    parts.append(f"**Web {i}** · `{title}` · `{url}`\n\n{snippet[:800]}")

            if not local_results and not web_results:
                parts.append("\n未检索到任何结果。")

            answer = "\n\n".join(parts)

            final_state = {
                "final_answer": answer,
                "blackboard": {
                    "local_results": local_results,
                    "web_results": web_results,
                    "citations": [c.to_dict() for c in citations],
                    "evaluation": state.blackboard.get("evaluation", {}),
                },
                "execution_trace": state.execution_trace,
            }
        else:
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

        image_data = _extract_images_from_local_results(local_results)

        return {
            "success": bool(answer),
            "answer": answer or "抱歉，我暂时无法回答这个问题。",
            "citations": citations or [],
            "execution_trace": execution_trace,
            "thinking_process": execution_trace,
            "metrics": metrics,
            "image_data": image_data,
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
            if message.get("attached_files"):
                file_tags = " ".join(
                    f"`📎 {f}`" for f in message["attached_files"]
                )
                st.markdown(f"{file_tags}")
            st.markdown(content)
    elif role == "assistant":
        with st.chat_message("assistant"):
            st.markdown(content)

            # 🆕 思考过程 toggle (默认收起)
            thinking_process = message.get("thinking_process", [])
            if thinking_process:
                with st.expander("🧠 展示推理过程", expanded=False):
                    agent_icons = {
                        "router": "🎯", "RouterAgent": "🎯", "supervisor": "🎯",
                        "search": "🔍", "SearchAgent": "🔍", "RAG": "🔍",
                        "web": "🌐", "WebAgent": "🌐",
                        "eval": "📏", "EvalAgent": "📏",
                        "refine": "✨", "RefineAgent": "✨",
                        "generate": "✍️", "GenerateAgent": "✍️",
                        "finance_data": "💰", "FinanceData": "💰", "FinanceDataAgent": "💰",
                        "business_compute": "📊", "BusinessCompute": "📊",
                        "conflict": "⚠️", "ConflictDetector": "⚠️",
                        "ChartCritic": "🎨", "chart_critic": "🎨",
                        "System": "⚙️",
                    }
                    for step in thinking_process:
                        agent = step.get("agent", "")
                        action = step.get("action", "")
                        icon = agent_icons.get(agent, "🤖")
                        status = step.get("status", "done")
                        description = step.get("description", action)

                        if status == "running":
                            st.markdown(f"{icon} **{agent}** → {description} ⏳")
                        elif status == "done":
                            st.markdown(f"{icon} ✅ **{agent}** → {description}")
                        elif status == "error":
                            st.markdown(f"{icon} ❌ **{agent}** → {description}")

                        detail = {k: v for k, v in step.items()
                                  if k not in ("agent", "action", "status", "description", "timestamp")}
                        if detail:
                            with st.expander("详情", expanded=False):
                                st.json(detail)

            if "image_data" in message and message["image_data"]:
                with st.expander("🖼️ 图片内容", expanded=True):
                    cols = st.columns(min(len(message["image_data"]), 3))
                    for i, img in enumerate(message["image_data"]):
                        import base64
                        col = cols[i % 3]
                        b64 = img.get("data", "")
                        mime = img.get("mime_type", "image/png")
                        caption = img.get("caption", "")
                        if b64:
                            col.markdown(
                                f'<img src="data:{mime};base64,{b64}" '
                                f'style="max-width:100%;border-radius:8px;">',
                                unsafe_allow_html=True,
                            )
                            if caption:
                                col.caption(caption[:120])

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
        answer_mode = st.selectbox(
            "Answer mode",
            options=[
                "agentic",
                "baseline_rag",
                "retrieval_only",
            ],
            index=0,
            help=(
                "agentic = 多智能体闭环（plan/read/eval/refine/web/generate）\n"
                "baseline_rag = 普通 RAG（仅本地检索 + 单次生成，不做多步循环）\n"
                "retrieval_only = 仅检索（走 agentic 的检索/反思，但不让 LLM 生成最终答案）"
            ),
        )
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
            if st.session_state.compaction_memory:
                st.session_state.compaction_memory.clear()
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

        st.divider()
        st.subheader("🧠 记忆系统")
        mem = st.session_state.compaction_memory
        if mem:
            from src.agent.compaction_memory import _estimate_messages_tokens
            raw_tokens = _estimate_messages_tokens(mem.messages)
            st.caption(f"短期记忆: {len(mem.messages)} 条 (~{raw_tokens} tokens)")
            if mem.compaction_summary:
                with st.expander("📝 Compaction 摘要", expanded=False):
                    st.markdown(mem.compaction_summary[:600])
            else:
                st.caption("暂无压缩摘要 (对话较短)")
        else:
            st.caption("记忆系统未激活")
        ltm_store: LangGraphMemoryStore = st.session_state.long_term_memory
        counts = ltm_store.get_memory_count()
        c_mem1, c_mem2 = st.columns(2)
        with c_mem1:
            st.metric("会话记忆", counts["conversations"])
        with c_mem2:
            st.metric("知识事实", counts["facts"])
        st.caption("LangGraph Store · JSON 持久化")

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
        _get_compaction_memory().add("user", prompt)

        config = {"routing_mode": routing_mode, "top_k": top_k, "answer_mode": answer_mode}

        with st.chat_message("assistant"):
            with st.spinner("正在思考..."):
                response = process_query(prompt, config)

        if response["success"]:
            assistant_message = {
                "role": "assistant",
                "content": response["answer"],
                "citations": response["citations"],
                "metrics": response["metrics"],
                "image_data": response.get("image_data", []),
                "thinking_process": response.get("thinking_process", []),
            }
            st.session_state.chat_history.append(assistant_message)
            st.session_state.execution_traces.extend(response["execution_trace"])
            _get_compaction_memory().add("assistant", response["answer"])
            ltm_store: LangGraphMemoryStore = st.session_state.long_term_memory
            ltm_store.save_session(
                _get_compaction_memory().messages,
                compaction_summary=_get_compaction_memory().compaction_summary,
            )
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
