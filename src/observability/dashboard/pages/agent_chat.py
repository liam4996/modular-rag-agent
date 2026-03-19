"""Multi-Agent Chat page – Interactive chat with the RAG Agent.

Layout:
1. Chat interface with message history
2. Configuration options (routing mode, retrieval settings)
3. Real-time agent responses with citations
4. Execution trace visualization
5. Metrics display
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# Import multi-agent system
from src.agent.multi_agent import (
    AgentState,
    MultiAgentRAG,
    Citation,
    CitationManager,
)

logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def initialize_session_state() -> None:
    """Initialize chat session state."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "agent_state" not in st.session_state:
        st.session_state.agent_state = None
    
    if "agent" not in st.session_state:
        st.session_state.agent = None
    
    if "execution_traces" not in st.session_state:
        st.session_state.execution_traces = []


def get_agent() -> MultiAgentRAG:
    """Get or create agent instance."""
    if st.session_state.agent is None:
        try:
            # 加载配置
            from src.core.settings import load_settings
            
            settings = load_settings()
            
            # 初始化 LLM
            from langchain_openai import ChatOpenAI
            
            llm = ChatOpenAI(
                model=settings.llm.model,
                temperature=settings.llm.temperature,
                max_tokens=settings.llm.max_tokens,
                api_key=settings.llm.api_key,
                base_url=settings.llm.base_url if settings.llm.base_url else None,
            )
            
            # 初始化 MultiAgentRAG
            st.session_state.agent = MultiAgentRAG(
                llm=llm,
                settings=settings,
                enable_logging=True
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
    """Extract recent conversation turns for the agent's memory.

    Keeps the most recent ``max_turns`` user/assistant pairs so the
    context window stays manageable.
    """
    max_turns = 10
    history: List[Dict[str, str]] = []
    for msg in st.session_state.get("chat_history", []):
        role = msg.get("role")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            history.append({"role": role, "content": content[:500]})
    return history[-max_turns * 2:]


def process_query(query: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Process user query through the agent with conversation memory."""
    agent = get_agent()
    if not agent:
        return {
            "success": False,
            "error": "Agent not initialized",
        }
    
    try:
        start_time = time.time()
        
        conversation_history = _build_conversation_history()
        final_state = agent.run(
            user_input=query,
            conversation_history=conversation_history,
        )
        
        # LangGraph returns dict representation of AgentState
        if isinstance(final_state, dict):
            answer = final_state.get('final_answer', '')
            bb = final_state.get('blackboard', {})
            local_results = bb.get('local_results', [])
            web_results = bb.get('web_results', [])
            citations = bb.get('citations', [])
            evaluation = bb.get('evaluation', {})
            execution_trace = final_state.get('execution_trace', [])
        else:
            # It's an AgentState object
            answer = final_state.final_answer
            local_results = final_state.local_results
            web_results = final_state.web_results
            citations = final_state.blackboard.get('citations', [])
            evaluation = final_state.evaluation
            execution_trace = final_state.execution_trace
        
        # Record timing
        elapsed = time.time() - start_time
        
        # Build metrics
        metrics = {
            "retrieval_count": len(local_results) + len(web_results),
            "citation_count": len(citations) if citations else 0,
            "confidence": evaluation.get("confidence", 0) if evaluation else 0,
            "total_time": elapsed,
        }
        
        # Build response
        response = {
            "success": bool(answer),
            "answer": answer if answer else "抱歉，我暂时无法回答这个问题。",
            "citations": citations if citations else [],
            "execution_trace": execution_trace,
            "metrics": metrics,
            "state": final_state,
        }
        
        return response
        
    except Exception as e:
        logger.exception(f"Error processing query: {e}")
        return {
            "success": False,
            "error": str(e),
        }


def simulate_agent_workflow(state: AgentState, agent: MultiAgentRAG) -> Dict[str, Any]:
    """
    Simulate the agent workflow for demonstration.
    
    This is a placeholder that will be replaced with actual graph execution.
    """
    # Simulate Router Agent
    state.add_execution_trace({
        "agent": "RouterAgent",
        "action": "classify_intent",
        "timestamp": time.time(),
        "data": {
            "intent": "local_search",
            "confidence": 0.95,
        }
    })
    
    # Simulate Search Agent
    mock_results = [
        {
            "content": f"这是关于「{state.user_input}」的检索结果 1",
            "source": "文档 A",
            "score": 0.95,
        },
        {
            "content": f"这是关于「{state.user_input}」的检索结果 2",
            "source": "文档 B",
            "score": 0.88,
        },
    ]
    state.add_to_blackboard("local_results", mock_results, "SearchAgent")
    
    state.add_execution_trace({
        "agent": "SearchAgent",
        "action": "retrieve",
        "timestamp": time.time(),
        "data": {
            "results_count": len(mock_results),
        }
    })
    
    # Simulate Eval Agent
    evaluation = {
        "relevance": 0.90,
        "confidence": 0.85,
    }
    state.add_to_blackboard("evaluation", evaluation, "EvalAgent")
    
    state.add_execution_trace({
        "agent": "EvalAgent",
        "action": "evaluate",
        "timestamp": time.time(),
        "data": evaluation
    })
    
    # Create citations
    citation_manager = CitationManager()
    from src.agent.multi_agent.citation import CitationType
    citations = [
        Citation(
            type=CitationType.LOCAL,
            source="文档 A",
            content=mock_results[0]["content"],
            confidence=0.95,
        ),
        Citation(
            type=CitationType.LOCAL,
            source="文档 B",
            content=mock_results[1]["content"],
            confidence=0.88,
        ),
    ]
    citation_manager.add_citations(citations)
    
    # Generate response
    answer = (
        f"根据检索到的信息，关于「{state.user_input}」：\n\n"
        f"1. {mock_results[0]['content']} [Local: {mock_results[0]['source']}]\n"
        f"2. {mock_results[1]['content']} [Local: {mock_results[1]['source']}]\n\n"
        f"以上信息基于本地知识库。"
    )
    
    state.final_answer = answer
    
    state.add_execution_trace({
        "agent": "GenerateAgent",
        "action": "generate_answer",
        "timestamp": time.time(),
        "data": {
            "answer_length": len(answer),
            "citations_count": len(citations),
        }
    })
    
    return {
        "success": True,
        "answer": answer,
        "citations": citations,
        "execution_trace": state.execution_trace,
        "metrics": {
            "retrieval_count": len(mock_results),
            "citation_count": len(citations),
            "confidence": evaluation["confidence"],
        },
        "state": state,
    }


def display_chat_message(message: Dict[str, Any]) -> None:
    """Display a chat message."""
    role = message["role"]
    content = message["content"]
    
    if role == "user":
        st.chat_message("user").markdown(content)
    elif role == "assistant":
        with st.chat_message("assistant"):
            st.markdown(content)
            
            # Show citations if available
            if "citations" in message and message["citations"]:
                with st.expander("📚 查看引用来源", expanded=False):
                    for idx, citation in enumerate(message["citations"], 1):
                        if isinstance(citation, dict):
                            c = Citation.from_dict(citation)
                        else:
                            c = citation
                        st.markdown(f"**{idx}.** {c.format_citation()}")
                        with st.expander(f"查看内容 #{idx}", expanded=False):
                            st.markdown(c.content[:300] + "..." if len(c.content) > 300 else c.content)
            
            # Show metrics if available
            if "metrics" in message and message["metrics"]:
                with st.expander("📊 执行指标", expanded=False):
                    metrics = message["metrics"]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("检索文档数", metrics.get("retrieval_count", 0))
                    with col2:
                        st.metric("引用数量", metrics.get("citation_count", 0))
                    with col3:
                        st.metric("置信度", f"{metrics.get('confidence', 0):.2f}")


def display_execution_trace(trace: List[Dict[str, Any]]) -> None:
    """Display execution trace visualization."""
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
        
        # Show all fields except agent/action as detail data
        detail = {k: v for k, v in step.items() if k not in ("agent", "action")}
        
        with st.expander(f"{icon} **{agent}** → {action}", expanded=False):
            if detail:
                st.json(detail)
            else:
                st.caption("无详细数据")


def render() -> None:
    """Render the Multi-Agent Chat page."""
    st.header("🤖 多智能体 RAG 对话")
    st.markdown("与 RAG Agent 进行交互式对话")
    
    # Initialize session state
    initialize_session_state()
    
    # ── Configuration Sidebar ────────────────────────────────────
    with st.sidebar:
        st.subheader("⚙️ 配置")
        
        # Routing mode
        routing_mode = st.selectbox(
            "路由模式",
            options=["auto", "local_search", "web_search", "hybrid_search"],
            index=0,
            help="选择搜索模式",
        )
        
        # Top-K
        top_k = st.slider(
            "检索数量",
            min_value=1,
            max_value=20,
            value=5,
            help="返回的检索结果数量",
        )
        
        # Clear chat button
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.execution_traces = []
            st.rerun()
        
        st.divider()
        
        # System status
        st.subheader("📊 系统状态")
        agent = get_agent()
        if agent:
            st.success("✅ Agent 已初始化")
        else:
            st.error("❌ Agent 未初始化")
        
        st.metric("对话轮数", len(st.session_state.chat_history))
        st.metric("执行轨迹", len(st.session_state.execution_traces))
    
    # ── Chat Interface ───────────────────────────────────────────
    
    # Display chat history
    for message in st.session_state.chat_history:
        display_chat_message(message)
    
    # Chat input
    if prompt := st.chat_input("输入您的问题..."):
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt,
        })
        
        # Process query
        config = {
            "routing_mode": routing_mode,
            "top_k": top_k,
        }
        
        with st.chat_message("assistant"):
            with st.spinner("正在思考..."):
                response = process_query(prompt, config)
        
        if response["success"]:
            # Display assistant response
            answer = response["answer"]
            citations = response["citations"]
            metrics = response["metrics"]
            execution_trace = response["execution_trace"]
            
            # Store in history
            assistant_message = {
                "role": "assistant",
                "content": answer,
                "citations": citations,
                "metrics": metrics,
            }
            st.session_state.chat_history.append(assistant_message)
            
            # Store execution trace
            st.session_state.execution_traces.extend(execution_trace)
            
            # Display response
            display_chat_message(assistant_message)
            
            # Show execution trace
            if execution_trace:
                display_execution_trace(execution_trace)
            
            # Auto-scroll
            st.rerun()
            
        else:
            # Display error
            error_message = {
                "role": "assistant",
                "content": f"❌ 处理失败：{response.get('error', '未知错误')}",
            }
            st.session_state.chat_history.append(error_message)
            display_chat_message(error_message)
            st.rerun()


def main() -> None:
    """Main entry point."""
    render()


if __name__ == "__main__":
    main()
