"""LangGraph-based Agent for Modular RAG.

This module implements a state-machine workflow using LangGraph:

1. Intent Classification → 2. Conditional Routing → 3. Tool Execution → 4. Response Generation

Key Features:
- Visualizable state graph for debugging
- Support for complex branching, loops, and parallel execution
- Native checkpointing for long-running conversations
- Compatible with existing SimpleAgent interface

Example:
    >>> from src.agent.langgraph_agent import LangGraphAgent
    >>> from src.core.settings import load_settings
    
    >>> settings = load_settings()
    >>> agent = LangGraphAgent(settings)
    
    >>> # Simple query
    >>> response = agent.run("什么是 RAG？")
    >>> print(response.content)
    
    >>> # Multi-turn conversation with pronoun resolution
    >>> response = agent.run("它有什么优势？")  # "它" resolved from context
    >>> print(response.content)
    
    >>> # Check execution trace
    >>> for step in response.steps:
    ...     print(f"Step {step.step}: {step.thought}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Annotated, Literal
from enum import Enum

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from src.agent.intent_classifier import IntentClassifier, IntentResult, IntentType
from src.agent.tool_caller import ToolRegistry, ToolName
from src.agent.memory import ConversationMemory
from src.core.settings import Settings


class WorkflowNodeType(Enum):
    """Types of nodes in the workflow graph."""
    CLASSIFY_INTENT = "classify_intent"
    ROUTE = "route"
    HANDLE_CHAT = "handle_chat"
    SEARCH_KNOWLEDGE = "search_knowledge"
    RERANK_RESULTS = "rerank_results"
    GENERATE_RESPONSE = "generate_response"
    GET_SUMMARY = "get_summary"
    LIST_COLLECTIONS = "list_collections"


@dataclass
class AgentStep:
    """Single step in the agent's execution trace."""
    step: int
    node: str
    thought: str
    action: Optional[str] = None
    observation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Structured agent response."""
    content: str
    intent: IntentType
    tool_called: Optional[str]
    tool_result: Optional[Dict[str, Any]]
    confidence: float
    steps: List[AgentStep] = field(default_factory=list)


class AgentState(dict):
    """State container for LangGraph workflow.
    
    Extends dict to support message accumulation and state updates.
    
    State Fields:
        - messages: Conversation history (accumulated via add_messages)
        - intent: Classified intent type
        - intent_confidence: Confidence score from classifier
        - intent_params: Extracted parameters from intent
        - query: Current query (may be rewritten from context)
        - search_results: Results from knowledge base search
        - reranked_results: Results after reranking
        - final_answer: Generated response
        - tool_called: Name of tool that was called
        - tool_result: Raw result from tool execution
        - execution_trace: List of AgentStep for observability
        - error: Error message if any step failed
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setdefault("messages", [])
        self.setdefault("execution_trace", [])
        self.setdefault("intent", None)
        self.setdefault("intent_confidence", 0.0)
        self.setdefault("intent_params", {})
        self.setdefault("query", None)
        self.setdefault("search_results", None)
        self.setdefault("reranked_results", None)
        self.setdefault("final_answer", None)
        self.setdefault("tool_called", None)
        self.setdefault("tool_result", None)
        self.setdefault("error", None)
    
    @property
    def messages(self) -> List[BaseMessage]:
        return self.get("messages", [])
    
    @messages.setter
    def messages(self, value: List[BaseMessage]):
        self["messages"] = value
    
    @property
    def execution_trace(self) -> List[AgentStep]:
        return self.get("execution_trace", [])
    
    @execution_trace.setter
    def execution_trace(self, value: List[AgentStep]):
        self["execution_trace"] = value
    
    def add_step(self, step: AgentStep):
        """Add a step to the execution trace."""
        self["execution_trace"].append(step)
    
    def update(self, **kwargs):
        """Update state with new values."""
        for key, value in kwargs.items():
            if value is not None:
                self[key] = value


class LangGraphAgent:
    """LangGraph-based Agent for Modular RAG.
    
    This agent uses a state machine approach:
    
    ```
    [START] → classify_intent → route → [branch based on intent]
                                              ↓
        ┌─────────────────────────────────────┼─────────────────────────────────────┐
        ↓                                     ↓                                     ↓
    handle_chat                         search_knowledge                      get_summary
        ↓                                     ↓                                     ↓
    [END] ←─────────────────────── generate_response ←──────────────────────────────┘
    ```
    
    Features:
        - Multi-turn conversation with context-aware query rewriting
        - Conditional routing based on intent classification
        - Hybrid search (dense + sparse) with optional reranking
        - Execution trace for observability
        - Compatible with SimpleAgent interface
    
    Usage:
        >>> settings = load_settings()
        >>> agent = LangGraphAgent(settings, enable_rerank=True)
        >>> response = agent.run("查询关于 RAG 的论文")
        >>> print(response.content)
        >>> print(f"Intent: {response.intent.value}")
        >>> print(f"Steps: {len(response.steps)}")
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        enable_rerank: bool = False,
        max_search_results: int = 10,
        enable_logging: bool = True,
    ):
        """Initialize LangGraph Agent.
        
        Args:
            settings: Application settings
            enable_rerank: Whether to enable reranking step
            max_search_results: Maximum number of search results
            enable_logging: Whether to enable logging
        """
        self.settings = settings or Settings()
        self.enable_rerank = enable_rerank
        self.max_search_results = max_search_results
        
        # Initialize components (reuse existing implementations)
        self.intent_classifier = IntentClassifier(self.settings)
        self.tool_registry = ToolRegistry(self.settings)
        self.memory = ConversationMemory(self.settings)
        
        # Build the workflow graph
        self.workflow = self._build_graph()
        
        self.enable_logging = enable_logging
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine.
        
        Returns:
            Compiled StateGraph ready for execution
        """
        # Initialize state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node(WorkflowNodeType.CLASSIFY_INTENT.value, self._node_classify_intent)
        workflow.add_node(WorkflowNodeType.ROUTE.value, self._node_route)
        workflow.add_node(WorkflowNodeType.HANDLE_CHAT.value, self._node_handle_chat)
        workflow.add_node(WorkflowNodeType.SEARCH_KNOWLEDGE.value, self._node_search_knowledge)
        workflow.add_node(WorkflowNodeType.RERANK_RESULTS.value, self._node_rerank_results)
        workflow.add_node(WorkflowNodeType.GENERATE_RESPONSE.value, self._node_generate_response)
        workflow.add_node(WorkflowNodeType.GET_SUMMARY.value, self._node_get_summary)
        workflow.add_node(WorkflowNodeType.LIST_COLLECTIONS.value, self._node_list_collections)
        
        # Set entry point
        workflow.set_entry_point(WorkflowNodeType.CLASSIFY_INTENT.value)
        
        # Add edges
        workflow.add_edge(WorkflowNodeType.CLASSIFY_INTENT.value, WorkflowNodeType.ROUTE.value)
        
        # Conditional routing based on intent
        workflow.add_conditional_edges(
            WorkflowNodeType.ROUTE.value,
            self._route_by_intent,
            {
                "chat": WorkflowNodeType.HANDLE_CHAT.value,
                "query": WorkflowNodeType.SEARCH_KNOWLEDGE.value,
                "summary": WorkflowNodeType.GET_SUMMARY.value,
                "list_collections": WorkflowNodeType.LIST_COLLECTIONS.value,
                "unknown": WorkflowNodeType.HANDLE_CHAT.value,
            }
        )
        
        # Search → (Rerank) → Generate
        if self.enable_rerank:
            workflow.add_edge(
                WorkflowNodeType.SEARCH_KNOWLEDGE.value,
                WorkflowNodeType.RERANK_RESULTS.value
            )
            workflow.add_edge(
                WorkflowNodeType.RERANK_RESULTS.value,
                WorkflowNodeType.GENERATE_RESPONSE.value
            )
        else:
            workflow.add_edge(
                WorkflowNodeType.SEARCH_KNOWLEDGE.value,
                WorkflowNodeType.GENERATE_RESPONSE.value
            )
        
        # Summary/Collections → Generate
        workflow.add_edge(WorkflowNodeType.GET_SUMMARY.value, WorkflowNodeType.GENERATE_RESPONSE.value)
        workflow.add_edge(WorkflowNodeType.LIST_COLLECTIONS.value, WorkflowNodeType.GENERATE_RESPONSE.value)
        
        # Chat → END
        workflow.add_edge(WorkflowNodeType.HANDLE_CHAT.value, END)
        
        # Generate → END
        workflow.add_edge(WorkflowNodeType.GENERATE_RESPONSE.value, END)
        
        return workflow.compile()
    
    def run(self, user_input: str) -> AgentResponse:
        """Main entry point: process user input through the workflow.
        
        Args:
            user_input: User's query or message
            
        Returns:
            AgentResponse with content, intent, tool calls, and execution trace
        """
        # Rewrite query based on conversation context
        rewritten_query = self.memory.rewrite_query(user_input)
        
        # Get conversation context for intent classification
        memory_context = self.memory.to_dict().get("turns", [])
        
        # Initialize state
        initial_state = AgentState(
            messages=[HumanMessage(content=rewritten_query)],
            query=rewritten_query,
        )
        
        # Execute workflow
        final_state = self.workflow.invoke(initial_state)
        
        # Save to memory
        intent_type = final_state.get("intent", IntentType.UNKNOWN)
        self.memory.add_user_message(user_input, intent=intent_type.value if intent_type else "unknown")
        self.memory.add_assistant_message(
            final_state.get("final_answer", ""),
            intent=intent_type.value if intent_type else "unknown",
            tool_called=final_state.get("tool_called")
        )
        
        # Build response
        return AgentResponse(
            content=final_state.get("final_answer", "抱歉，我暂时无法回答这个问题。"),
            intent=final_state.get("intent", IntentType.UNKNOWN),
            tool_called=final_state.get("tool_called"),
            tool_result=final_state.get("tool_result"),
            confidence=final_state.get("intent_confidence", 0.0),
            steps=final_state.get("execution_trace", []),
        )
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.memory.clear()
    
    def get_history(self) -> List[Dict[str, str]]:
        """Return conversation history."""
        return [
            {"role": turn.role, "content": turn.content, "intent": turn.intent}
            for turn in self.memory.turns
        ]
    
    def get_memory(self) -> ConversationMemory:
        """Return the conversation memory instance."""
        return self.memory
    
    # ------------------------------------------------------------------
    # Node Implementations
    # ------------------------------------------------------------------
    
    def _node_classify_intent(self, state: AgentState) -> AgentState:
        """Classify user intent.
        
        This node:
        1. Uses IntentClassifier to determine intent
        2. Extracts parameters
        3. Records execution trace
        
        Returns:
            Updated state with intent information
        """
        query = state.get("query")
        memory_context = self.memory.to_dict().get("turns", [])
        
        # Classify intent
        intent_result = self.intent_classifier.classify(query, context=memory_context)
        
        # Record trace
        step = AgentStep(
            step=len(state.execution_trace) + 1,
            node=WorkflowNodeType.CLASSIFY_INTENT.value,
            thought=f"分析用户意图：{query[:50]}...",
            action=f"IntentClassifier.classify()",
            observation=f"Intent: {intent_result.intent.value} (confidence: {intent_result.confidence:.2f})",
            metadata={
                "intent": intent_result.intent.value,
                "confidence": intent_result.confidence,
                "parameters": intent_result.parameters,
            }
        )
        state.add_step(step)
        
        # Update state
        state.update(
            intent=intent_result.intent,
            intent_confidence=intent_result.confidence,
            intent_params=intent_result.parameters,
        )
        
        if self.enable_logging:
            print(f"[Intent] {intent_result.intent.value} (conf: {intent_result.confidence:.2f})")
        
        return state
    
    def _node_route(self, state: AgentState) -> AgentState:
        """Routing node (placeholder for conditional edges).
        
        This node doesn't modify state, just serves as a branching point.
        """
        step = AgentStep(
            step=len(state.execution_trace) + 1,
            node=WorkflowNodeType.ROUTE.value,
            thought=f"根据意图路由到对应处理节点",
            action=f"route_by_intent({state.get('intent')})",
        )
        state.add_step(step)
        
        return state
    
    def _node_handle_chat(self, state: AgentState) -> AgentState:
        """Handle simple chat without tool use.
        
        For now, uses rule-based responses. Can be enhanced with LLM.
        """
        query = state.get("query")
        
        # Simple rule-based responses
        if any(greeting in query.lower() for greeting in ["你好", "hello", "hi", "您好"]):
            response = "你好！我是你的 RAG 助手，可以帮你查询知识库、浏览文档集合或获取文档摘要。有什么可以帮你的吗？"
        elif any(thanks in query.lower() for thanks in ["谢谢", "thank", "感谢"]):
            response = "不客气！如果还有其他问题，随时问我。"
        elif "你能做什么" in query or "what can you do" in query.lower():
            response = "我可以帮你：\n1. 查询知识库（例如：'什么是 RAG？'）\n2. 列出文档集合（例如：'有哪些文档？'）\n3. 获取文档摘要（例如：'总结这篇论文'）"
        else:
            response = "我收到你的消息了。如果你想查询知识库、浏览文档或获取摘要，请随时告诉我！"
        
        step = AgentStep(
            step=len(state.execution_trace) + 1,
            node=WorkflowNodeType.HANDLE_CHAT.value,
            thought="判断为普通对话，直接回复",
            action="rule_based_response",
            observation="已生成回复",
        )
        state.add_step(step)
        
        state.update(
            final_answer=response,
        )
        
        return state
    
    def _node_search_knowledge(self, state: AgentState) -> AgentState:
        """Search knowledge base using hybrid search (Dense + Sparse in parallel).
        
        Executes the query_knowledge_hub tool which internally uses:
        1. ThreadPoolExecutor for parallel Dense + Sparse retrieval
        2. RRF Fusion for result combination
        3. Post-filtering for metadata constraints
        
        Returns:
            Updated state with search results
        """
        query = state.get("query")
        params = state.get("intent_params", {})
        top_k = params.get("top_k", self.max_search_results)
        collection = params.get("collection", "default")
        
        step = AgentStep(
            step=len(state.execution_trace) + 1,
            node=WorkflowNodeType.SEARCH_KNOWLEDGE.value,
            thought=f"并行检索：Dense(语义) + Sparse(BM25) → RRF 融合",
            action=f"hybrid_search(query='{query[:30]}...', top_k={top_k})",
        )
        
        try:
            # Execute tool (internally uses ThreadPoolExecutor for parallel retrieval)
            tool_result = self.tool_registry.execute(
                ToolName.QUERY_KNOWLEDGE_HUB.value,
                query=query,
                top_k=top_k,
                collection=collection,
            )
            
            # Extract detailed metadata from hybrid search result
            result_count = len(tool_result.data.get('results', [])) if tool_result.success else 0
            dense_count = tool_result.data.get('dense_count', 0) if tool_result.success else 0
            sparse_count = tool_result.data.get('sparse_count', 0) if tool_result.success else 0
            
            step.observation = (
                f"并行检索完成 | Dense: {dense_count} | Sparse: {sparse_count} | "
                f"融合后：{result_count}" if tool_result.success else f"错误：{tool_result.error}"
            )
            step.metadata = {
                "success": tool_result.success,
                "result_count": result_count,
                "dense_count": dense_count,
                "sparse_count": sparse_count,
                "fusion_method": "RRF",
                "parallel_execution": True,
            }
            
            state.update(
                search_results=tool_result.data.get("results") if tool_result.success else [],
                tool_result=tool_result.data if tool_result.success else None,
                error=tool_result.error if not tool_result.success else None,
            )
            
        except Exception as e:
            step.observation = f"异常：{str(e)}"
            state.update(
                search_results=[],
                error=str(e),
            )
        
        state.add_step(step)
        
        if self.enable_logging:
            print(f"[Search] Parallel: Dense({step.metadata.get('dense_count', 0)}) + Sparse({step.metadata.get('sparse_count', 0)}) → RRF({len(state.get('search_results', []))})")
        
        return state
    
    def _node_rerank_results(self, state: AgentState) -> AgentState:
        """Rerank search results (optional step).
        
        This is a placeholder for reranking logic.
        Currently just passes through the search results.
        """
        search_results = state.get("search_results", [])
        
        step = AgentStep(
            step=len(state.execution_trace) + 1,
            node=WorkflowNodeType.RERANK_RESULTS.value,
            thought=f"对 {len(search_results)} 条结果进行重排序",
            action="rerank_results",
            observation=f"重排序完成（当前为 pass-through）",
        )
        state.add_step(step)
        
        # TODO: Integrate actual reranker
        # from src.core.query_engine.reranker import Reranker
        # reranker = Reranker(self.settings)
        # reranked = reranker.rerank(query, search_results)
        
        state.update(
            reranked_results=search_results,  # Pass-through for now
        )
        
        return state
    
    def _node_generate_response(self, state: AgentState) -> AgentState:
        """Generate final response from search results or tool outputs.
        
        Formats the results into a user-friendly response.
        """
        query = state.get("query")
        intent = state.get("intent")
        search_results = state.get("search_results") or state.get("reranked_results", [])
        tool_result = state.get("tool_result")
        error = state.get("error")
        
        step = AgentStep(
            step=len(state.execution_trace) + 1,
            node=WorkflowNodeType.GENERATE_RESPONSE.value,
            thought="整理结果并生成最终回复",
            action="format_response",
        )
        
        if error:
            response = f"抱歉，在处理你的请求时出错了：{error}"
            step.observation = f"错误回复：{error}"
        
        elif intent == IntentType.QUERY:
            response = self._format_query_response(query, search_results)
            step.observation = f"生成查询回复 ({len(search_results)} 条结果)"
            state.update(tool_called=ToolName.QUERY_KNOWLEDGE_HUB.value)
        
        elif intent == IntentType.GET_SUMMARY:
            response = self._format_summary_response(
                state.get("intent_params", {}).get("doc_id", query),
                tool_result,
            )
            step.observation = "生成文档摘要回复"
            state.update(tool_called=ToolName.GET_DOCUMENT_SUMMARY.value)
        
        elif intent == IntentType.LIST_COLLECTIONS:
            response = self._format_collections_response(tool_result)
            step.observation = "生成集合列表回复"
            state.update(tool_called=ToolName.LIST_COLLECTIONS.value)
        
        else:
            response = "抱歉，我暂时无法回答这个问题。"
            step.observation = "未知意图的兜底回复"
        
        state.add_step(step)
        state.update(final_answer=response)
        
        return state
    
    def _node_get_summary(self, state: AgentState) -> AgentState:
        """Get document summary.
        
        Executes get_document_summary tool.
        """
        params = state.get("intent_params", {})
        doc_id = params.get("doc_id") or state.get("query")
        
        step = AgentStep(
            step=len(state.execution_trace) + 1,
            node=WorkflowNodeType.GET_SUMMARY.value,
            thought=f"获取文档摘要：{doc_id[:50]}...",
            action=f"get_document_summary(source_path='{doc_id}')",
        )
        
        try:
            tool_result = self.tool_registry.execute(
                ToolName.GET_DOCUMENT_SUMMARY.value,
                source_path=doc_id,
            )
            
            step.observation = "已获取文档摘要" if tool_result.success else f"错误：{tool_result.error}"
            
            state.update(
                tool_result=tool_result.data if tool_result.success else None,
                error=tool_result.error if not tool_result.success else None,
            )
            
        except Exception as e:
            step.observation = f"异常：{str(e)}"
            state.update(error=str(e))
        
        state.add_step(step)
        return state
    
    def _node_list_collections(self, state: AgentState) -> AgentState:
        """List all document collections.
        
        Executes list_collections tool.
        """
        step = AgentStep(
            step=len(state.execution_trace) + 1,
            node=WorkflowNodeType.LIST_COLLECTIONS.value,
            thought="获取当前向量库中的集合列表",
            action="list_collections()",
        )
        
        try:
            tool_result = self.tool_registry.execute(ToolName.LIST_COLLECTIONS.value)
            
            step.observation = "已获取集合列表" if tool_result.success else f"错误：{tool_result.error}"
            
            state.update(
                tool_result=tool_result.data if tool_result.success else None,
                error=tool_result.error if not tool_result.success else None,
            )
            
        except Exception as e:
            step.observation = f"异常：{str(e)}"
            state.update(error=str(e))
        
        state.add_step(step)
        return state
    
    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------
    
    def _route_by_intent(self, state: AgentState) -> Literal["chat", "query", "summary", "list_collections", "unknown"]:
        """Determine next node based on intent.
        
        Returns:
            Name of the next node to execute
        """
        intent = state.get("intent")
        
        if intent == IntentType.QUERY:
            return "query"
        elif intent == IntentType.GET_SUMMARY:
            return "summary"
        elif intent == IntentType.LIST_COLLECTIONS:
            return "list_collections"
        elif intent in (IntentType.CHAT, IntentType.UNKNOWN):
            return "chat"
        else:
            return "unknown"
    
    def _format_query_response(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Format search results into response."""
        if not results:
            return f"抱歉，没有找到与「{query}」相关的文档。"
        
        response_lines = [
            f"为你找到 {len(results)} 条与「{query}」相关的内容，这里是最相关的几条：\n"
        ]
        
        for idx, item in enumerate(results[:3], start=1):
            content = item.get("content", "")
            score = item.get("score", 0.0)
            source = item.get("source", "unknown")
            
            response_lines.append(f"**{idx}. 片段预览**")
            response_lines.append(content.strip()[:200] + "...\n")
            response_lines.append(f"   📄 来源：{source} | 相关度：{score:.3f}\n")
        
        if len(results) > 3:
            response_lines.append(f"... 另外还有 {len(results) - 3} 条结果未展示。")
        
        response_lines.append("\n如果你需要，我可以帮你围绕这些结果继续总结、对比或写一个小结。")
        
        return "\n".join(response_lines)
    
    def _format_summary_response(self, doc_id: str, tool_result: Optional[Dict[str, Any]]) -> str:
        """Format document summary response."""
        if not tool_result:
            return f"抱歉，无法获取文档「{doc_id}」的摘要。"
        
        summary = tool_result.get("summary", "暂无摘要信息")
        metadata = tool_result.get("metadata", {})
        
        response = f"📄 **文档摘要**: {doc_id}\n\n"
        response += f"{summary}\n\n"
        
        if metadata:
            response += "**元数据信息**:\n"
            for key, value in metadata.items():
                response += f"- {key}: {value}\n"
        
        return response
    
    def _format_collections_response(self, tool_result: Optional[Dict[str, Any]]) -> str:
        """Format collections list response."""
        if not tool_result:
            return "抱歉，无法获取集合列表。"
        
        collections = tool_result.get("collections", {})
        
        if not collections:
            return "当前向量库中没有集合。"
        
        response = "📚 **当前文档集合**:\n\n"
        for name, stats in collections.items():
            response += f"- **{name}**: {stats.get('count', 0)} 个文档\n"
        
        return response
