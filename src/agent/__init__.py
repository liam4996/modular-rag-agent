"""Agent package for Modular RAG MCP Server.

当前系统使用的 Agent 全部在 multi_agent/ 子包中。
此 __init__.py 仅作为包标记，主要入口请使用 `from src.agent.multi_agent import ...`。
"""

from .multi_agent import (
    AgentState,
    FallbackReason,
    RouterAgent,
    AgentType,
    RoutingDecision,
    SearchAgent,
    WebSearchAgent,
    ParallelFusionController,
    EvalAgent,
    EvaluationResult,
    RefineAgent,
    RefinementResult,
    SupervisorAgent,
    FinanceDataAgent,
    BusinessComputeAgent,
    MultiAgentRAG,
    Citation,
    CitationType,
    CitationManager,
    FaithfulnessCheck,
    format_answer_with_citations,
)

__all__ = [
    "AgentState",
    "FallbackReason",
    "RouterAgent",
    "AgentType",
    "RoutingDecision",
    "SearchAgent",
    "WebSearchAgent",
    "ParallelFusionController",
    "EvalAgent",
    "EvaluationResult",
    "RefineAgent",
    "RefinementResult",
    "SupervisorAgent",
    "FinanceDataAgent",
    "BusinessComputeAgent",
    "MultiAgentRAG",
    "Citation",
    "CitationType",
    "CitationManager",
    "FaithfulnessCheck",
    "format_answer_with_citations",
]
__version__ = "0.5.0"
