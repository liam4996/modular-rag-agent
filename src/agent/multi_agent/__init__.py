"""
多智能体 RAG 系统

提供基于 LangGraph 的多智能体编排，支持：
- 并行融合检索
- 共享状态（Blackboard Pattern）
- 容错机制（重试 + 兜底）
- 溯源与忠实度保证
"""

from .state import AgentState, FallbackReason
from .router_agent import RouterAgent, AgentType, RoutingDecision
from .search_agent import SearchAgent
from .web_agent import WebSearchAgent
from .parallel_controller import ParallelFusionController
from .eval_agent import EvalAgent, EvaluationResult
from .refine_agent import RefineAgent, RefinementResult
from .multi_agent_system import MultiAgentRAG
from .citation import (
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
    "MultiAgentRAG",
    "Citation",
    "CitationType",
    "CitationManager",
    "FaithfulnessCheck",
    "format_answer_with_citations",
]
