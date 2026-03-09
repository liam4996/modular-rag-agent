"""Agent package for Modular RAG MCP Server.

说明：
    这里提供了 Agent 封装，把底层的 RAG / 向量库工具
    （如 `query_knowledge_hub`、`list_collections` 等）包装成一个
    可规划、可追踪的智能 Agent。

常用入口：
    >>> from src.agent import SimpleAgent
    >>> agent = SimpleAgent()
    >>> resp = agent.run("查询论文结论")
    
    >>> from src.agent import ReActAgent
    >>> agent = ReActAgent()
    >>> resp = agent.run("总结这份文档")
    >>> for step in resp.steps:
    >>>     print(f"Step {step.step}: {step.thought}")
"""

from .simple_agent import SimpleAgent, AgentResponse, AgentStep
from .react_agent import ReActAgent, ReActResponse, ReActStep
from .memory import ConversationMemory
from .tool_chain import ToolChainExecutor, ChainStep, ChainStepType, ChainResult

__all__ = [
    "SimpleAgent", 
    "AgentResponse", 
    "AgentStep",
    "ReActAgent",
    "ReActResponse",
    "ReActStep",
    "ConversationMemory",
    "ToolChainExecutor",
    "ChainStep",
    "ChainStepType",
    "ChainResult",
]
__version__ = "0.4.0"
