"""Agent 流式事件定义。

提供 Agent 执行过程的实时事件流，支持：
- Agent 开始/步骤/完成事件
- 思考过程片段
- LLM 部分文本输出
- 错误/完成信号
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class EventType(Enum):
    AGENT_START = "agent_start"
    AGENT_STEP = "agent_step"
    AGENT_RESULT = "agent_result"
    THINKING = "thinking"
    CONFLICT = "conflict"
    PARTIAL_TEXT = "partial_text"
    CHART_GENERATED = "chart_generated"
    DONE = "done"
    ERROR = "error"


@dataclass
class AgentEvent:
    type: EventType
    agent_name: str
    timestamp: str = ""
    content: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    trace_id: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class StreamingSession:
    events: List[AgentEvent] = field(default_factory=list)
    partial_answer: str = ""
    current_agent: str = ""
    finished: bool = False

    def add_event(self, event: AgentEvent):
        self.events.append(event)
        if event.type == EventType.PARTIAL_TEXT:
            self.partial_answer += event.content
        if event.type == EventType.DONE:
            self.finished = True
