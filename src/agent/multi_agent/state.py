"""
多智能体 RAG 系统 - 共享状态容器

提供所有智能体共享的状态管理，支持：
- 黑板模式（Blackboard Pattern）
- 重试控制
- 执行追踪
- 指标记录
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum


class FallbackReason(Enum):
    """兜底触发原因"""
    MAX_RETRIES_EXCEEDED = "max_retries_exceeded"
    NO_RESULTS_FOUND = "no_results_found"
    LOW_CONFIDENCE = "low_confidence"
    USER_ASKED_UNKNOWN = "user_asked_unknown"


@dataclass
class AgentState:
    """
    增强版共享状态容器
    
    所有智能体共享同一个 AgentState 实例，通过 blackboard 模式
    实现数据共享和通信。
    
    Attributes:
        user_input: 用户原始输入
        conversation_history: 对话历史（所有 Agent 共享的上下文）
        blackboard: 黑板 - 所有 Agent 都可以读写的共享空间
        retry_count: 当前重试次数
        max_retries: 最大重试次数（默认 2）
        fallback_triggered: 是否触发兜底机制
        fallback_reason: 兜底触发原因
        execution_log: 执行日志（所有 Agent 都会写入）
        execution_trace: 详细的执行轨迹
        metrics: 执行指标
        final_answer: 最终生成的回答
    """
    
    # ========== 输入部分 ==========
    # 用户原始输入
    user_input: str = ""
    
    # 对话历史（所有 Agent 共享的上下文）
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    
    # ========== 黑板模式 - 共享数据 ==========
    # 所有 Agent 都可以读写的共享空间
    blackboard: Dict[str, Any] = field(default_factory=dict)
    
    # ========== 重试控制 ==========
    retry_count: int = 0  # 当前重试次数
    max_retries: int = 2  # 最大重试次数（可配置）
    fallback_triggered: bool = False  # 是否触发兜底
    fallback_reason: Optional[FallbackReason] = None  # 兜底原因
    
    # ========== 执行追踪 ==========
    # 执行日志（所有 Agent 都会写入）
    execution_log: List[str] = field(default_factory=list)
    
    # 详细的执行轨迹
    execution_trace: List[Dict[str, Any]] = field(default_factory=list)
    
    # 执行指标
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # ========== 输出部分 ==========
    # 最终生成的回答
    final_answer: str = ""
    
    # ========== 只读属性 ==========
    @property
    def intent(self) -> Optional[str]:
        """获取意图识别结果"""
        return self.blackboard.get("intent")
    
    @property
    def local_results(self) -> List:
        """获取本地搜索结果"""
        return self.blackboard.get("local_results", [])
    
    @property
    def web_results(self) -> List:
        """获取联网搜索结果"""
        return self.blackboard.get("web_results", [])
    
    @property
    def evaluation(self) -> Dict:
        """获取评估结果"""
        return self.blackboard.get("evaluation", {})
    
    @property
    def refined_query(self) -> str:
        """获取优化后的查询"""
        return self.blackboard.get("refined_query", self.user_input)
    
    @property
    def should_fallback(self) -> bool:
        """判断是否应该触发兜底"""
        return (
            self.retry_count >= self.max_retries or
            self.fallback_triggered
        )
    
    # ========== 辅助方法 ==========
    
    def add_to_blackboard(self, key: str, value: Any, agent: str):
        """
        Agent 写入数据到黑板
        
        Args:
            key: 数据键
            value: 数据值
            agent: 写入数据的 Agent 名称
        """
        self.blackboard[key] = value
        self.execution_log.append(f"{agent}: wrote {key}")
    
    def read_from_blackboard(self, key: str) -> Any:
        """
        Agent 从黑板读取数据
        
        Args:
            key: 数据键
            
        Returns:
            数据值，如果不存在则返回 None
        """
        return self.blackboard.get(key)
    
    def increment_retry(self, agent: str):
        """
        增加重试次数
        
        Args:
            agent: 调用此方法的 Agent 名称
        """
        self.retry_count += 1
        self.execution_log.append(
            f"{agent}: retry_count incremented to {self.retry_count}"
        )
    
    def trigger_fallback(self, reason: FallbackReason, agent: str):
        """
        触发兜底机制
        
        Args:
            reason: 兜底原因
            agent: 触发兜底的 Agent 名称
        """
        self.fallback_triggered = True
        self.fallback_reason = reason
        self.execution_log.append(
            f"{agent}: fallback triggered - {reason.value}"
        )
    
    def add_execution_trace(self, step: Dict[str, Any]):
        """
        添加详细的执行轨迹
        
        Args:
            step: 执行步骤信息
        """
        self.execution_trace.append(step)
    
    def add_metric(self, key: str, value: Any):
        """
        记录执行指标
        
        Args:
            key: 指标名称
            value: 指标值
        """
        self.metrics[key] = value
    
    def get_all_context(self) -> Dict[str, Any]:
        """
        获取所有上下文信息（用于最终生成）
        
        Returns:
            包含所有上下文信息的字典
        """
        return {
            "user_input": self.user_input,
            "intent": self.intent,
            "local_results": self.local_results,
            "web_results": self.web_results,
            "evaluation": self.evaluation,
            "refined_query": self.refined_query,
            "conversation_history": self.conversation_history,
            "metrics": self.metrics,
            "retry_count": self.retry_count,
            "fallback_triggered": self.fallback_triggered,
            "fallback_reason": self.fallback_reason.value if self.fallback_reason else None,
        }
    
    def reset(self):
        """重置状态（用于多轮对话）"""
        self.blackboard.clear()
        self.execution_log.clear()
        self.execution_trace.clear()
        self.metrics.clear()
        self.retry_count = 0
        self.fallback_triggered = False
        self.fallback_reason = None
        self.final_answer = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将状态转换为字典（用于序列化）
        
        Returns:
            状态的字典表示
        """
        return {
            "user_input": self.user_input,
            "blackboard": self.blackboard,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "fallback_triggered": self.fallback_triggered,
            "fallback_reason": self.fallback_reason.value if self.fallback_reason else None,
            "execution_log": self.execution_log,
            "execution_trace": self.execution_trace,
            "metrics": self.metrics,
            "final_answer": self.final_answer,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentState":
        """
        从字典创建状态
        
        Args:
            data: 状态的字典表示
            
        Returns:
            AgentState 实例
        """
        state = cls()
        state.user_input = data.get("user_input", "")
        state.blackboard = data.get("blackboard", {})
        state.retry_count = data.get("retry_count", 0)
        state.max_retries = data.get("max_retries", 2)
        state.fallback_triggered = data.get("fallback_triggered", False)
        
        fallback_reason = data.get("fallback_reason")
        if fallback_reason:
            state.fallback_reason = FallbackReason(fallback_reason)
        
        state.execution_log = data.get("execution_log", [])
        state.execution_trace = data.get("execution_trace", [])
        state.metrics = data.get("metrics", {})
        state.final_answer = data.get("final_answer", "")
        
        return state
