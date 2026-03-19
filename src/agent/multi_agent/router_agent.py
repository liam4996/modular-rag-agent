"""
多智能体 RAG 系统 - Router Agent

职责：
- 意图识别
- 路由决策（支持并行路由）
- 复杂查询识别（如"结合内部文档和网上资料"）
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseLLM
import json


class AgentType(Enum):
    """智能体类型"""
    CHAT = "chat"
    SEARCH = "search"
    WEB = "web"
    EVAL = "eval"
    REFINE = "refine"
    GENERATE = "generate"


@dataclass
class RoutingDecision:
    """
    路由决策结果
    
    Attributes:
        intent: 识别的意图
        agents_to_invoke: 要调用的 Agent 列表（可以是多个！）
        parallel: 是否并行执行
        confidence: 置信度
        reasoning: 推理过程
        parameters: 额外参数
    """
    intent: str
    agents_to_invoke: List[AgentType]
    parallel: bool = False
    confidence: float = 0.0
    reasoning: str = ""
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class RouterAgent:
    """
    增强版 Router Agent
    
    支持：
    - 识别 5 种意图：CHAT, LOCAL_SEARCH, WEB_SEARCH, HYBRID_SEARCH, UNKNOWN
    - 返回 RoutingDecision 包含 agents_to_invoke（可以是多个 Agent）
    - 支持并行标记 parallel: bool
    - 复杂查询识别（如"结合内部文档和网上资料"）
    """
    
    SYSTEM_PROMPT = """You are an intent classifier and router for a multi-agent RAG system.

Available intents and routing strategies:

1. CHAT - Simple conversation
   - Invoke: ChatAgent only
   - Parallel: False
   - Examples: "你好", "谢谢", "你是谁", "聊聊天"

2. LOCAL_SEARCH - Search internal knowledge base
   - Invoke: SearchAgent only
   - Parallel: False
   - Examples: "公司文档里关于 RAG 的说明", "我们的 2026 战略", "本地知识库中..."

3. WEB_SEARCH - Search internet for real-time information
   - Invoke: WebAgent only
   - Parallel: False
   - Examples: "今天 AI 领域的最新新闻", "实时股票价格", "天气", "2025 年 AI 发展"
   - Keywords: "最新", "今天", "本周", "实时", "新闻", "天气", "股票"

4. HYBRID_SEARCH - Search BOTH local and web ⭐ IMPORTANT
   - Invoke: SearchAgent AND WebAgent
   - Parallel: True  # Critical: execute in parallel
   - Examples: 
     - "结合我们内部的《2026 战略文档》和网上最新的'DeepSeek 行业分析'，写一份对比报告"
     - "我们公司的产品和技术 + 网上竞品分析"
     - "内部文档里的 RAG 原理 + 2025 年最新研究进展"
     - "我们团队的文档和网上的相关资料"

5. UNKNOWN - Cannot find answer or impossible query
   - Will trigger fallback after retries
   - Examples: "我昨天晚饭吃了什么" (user's private information)

Respond in JSON format:
{{
    "intent": "intent_type",
    "agents_to_invoke": ["SearchAgent", "WebAgent"],  # Can be multiple!
    "parallel": true,  # Whether to execute in parallel
    "confidence": 0.95,
    "reasoning": "Why this intent was chosen",
    "parameters": {{}}  # Optional extra parameters
}}"""
    
    def __init__(self, llm: BaseLLM):
        """
        初始化 Router Agent
        
        Args:
            llm: 语言模型实例
        """
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("user", "Query: {query}\nContext: {context}")
        ])
        self.chain = self.prompt | self.llm
    
    def classify(self, query: str, context: Optional[List[Dict]] = None) -> RoutingDecision:
        """
        分类意图并决定路由策略
        
        Args:
            query: 用户查询
            context: 对话历史上下文（可选）
        
        Returns:
            RoutingDecision 包含：
            - agents_to_invoke: 要调用的 Agent 列表（可以是多个！）
            - parallel: 是否并行执行
        """
        # 构建提示
        context_str = str(context) if context else "No context"
        
        # 调用 LLM 进行分类
        response = self.chain.invoke({
            "query": query,
            "context": context_str
        })
        
        # 解析响应
        result = self._parse_response(response.content)
        
        # 创建 RoutingDecision
        return RoutingDecision(
            intent=result.get("intent", "unknown"),
            agents_to_invoke=[
                AgentType(agent.lower().replace("agent", ""))
                for agent in result.get("agents_to_invoke", [])
            ],
            parallel=result.get("parallel", False),
            confidence=float(result.get("confidence", 0.0)),
            reasoning=result.get("reasoning", ""),
            parameters=result.get("parameters", {})
        )
    
    def _parse_response(self, content: str) -> Dict:
        """
        解析 LLM 响应
        
        Args:
            content: LLM 返回的文本内容
            
        Returns:
            解析后的字典
        """
        try:
            # 尝试直接解析 JSON
            return json.loads(content)
        except json.JSONDecodeError:
            # 尝试提取 JSON 部分
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            # 如果还是失败，返回默认值
            return {
                "intent": "unknown",
                "agents_to_invoke": [],
                "parallel": False,
                "confidence": 0.0,
                "reasoning": "Failed to parse response",
                "parameters": {}
            }
    
    def classify_simple(self, query: str) -> str:
        """
        简单分类（用于快速判断）
        
        Args:
            query: 用户查询
            
        Returns:
            意图类型字符串
        """
        # 快速规则匹配
        query_lower = query.lower()
        
        # 闲聊
        if any(kw in query_lower for kw in ["你好", "谢谢", "再见", "你是谁", "聊聊天"]):
            return "chat"
        
        # 联网搜索
        if any(kw in query_lower for kw in ["最新", "今天", "本周", "实时", "新闻", "天气", "股票"]):
            return "web_search"
        
        # 混合搜索
        if any(kw in query_lower for kw in ["结合", "和网上", "+ 网上", "内部文档和网上"]):
            return "hybrid_search"
        
        # 本地搜索
        if any(kw in query_lower for kw in ["公司文档", "本地", "我们的", "内部"]):
            return "local_search"
        
        # 未知
        return "unknown"
