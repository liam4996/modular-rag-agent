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

from .tools.json_parser import safe_parse_json_with_default


class AgentType(Enum):
    """智能体类型"""
    CHAT = "chat"
    SEARCH = "search"
    WEB = "web"
    EVAL = "eval"
    REFINE = "refine"
    GENERATE = "generate"
    FINANCE_DATA = "finance_data"
    FINANCE_COMPUTE = "finance_compute"


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
    needs_local: bool = False
    needs_web: bool = False
    complexity: str = "simple"
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

Classify the user query into FOUR dimensions:

1. intent
   - chat: simple chitchat / general conversation
   - fact_query: direct factual question
   - document_qa: asks about uploaded/local documents or internal knowledge
   - summarization: asks to summarize a paper, file, report, or document
   - comparison: compare two methods / products / documents / viewpoints
   - analysis: asks for deeper judgment, evaluation, recommendation, tradeoff analysis
   - financial_analysis: asks to analyze financial reports, stock performance, company fundamentals,
     or make investment/trading judgments (signals: "分析财报", "PE", "ROE", "毛利率", "估值",
     "投资价值", "行业对比", "财务指标", "营收", "利润", "K线", "行情")
   - financial_market: asks for real-time market data, stock prices, or market indices
     (signals: "股价", "涨跌", "市值", "换手率", "市盈率", "市净率", "港股", "美股", "大盘")
   - financial_report: asks to generate a structured financial report or visualization
     (signals: "生成报告", "画图", "可视化", "K线图", "对比图", "图表", "出具报告")
   - unknown: impossible or unclear request

2. needs_local
   - true if the answer should use local knowledge base / uploaded files / internal docs
   - signals: "根据文档", "上传的 PDF", "内部资料", "本地知识库", paper/file names

3. needs_web
   - true if the answer needs public/latest/external information from the internet
   - signals: "最新", "今天", "本周", "实时", "新闻", "2026", "官网", "公开资料", "竞品"

4. complexity
   - simple: single-hop, direct answer likely enough
   - medium: may need synthesis or comparison, but usually one retrieval round is enough
   - complex: multi-step reasoning, comparison + judgment, planning, or local + web synthesis
   - financial financial_* intents are almost always "complex"

Routing guidance:
- chat usually means needs_local=false and needs_web=false
- document_qa / summarization usually means needs_local=true
- latest/public/current/trend/comparison with outside world often means needs_web=true
- if the query explicitly asks to combine internal/local docs with web/public/latest info, set both needs_local=true and needs_web=true
- comparison / analysis involving "latest", "industry", "public", "research progress" is often complex
- financial_analysis: set needs_local=true, complexity=complex, agents=["SearchAgent", "FinanceDataAgent"]
- financial_market: set needs_web=true, complexity=simple, agents=["FinanceDataAgent"]
- financial_report: set needs_local=true, complexity=complex, agents=["SearchAgent", "FinanceDataAgent"]

Respond ONLY in JSON:
{{
    "intent": "analysis",
    "needs_local": true,
    "needs_web": true,
    "complexity": "complex",
    "agents_to_invoke": ["SearchAgent", "WebAgent"],
    "parallel": true,
    "confidence": 0.92,
    "reasoning": "The question requires both uploaded documents and latest public information.",
    "parameters": {{}}
}}"""
    
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT + "\n\nCurrent datetime: {current_datetime}"),
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
        context_str = str(context) if context else "No context"

        from datetime import datetime
        now = datetime.now()
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        current_dt = f"{now.strftime('%Y-%m-%d')} {weekdays[now.weekday()]} {now.strftime('%H:%M')}"

        response = self.chain.invoke({
            "query": query,
            "context": context_str,
            "current_datetime": current_dt,
        })
        
        # 解析响应
        result = self._parse_response(response.content)
        
        # 创建 RoutingDecision
        return RoutingDecision(
            intent=result.get("intent", "unknown"),
            agents_to_invoke=self._build_agents_to_invoke(result),
            needs_local=bool(result.get("needs_local", False)),
            needs_web=bool(result.get("needs_web", False)),
            complexity=str(result.get("complexity", "simple")).lower(),
            parallel=result.get("parallel", False),
            confidence=float(result.get("confidence", 0.0)),
            reasoning=result.get("reasoning", ""),
            parameters=result.get("parameters", {})
        )
    
    def _parse_response(self, content: str) -> Dict:
        """解析 LLM 响应为字典"""
        return safe_parse_json_with_default(content, {
            "intent": "unknown",
            "needs_local": False,
            "needs_web": False,
            "complexity": "simple",
            "agents_to_invoke": [],
            "parallel": False,
            "confidence": 0.0,
            "reasoning": "Failed to parse response",
            "parameters": {}
        })

    def _build_agents_to_invoke(self, result: Dict[str, Any]) -> List[AgentType]:
        """Build invoked agents from explicit source flags, with backward compatibility."""
        needs_local = bool(result.get("needs_local", False))
        needs_web = bool(result.get("needs_web", False))
        intent = str(result.get("intent", "unknown")).lower()

        agents: List[AgentType] = []
        if needs_local:
            agents.append(AgentType.SEARCH)
        if needs_web:
            agents.append(AgentType.WEB)

        if agents:
            return agents

        # Backward compatibility with older prompt outputs / fallback parsing
        for agent in result.get("agents_to_invoke", []):
            try:
                parsed = AgentType(agent.lower().replace("agent", ""))
                if parsed not in agents:
                    agents.append(parsed)
            except ValueError:
                continue

        if agents:
            return agents

        if intent in ("document_qa", "summarization", "local_search"):
            return [AgentType.SEARCH]
        if intent in ("web_search",):
            return [AgentType.WEB]
        if intent in ("comparison", "analysis", "hybrid_search"):
            return [AgentType.SEARCH, AgentType.WEB]
        return []
    
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
