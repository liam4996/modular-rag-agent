"""
多智能体 RAG 系统 - Router Agent

职责：
- 意图识别
- 路由决策（支持并行路由）
- 复杂查询识别（如"结合内部文档和网上资料"）
- 金融意图识别（新增）
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
    FINANCE_DATA = "finance_data"
    FINANCE_COMPUTE = "finance_compute"


@dataclass
class RoutingDecision:
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
    - 识别 5 种意图 + 3 种金融意图
    - 返回 RoutingDecision
    - 支持并行标记
    - 复杂查询识别
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

3. needs_web
   - true if the answer needs public/latest/external information from the internet

4. complexity
   - simple: single-hop
   - medium: may need synthesis
   - complex: multi-step reasoning
   - financial financial_* intents are almost always "complex"

Routing guidance:
- chat usually means needs_local=false and needs_web=false
- document_qa / summarization usually means needs_local=true
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
    "reasoning": "...",
    "parameters": {{}}
}}"""
    
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT + "\n\nCurrent datetime: {current_datetime}"),
            ("user", "Query: {query}\nContext: {context}")
        ])
        self.chain = self.prompt | self.llm
    
    def classify(
        self,
        query: str,
        context: Optional[List[Dict[str, str]]] = None,
    ) -> RoutingDecision:
        from datetime import datetime
        current_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        context_str = json.dumps(context or [], ensure_ascii=False)
        
        try:
            response = self.chain.invoke({
                "query": query,
                "context": context_str,
                "current_datetime": current_dt,
            })
            content = response.content if hasattr(response, "content") else str(response)
            data = json.loads(content)
            
            agents_str = data.get("agents_to_invoke", ["SearchAgent"])
            agents = []
            for a in agents_str:
                try:
                    agents.append(AgentType(a.lower().replace("agent", "")))
                except ValueError:
                    pass
            
            return RoutingDecision(
                intent=data.get("intent", "fact_query"),
                agents_to_invoke=agents or [AgentType.SEARCH],
                needs_local=data.get("needs_local", True),
                needs_web=data.get("needs_web", False),
                complexity=data.get("complexity", "simple"),
                parallel=data.get("parallel", False),
                confidence=data.get("confidence", 0.5),
                reasoning=data.get("reasoning", ""),
                parameters=data.get("parameters", {}),
            )
        except Exception:
            return RoutingDecision(
                intent="fact_query",
                agents_to_invoke=[AgentType.SEARCH],
                needs_local=True,
                confidence=0.3,
            )
