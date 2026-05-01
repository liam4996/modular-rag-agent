"""
Financial Data Agent

Encapsulates the query_market_data MCP tool for use within
the multi-agent LangGraph system.

与 RAG Agent 的关系：
- RAG Agent 回答"为什么"（财报原文、研报观点）
- Finance Data Agent 回答"是什么"（股价、PE、行业均值）
- 两者结果汇合后交给 Business Compute Agent 做交叉分析
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.core.settings import Settings


class FinanceDataAgent:
    """
    Finance Data Agent — 实时行情与基本面数据获取。

    直接调用 MCP Tool `query_market_data`，拿到原始数据后做基本清洗，
    写入 blackboard["market_data"]。

    使用场景：
    - FINANCIAL_MARKET 意图
    - 配合 RAG Agent 做财报原文 + 行情交叉分析
    """

    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings
        self._tool = None

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            from src.core.settings import load_settings
            self._settings = load_settings()
        return self._settings

    @property
    def tool(self):
        if self._tool is None:
            from src.mcp_server.tools.query_market_data import QueryMarketDataTool
            self._tool = QueryMarketDataTool(settings=self.settings)
        return self._tool

    def query(
        self,
        symbols: List[str],
        data_types: Optional[List[str]] = None,
        period: str = "3mo",
    ) -> Dict[str, Any]:
        """
        执行金融数据查询。

        Args:
            symbols: 股票代码列表，如 ["300750.SZ", "AAPL"]
            data_types: 数据类型，默认全部（quote + fundamentals + history）
            period: 历史数据周期

        Returns:
            结构化金融数据 dict，写入 blackboard["market_data"]
        """
        effective_types = data_types or ["quote", "fundamentals", "history"]

        result: Dict[str, Any] = {
            "symbols": symbols,
            "quote": [],
            "fundamentals": [],
            "history": [],
            "industry_comparison": {},
        }

        for symbol in symbols:
            symbol_result = self._query_single(symbol, effective_types, period)
            self._merge_symbol_result(result, symbol_result)

        return result

    def _query_single(
        self,
        symbol: str,
        data_types: List[str],
        period: str,
    ) -> Dict[str, Any]:
        from src.mcp_server.tools.query_market_data import QueryMarketDataTool

        market = QueryMarketDataTool.classify_symbol(symbol)
        result: Dict[str, Any] = {"symbol": symbol, "quote": None, "fundamentals": None, "history": None}

        is_a_share = str(market).endswith("A_SHARE') or market.value == "a_share"

        if "quote" in data_types:
            if is_a_share:
                quotes = self.tool._fetch_a_share_quote([symbol])
                if quotes:
                    result["quote"] = quotes[0].to_dict()
            else:
                quotes = self.tool._fetch_yfinance_quote([symbol])
                if quotes:
                    result["quote"] = quotes[0].to_dict()

        if "fundamentals" in data_types:
            if is_a_share:
                funds = self.tool._fetch_a_share_fundamentals([symbol])
                if funds:
                    result["fundamentals"] = funds[0].to_dict()
            else:
                funds = self.tool._fetch_yfinance_fundamentals([symbol])
                if funds:
                    result["fundamentals"] = funds[0].to_dict()

        if "history" in data_types:
            if is_a_share:
                histories = self.tool._fetch_a_share_history([symbol], period)
                if histories:
                    result["history"] = histories[0].to_dict()
            else:
                histories = self.tool._fetch_yfinance_history([symbol], period)
                if histories:
                    result["history"] = histories[0].to_dict()

        return result

    def _merge_symbol_result(self, aggregate: Dict[str, Any], single: Dict[str, Any]) -> None:
        if single.get("quote"):
            aggregate["quote"].append(single["quote"])
        if single.get("fundamentals"):
            aggregate["fundamentals"].append(single["fundamentals"])
        if single.get("history"):
            aggregate["history"].append(single["history"])

    def query_and_store(
        self,
        state: Any,
        symbols: List[str],
        data_types: Optional[List[str]] = None,
        period: str = "3mo",
    ) -> Any:
        """
        查询金融数据并写入 AgentState.blackboard["market_data"]。

        Args:
            state: AgentState 实例
            symbols: 股票代码列表
            data_types: 数据类型
            period: 历史数据周期

        Returns:
            更新后的 AgentState
        """
        data = self.query(symbols=symbols, data_types=data_types, period=period)

        existing = state.blackboard.get("market_data", {})
        existing.update(data)
        state.add_to_blackboard("market_data", existing, "finance_data")

        state.add_execution_trace({
            "agent": "finance_data",
            "action": "query_market",
            "symbols": symbols,
            "data_types": data_types,
            "quote_count": len(data.get("quote", [])),
            "fundamentals_count": len(data.get("fundamentals", [])),
            "history_count": len(data.get("history", [])),
        })

        return state
