"""MCP Tool: query_market_data

Unified financial market data query interface.
Covers A-shares (akshare), US stocks and HK stocks (yfinance).

Usage via MCP:
    Tool name: query_market_data
    Input schema:
        - symbols (array[string], required): Stock symbols
        - data_types (array[string], required): quote/fundamentals/history/industry_comparison
        - period (string, optional): History period (default: 3mo)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from mcp import types

if TYPE_CHECKING:
    from src.mcp_server.protocol_handler import ProtocolHandler
    from src.core.settings import Settings

logger = logging.getLogger(__name__)

TOOL_NAME = "query_market_data"
TOOL_DESCRIPTION = """Query real-time and historical financial market data.

Unified interface for A-shares, US stocks, and HK stocks.

Data types:
- quote: Real-time price, change%, volume, market cap
- fundamentals: PE, PB, ROE, revenue, profit, debt ratio
- history: Historical OHLCV data (K-line)
- industry_comparison: Industry average metrics for comparison

Parameters:
- symbols: List of stock symbols (e.g. ["300750.SZ", "AAPL", "0700.HK"])
- data_types: Types of data to fetch
- period: History data period (1mo/3mo/6mo/1y/3y/5y, default: 3mo)
"""

TOOL_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "symbols": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Stock symbols to query. A-share: '000001.SZ' or '600000.SH'. US: 'AAPL'. HK: '0700.HK'.",
        },
        "data_types": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["quote", "fundamentals", "history", "industry_comparison"],
            },
            "description": "Types of data to fetch.",
        },
        "period": {
            "type": "string",
            "enum": ["1mo", "3mo", "6mo", "1y", "3y", "5y"],
            "description": "History data period.",
            "default": "3mo",
        },
    },
    "required": ["symbols", "data_types"],
}


class Market(Enum):
    A_SHARE = "a_share"
    US = "us"
    HK = "hk"
    UNKNOWN = "unknown"


@dataclass
class QueryMarketDataConfig:
    default_period: str = "3mo"
    request_timeout: int = 15


@dataclass
class MarketQuote:
    symbol: str
    name: str = ""
    price: Optional[float] = None
    change_pct: Optional[float] = None
    volume: Optional[float] = None
    market_cap: Optional[float] = None
    currency: str = "CNY"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "price": self.price,
            "change_pct": self.change_pct,
            "volume": self.volume,
            "market_cap": self.market_cap,
            "currency": self.currency,
        }


@dataclass
class MarketFundamentals:
    symbol: str
    pe: Optional[float] = None
    pb: Optional[float] = None
    ps: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    gross_margin: Optional[float] = None
    net_margin: Optional[float] = None
    debt_to_asset: Optional[float] = None
    current_ratio: Optional[float] = None
    revenue: Optional[float] = None
    net_profit: Optional[float] = None
    total_assets: Optional[float] = None
    total_equity: Optional[float] = None
    eps: Optional[float] = None
    bvps: Optional[float] = None
    currency: str = "CNY"
    report_period: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result = {"symbol": self.symbol, "currency": self.currency}
        if self.report_period:
            result["report_period"] = self.report_period
        for field_name in [
            "pe", "pb", "ps", "roe", "roa", "gross_margin", "net_margin",
            "debt_to_asset", "current_ratio", "revenue", "net_profit",
            "total_assets", "total_equity", "eps", "bvps",
        ]:
            val = getattr(self, field_name, None)
            if val is not None:
                result[field_name] = val
        return result


@dataclass
class MarketHistory:
    symbol: str
    period: str = ""
    data: List[Dict[str, Any]] = field(default_factory=list)
    data_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "period": self.period,
            "data_count": self.data_count,
            "data": self.data,
        }


class QueryMarketDataTool:
    """MCP Tool for querying financial market data.

    Data sources:
    - A-shares: akshare (free)
    - US/HK stocks: yfinance (free)

    Returns unified schema regardless of underlying source.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        config: Optional[QueryMarketDataConfig] = None,
    ) -> None:
        self._settings = settings
        self._config = config or QueryMarketDataConfig()

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            from src.core.settings import load_settings
            self._settings = load_settings()
        return self._settings

    @staticmethod
    def classify_symbol(symbol: str) -> Market:
        upper = symbol.upper()
        if upper.endswith(".SZ") or upper.endswith(".SH"):
            return Market.A_SHARE
        if upper.endswith(".HK"):
            return Market.HK
        if "." not in upper and upper.isalpha():
            if len(upper) <= 5:
                return Market.US
        return Market.UNKNOWN

    def _fetch_a_share_quote(self, symbols: List[str]) -> List[MarketQuote]:
        try:
            import akshare as ak
        except ImportError:
            logger.warning("akshare not installed, skipping A-share quote")
            return []

        results = []
        try:
            df = ak.stock_zh_a_spot_em()
            if df is None or df.empty:
                return results
            clean_symbols = {s.replace(".SZ", "").replace(".SH", "") for s in symbols}
            for _, row in df.iterrows():
                code = str(row.get("代码", ""))
                if code in clean_symbols:
                    suffix = ".SZ" if code.startswith(("0", "3")) else ".SH"
                    results.append(MarketQuote(
                        symbol=f"{code}{suffix}",
                        name=str(row.get("名称", "")),
                        price=_safe_float(row.get("最新价")),
                        change_pct=_safe_float(row.get("涨跌幅")),
                        volume=_safe_float(row.get("成交量")),
                        market_cap=_safe_float(row.get("总市值")),
                        currency="CNY",
                    ))
        except Exception as e:
            logger.warning(f"A-share quote failed: {e}")
        return results

    def _fetch_a_share_fundamentals(self, symbols: List[str]) -> List[MarketFundamentals]:
        try:
            import akshare as ak
        except ImportError:
            return []

        results = []
        clean_symbols = {s.replace(".SZ", "").replace(".SH", "") for s in symbols}
        for code in clean_symbols:
            try:
                df = ak.stock_financial_abstract_ths(symbol=code, indicator="按报告期")
                if df is None or df.empty:
                    continue
                row = df.iloc[0]
                suffix = ".SZ" if code.startswith(("0", "3")) else ".SH"
                results.append(MarketFundamentals(
                    symbol=f"{code}{suffix}",
                    pe=_safe_float(row.get("市盈率")),
                    pb=_safe_float(row.get("市净率")),
                    roe=_safe_float(row.get("净资产收益率")),
                    gross_margin=_safe_float(row.get("销售毛利率")),
                    net_margin=_safe_float(row.get("销售净利率")),
                    revenue=_safe_float(row.get("营业总收入")),
                    net_profit=_safe_float(row.get("净利润")),
                    eps=_safe_float(row.get("基本每股收益")),
                    currency="CNY",
                    report_period=str(row.get("报告期", "")),
                ))
            except Exception as e:
                logger.warning(f"A-share fundamentals failed for {code}: {e}")
        return results

    def _fetch_a_share_history(self, symbols: List[str], period: str) -> List[MarketHistory]:
        try:
            import akshare as ak
        except ImportError:
            return []

        period_map = {"1mo": "daily", "3mo": "daily", "6mo": "daily", "1y": "weekly", "3y": "weekly", "5y": "monthly"}
        freq = period_map.get(period, "daily")

        results = []
        clean_symbols = {s.replace(".SZ", "").replace(".SH", "") for s in symbols}
        for code in clean_symbols:
            try:
                df = ak.stock_zh_a_hist(symbol=code, period=freq, adjust="qfq")
                if df is None or df.empty:
                    continue
                data = []
                for _, row in df.tail(120).iterrows():
                    data.append({
                        "date": str(row.get("日期", "")),
                        "open": _safe_float(row.get("开盘")),
                        "high": _safe_float(row.get("最高")),
                        "low": _safe_float(row.get("最低")),
                        "close": _safe_float(row.get("收盘")),
                        "volume": _safe_float(row.get("成交量")),
                    })
                suffix = ".SZ" if code.startswith(("0", "3")) else ".SH"
                results.append(MarketHistory(
                    symbol=f"{code}{suffix}",
                    period=period,
                    data=data,
                    data_count=len(data),
                ))
            except Exception as e:
                logger.warning(f"A-share history failed for {code}: {e}")
        return results

    def _fetch_a_share_industry_comparison(self, symbols: List[str]) -> Dict[str, Any]:
        try:
            import akshare as ak
        except ImportError:
            return {}

        result = {}
        for symbol in symbols:
            try:
                code = symbol.replace(".SZ", "").replace(".SH", "")
                df = ak.stock_individual_info_em(symbol=code)
                if df is not None and not df.empty:
                    info = {}
                    for _, row in df.iterrows():
                        info[str(row.get("item", ""))] = str(row.get("value", ""))
                    result[symbol] = info
            except Exception as e:
                logger.warning(f"Industry comparison failed for {symbol}: {e}")
        return result

    def _fetch_yfinance_quote(self, symbols: List[str]) -> List[MarketQuote]:
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance not installed, skipping US/HK quote")
            return []

        results = []
        clean_symbols = [s for s in symbols if self.classify_symbol(s) in (Market.US, Market.HK)]
        if not clean_symbols:
            return results

        try:
            tickers = yf.Tickers(" ".join(clean_symbols))
            for sym in clean_symbols:
                try:
                    t = tickers.tickers.get(sym)
                    if t is None:
                        continue
                    info = t.info or {}
                    fast = t.fast_info or {}
                    cur = info.get("currency", "USD") if info else "USD"
                    results.append(MarketQuote(
                        symbol=sym,
                        name=info.get("longName", info.get("shortName", "")),
                        price=_safe_float(fast.get("last_price") if fast else info.get("currentPrice")),
                        change_pct=_safe_float(info.get("regularMarketChangePercent")),
                        volume=_safe_float(fast.get("last_volume") if fast else info.get("volume")),
                        market_cap=_safe_float(info.get("marketCap")),
                        currency=cur,
                    ))
                except Exception as e:
                    logger.warning(f"yfinance quote failed for {sym}: {e}")
        except Exception as e:
            logger.warning(f"yfinance batch query failed: {e}")
        return results

    def _fetch_yfinance_fundamentals(self, symbols: List[str]) -> List[MarketFundamentals]:
        try:
            import yfinance as yf
        except ImportError:
            return []

        results = []
        clean_symbols = [s for s in symbols if self.classify_symbol(s) in (Market.US, Market.HK)]
        if not clean_symbols:
            return results

        for sym in clean_symbols:
            try:
                t = yf.Ticker(sym)
                info = t.info or {}
                cur = info.get("currency", "USD")
                results.append(MarketFundamentals(
                    symbol=sym,
                    pe=_safe_float(info.get("trailingPE") or info.get("forwardPE")),
                    pb=_safe_float(info.get("priceToBook")),
                    ps=_safe_float(info.get("priceToSalesTrailing12Months")),
                    roe=_safe_float(info.get("returnOnEquity")),
                    roa=_safe_float(info.get("returnOnAssets")),
                    gross_margin=_safe_float(info.get("grossMargins")),
                    net_margin=_safe_float(info.get("profitMargins")),
                    debt_to_asset=_safe_float(info.get("debtToEquity")),
                    current_ratio=_safe_float(info.get("currentRatio")),
                    revenue=_safe_float(info.get("totalRevenue")),
                    net_profit=_safe_float(info.get("netIncomeToCommon")),
                    total_assets=_safe_float(info.get("totalAssets")),
                    total_equity=_safe_float(info.get("totalStockholderEquity")),
                    eps=_safe_float(info.get("trailingEps")),
                    bvps=_safe_float(info.get("bookValue")),
                    currency=cur,
                ))
            except Exception as e:
                logger.warning(f"yfinance fundamentals failed for {sym}: {e}")
        return results

    def _fetch_yfinance_history(self, symbols: List[str], period: str) -> List[MarketHistory]:
        try:
            import yfinance as yf
        except ImportError:
            return []

        results = []
        clean_symbols = [s for s in symbols if self.classify_symbol(s) in (Market.US, Market.HK)]
        if not clean_symbols:
            return results

        for sym in clean_symbols:
            try:
                t = yf.Ticker(sym)
                df = t.history(period=period)
                if df is None or df.empty:
                    continue
                data = []
                for idx, row in df.tail(120).iterrows():
                    data.append({
                        "date": str(idx.date()) if hasattr(idx, "date") else str(idx),
                        "open": _safe_float(row.get("Open")),
                        "high": _safe_float(row.get("High")),
                        "low": _safe_float(row.get("Low")),
                        "close": _safe_float(row.get("Close")),
                        "volume": _safe_float(row.get("Volume")),
                    })
                results.append(MarketHistory(
                    symbol=sym,
                    period=period,
                    data=data,
                    data_count=len(data),
                ))
            except Exception as e:
                logger.warning(f"yfinance history failed for {sym}: {e}")
        return results

    async def execute(
        self,
        symbols: List[str],
        data_types: List[str],
        period: str = "3mo",
    ) -> types.CallToolResult:
        logger.info(
            f"Executing query_market_data "
            f"(symbols={symbols}, data_types={data_types}, period={period})"
        )

        a_share_symbols = [s for s in symbols if self.classify_symbol(s) == Market.A_SHARE]
        global_symbols = [s for s in symbols if self.classify_symbol(s) in (Market.US, Market.HK)]

        result: Dict[str, Any] = {"symbols": symbols, "quote": [], "fundamentals": [], "history": [], "industry_comparison": {}}

        if "quote" in data_types:
            if a_share_symbols:
                quote_a = await asyncio.to_thread(self._fetch_a_share_quote, a_share_symbols)
                result["quote"].extend(q.to_dict() for q in quote_a)
            if global_symbols:
                quote_g = await asyncio.to_thread(self._fetch_yfinance_quote, global_symbols)
                result["quote"].extend(q.to_dict() for q in quote_g)

        if "fundamentals" in data_types:
            if a_share_symbols:
                fund_a = await asyncio.to_thread(self._fetch_a_share_fundamentals, a_share_symbols)
                result["fundamentals"].extend(f.to_dict() for f in fund_a)
            if global_symbols:
                fund_g = await asyncio.to_thread(self._fetch_yfinance_fundamentals, global_symbols)
                result["fundamentals"].extend(f.to_dict() for f in fund_g)

        if "history" in data_types:
            if a_share_symbols:
                hist_a = await asyncio.to_thread(self._fetch_a_share_history, a_share_symbols, period)
                result["history"].extend(h.to_dict() for h in hist_a)
            if global_symbols:
                hist_g = await asyncio.to_thread(self._fetch_yfinance_history, global_symbols, period)
                result["history"].extend(h.to_dict() for h in hist_g)

        if "industry_comparison" in data_types:
            if a_share_symbols:
                ind_a = await asyncio.to_thread(self._fetch_a_share_industry_comparison, a_share_symbols)
                result["industry_comparison"] = ind_a

        response_text = self._format_response(result, data_types)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=response_text)],
            isError=False,
        )

    def _format_response(self, result: Dict[str, Any], data_types: List[str]) -> str:
        lines = [f"## 金融市场数据\n"]
        lines.append(f"查询标的: {', '.join(result['symbols'])}\n")

        if result["quote"]:
            lines.append("### 实时行情\n")
            lines.append("| 代码 | 名称 | 最新价 | 涨跌幅 | 市值 |")
            lines.append("|------|------|--------|--------|------|")
            for q in result["quote"]:
                price = f"{q['price']:.2f}" if q.get("price") else "-"
                change = f"{q['change_pct']:+.2f}%" if q.get("change_pct") is not None else "-"
                cap = _fmt_cap(q.get("market_cap"))
                lines.append(f"| {q['symbol']} | {q.get('name', '-')} | {price} | {change} | {cap} |")
            lines.append("")

        if result["fundamentals"]:
            lines.append("### 基本面数据\n")
            for f in result["fundamentals"]:
                lines.append(f"**{f['symbol']}** ({f.get('report_period', '-')})")
                items = []
                if f.get("pe") is not None:
                    items.append(f"PE: {f['pe']:.2f}")
                if f.get("pb") is not None:
                    items.append(f"PB: {f['pb']:.2f}")
                if f.get("roe") is not None:
                    items.append(f"ROE: {f['roe']:.2f}%")
                if f.get("gross_margin") is not None:
                    items.append(f"毛利率: {f['gross_margin']:.2f}%")
                if f.get("net_margin") is not None:
                    items.append(f"净利率: {f['net_margin']:.2f}%")
                if f.get("revenue") is not None:
                    items.append(f"营收: {_fmt_cap(f['revenue'])}")
                if f.get("net_profit") is not None:
                    items.append(f"净利润: {_fmt_cap(f['net_profit'])}")
                if f.get("eps") is not None:
                    items.append(f"EPS: {f['eps']:.2f}")
                if items:
                    lines.append(" | ".join(items))
                lines.append("")

        if result["history"]:
            lines.append("### 历史数据\n")
            for h in result["history"]:
                lines.append(f"**{h['symbol']}** ({h['data_count']} 条记录)")
            lines.append("")

        lines.append("---\n")
        import json
        lines.append(f"```json\n{json.dumps(result, ensure_ascii=False, indent=2)}\n```")

        return "\n".join(lines)


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        v = float(val)
        if v != v:
            return None
        return v
    except (ValueError, TypeError):
        return None


def _fmt_cap(val: Optional[float]) -> str:
    if val is None:
        return "-"
    v = float(val)
    if abs(v) >= 1e12:
        return f"{v / 1e12:.2f}万亿"
    if abs(v) >= 1e8:
        return f"{v / 1e8:.2f}亿"
    if abs(v) >= 1e4:
        return f"{v / 1e4:.2f}万"
    return f"{v:.2f}"


def register_tool(protocol_handler: ProtocolHandler) -> None:
    """Register the query_market_data tool with the protocol handler."""
    tool = QueryMarketDataTool()

    async def handler(
        symbols: List[str],
        data_types: List[str],
        period: str = "3mo",
    ) -> types.CallToolResult:
        return await tool.execute(
            symbols=symbols,
            data_types=data_types,
            period=period,
        )

    protocol_handler.register_tool(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=TOOL_INPUT_SCHEMA,
        handler=handler,
    )

    logger.info(f"Registered MCP tool: {TOOL_NAME}")
