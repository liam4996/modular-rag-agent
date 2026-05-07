"""Tushare A 股数据客户端。

作为 A 股主数据源，akshare 作为备选降级。

API Token: 从 settings.yaml 或环境变量 TUSHARE_API_TOKEN 读取
"""

from __future__ import annotations

import functools
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def _retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
) -> Callable[[F], F]:

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = base_delay * (backoff_factor ** attempt)
                        logger.warning(
                            "%s 第 %d/%d 次失败: %s，%.1fs 后重试...",
                            func.__name__,
                            attempt + 1,
                            max_retries,
                            e,
                            delay,
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "%s 重试 %d 次后仍失败: %s",
                            func.__name__,
                            max_retries,
                            e,
                        )
            raise last_exception  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator


class TushareClient:
    """
    Tushare A 股数据客户端。
    
    作为 A 股主数据源，akshare 作为备选降级。
    """

    def __init__(self, token: Optional[str] = None):
        self._token = token or os.getenv("TUSHARE_API_TOKEN")
        self._api = None
        self._akshare_available: Optional[bool] = None
        if not self._token:
            logger.warning("TUSHARE_API_TOKEN 未设置！A股数据将使用后备方案。请设置环境变量 TUSHARE_API_TOKEN 或在 .env 文件中配置。")

    @property
    def api(self):
        if self._api is None:
            try:
                import tushare as ts
                self._api = ts.pro_api(self._token)
            except ImportError:
                logger.warning("tushare not installed, A-share data unavailable")
                self._api = None
        return self._api

    @property
    def available(self) -> bool:
        return self.api is not None

    @property
    def akshare_available(self) -> bool:
        if self._akshare_available is None:
            try:
                import akshare as ak  # noqa: F401
                self._akshare_available = True
            except ImportError:
                logger.warning("akshare not installed, fallback data unavailable")
                self._akshare_available = False
        return self._akshare_available

    # ==================================================================
    # Public methods (Tushare, with retry)
    # ==================================================================

    @_retry_with_backoff(max_retries=3, base_delay=1.0, backoff_factor=2.0)
    def fetch_quote(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Tushare 实时行情（含自动重试）"""
        if not self.available:
            return []

        results = []
        clean_symbols = {s.replace(".SZ", "").replace(".SH", "") for s in symbols}
        for code in clean_symbols:
            try:
                market = "SZ" if code.startswith(("0", "3")) else "SH"
                df = self.api.daily(ts_code=f"{code}.{market}")
                if df is not None and not df.empty:
                    row = df.iloc[0]
                    results.append({
                        "symbol": f"{code}.{market}",
                        "name": "",
                        "price": _safe_float(row.get("close")),
                        "change_pct": _safe_float(row.get("pct_chg")),
                        "volume": _safe_float(row.get("vol")),
                        "currency": "CNY",
                        "_source": "tushare",
                    })
            except Exception as e:
                logger.warning(f"Tushare quote failed for {code}: {e}")
        return results

    @_retry_with_backoff(max_retries=3, base_delay=1.0, backoff_factor=2.0)
    def fetch_fundamentals(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Tushare 基本面数据（含自动重试）"""
        if not self.available:
            return []

        results = []
        clean_symbols = {s.replace(".SZ", "").replace(".SH", "") for s in symbols}
        for code in clean_symbols:
            try:
                market = "SZ" if code.startswith(("0", "3")) else "SH"
                df = self.api.fina_indicator(ts_code=f"{code}.{market}", limit=1)
                if df is not None and not df.empty:
                    row = df.iloc[0]
                    results.append({
                        "symbol": f"{code}.{market}",
                        "pe": _safe_float(row.get("pe")),
                        "pb": _safe_float(row.get("pb")),
                        "roe": _safe_float(row.get("roe")),
                        "eps": _safe_float(row.get("eps")),
                        "revenue": _safe_float(row.get("revenue")),
                        "net_profit": _safe_float(row.get("net_profit")),
                        "gross_margin": _safe_float(row.get("gross_margin")),
                        "debt_to_asset": _safe_float(row.get("debt_to_assets")),
                        "currency": "CNY",
                        "_source": "tushare",
                    })
            except Exception as e:
                logger.warning(f"Tushare fundamentals failed for {code}: {e}")
        return results

    @_retry_with_backoff(max_retries=3, base_delay=1.0, backoff_factor=2.0)
    def fetch_history(self, symbols: List[str], period: str) -> List[Dict[str, Any]]:
        """Tushare 历史 K 线数据（含自动重试）"""
        if not self.available:
            return []

        from datetime import datetime, timedelta

        period_days = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "3y": 1095, "5y": 1825}
        end = datetime.now()
        start = end - timedelta(days=period_days.get(period, 90))

        results = []
        clean_symbols = {s.replace(".SZ", "").replace(".SH", "") for s in symbols}
        for code in clean_symbols:
            try:
                market = "SZ" if code.startswith(("0", "3")) else "SH"
                df = self.api.daily(
                    ts_code=f"{code}.{market}",
                    start_date=start.strftime("%Y%m%d"),
                    end_date=end.strftime("%Y%m%d"),
                )
                if df is not None and not df.empty:
                    data = []
                    for _, row in df.iterrows():
                        data.append({
                            "date": str(row.get("trade_date", "")),
                            "open": _safe_float(row.get("open")),
                            "high": _safe_float(row.get("high")),
                            "low": _safe_float(row.get("low")),
                            "close": _safe_float(row.get("close")),
                            "volume": _safe_float(row.get("vol")),
                        })
                    results.append({
                        "symbol": f"{code}.{market}",
                        "period": period,
                        "data": data,
                        "data_count": len(data),
                        "_source": "tushare",
                    })
            except Exception as e:
                logger.warning(f"Tushare history failed for {code}: {e}")
        return results

    # ==================================================================
    # akshare fallback methods
    # ==================================================================

    def fetch_quote_akshare(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """akshare 实时行情（降级后备）"""
        if not self.akshare_available:
            logger.warning("akshare 不可用，无法获取行情数据")
            return []

        try:
            import akshare as ak
        except ImportError:
            logger.warning("akshare 未安装，无法使用备选行情源")
            return []

        logger.warning("使用 akshare 作为备选行情源")

        results: List[Dict[str, Any]] = []
        clean_symbols = {s.replace(".SZ", "").replace(".SH", "") for s in symbols}

        try:
            df = ak.stock_zh_a_spot_em()
            if df is None or df.empty:
                logger.warning("akshare stock_zh_a_spot_em 返回空数据")
                return results

            code_col = _find_col(df, ["代码", "code"])
            name_col = _find_col(df, ["名称", "name"])
            price_col = _find_col(df, ["最新价", "latest"])
            pct_col = _find_col(df, ["涨跌幅", "pct_chg"])
            volume_col = _find_col(df, ["成交量", "volume"])

            if code_col is None:
                logger.warning("akshare 行情数据中未找到代码列")
                return results

            for code in clean_symbols:
                match = df[df[code_col].astype(str).str.contains(code, na=False)]
                if not match.empty:
                    row = match.iloc[0]
                    results.append({
                        "symbol": code,
                        "name": str(row[name_col]) if name_col else "",
                        "price": _safe_float(row[price_col]) if price_col else None,
                        "change_pct": _safe_float(row[pct_col]) if pct_col else None,
                        "volume": _safe_float(row[volume_col]) if volume_col else None,
                        "currency": "CNY",
                        "_source": "akshare",
                    })
        except Exception as e:
            logger.warning(f"akshare 行情获取失败: {e}")

        return results

    def fetch_fundamentals_akshare(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """akshare 基本面数据（降级后备）"""
        if not self.akshare_available:
            logger.warning("akshare 不可用，无法获取基本面数据")
            return []

        try:
            import akshare as ak
        except ImportError:
            logger.warning("akshare 未安装，无法使用备选基本面源")
            return []

        logger.warning("使用 akshare 作为备选基本面源")

        results: List[Dict[str, Any]] = []
        clean_symbols = {s.replace(".SZ", "").replace(".SH", "") for s in symbols}

        for code in clean_symbols:
            try:
                df = ak.stock_financial_abstract_ths(symbol=code, indicator="按报告期")
                if df is not None and not df.empty:
                    row = df.iloc[0]
                    results.append({
                        "symbol": code,
                        "pe": _safe_float(row.get("市盈率")),
                        "pb": _safe_float(row.get("市净率")),
                        "roe": _safe_float(row.get("净资产收益率")),
                        "eps": _safe_float(row.get("每股收益")),
                        "revenue": _safe_float(row.get("营业总收入")),
                        "net_profit": _safe_float(row.get("净利润")),
                        "gross_margin": _safe_float(row.get("毛利率")),
                        "debt_to_asset": _safe_float(row.get("资产负债率")),
                        "currency": "CNY",
                        "_source": "akshare",
                    })
            except Exception as e:
                logger.warning(f"akshare 基本面获取失败 for {code}: {e}")

        if not results:
            try:
                df = ak.stock_a_lg_indicator(symbol="all")
                if df is not None and not df.empty:
                    code_col = _find_col(df, ["code", "代码", "symbol"])
                    if code_col is not None:
                        for code in clean_symbols:
                            match = df[df[code_col].astype(str).str.contains(code, na=False)]
                            if not match.empty:
                                row = match.iloc[0]
                                results.append({
                                    "symbol": code,
                                    "pe": _safe_float(row.get("pe")),
                                    "pb": _safe_float(row.get("pb")),
                                    "roe": _safe_float(row.get("roe")),
                                    "eps": _safe_float(row.get("eps")),
                                    "revenue": _safe_float(row.get("revenue")),
                                    "net_profit": _safe_float(row.get("net_profit")),
                                    "gross_margin": _safe_float(row.get("gross_margin")),
                                    "debt_to_asset": _safe_float(row.get("debt_to_assets")),
                                    "currency": "CNY",
                                    "_source": "akshare",
                                })
            except Exception as e:
                logger.warning(f"akshare stock_a_lg_indicator 获取失败: {e}")

        return results

    # ==================================================================
    # Unified fallback entry
    # ==================================================================

    def fetch_with_fallback(
        self,
        data_type: str,
        symbols: List[str],
        period: Optional[str] = None,
    ) -> Dict[str, Any]:
        trace: List[Dict[str, Any]] = []
        data: List[Dict[str, Any]] = []
        source: str = "none"

        tushare_start = time.time()
        try:
            if data_type == "quote":
                data = self.fetch_quote(symbols)
            elif data_type == "fundamentals":
                data = self.fetch_fundamentals(symbols)
            elif data_type == "history":
                if period is None:
                    raise ValueError("data_type='history' 时必须提供 period 参数")
                data = self.fetch_history(symbols, period)
            else:
                raise ValueError(f"未知 data_type: {data_type}")

            if data:
                source = "tushare"
        except Exception as e:
            logger.warning(f"Tushare {data_type} 全部失败: {e}")

        tushare_elapsed = time.time() - tushare_start
        trace.append({
            "provider": "tushare",
            "elapsed_s": round(tushare_elapsed, 3),
            "result_count": len(data),
            "success": source == "tushare",
        })

        if source != "tushare":
            akshare_start = time.time()
            try:
                if data_type == "quote":
                    data = self.fetch_quote_akshare(symbols)
                elif data_type == "fundamentals":
                    data = self.fetch_fundamentals_akshare(symbols)
                elif data_type == "history":
                    logger.warning("akshare 暂不支持历史K线回退")
                    data = []

                if data:
                    source = "akshare"
            except Exception as e:
                logger.warning(f"akshare {data_type} 也失败: {e}")

            akshare_elapsed = time.time() - akshare_start
            trace.append({
                "provider": "akshare",
                "elapsed_s": round(akshare_elapsed, 3),
                "result_count": len(data),
                "success": source == "akshare",
            })

        return {
            "data": data,
            "source": source,
            "trace": trace,
        }


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


def _find_col(df: Any, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None
