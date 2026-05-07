"""联网获取财报文档并自动存入 RAG 向量库。

使用策略：
1. 通过 FinanceDataAgent（内部走 MCP query_market_data）获取结构化基本面数据
   - A 股: Tushare (主) → akshare (备) 自动降级
   - 美股/港股: yfinance 自动检测
2. 结构化数据不足时，通过 WebSearchAgent 搜索财报原文
3. 通过现有 Pipeline.run() 将文本文档入库

用途：当本地知识库检索不到目标公司财报时，自动从网络拉取并入库。
"""

from __future__ import annotations

import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.mcp_server.tools.query_market_data import QueryMarketDataTool, Market

logger = logging.getLogger(__name__)


class FinancialDocumentFetcher:
    """通过 MCP Tool (query_market_data) + WebSearchAgent 获取财报数据并入库 RAG。

    不直接调用 TushareClient — 所有数据获取走 MCP Tool 的统一降级链路。
    """

    def __init__(self, settings=None, finance_data_agent=None):
        self._settings = settings
        self._finance_agent = finance_data_agent
        self._web_agent = None

    @property
    def finance_agent(self):
        if self._finance_agent is None:
            from src.agent.multi_agent.finance_data_agent import FinanceDataAgent
            self._finance_agent = FinanceDataAgent(settings=self._settings)
        return self._finance_agent

    @property
    def web_agent(self):
        if self._web_agent is None:
            from src.agent.multi_agent.web_agent import WebSearchAgent
            self._web_agent = WebSearchAgent(self._settings)
        return self._web_agent

    def fetch_and_ingest(
        self,
        company: str,
        symbols: Optional[List[str]] = None,
        period: str = "3mo",
        state: Any = None,
    ) -> Dict[str, Any]:
        """
        获取目标公司财报并尝试存入 RAG 向量库。

        Args:
            company: 公司名称，如 "宁德时代"
            symbols: 股票代码列表，如 ["300750.SZ", "AAPL"]
            period: 历史数据周期
            state: AgentState（用于写入 execution_trace）

        Returns:
            {
                "success": bool,
                "source": "mcp_fundamentals" | "web_search" | "none",
                "market": "a_share" | "us" | "hk" | "unknown",
                "documents": [{"content": str, "source": str, "symbol": str}],
                "ingested": bool,
                "chunk_count": int,
            }
        """
        result: Dict[str, Any] = {
            "success": False,
            "source": "none",
            "market": "unknown",
            "documents": [],
            "ingested": False,
            "chunk_count": 0,
        }

        syms = symbols or []

        if state:
            state.add_execution_trace({
                "agent": "financial_fetcher",
                "action": "fetch_start",
                "company": company,
                "symbols": syms,
                "status": "running",
            })

        # ── 策略 1: MCP query_market_data（Tushare/yfinance → 自动降级） ──
        docs, market = self._fetch_via_mcp(syms)
        if docs:
            result["documents"] = docs
            result["source"] = "mcp_fundamentals"
            result["market"] = market
            result["success"] = True
        else:
            # ── 策略 2: WebSearchAgent 联网搜索 ──
            docs = self._fetch_via_web(company, syms)
            if docs:
                result["documents"] = docs
                result["source"] = "web_search"
                result["success"] = True

        # ── 存入 RAG 向量库 ──
        if result["documents"] and result["success"]:
            ingest_result = self._ingest_documents(
                result["documents"],
                collection="financial_reports",
                company=company,
            )
            result["ingested"] = ingest_result.get("ingested", False)
            result["chunk_count"] = ingest_result.get("chunk_count", 0)

        if state:
            state.add_execution_trace({
                "agent": "financial_fetcher",
                "action": "fetch_done",
                "source": result["source"],
                "market": result["market"],
                "doc_count": len(result["documents"]),
                "ingested": result["ingested"],
                "chunk_count": result["chunk_count"],
                "status": "done" if result["success"] else "error",
            })

        return result

    def _fetch_via_mcp(self, symbols: List[str]) -> tuple:
        """通过 FinanceDataAgent → MCP query_market_data 获取基本面数据。"""
        if not symbols:
            return [], "unknown"

        classified = {}
        for s in symbols:
            m = QueryMarketDataTool.classify_symbol(s)
            classified.setdefault(m, []).append(s)

        if Market.A_SHARE in classified:
            primary_market = "a_share"
            target_symbols = classified[Market.A_SHARE]
        elif Market.US in classified:
            primary_market = "us"
            target_symbols = classified[Market.US]
        elif Market.HK in classified:
            primary_market = "hk"
            target_symbols = classified[Market.HK]
        else:
            return [], "unknown"

        try:
            data = self.finance_agent.query(
                symbols=target_symbols,
                data_types=["fundamentals"],
            )
        except Exception as e:
            logger.warning(f"MCP fundamentals fetch failed: {e}")
            return [], primary_market

        fundamentals = data.get("fundamentals", [])
        if not fundamentals:
            return [], primary_market

        docs = []
        for f in fundamentals:
            if not isinstance(f, dict):
                continue
            sym = f.get("symbol", "unknown")
            content_parts = [
                f"# {sym} 基本面数据",
                f"数据来源: MCP query_market_data ({primary_market})",
                f"获取时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "",
                "## 估值指标",
                f"- 市盈率(PE): {_fmt_val(f.get('pe'))}",
                f"- 市净率(PB): {_fmt_val(f.get('pb'))}",
            ]
            if f.get("ps") is not None:
                content_parts.append(f"- 市销率(PS): {_fmt_val(f.get('ps'))}")

            content_parts.extend([
                "",
                "## 盈利能力",
                f"- 净资产收益率(ROE): {_fmt_val(f.get('roe'), '%')}",
            ])
            if f.get("roa") is not None:
                content_parts.append(f"- 总资产收益率(ROA): {_fmt_val(f.get('roa'), '%')}")
            content_parts.extend([
                f"- 毛利率: {_fmt_val(f.get('gross_margin'), '%')}",
                f"- 净利率: {_fmt_val(f.get('net_margin'), '%')}",
                f"- 每股收益(EPS): {_fmt_val(f.get('eps'))}",
                "",
                "## 财务规模",
                f"- 营业收入: {_fmt_val(f.get('revenue'))}",
                f"- 净利润: {_fmt_val(f.get('net_profit'))}",
            ])
            if f.get("total_assets") is not None:
                content_parts.append(f"- 总资产: {_fmt_val(f.get('total_assets'))}")
            if f.get("total_equity") is not None:
                content_parts.append(f"- 净资产: {_fmt_val(f.get('total_equity'))}")

            content_parts.extend([
                "",
                "## 偿债能力",
                f"- 资产负债率: {_fmt_val(f.get('debt_to_asset'), '%')}",
            ])
            if f.get("current_ratio") is not None:
                content_parts.append(f"- 流动比率: {_fmt_val(f.get('current_ratio'))}")

            if f.get("report_period"):
                content_parts.append(f"\n报告期: {f['report_period']}")

            docs.append({
                "content": "\n".join(content_parts),
                "source": f"mcp://{primary_market}/{sym}",
                "symbol": sym,
                "company": sym,
            })

        return docs, primary_market

    def _fetch_via_web(
        self, company: str, symbols: List[str]
    ) -> List[Dict[str, Any]]:
        """通过 WebSearchAgent 联网搜索财报信息。"""
        if not self.web_agent:
            logger.info("WebSearchAgent not available")
            return []

        queries = [
            f"{company} 最新季度财报 营收 利润 毛利率",
            f"{company} 财务报表 PE ROE 资产负债率",
        ]
        if symbols:
            queries.append(f"{' '.join(symbols)} 财务指标 财务数据")

        all_results = []
        for q in queries[:2]:
            try:
                results = self.web_agent.search(query=q, num_results=3, time_range="y")
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Web search failed for '{q}': {e}")

        if not all_results:
            return []

        doc = {
            "content": self._format_web_results(company, all_results),
            "source": f"web_search://{company}",
            "symbol": symbols[0] if symbols else "unknown",
            "company": company,
        }
        return [doc]

    @staticmethod
    def _format_web_results(company: str, results: List[Any]) -> str:
        parts = [
            f"# {company} 网络检索结果",
            f"获取时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
        ]
        for i, r in enumerate(results[:5], 1):
            if isinstance(r, dict):
                title = r.get("title", "")
                snippet = r.get("snippet", r.get("content", ""))
                url = r.get("url", r.get("source", ""))
            else:
                title = getattr(r, "title", "")
                snippet = getattr(r, "snippet", getattr(r, "content", ""))
                url = getattr(r, "url", getattr(r, "source", ""))
            parts.append(f"## [{i}] {title}")
            parts.append(f"来源: {url}")
            parts.append(str(snippet)[:800])
            parts.append("")
        return "\n".join(parts)

    @staticmethod
    def _ingest_documents(
        docs: List[Dict[str, Any]],
        collection: str = "financial_reports",
        company: str = "",
    ) -> Dict[str, Any]:
        """通过 Pipeline.run() 将文档存入 RAG 向量库。"""
        try:
            from src.core.settings import load_settings
            from src.core.types import Document
            from src.ingestion.chunking.document_chunker import DocumentChunker
            from src.ingestion.embedding.batch_processor import BatchProcessor
            from src.ingestion.storage.vector_upserter import VectorUpserter

            settings = load_settings()
            chunker = DocumentChunker(settings)
            processor = BatchProcessor(settings)
            upserter = VectorUpserter(settings, collection_name=collection)

            total_chunk_count = 0
            for doc_data in docs:
                content = doc_data.get("content", "")
                if not content.strip():
                    continue

                source = doc_data.get("source", f"finance://{company}")
                symbol = doc_data.get("symbol", "")

                document = Document(
                    text=content,
                    metadata={
                        "source_path": source,
                        "title": f"{company} ({symbol}) 财务数据",
                        "source_type": "financial_fetcher",
                        "collection": collection,
                        "company": company,
                        "symbol": symbol,
                        "fetched_at": datetime.now().isoformat(),
                    },
                    page_number=0,
                )

                chunks = chunker.split_document(document)
                if not chunks:
                    continue

                batch_result = processor.process(chunks)
                if batch_result.dense_vectors:
                    chunk_ids = upserter.upsert(chunks, batch_result.dense_vectors)
                    total_chunk_count += len(chunk_ids)
                    logger.info(
                        f"Ingested {len(chunk_ids)} chunks for {company} ({symbol}) "
                        f"from {source}"
                    )

            return {
                "ingested": total_chunk_count > 0,
                "chunk_count": total_chunk_count,
                "collection": collection,
            }

        except ImportError as e:
            logger.warning(f"Cannot ingest documents (missing deps): {e}")
            return {"ingested": False, "chunk_count": 0, "error": str(e)}
        except Exception as e:
            logger.exception(f"Document ingestion failed: {e}")
            return {"ingested": False, "chunk_count": 0, "error": str(e)}


def _fmt_val(val: Any, suffix: str = "") -> str:
    if val is None:
        return "N/A"
    try:
        v = float(val)
        if v != v:
            return "N/A"
        return f"{v:.2f}{suffix}"
    except (ValueError, TypeError):
        return "N/A"
