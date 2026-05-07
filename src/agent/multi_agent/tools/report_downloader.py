"""财报 PDF 下载器 — 搜索并下载公司财报 PDF，通过 Pipeline 入库 RAG。

策略：
1. 通过 WebSearchAgent 搜索 "公司名 年报/季报/10-K PDF"
2. 尝试下载 PDF → 用现有 Pipeline.run() 入库
3. 超时 15s / 下载失败 → 降级到纯文本搜索摘要

支持: A 股年报/季报、美股 10-K/10-Q、港股年报
"""

from __future__ import annotations

import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class ReportDownloader:
    """搜索并下载公司财报 PDF，通过 Pipeline.run() 入库 RAG。"""

    _PDF_SEARCH_TEMPLATES = {
        "a_share": [
            "{company} 年度报告 PDF site:sse.com.cn OR site:szse.cn",
            "{company} 年报 site:cninfo.com.cn",
            "{company} {year} 年年度报告 PDF",
        ],
        "us": [
            "{company} 10-K annual report PDF site:sec.gov",
            "{company} annual report {year} PDF site:sec.gov",
            "{company} investor relations annual report PDF",
        ],
        "hk": [
            "{company} 年度報告 PDF site:hkexnews.hk",
            "{company} annual report PDF site:hkexnews.hk",
            "{company} {year} 年报 PDF",
        ],
    }

    _DOWNLOAD_TIMEOUT = 15

    def __init__(self, settings=None, web_agent=None):
        self._settings = settings
        self._web_agent = web_agent

    @property
    def web_agent(self):
        if self._web_agent is None:
            from src.agent.multi_agent.web_agent import WebSearchAgent
            self._web_agent = WebSearchAgent(self._settings)
        return self._web_agent

    def download_and_ingest(
        self,
        company: str,
        symbols: Optional[List[str]] = None,
        market: str = "a_share",
        year: str = "",
        state: Any = None,
    ) -> Dict[str, Any]:
        """
        搜索并下载公司财报 PDF，入库 RAG。

        Args:
            company: 公司名称
            symbols: 股票代码列表
            market: 市场类型 (a_share / us / hk)
            year: 报告年份
            state: AgentState

        Returns:
            {"success": bool, "source": "pdf_ingested" | "web_snippets" | "none",
             "file_paths": [...], "chunk_count": int}
        """
        result: Dict[str, Any] = {
            "success": False,
            "source": "none",
            "file_paths": [],
            "chunk_count": 0,
        }

        if state:
            state.add_execution_trace({
                "agent": "report_downloader",
                "action": "download_start",
                "company": company,
                "market": market,
                "status": "running",
            })

        y = year or str(datetime.now().year - 1)
        templates = self._PDF_SEARCH_TEMPLATES.get(market, self._PDF_SEARCH_TEMPLATES["a_share"])

        pdf_urls: List[str] = []
        for tmpl in templates[:2]:
            query = tmpl.format(company=company, year=y)
            try:
                results = self.web_agent.search(query=query, num_results=3, time_range="y")
            except Exception as e:
                logger.warning(f"PDF search failed for '{query}': {e}")
                continue
            for r in results:
                url = r.get("url", "") if isinstance(r, dict) else getattr(r, "url", "")
                if url and self._is_pdf_url(url):
                    if url not in pdf_urls:
                        pdf_urls.append(url)

        if pdf_urls:
            downloaded = self._download_pdfs(pdf_urls, company)
            if downloaded:
                result["file_paths"] = downloaded
                result["source"] = "pdf_ingested"

                # 尝试通过 Pipeline 入库
                if self._settings and downloaded:
                    try:
                        for fp in downloaded[:2]:
                            chunk_cnt = self._ingest_pdf_via_pipeline(fp, company)
                            if chunk_cnt:
                                result["chunk_count"] += chunk_cnt
                    except Exception as e:
                        logger.warning(f"Pipeline ingest failed for PDFs: {e}")

                result["success"] = True

        if not result["success"]:
            result["source"] = "web_snippets"

        if state:
            state.add_execution_trace({
                "agent": "report_downloader",
                "action": "download_done",
                "source": result["source"],
                "pdf_count": len(result["file_paths"]),
                "chunk_count": result["chunk_count"],
                "status": "done" if result["success"] else "error",
            })

        return result

    @staticmethod
    def _is_pdf_url(url: str) -> bool:
        low = url.lower()
        if low.endswith(".pdf"):
            return True
        if ".pdf?" in low or "format=pdf" in low:
            return True
        return False

    def _download_pdfs(self, urls: List[str], company: str) -> List[str]:
        downloaded = []
        for url in urls[:3]:
            try:
                import urllib.request
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; FinanceBot/1.0)"},
                )
                with urllib.request.urlopen(req, timeout=self._DOWNLOAD_TIMEOUT) as resp:
                    content = resp.read()
                    if len(content) < 1024:
                        continue
                    parsed = urlparse(url)
                    fname = f"{company}_{Path(parsed.path).name}"
                    if not fname.endswith(".pdf"):
                        fname += ".pdf"
                    tmpdir = Path(tempfile.gettempdir()) / "finance_reports"
                    tmpdir.mkdir(parents=True, exist_ok=True)
                    fpath = tmpdir / fname
                    fpath.write_bytes(content)
                    downloaded.append(str(fpath))
                    logger.info(f"Downloaded PDF: {fpath} ({len(content)} bytes)")
            except Exception as e:
                logger.warning(f"Failed to download {url}: {e}")
        return downloaded

    @staticmethod
    def _ingest_pdf_via_pipeline(pdf_path: str, company: str) -> int:
        try:
            from src.ingestion.pipeline import Document, Pipeline
            from pathlib import Path

            fp = Path(pdf_path)
            if not fp.exists() or fp.stat().st_size < 1024:
                return 0

            pipeline = Pipeline()
            result = pipeline.run(str(fp))

            if result and hasattr(result, "chunk_count"):
                return result.chunk_count
            return 0
        except ImportError as e:
            logger.warning(f"Pipeline not available: {e}")
        except Exception as e:
            logger.warning(f"Pipeline ingest failed: {e}")
        return 0
