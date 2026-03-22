"""
多智能体 RAG 系统 - Web Search 工具

支持多种搜索引擎：
- Tavily（为 AI Agent 设计，推荐）
- DuckDuckGo（免费备选）
- Google Custom Search API（付费）
- Bing Search API（付费）
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import requests
from urllib.parse import urlparse


@dataclass
class WebSearchResult:
    """联网搜索结果"""
    title: str
    url: str
    snippet: str
    source: str
    published_date: Optional[str] = None


class WebSearchTool:
    """
    联网搜索工具

    默认使用 Tavily（专为 RAG / AI Agent 设计的搜索 API，国内可用）。
    当 Tavily 未配置时自动降级到 DuckDuckGo。
    """

    def __init__(
        self,
        search_engine: str = "tavily",
        tavily_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        google_search_engine_id: Optional[str] = None,
        bing_api_key: Optional[str] = None,
    ):
        self.search_engine = search_engine
        self.tavily_api_key = tavily_api_key
        self.google_api_key = google_api_key
        self.google_search_engine_id = google_search_engine_id
        self.bing_api_key = bing_api_key

    def search(
        self,
        query: str,
        num_results: int = 5,
        time_range: str = "y",
    ) -> List[WebSearchResult]:
        engine = self.search_engine

        if engine == "tavily" and self.tavily_api_key:
            return self._search_tavily(query, num_results)
        if engine == "google" and self.google_api_key:
            return self._search_google(query, num_results)
        if engine == "bing" and self.bing_api_key:
            return self._search_bing(query, num_results)

        # Fallback chain: tavily → duckduckgo
        if self.tavily_api_key:
            return self._search_tavily(query, num_results)
        return self._search_duckduckgo(query, num_results, time_range)

    # ======================== Tavily ========================

    def _search_tavily(
        self, query: str, num_results: int
    ) -> List[WebSearchResult]:
        """Tavily Search API — 专为 AI Agent / RAG 设计。"""
        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=self.tavily_api_key)
            response = client.search(
                query=query,
                max_results=num_results,
                search_depth="basic",
                include_answer=False,
            )

            results: List[WebSearchResult] = []
            for item in response.get("results", []):
                results.append(WebSearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    source=self._extract_domain(item.get("url", "")),
                    published_date=item.get("published_date"),
                ))
            return results
        except Exception as e:
            print(f"Tavily search error: {e}")
            return self._search_duckduckgo(query, num_results, "y")

    # ======================== DuckDuckGo ========================

    def _search_duckduckgo(
        self, query: str, num_results: int, time_range: str
    ) -> List[WebSearchResult]:
        """DuckDuckGo 搜索（免费备选）"""
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                ddg_results = ddgs.text(
                    query, max_results=num_results, timelimit=time_range
                )
                results = [
                    WebSearchResult(
                        title=item.get("title", ""),
                        url=item.get("href", ""),
                        snippet=item.get("body", ""),
                        source=self._extract_domain(item.get("href", "")),
                    )
                    for item in ddg_results
                ]
                if results:
                    return results
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")

        return self._search_duckduckgo_fallback(query, num_results)

    def _search_duckduckgo_fallback(
        self, query: str, num_results: int
    ) -> List[WebSearchResult]:
        """DuckDuckGo Instant Answer API（公开、无需 key）"""
        try:
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            results: List[WebSearchResult] = []
            for item in data.get("RelatedTopics", [])[:num_results]:
                if "Text" in item and "FirstURL" in item:
                    text = item["Text"]
                    results.append(WebSearchResult(
                        title=text.split(" - ")[0] if " - " in text else text,
                        url=item["FirstURL"],
                        snippet=text,
                        source=self._extract_domain(item["FirstURL"]),
                    ))
            return results
        except Exception as e:
            print(f"DuckDuckGo fallback error: {e}")
            return []

    # ======================== Google ========================

    def _search_google(
        self, query: str, num_results: int
    ) -> List[WebSearchResult]:
        if not self.google_api_key or not self.google_search_engine_id:
            return []
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "q": query,
                "key": self.google_api_key,
                "cx": self.google_search_engine_id,
                "num": min(num_results, 10),
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return [
                WebSearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    source=self._extract_domain(item.get("link", "")),
                )
                for item in data.get("items", [])
            ]
        except Exception as e:
            print(f"Google search error: {e}")
            return []

    # ======================== Bing ========================

    def _search_bing(
        self, query: str, num_results: int
    ) -> List[WebSearchResult]:
        if not self.bing_api_key:
            return []
        try:
            url = "https://api.bing.microsoft.com/v7.0/search"
            headers = {"Ocp-Apim-Subscription-Key": self.bing_api_key}
            params = {"q": query, "count": min(num_results, 10)}
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return [
                WebSearchResult(
                    title=item.get("name", ""),
                    url=item.get("url", ""),
                    snippet=item.get("snippet", ""),
                    source=self._extract_domain(item.get("url", "")),
                    published_date=item.get("datePublished"),
                )
                for item in data.get("webPages", {}).get("value", [])
            ]
        except Exception as e:
            print(f"Bing search error: {e}")
            return []

    # ======================== Helpers ========================

    def _extract_domain(self, url: str) -> str:
        if not url:
            return "unknown"
        try:
            return urlparse(url).netloc.replace("www.", "")
        except Exception:
            return "unknown"
