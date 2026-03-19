"""
多智能体 RAG 系统 - Web Search 工具

支持多种搜索引擎：
- DuckDuckGo（免费）
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
    
    支持多种搜索引擎：
    - DuckDuckGo（免费，默认）
    - Google Custom Search API（付费）
    - Bing Search API（付费）
    """
    
    def __init__(
        self,
        search_engine: str = "duckduckgo",
        google_api_key: Optional[str] = None,
        google_search_engine_id: Optional[str] = None,
        bing_api_key: Optional[str] = None
    ):
        """
        初始化搜索工具
        
        Args:
            search_engine: 搜索引擎类型 ("duckduckgo", "google", "bing")
            google_api_key: Google API key
            google_search_engine_id: Google Search Engine ID
            bing_api_key: Bing API key
        """
        self.search_engine = search_engine
        self.google_api_key = google_api_key
        self.google_search_engine_id = google_search_engine_id
        self.bing_api_key = bing_api_key
    
    def search(
        self,
        query: str,
        num_results: int = 5,
        time_range: str = "y"
    ) -> List[WebSearchResult]:
        """
        执行联网搜索
        
        Args:
            query: 搜索查询
            num_results: 返回结果数量
            time_range: 时间范围
                - 'd': 今天
                - 'w': 本周
                - 'm': 本月
                - 'y': 今年
                - 'a': 所有时间
        
        Returns:
            搜索结果列表
        """
        if self.search_engine == "duckduckgo":
            return self._search_duckduckgo(query, num_results, time_range)
        elif self.search_engine == "google" and self.google_api_key:
            return self._search_google(query, num_results)
        elif self.search_engine == "bing" and self.bing_api_key:
            return self._search_bing(query, num_results)
        else:
            # 默认使用 DuckDuckGo
            return self._search_duckduckgo(query, num_results, time_range)
    
    def _search_duckduckgo(
        self,
        query: str,
        num_results: int,
        time_range: str
    ) -> List[WebSearchResult]:
        """使用 DuckDuckGo 搜索（免费）"""
        try:
            # 尝试导入 duckduckgo_search
            from duckduckgo_search import DDGS
            
            with DDGS() as ddgs:
                ddg_results = ddgs.text(
                    query,
                    max_results=num_results,
                    timelimit=time_range
                )
                
                results = []
                for item in ddg_results:
                    results.append(WebSearchResult(
                        title=item.get('title', ''),
                        url=item.get('href', ''),
                        snippet=item.get('body', ''),
                        source=self._extract_domain(item.get('href', '')),
                    ))
                
                return results
        except ImportError:
            # 如果没有安装 duckduckgo_search，返回模拟结果
            return self._search_duckduckgo_fallback(query, num_results)
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            return []
    
    def _search_duckduckgo_fallback(
        self,
        query: str,
        num_results: int
    ) -> List[WebSearchResult]:
        """使用备用方法搜索（通过 API）"""
        try:
            # 使用公开的 DuckDuckGo API
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1',
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('RelatedTopics', [])[:num_results]:
                if 'Text' in item and 'FirstURL' in item:
                    results.append(WebSearchResult(
                        title=item.get('Text', '').split(' - ')[0] if ' - ' in item.get('Text', '') else item.get('Text', ''),
                        url=item.get('FirstURL', ''),
                        snippet=item.get('Text', ''),
                        source=self._extract_domain(item.get('FirstURL', '')),
                    ))
            
            return results
        except Exception as e:
            print(f"DuckDuckGo fallback search error: {e}")
            return []
    
    def _search_google(
        self,
        query: str,
        num_results: int
    ) -> List[WebSearchResult]:
        """使用 Google Custom Search API（需要 API key）"""
        if not self.google_api_key or not self.google_search_engine_id:
            return []
        
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'q': query,
            'key': self.google_api_key,
            'cx': self.google_search_engine_id,
            'num': min(num_results, 10),
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('items', []):
                results.append(WebSearchResult(
                    title=item.get('title', ''),
                    url=item.get('link', ''),
                    snippet=item.get('snippet', ''),
                    source=self._extract_domain(item.get('link', '')),
                ))
            
            return results
        except Exception as e:
            print(f"Google search error: {e}")
            return []
    
    def _search_bing(
        self,
        query: str,
        num_results: int
    ) -> List[WebSearchResult]:
        """使用 Bing Search API（需要 API key）"""
        if not self.bing_api_key:
            return []
        
        url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {'Ocp-Apim-Subscription-Key': self.bing_api_key}
        params = {
            'q': query,
            'count': min(num_results, 10),
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('webPages', {}).get('value', []):
                results.append(WebSearchResult(
                    title=item.get('name', ''),
                    url=item.get('url', ''),
                    snippet=item.get('snippet', ''),
                    source=self._extract_domain(item.get('url', '')),
                    published_date=item.get('datePublished'),
                ))
            
            return results
        except Exception as e:
            print(f"Bing search error: {e}")
            return []
    
    def _extract_domain(self, url: str) -> str:
        """从 URL 提取域名"""
        if not url:
            return "unknown"
        try:
            parsed = urlparse(url)
            return parsed.netloc.replace('www.', '')
        except Exception:
            return "unknown"
