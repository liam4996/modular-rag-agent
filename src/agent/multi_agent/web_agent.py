"""
多智能体 RAG 系统 - Web Agent

职责：
- 搜索互联网获取实时信息
- 支持 DuckDuckGo（免费）、Google/Bing API
- 返回格式适配新的 AgentState
- 写入结果到 blackboard["web_results"]
"""

import os
from typing import List, Dict, Any, Optional
from src.agent.tools.web_search import WebSearchTool, WebSearchResult
from src.core.settings import Settings


class WebSearchAgent:
    """
    Web Search Agent - 联网搜索
    
    支持：
    - DuckDuckGo 搜索（免费，默认）
    - Google Custom Search API（付费）
    - Bing Search API（付费）
    
    使用场景：
    - WEB_SEARCH 意图
    - HYBRID_SEARCH 意图（与 Search Agent 并行）
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        search_engine: str = "tavily",
    ):
        self.settings = settings or Settings()

        ws = getattr(self.settings, "web_search", None)
        tavily_raw = getattr(ws, "tavily_api_key", None) if ws else None
        tavily_api_key = tavily_raw or os.environ.get("TAVILY_API_KEY")
        google_api_key = getattr(ws, "google_api_key", None) if ws else None
        google_search_engine_id = getattr(ws, "google_search_engine_id", None) if ws else None
        bing_api_key = getattr(ws, "bing_api_key", None) if ws else None

        engine = (getattr(ws, "engine", None) if ws else None) or search_engine

        self.search_tool = WebSearchTool(
            search_engine=engine,
            tavily_api_key=tavily_api_key,
            google_api_key=google_api_key,
            google_search_engine_id=google_search_engine_id,
            bing_api_key=bing_api_key,
        )
    
    def search(
        self,
        query: str,
        num_results: int = 5,
        time_range: str = "y",
        local_results: Optional[List[Dict]] = None
    ) -> List[Dict[str, Any]]:
        """
        执行联网搜索
        
        Args:
            query: 搜索查询
            num_results: 返回结果数量
            time_range: 时间范围 ('d', 'w', 'm', 'y', 'a')
            local_results: 本地搜索结果（用于优化查询）
        
        Returns:
            格式化的搜索结果列表
        """
        # 如果有本地结果，可以优化查询
        if local_results:
            # 本地已有基础原理，搜索最新进展
            refined_query = f"{query} 2025 2026 最新进展"
        else:
            refined_query = query
        
        # 执行搜索
        results = self.search_tool.search(
            query=refined_query,
            num_results=num_results,
            time_range=time_range
        )
        
        # 格式化结果
        return self._format_results(results)
    
    def _format_results(self, results: List[WebSearchResult]) -> List[Dict[str, Any]]:
        """
        格式化搜索结果
        
        Args:
            results: WebSearchResult 列表
            
        Returns:
            格式化的字典列表
        """
        formatted = []
        for result in results:
            formatted.append({
                "title": result.title,
                "url": result.url,
                "snippet": result.snippet,
                "source": result.source,
                "published_date": result.published_date,
                "metadata": {
                    "type": "web",
                    "search_engine": self.search_tool.search_engine,
                }
            })
        
        return formatted
    
    def search_with_metadata(
        self,
        query: str,
        num_results: int = 5,
        time_range: str = "y",
        local_results: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        执行搜索并返回详细元数据
        
        Args:
            query: 搜索查询
            num_results: 返回结果数量
            time_range: 时间范围
            local_results: 本地搜索结果
        
        Returns:
            包含结果和元数据的字典
        """
        results = self.search(query, num_results, time_range, local_results)
        
        return {
            "results": results,
            "total_results": len(results),
            "query": query,
            "num_results": num_results,
            "time_range": time_range,
            "metadata": {
                "search_type": "web",
                "search_engine": self.search_tool.search_engine,
            }
        }
    
    def get_trending_topics(self) -> List[str]:
        """
        获取热门话题（可选功能）
        
        Returns:
            热门话题列表
        """
        # TODO: 实现获取 trending topics
        return []
    
    def search_news(
        self,
        query: str,
        num_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        搜索新闻（时间范围：今天）
        
        Args:
            query: 搜索查询
            num_results: 返回结果数量
        
        Returns:
            新闻搜索结果
        """
        return self.search(query, num_results, time_range="d")
    
    def search_recent(
        self,
        query: str,
        num_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        搜索最近内容（时间范围：本周）
        
        Args:
            query: 搜索查询
            num_results: 返回结果数量
        
        Returns:
            最近搜索结果
        """
        return self.search(query, num_results, time_range="w")
