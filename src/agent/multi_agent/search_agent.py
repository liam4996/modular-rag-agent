"""
多智能体 RAG 系统 - Search Agent

职责：
- 封装现有的 hybrid_search 功能
- 返回格式适配新的 AgentState
- 支持从 Blackboard 读取上下文
- 写入结果到 blackboard["local_results"]
"""

from typing import List, Dict, Optional, Any
from src.agent.tool_caller import QueryKnowledgeHubTool, ToolResult
from src.core.settings import Settings


class SearchAgent:
    """
    Search Agent - 本地知识库检索
    
    基于现有的 ToolCaller 封装，提供：
    - 并行 Dense + Sparse 检索
    - RRF 融合
    - 结果格式化
    - 与 AgentState 集成
    
    使用场景：
    - LOCAL_SEARCH 意图
    - HYBRID_SEARCH 意图（与 Web Agent 并行）
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        初始化 Search Agent
        
        Args:
            settings: 系统配置
        """
        self.settings = settings or Settings()
        self.tool = QueryKnowledgeHubTool(self.settings)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        collection: Optional[str] = None,
        context: Optional[List[Dict]] = None
    ) -> List[Dict[str, Any]]:
        """
        执行本地知识库检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            collection: 集合名称（可选）
            context: 对话历史上下文（用于理解指代等）
        
        Returns:
            格式化的搜索结果列表
        """
        # 如果有上下文，可以进行查询优化（指代消解等）
        if context:
            query = self._resolve_pronouns(query, context)
        
        # 执行检索
        result = self.tool.execute(
            query=query,
            top_k=top_k,
            collection=collection
        )
        
        # 处理结果
        if result.success:
            # 返回详细结果
            return self._format_results(result.data)
        else:
            # 检索失败
            raise Exception(f"Search failed: {result.error}")
    
    def _format_results(self, data: Dict) -> List[Dict[str, Any]]:
        """
        格式化检索结果
        
        Args:
            data: 原始检索数据
            
        Returns:
            格式化的结果列表
        """
        results = data.get("results", [])
        
        # 添加元数据
        formatted = []
        for item in results:
            formatted.append({
                "content": item.get("content", ""),
                "score": item.get("score", 0.0),
                "source": item.get("source", "unknown"),
                "chunk_index": item.get("chunk_index", -1),
                "metadata": {
                    "type": "local",
                    "dense_sparse_fusion": True,
                    "fusion_method": "RRF",
                }
            })
        
        return formatted
    
    def _resolve_pronouns(self, query: str, context: List[Dict]) -> str:
        """Resolve pronouns and elliptical references using conversation history.

        When the user says things like "它的结论", "这篇文章", "继续",
        "还有呢", we prepend the topic from the most recent user query
        so the search engine gets a self-contained query.
        """
        if not context:
            return query

        pronouns = [
            "它", "这个", "那个", "其", "该",
            "这篇", "那篇", "上面", "刚才",
            "还有呢", "继续", "接着说",
        ]

        needs_context = (
            any(p in query for p in pronouns)
            or len(query.strip()) < 6  # very short follow-up like "结论呢"
        )

        if not needs_context:
            return query

        # Find the most recent substantive user query (>5 chars, not a pronoun itself)
        prev_query = ""
        for msg in reversed(context):
            if msg.get("role") == "user":
                candidate = msg.get("content", "").strip()
                if len(candidate) > 5 and candidate != query:
                    prev_query = candidate
                    break

        if not prev_query:
            return query

        # Extract key topic from previous query (remove common question words)
        import re
        topic = prev_query
        remove_patterns = [
            r"^(请问|请|帮我|告诉我|说说|介绍一下|解释一下)",
            r"(是什么|是啥|有哪些|怎么样|如何|吗|呢|吧|呀)$",
            r"(这篇|那篇|该)(文献|论文|文章|文档)(的)?",
        ]
        for pat in remove_patterns:
            topic = re.sub(pat, "", topic).strip()

        # Keep the topic concise — if still too long, take the first ~50 chars
        if len(topic) > 50:
            topic = topic[:50]

        return f"{topic} {query}"
    
    def search_with_metadata(
        self,
        query: str,
        top_k: int = 5,
        collection: Optional[str] = None,
        context: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        执行检索并返回详细元数据
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            collection: 集合名称
            context: 对话历史上下文
        
        Returns:
            包含结果和元数据的字典
        """
        results = self.search(query, top_k, collection, context)
        
        return {
            "results": results,
            "total_results": len(results),
            "query": query,
            "top_k": top_k,
            "collection": collection or "default",
            "metadata": {
                "search_type": "hybrid",
                "dense_sparse_fusion": True,
                "fusion_method": "RRF",
                "parallel_execution": True,
            }
        }
    
    def batch_search(
        self,
        queries: List[str],
        top_k: int = 5,
        collection: Optional[str] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        批量检索
        
        Args:
            queries: 查询列表
            top_k: 每个查询返回结果数量
            collection: 集合名称
        
        Returns:
            每个查询的结果列表
        """
        all_results = []
        
        for query in queries:
            try:
                results = self.search(query, top_k, collection)
                all_results.append(results)
            except Exception as e:
                # 单个查询失败不影响其他查询
                all_results.append([])
        
        return all_results
