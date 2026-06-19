"""Tool Calling Framework for Agent.

Provides standardized tool definition, calling, and result parsing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable
from enum import Enum

from src.core.settings import Settings


def _filename_from_path(path: str) -> str:
    """Extract a human-readable filename from a full path, or return empty."""
    if not path:
        return ""
    import os
    return os.path.basename(path)


class ToolName(Enum):
    """Available tool names."""
    QUERY_KNOWLEDGE_HUB = "query_knowledge_hub"
    LIST_COLLECTIONS = "list_collections"
    GET_DOCUMENT_SUMMARY = "get_document_summary"
    OPEN_ORIGINAL_DOCUMENT = "open_original_document"


@dataclass
class ToolDefinition:
    """Tool definition for LLM function calling."""
    name: str
    description: str
    parameters: Dict[str, Any]
    required: List[str]


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    data: Any
    error: Optional[str] = None


class BaseTool(ABC):
    """Base class for all tools."""

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Return tool definition."""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass


class QueryKnowledgeHubTool(BaseTool):
    """Tool for querying the knowledge base."""

    def __init__(self, settings: Settings):
        self.settings = settings
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        from src.libs.embedding.embedding_factory import EmbeddingFactory
        from src.core.query_engine.dense_retriever import create_dense_retriever
        from src.core.query_engine.sparse_retriever import create_sparse_retriever
        from src.core.query_engine.hybrid_search import create_hybrid_search
        from src.core.query_engine.query_processor import QueryProcessor
        from src.ingestion.storage.bm25_indexer import BM25Indexer

        collection = "default"
        self.vector_store = VectorStoreFactory.create(settings, collection_name=collection)
        embedding_client = EmbeddingFactory.create(settings)

        dense_retriever = create_dense_retriever(
            settings=settings,
            embedding_client=embedding_client,
            vector_store=self.vector_store,
        )

        bm25_indexer = BM25Indexer(index_dir=f"data/db/bm25/{collection}")
        sparse_retriever = create_sparse_retriever(
            settings=settings,
            bm25_indexer=bm25_indexer,
            vector_store=self.vector_store,
        )
        sparse_retriever.default_collection = collection

        query_processor = QueryProcessor()
        self.hybrid_search = create_hybrid_search(
            settings=settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
        )

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=ToolName.QUERY_KNOWLEDGE_HUB.value,
            description="Search the knowledge base for relevant documents using hybrid search (semantic + keyword).",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query or question"},
                    "top_k": {"type": "integer", "description": "Number of results to return (default: 5)", "default": 5, "minimum": 1, "maximum": 20},
                    "collection": {"type": "string", "description": "Optional collection name to limit search scope"},
                },
            },
            required=["query"],
        )

    def execute(self, query: str, top_k: int = 5, collection: Optional[str] = None, **kwargs) -> ToolResult:
        try:
            from src.core.trace import TraceContext
            trace = TraceContext(trace_type="query")
            result = self.hybrid_search.search(query=query, top_k=top_k, filters=None, trace=trace, return_details=True)
            if hasattr(result, 'results'):
                search_results = result.results
                dense_results = result.dense_results or []
                sparse_results = result.sparse_results or []
            else:
                search_results = result
                dense_results = []
                sparse_results = []
            formatted_results = []
            for doc in search_results:
                meta = doc.metadata or {}
                source = (
                    meta.get("source") or meta.get("title") or meta.get("source_ref")
                    or _filename_from_path(meta.get("source_path", "")) or "unknown"
                )
                formatted_results.append({
                    "content": doc.text[:500] + "..." if len(doc.text) > 500 else doc.text,
                    "score": doc.score,
                    "source": source,
                    "chunk_index": meta.get("chunk_index", -1),
                    "images": meta.get("images"),
                    "image_refs": meta.get("image_refs"),
                    "doc_hash": meta.get("doc_hash"),
                    "source_path": meta.get("source_path"),
                })
            return ToolResult(success=True, data={
                "query": query, "results": formatted_results, "total_results": len(formatted_results),
                "dense_count": len(dense_results), "sparse_count": len(sparse_results),
                "fusion_method": "RRF", "parallel_execution": True,
            })
        except Exception as e:
            import traceback
            return ToolResult(success=False, data=None, error=f"{str(e)}\n{traceback.format_exc()}")


class ListCollectionsTool(BaseTool):
    """Tool for listing all document collections."""

    def __init__(self, settings: Settings):
        self.settings = settings
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        self.vector_store = VectorStoreFactory.create(settings)

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=ToolName.LIST_COLLECTIONS.value,
            description="List all document collections in the knowledge base.",
            parameters={"type": "object", "properties": {}},
            required=[],
        )

    def execute(self, **kwargs) -> ToolResult:
        try:
            collections: Dict[str, Dict[str, int]] = {}
            if hasattr(self.vector_store, "get_collection_stats"):
                stats = self.vector_store.get_collection_stats()
                name = stats.get("name") or getattr(
                    getattr(self.settings, "vector_store", None), "collection_name", "default"
                )
                collections[name] = {"document_count": 0, "total_chunks": int(stats.get("count", 0)), "total_images": 0}
            else:
                name = getattr(getattr(self.settings, "vector_store", None), "collection_name", "default")
                collections[name] = {"document_count": 0, "total_chunks": 0, "total_images": 0}
            return ToolResult(success=True, data={"collections": collections, "total_collections": len(collections)})
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class GetDocumentSummaryTool(BaseTool):
    """Tool for getting document summary."""

    def __init__(self, settings: Settings):
        self.settings = settings
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        self.vector_store = VectorStoreFactory.create(settings)

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=ToolName.GET_DOCUMENT_SUMMARY.value,
            description="Get summary information about a specific document.",
            parameters={"type": "object", "properties": {"source_path": {"type": "string", "description": "The source path or name of the document"}}},
            required=["source_path"],
        )

    def execute(self, source_path: str, **kwargs) -> ToolResult:
        try:
            results = self.vector_store.query(query="", top_k=100, collection="default", metadata_filter={"source": source_path})
            if not results:
                return ToolResult(success=False, data=None, error=f"Document not found: {source_path}")
            sources = set()
            for r in results:
                source = r.metadata.get("source", "unknown")
                sources.add(source)
            return ToolResult(success=True, data={"source_path": source_path, "collection": "default", "chunk_count": len(results), "image_count": 0, "matching_sources": list(sources)})
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class OpenOriginalDocumentTool(BaseTool):
    """Tool for opening and viewing the full original document from a citation."""

    def __init__(self, settings: Settings):
        self.settings = settings
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        self.vector_store = VectorStoreFactory.create(settings)

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=ToolName.OPEN_ORIGINAL_DOCUMENT.value,
            description=(
                "Open and view the full original document from a citation source. "
                "Given a chunk_id (from search result citations), returns the complete "
                "document content with the target chunk highlighted."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string", "description": "The chunk ID from a citation reference"},
                    "doc_hash": {"type": "string", "description": "Alternative: the document hash directly"},
                    "collection": {"type": "string", "description": "Collection name (default: 'default')"},
                },
            },
            required=[],
        )

    def execute(self, chunk_id: Optional[str] = None, doc_hash: Optional[str] = None, collection: Optional[str] = None, **kwargs) -> ToolResult:
        try:
            from src.mcp_server.tools.open_original_document import OpenOriginalDocumentTool as MCPOpenDocTool
            tool = MCPOpenDocTool(settings=self.settings)
            result = tool.get_document_content(chunk_id=chunk_id, doc_hash=doc_hash, collection=collection)
            return ToolResult(success=True, data={
                "title": result["title"], "total_chunks": result["total_chunks"],
                "total_chars": result["total_chars"], "source_path": result["source_path"],
                "doc_hash": result["doc_hash"], "highlighted_chunks": result["highlighted_chunks"],
                "truncated": result["truncated"],
            })
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._tools: Dict[str, BaseTool] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register default tools."""
        self.register(QueryKnowledgeHubTool(self.settings))
        self.register(ListCollectionsTool(self.settings))
        self.register(GetDocumentSummaryTool(self.settings))
        self.register(OpenOriginalDocumentTool(self.settings))

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.definition.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def list_tools(self) -> List[ToolDefinition]:
        return [tool.definition for tool in self._tools.values()]

    def execute(self, name: str, max_retries: int = 2, **kwargs) -> ToolResult:
        tool = self.get(name)
        if not tool:
            return ToolResult(success=False, data=None, error=f"Tool not found: {name}")
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                result = tool.execute(**kwargs)
                if result.success:
                    return result
                return result
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries:
                    import time
                    time.sleep(0.5 * (attempt + 1))
                    continue
        return ToolResult(success=False, data=None, error=f"Tool execution failed after {max_retries + 1} attempts: {last_error}")

    def execute_with_fallback(self, primary_tool: str, fallback_tool: str, **kwargs) -> ToolResult:
        result = self.execute(primary_tool, **kwargs)
        if result.success:
            return result
        fallback_result = self.execute(fallback_tool, **kwargs)
        if fallback_result.success:
            return ToolResult(success=True, data=fallback_result.data, error=None)
        return ToolResult(success=False, data=None, error=f"Primary tool '{primary_tool}' failed: {result.error}. Fallback tool '{fallback_tool}' also failed: {fallback_result.error}")
