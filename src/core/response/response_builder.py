"""Response Builder for constructing MCP-formatted responses.

This module builds structured responses for MCP tools, combining:
- Human-readable Markdown content with citation markers
- Structured citation data for machine consumption
- Multimodal content (text + images) support
- Proper handling of empty results and error cases
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from mcp import types

from src.core.response.citation_generator import Citation, CitationGenerator
from src.core.types import RetrievalResult


@dataclass
class MCPToolResponse:
    """Structured response for MCP tools.

    Attributes:
        content: Human-readable Markdown content with citation markers [1], [2], etc.
        citations: List of structured citations for reference
        metadata: Additional response metadata (query, result_count, etc.)
        is_empty: Whether the search returned no results
        image_contents: List of MCP ImageContent blocks for multimodal responses
    """
    content: str
    citations: List[Citation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_empty: bool = False
    image_contents: List[types.ImageContent] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "structuredContent": {
                "citations": [c.to_dict() for c in self.citations],
                "metadata": self.metadata,
                "isEmpty": self.is_empty,
            }
        }

    def to_mcp_content(self) -> List[Union[types.TextContent, types.ImageContent]]:
        blocks: List[Union[types.TextContent, types.ImageContent]] = [
            types.TextContent(type="text", text=self.content)
        ]
        if self.image_contents:
            blocks.extend(self.image_contents)
        if self.citations or self.metadata:
            import json
            structured = {
                "citations": [c.to_dict() for c in self.citations],
                "metadata": self.metadata,
                "has_images": len(self.image_contents) > 0,
                "image_count": len(self.image_contents),
            }
            blocks.append(
                types.TextContent(
                    type="text",
                    text=f"\n---\n**References (JSON):**\n```json\n{json.dumps(structured, ensure_ascii=False, indent=2)}\n```",
                )
            )
        return blocks

    @property
    def has_images(self) -> bool:
        return len(self.image_contents) > 0


class ResponseBuilder:
    def __init__(
        self,
        citation_generator: Optional[CitationGenerator] = None,
        multimodal_assembler: Optional["MultimodalAssembler"] = None,
        max_results_in_content: int = 5,
        snippet_max_length: int = 300,
        enable_multimodal: bool = True,
    ) -> None:
        self.citation_generator = citation_generator or CitationGenerator()
        self.max_results_in_content = max_results_in_content
        self.snippet_max_length = snippet_max_length
        self.enable_multimodal = enable_multimodal
        self._multimodal_assembler = multimodal_assembler

    @property
    def multimodal_assembler(self) -> "MultimodalAssembler":
        if self._multimodal_assembler is None:
            from src.core.response.multimodal_assembler import MultimodalAssembler
            self._multimodal_assembler = MultimodalAssembler()
        return self._multimodal_assembler

    def build(
        self,
        results: List[RetrievalResult],
        query: str,
        collection: Optional[str] = None,
        include_images: bool = True,
    ) -> MCPToolResponse:
        if not results:
            return self._build_empty_response(query, collection)
        citations = self.citation_generator.generate(results)
        content = self._build_markdown_content(results, citations, query)
        metadata = self._build_metadata(query, collection, len(results))
        image_contents: List[types.ImageContent] = []
        if self.enable_multimodal and include_images:
            image_blocks = self.multimodal_assembler.assemble(results, collection)
            image_contents = [
                block for block in image_blocks
                if isinstance(block, types.ImageContent)
            ]
            if image_contents:
                metadata["has_images"] = True
                metadata["image_count"] = len(image_contents)
        return MCPToolResponse(
            content=content,
            citations=citations,
            metadata=metadata,
            is_empty=False,
            image_contents=image_contents,
        )

    def _build_empty_response(self, query: str, collection: Optional[str] = None) -> MCPToolResponse:
        content = f"## 未找到相关结果\n\n"
        content += f"查询: **{query}**\n\n"
        if collection:
            content += f"在集合 `{collection}` 中未找到与查询相关的文档。\n\n"
        else:
            content += "未找到与查询相关的文档。\n\n"
        content += "**建议:**\n"
        content += "- 尝试使用不同的关键词\n"
        content += "- 检查是否已摄取相关文档\n"
        content += "- 扩大搜索范围（如不指定 collection）\n"
        metadata = self._build_metadata(query, collection, 0)
        return MCPToolResponse(content=content, citations=[], metadata=metadata, is_empty=True)

    def _build_markdown_content(
        self, results: List[RetrievalResult], citations: List[Citation], query: str
    ) -> str:
        lines = []
        lines.append(f"## 检索结果\n")
        lines.append(f"针对查询 **\"{query}\"** 找到 {len(results)} 条相关结果:\n")
        display_count = min(len(results), self.max_results_in_content)
        for i, (result, citation) in enumerate(zip(results[:display_count], citations[:display_count])):
            marker = self.citation_generator.format_citation_marker(citation.index)
            lines.append(f"### {marker} 结果 {citation.index}")
            lines.append(f"**相关度:** {citation.score:.2%}")
            lines.append(f"**来源:** `{citation.source}`")
            if citation.page is not None:
                lines.append(f"**页码:** {citation.page}")
            snippet = self._truncate_text(result.text, self.snippet_max_length)
            lines.append(f"\n> {snippet}\n")
        if len(results) > display_count:
            remaining = len(results) - display_count
            lines.append(f"\n*...还有 {remaining} 条结果未显示*\n")
        lines.append("\n---\n")
        lines.append("## 引用来源\n")
        lines.append("> 💡 要查看某条引用的完整原文，请使用 `open_original_document` 工具并传入对应的 `chunk_id`。\n")
        for citation in citations:
            source_info = f"`{citation.source}`"
            if citation.page is not None:
                source_info += f" (p.{citation.page})"
            chunk_hint = f"`chunk_id: {citation.chunk_id}`"
            lines.append(f"- [{citation.index}] {source_info} — {chunk_hint}")
        return "\n".join(lines)

    def _build_metadata(self, query: str, collection: Optional[str], result_count: int) -> Dict[str, Any]:
        metadata = {"query": query, "result_count": result_count}
        if collection:
            metadata["collection"] = collection
        return metadata

    def _truncate_text(self, text: str, max_length: int) -> str:
        if not text:
            return ""
        cleaned = " ".join(text.split())
        if len(cleaned) <= max_length:
            return cleaned
        return cleaned[:max_length].rsplit(" ", 1)[0] + "..."
