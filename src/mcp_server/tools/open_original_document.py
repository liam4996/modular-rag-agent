"""MCP Tool: open_original_document

Click a citation source to open and view the full original document.
Reconstructs the complete document from ChromaDB chunks by doc_hash,
with optional highlighting of target chunks.

Usage via MCP:
    Tool name: open_original_document
    Input schema:
        - chunk_id (string, optional): Chunk ID from a citation marker
        - doc_hash (string, optional): Document hash (falls back if chunk_id not given)
        - highlight_chunk_ids (array[string], optional): Additional chunk IDs to highlight
        - collection (string, optional): Collection name
        - max_chars (integer, optional): Max characters to return (default: 30000)
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from mcp import types

if TYPE_CHECKING:
    from src.mcp_server.protocol_handler import ProtocolHandler
    from src.core.settings import Settings

logger = logging.getLogger(__name__)

TOOL_NAME = "open_original_document"
TOOL_DESCRIPTION = """Open and view the full original document from a citation source.

Given a chunk_id (from search result citations), this tool reconstructs and returns
the COMPLETE original document content by pulling all chunks from the same document.
Target chunks are highlighted with >>> markers.

Use this when you want to:
- Read the full original document that a search result came from
- See the surrounding context of a specific chunk
- Verify the accuracy of a citation by reading the original source

Parameters:
- chunk_id: The chunk ID from a citation (e.g., "a1b2c3d4_0003_e5f6g7h8")
- doc_hash: Alternative - the document hash directly
- highlight_chunk_ids: Additional chunk IDs to highlight in the output
- collection: The collection name (defaults to "default")
- max_chars: Maximum characters to return (default: 30000, max: 100000)
"""

TOOL_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "chunk_id": {
            "type": "string",
            "description": "The chunk ID from a citation/source reference. Format: {source_hash}_{index:04d}_{content_hash}",
        },
        "doc_hash": {
            "type": "string",
            "description": "The document hash to open. Used as fallback if chunk_id is not provided.",
        },
        "highlight_chunk_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Additional chunk IDs to highlight with >>> markers in the output.",
        },
        "collection": {
            "type": "string",
            "description": "The collection name to search in. Defaults to 'default'.",
        },
        "max_chars": {
            "type": "integer",
            "description": "Maximum characters to return. Default: 30000, Max: 100000.",
            "default": 30000,
            "minimum": 1000,
            "maximum": 100000,
        },
    },
    "required": [],
}


@dataclass
class OpenDocumentConfig:
    """Configuration for open_original_document tool."""

    persist_directory: str = "./data/db/chroma"
    default_collection: str = "default"
    default_max_chars: int = 30000
    max_chars_limit: int = 100000


class DocumentNotFoundError(Exception):
    """Raised when a document cannot be found."""

    def __init__(self, identifier: str):
        self.identifier = identifier
        super().__init__(f"Document not found: {identifier}")


class OpenOriginalDocumentTool:
    """MCP Tool for opening and viewing full original documents from citations.

    Reconstructs complete documents from ChromaDB chunks by aggregating
    all chunks with the same doc_hash, sorted by chunk_index.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        config: Optional[OpenDocumentConfig] = None,
    ) -> None:
        self._settings = settings
        self._config = config
        self._chroma_client = None

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            from src.core.settings import load_settings
            self._settings = load_settings()
        return self._settings

    @property
    def config(self) -> OpenDocumentConfig:
        if self._config is None:
            try:
                persist_dir = getattr(
                    self.settings.vector_store, "persist_directory", "./data/db/chroma"
                )
                default_coll = getattr(
                    self.settings.vector_store, "collection_name", "default"
                )
            except AttributeError:
                persist_dir = "./data/db/chroma"
                default_coll = "default"
            self._config = OpenDocumentConfig(
                persist_directory=persist_dir,
                default_collection=default_coll,
            )
        return self._config

    def _get_chroma_client(self) -> Any:
        if self._chroma_client is not None:
            return self._chroma_client
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError:
            raise ImportError(
                "chromadb is required. Install with: pip install chromadb"
            )
        persist_path = Path(self.config.persist_directory).resolve()
        if not persist_path.exists():
            raise RuntimeError(f"ChromaDB directory not found: {persist_path}")
        self._chroma_client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        return self._chroma_client

    @staticmethod
    def parse_chunk_id(chunk_id: str) -> Dict[str, Any]:
        """Parse a chunk_id to extract source_hash and chunk_index.

        Format: {source_hash}_{chunk_index:04d}_{content_hash}

        Returns dict with keys: source_hash, chunk_index, content_hash.
        """
        parts = chunk_id.split("_")
        source_hash = ""
        chunk_index = -1
        content_hash = ""

        if len(parts) >= 2:
            content_hash = parts[-1] if len(parts[-1]) == 8 else ""
            if len(parts) >= 2 and len(parts[-2]) == 4 and parts[-2].isdigit():
                chunk_index = int(parts[-2])
                source_hash = "_".join(parts[:-2])
            else:
                for i, part in enumerate(parts):
                    if len(part) == 4 and part.isdigit() and i > 0:
                        chunk_index = int(part)
                        source_hash = "_".join(parts[:i])
                        content_hash = "_".join(parts[i + 1:])
                        break

        return {
            "source_hash": source_hash,
            "chunk_index": chunk_index,
            "content_hash": content_hash,
        }

    def _get_collection(self, collection_name: Optional[str] = None) -> Any:
        client = self._get_chroma_client()
        name = collection_name or self.config.default_collection
        try:
            return client.get_collection(name=name)
        except Exception as e:
            raise ValueError(f"Collection '{name}' does not exist: {e}") from e

    def _find_chunks_by_doc_hash(
        self, doc_hash: str, collection_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find all chunks belonging to a document.

        Uses multiple strategies:
        1. Metadata filter on source_ref (= document.id, e.g. "doc_a1b2c3d4...")
        2. Metadata filter on doc_hash (full SHA256 of file content)
        3. Chunk ID prefix match (all chunks sharing the same {source_hash}_ prefix)
        """
        collection = self._get_collection(collection_name)

        # Strategy 1: Metadata filter on source_ref
        try:
            results = collection.get(
                where={"source_ref": doc_hash},
                include=["metadatas", "documents"],
            )
            if results and results.get("ids"):
                chunks = []
                for i, chunk_id in enumerate(results["ids"]):
                    chunks.append({
                        "id": chunk_id,
                        "text": results["documents"][i] if results.get("documents") else "",
                        "metadata": results["metadatas"][i] if results.get("metadatas") else {},
                    })
                if chunks:
                    return chunks
        except Exception as e:
            logger.debug(f"source_ref search failed: {e}")

        # Strategy 2: Metadata filter on doc_hash
        try:
            results = collection.get(
                where={"doc_hash": doc_hash},
                include=["metadatas", "documents"],
            )
            if results and results.get("ids"):
                chunks = []
                for i, chunk_id in enumerate(results["ids"]):
                    chunks.append({
                        "id": chunk_id,
                        "text": results["documents"][i] if results.get("documents") else "",
                        "metadata": results["metadatas"][i] if results.get("metadatas") else {},
                    })
                if chunks:
                    return chunks
        except Exception as e:
            logger.debug(f"doc_hash search failed: {e}")

        # Strategy 3: Chunk ID prefix match
        try:
            all_data = collection.get(include=["metadatas", "documents"])
            if all_data and all_data.get("ids"):
                chunks = []
                for i, chunk_id in enumerate(all_data["ids"]):
                    if chunk_id.startswith(doc_hash) or doc_hash in chunk_id:
                        chunks.append({
                            "id": chunk_id,
                            "text": all_data["documents"][i] if all_data.get("documents") else "",
                            "metadata": all_data["metadatas"][i] if all_data.get("metadatas") else {},
                        })
                if chunks:
                    return chunks
        except Exception as e:
            logger.debug(f"ID prefix search failed: {e}")

        return []

    def get_document_content(
        self,
        chunk_id: Optional[str] = None,
        doc_hash: Optional[str] = None,
        highlight_chunk_ids: Optional[List[str]] = None,
        collection: Optional[str] = None,
        max_chars: int = 30000,
    ) -> Dict[str, Any]:
        """Get the full document content reconstructed from chunks.

        Returns dict with:
            - title: Document title
            - content: Full text (truncated to max_chars)
            - total_chars: Total character count
            - total_chunks: Total number of chunks
            - source_path: Original file path
            - doc_hash: Document hash
            - highlighted_chunks: List of highlighted chunk indices
            - truncated: Whether content was truncated
        """
        highlight_ids = set(highlight_chunk_ids or [])

        resolved_doc_hash = doc_hash
        target_chunk_index = -1
        if chunk_id:
            parsed = self.parse_chunk_id(chunk_id)
            if not resolved_doc_hash and parsed["source_hash"]:
                resolved_doc_hash = parsed["source_hash"]
            target_chunk_index = parsed["chunk_index"]
            if chunk_id:
                highlight_ids.add(chunk_id)

        if not resolved_doc_hash:
            raise ValueError(
                "Either chunk_id or doc_hash must be provided and valid"
            )

        chunks = self._find_chunks_by_doc_hash(resolved_doc_hash, collection)
        if not chunks:
            raise DocumentNotFoundError(resolved_doc_hash)

        chunks.sort(key=lambda c: c.get("metadata", {}).get("chunk_index", 0))

        highlighted_indices = set()
        if target_chunk_index >= 0:
            highlighted_indices.add(target_chunk_index)
        for hid in highlight_ids:
            pid = self.parse_chunk_id(hid)
            if pid["chunk_index"] >= 0:
                highlighted_indices.add(pid["chunk_index"])

        first_meta = chunks[0].get("metadata", {}) if chunks else {}
        title = first_meta.get("title", "")
        if not title:
            source_path = first_meta.get("source_path", "")
            if source_path:
                title = Path(source_path).stem.replace("_", " ").replace("-", " ").title()

        source_path = first_meta.get("source_path", "")

        full_lines = []
        for chunk in chunks:
            chunk_meta = chunk.get("metadata", {})
            ci = chunk_meta.get("chunk_index", -1)
            text = chunk.get("text", "")

            if ci in highlighted_indices:
                full_lines.append(f">>> [CHUNK {ci}] >>>")
                full_lines.append(text)
                full_lines.append(f"<<< [CHUNK {ci}] <<<")
            else:
                full_lines.append(text)

        full_text = "\n\n".join(full_lines)
        total_chars = len(full_text)
        total_chunks = len(chunks)

        truncated = False
        if total_chars > max_chars:
            h_positions = []
            for i, chunk in enumerate(chunks):
                ci = chunk.get("metadata", {}).get("chunk_index", -1)
                if ci in highlighted_indices:
                    pos = sum(len(c.get("text", "")) + 2 for c in chunks[:i])
                    h_positions.append((ci, pos))

            if h_positions:
                center = h_positions[0][1]
                half = max_chars // 2
                start = max(0, center - half)
                end = min(total_chars, center + half)
                full_text = (
                    f"...(showing context around highlighted chunk, {total_chars} total chars)...\n\n"
                    + full_text[start:end]
                    + f"\n\n...(use higher max_chars to see more)..."
                )

            full_text = full_text[:max_chars] + (
                f"\n\n---\n*Content truncated ({total_chars} total chars). "
                f"Use `max_chars` parameter to see more (up to {self.config.max_chars_limit}).*"
            )

        return {
            "title": title or "Untitled Document",
            "content": full_text,
            "total_chars": total_chars,
            "total_chunks": total_chunks,
            "source_path": source_path,
            "doc_hash": resolved_doc_hash,
            "highlighted_chunks": sorted(highlighted_indices),
            "truncated": total_chars > max_chars,
        }

    def format_response(self, doc: Dict[str, Any]) -> str:
        """Format document content as a readable Markdown response."""
        lines = [
            f"# {doc['title']}",
            "",
            f"**Document Hash:** `{doc['doc_hash']}`",
        ]
        if doc["source_path"]:
            lines.append(f"**Source Path:** `{doc['source_path']}`")
        lines.extend([
            f"**Total Chunks:** {doc['total_chunks']}",
            f"**Total Characters:** {doc['total_chars']:,}",
        ])
        if doc["truncated"]:
            lines.append(f"**⚠️ Content Truncated** (showing first portion)")
        lines.append("")
        if doc["highlighted_chunks"]:
            highlighted_str = ", ".join(str(h) for h in doc["highlighted_chunks"])
            lines.append(f"**🔍 Highlighted Chunks:** {highlighted_str}")
            lines.append("")

        lines.extend([
            "---",
            "",
            doc["content"],
        ])

        return "\n".join(lines)

    def format_error(self, error: Exception) -> str:
        if isinstance(error, DocumentNotFoundError):
            return (
                f"## Document Not Found\n\n"
                f"Document '{error.identifier}' was not found in the knowledge base.\n\n"
                f"**Possible causes:**\n"
                f"- The document may have been ingested under a different hash\n"
                f"- The collection name may be incorrect\n"
                f"- Try using `list_collections` to find available collections\n"
                f"- Try using `query_knowledge_hub` to search and get valid chunk_ids"
            )
        elif isinstance(error, ValueError):
            return f"## Invalid Request\n\n{str(error)}"
        else:
            return f"## Error\n\nAn error occurred: {str(error)}"

    async def execute(
        self,
        chunk_id: Optional[str] = None,
        doc_hash: Optional[str] = None,
        highlight_chunk_ids: Optional[List[str]] = None,
        collection: Optional[str] = None,
        max_chars: Optional[int] = None,
    ) -> types.CallToolResult:
        logger.info(
            f"Executing open_original_document "
            f"(chunk_id={chunk_id}, doc_hash={doc_hash}, collection={collection})"
        )

        effective_max_chars = min(
            max_chars or self.config.default_max_chars,
            self.config.max_chars_limit,
        )

        try:
            doc = await asyncio.to_thread(
                self.get_document_content,
                chunk_id=chunk_id,
                doc_hash=doc_hash,
                highlight_chunk_ids=highlight_chunk_ids,
                collection=collection,
                max_chars=effective_max_chars,
            )
            response_text = self.format_response(doc)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=response_text)],
                isError=False,
            )
        except (DocumentNotFoundError, ValueError) as e:
            logger.warning(f"open_original_document failed: {e}")
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=self.format_error(e))],
                isError=True,
            )
        except Exception as e:
            logger.exception("Error executing open_original_document")
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=self.format_error(e))],
                isError=True,
            )


def register_tool(protocol_handler: ProtocolHandler) -> None:
    """Register the open_original_document tool with the protocol handler."""
    tool = OpenOriginalDocumentTool()

    async def handler(
        chunk_id: Optional[str] = None,
        doc_hash: Optional[str] = None,
        highlight_chunk_ids: Optional[List[str]] = None,
        collection: Optional[str] = None,
        max_chars: Optional[int] = None,
    ) -> types.CallToolResult:
        return await tool.execute(
            chunk_id=chunk_id,
            doc_hash=doc_hash,
            highlight_chunk_ids=highlight_chunk_ids,
            collection=collection,
            max_chars=max_chars,
        )

    protocol_handler.register_tool(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=TOOL_INPUT_SCHEMA,
        handler=handler,
    )

    logger.info(f"Registered MCP tool: {TOOL_NAME}")
