"""Compaction Memory — OpenClaw-inspired short-term + LangGraph Store long-term.

Short-term (Compaction):
  - Keep the most recent messages within a token budget (default 4000).
  - When messages exceed the budget, older messages are summarized by the LLM
    into a structured compaction that preserves task state, decisions, TODOs,
    and key identifiers.
  - The final context = [compaction summary] + [recent raw messages].

Long-term (LangGraph InMemoryStore):
  - Uses LangGraph's native Store API: JSON documents organized under
    hierarchical namespaces (user_id, category) with unique keys.
  - Supports vector semantic search when configured with embeddings.
  - Persistent to disk via JSON snapshot so data survives restarts.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langgraph.store.memory import InMemoryStore

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_STORE_PATH = _REPO_ROOT / "data" / "memory" / "langgraph_store.json"

# ─── Token estimation ────────────────────────────────────────────────
def _estimate_tokens(text: str) -> int:
    """Rough token count: ~1.5 chars per token for mixed CJK/English."""
    return max(1, int(len(text) / 1.5))


def _estimate_messages_tokens(messages: List[Dict[str, str]]) -> int:
    return sum(_estimate_tokens(m.get("content", "")) + 4 for m in messages)


# ═════════════════════════════════════════════════════════════════════
# Compaction Summary (Short-term) — unchanged
# ═════════════════════════════════════════════════════════════════════

_COMPACTION_PROMPT = """\
你是一个对话压缩助手。请将下面的旧对话历史压缩为一段**结构化摘要**。

要求：
1. **任务状态**：用户当前在做什么、进展到了哪一步。
2. **关键决策**：做过什么选择、拒绝了什么方案。
3. **待办 / TODO**：用户或助手提到的待办事项。
4. **关键标识符**：出现过的文件名、URL、专有名词、数字等。
5. 用中文输出；不要编造信息。
6. 输出不超过 500 字。

--- 旧对话 ---
{old_conversation}
--- 结束 ---

结构化摘要："""


@dataclass
class CompactionMemory:
    """Token-aware conversation memory with LLM compaction."""

    llm: Any = None
    token_budget: int = 4000
    compaction_summary: str = ""
    messages: List[Dict[str, str]] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self._maybe_compact()

    def build_history(self) -> List[Dict[str, str]]:
        result: List[Dict[str, str]] = []
        if self.compaction_summary:
            result.append({
                "role": "system",
                "content": f"[对话摘要]\n{self.compaction_summary}",
            })
        result.extend(self.messages)
        return result

    def build_history_dicts(self) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        if self.compaction_summary:
            out.append({"role": "user", "content": f"[之前的对话摘要] {self.compaction_summary}"})
            out.append({"role": "assistant", "content": "好的，我已了解之前的对话内容，请继续。"})
        for m in self.messages:
            if m["role"] in ("user", "assistant"):
                out.append(m)
        return out

    def clear(self) -> None:
        self.messages.clear()
        self.compaction_summary = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compaction_summary": self.compaction_summary,
            "messages": self.messages,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs) -> CompactionMemory:
        mem = cls(**kwargs)
        mem.compaction_summary = data.get("compaction_summary", "")
        mem.messages = data.get("messages", [])
        return mem

    def _maybe_compact(self) -> None:
        total = _estimate_messages_tokens(self.messages)
        if self.compaction_summary:
            total += _estimate_tokens(self.compaction_summary)
        if total <= self.token_budget:
            return
        self._run_compaction()

    def _run_compaction(self) -> None:
        keep_raw = 4
        if len(self.messages) <= keep_raw:
            return
        old_msgs = self.messages[:-keep_raw]
        recent_msgs = self.messages[-keep_raw:]
        old_text_parts: List[str] = []
        if self.compaction_summary:
            old_text_parts.append(f"[之前的摘要] {self.compaction_summary}")
        for m in old_msgs:
            label = "用户" if m["role"] == "user" else "助手"
            old_text_parts.append(f"{label}: {m['content']}")
        old_text = "\n".join(old_text_parts)
        new_summary = self._llm_summarize(old_text)
        if new_summary:
            self.compaction_summary = new_summary
        self.messages = recent_msgs

    def _llm_summarize(self, old_conversation: str) -> str:
        if not self.llm:
            return old_conversation[:800]
        try:
            from langchain_core.messages import HumanMessage
            prompt = _COMPACTION_PROMPT.format(old_conversation=old_conversation[:6000])
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            return resp.content if hasattr(resp, "content") else str(resp)
        except Exception:
            return old_conversation[:800]


# ═════════════════════════════════════════════════════════════════════
# Long-term Memory — LangGraph Native Store
# ═════════════════════════════════════════════════════════════════════
#
# Namespace hierarchy:
#   (user_id, "conversations")  — episodic: full session logs
#   (user_id, "facts")          — semantic: extracted key facts
#
# Each item is a JSON document with a unique key (UUID).
# InMemoryStore supports .put(), .get(), .search(), .delete().
# We add a thin persistence layer (JSON snapshot) so data survives restarts.

class LangGraphMemoryStore:
    """Long-term memory backed by LangGraph's InMemoryStore.

    Namespace layout::

        (user_id, "conversations")  → session-level episodic memories
        (user_id, "facts")          → extracted semantic facts / preferences

    Persisted to ``data/memory/langgraph_store.json`` on every write.
    """

    def __init__(
        self,
        user_id: str = "default",
        store_path: Path | str | None = None,
    ):
        self.user_id = user_id
        self.store_path = Path(store_path) if store_path else _STORE_PATH
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.store = InMemoryStore()
        self._load_snapshot()

    @property
    def conversations_ns(self) -> tuple:
        return (self.user_id, "conversations")

    @property
    def facts_ns(self) -> tuple:
        return (self.user_id, "facts")

    # ── Write ────────────────────────────────────────────────────

    def save_session(
        self,
        messages: List[Dict[str, str]],
        compaction_summary: str = "",
        session_id: str = "",
    ) -> str:
        """Store a conversation session as an episodic memory document."""
        key = session_id or str(uuid.uuid4())
        now = datetime.now().isoformat()
        doc = {
            "type": "session",
            "timestamp": now,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "compaction_summary": compaction_summary,
            "messages": [
                {"role": m["role"], "content": m.get("content", "")[:1000]}
                for m in messages
            ],
            "text": self._session_to_text(messages, compaction_summary),
        }
        self.store.put(self.conversations_ns, key, doc)
        self._save_snapshot()
        return key

    def save_fact(self, fact: str, category: str = "general") -> str:
        """Store a semantic fact / user preference."""
        key = str(uuid.uuid4())
        doc = {
            "type": "fact",
            "category": category,
            "content": fact,
            "timestamp": datetime.now().isoformat(),
        }
        self.store.put(self.facts_ns, key, doc)
        self._save_snapshot()
        return key

    # ── Read / Search ────────────────────────────────────────────

    def search_conversations(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """List recent conversation memories (newest first)."""
        items = self.store.search(self.conversations_ns, limit=top_k)
        results = []
        for item in items:
            val = item.value
            results.append({
                "key": item.key,
                "date": val.get("date", ""),
                "content": val.get("text", "")[:500],
                "compaction_summary": val.get("compaction_summary", ""),
            })
        results.sort(key=lambda x: x.get("date", ""), reverse=True)
        return results[:top_k]

    def search_facts(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """List all stored facts."""
        items = self.store.search(self.facts_ns, limit=top_k)
        return [
            {"key": item.key, **item.value}
            for item in items
        ]

    def recall(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search across conversations for relevant memories (keyword match).

        Uses simple keyword overlap since InMemoryStore vector search
        requires an embedding function configured at init time.
        """
        all_items = self.store.search(self.conversations_ns, limit=100)
        if not all_items:
            return []

        query_chars = set(query.lower())
        scored: list[tuple[float, Any]] = []
        for item in all_items:
            text = item.value.get("text", "").lower()
            if not text:
                continue
            overlap = sum(1 for c in query_chars if c in text)
            score = overlap / (len(query_chars) + 1)
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, item in scored[:top_k]:
            if score <= 0:
                continue
            val = item.value
            results.append({
                "key": item.key,
                "date": val.get("date", ""),
                "content": val.get("text", "")[:300],
                "relevance_score": round(score, 3),
            })
        return results

    def get_memory_count(self) -> Dict[str, int]:
        """Return counts of stored memories by namespace."""
        convs = self.store.search(self.conversations_ns, limit=1000)
        facts = self.store.search(self.facts_ns, limit=1000)
        return {
            "conversations": len(list(convs)),
            "facts": len(list(facts)),
        }

    def delete_memory(self, namespace: str, key: str) -> None:
        """Delete a specific memory item."""
        ns = self.conversations_ns if namespace == "conversations" else self.facts_ns
        self.store.delete(ns, key)
        self._save_snapshot()

    # ── Persistence (JSON snapshot) ──────────────────────────────

    def _save_snapshot(self) -> None:
        """Dump all store data to JSON file for persistence across restarts."""
        snapshot: Dict[str, Any] = {"namespaces": {}}
        for ns_tuple in [self.conversations_ns, self.facts_ns]:
            ns_key = "/".join(ns_tuple)
            items = self.store.search(ns_tuple, limit=10000)
            snapshot["namespaces"][ns_key] = {
                item.key: item.value for item in items
            }
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)

    def _load_snapshot(self) -> None:
        """Restore store data from JSON snapshot if it exists."""
        if not self.store_path.exists():
            return
        try:
            with open(self.store_path, "r", encoding="utf-8") as f:
                snapshot = json.load(f)
            for ns_key, items in snapshot.get("namespaces", {}).items():
                ns_tuple = tuple(ns_key.split("/"))
                for key, value in items.items():
                    self.store.put(ns_tuple, key, value)
        except (json.JSONDecodeError, KeyError):
            pass

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _session_to_text(
        messages: List[Dict[str, str]], summary: str = ""
    ) -> str:
        parts: List[str] = []
        if summary:
            parts.append(f"[摘要] {summary}")
        for m in messages:
            role = "用户" if m["role"] == "user" else "助手"
            parts.append(f"{role}: {m.get('content', '')[:500]}")
        return "\n".join(parts)
