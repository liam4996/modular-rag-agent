"""Conversation Memory Module for Agent.

Manages multi-turn conversation history with token limit control
and context-aware query rewriting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json

from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.base_llm import Message
from src.core.settings import Settings


@dataclass
class ConversationTurn:
    """Single turn in conversation."""
    role: str  # "user" or "assistant"
    content: str
    intent: Optional[str] = None
    tool_called: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationMemory:
    """Manages conversation history for multi-turn Agent.
    
    Features:
    - Token-aware context window management
    - Query rewriting for pronoun resolution
    - Conversation summarization for long contexts
    
    Example:
        >>> memory = ConversationMemory(settings)
        >>> memory.add_user_message("什么是RAG？")
        >>> memory.add_assistant_message("RAG是...", intent="query")
        >>> 
        >>> # Rewrite follow-up query
        >>> rewritten = memory.rewrite_query("它有什么优势？")
        >>> print(rewritten)  # "RAG有什么优势？"
    """
    
    def __init__(
        self, 
        settings: Settings,
        max_context_turns: int = 6,
        max_tokens: int = 2000
    ):
        """Initialize conversation memory.
        
        Args:
            settings: Application settings
            max_context_turns: Maximum number of turns to keep in context
            max_tokens: Approximate token limit for context window
        """
        self.settings = settings
        self.max_context_turns = max_context_turns
        self.max_tokens = max_tokens
        self.llm = LLMFactory.create(settings)
        
        self.turns: List[ConversationTurn] = []
        self.summary: Optional[str] = None  # For very long conversations
    
    def add_user_message(
        self, 
        content: str, 
        intent: Optional[str] = None
    ) -> None:
        """Add a user message to memory."""
        self.turns.append(ConversationTurn(
            role="user",
            content=content,
            intent=intent
        ))
    
    def add_assistant_message(
        self, 
        content: str, 
        intent: Optional[str] = None,
        tool_called: Optional[str] = None
    ) -> None:
        """Add an assistant message to memory."""
        self.turns.append(ConversationTurn(
            role="assistant",
            content=content,
            intent=intent,
            tool_called=tool_called
        ))
    
    def get_recent_turns(self, n: Optional[int] = None) -> List[ConversationTurn]:
        """Get recent n turns.
        
        Args:
            n: Number of turns to return. If None, uses max_context_turns.
            
        Returns:
            List of recent conversation turns.
        """
        if n is None:
            n = self.max_context_turns
        return self.turns[-n:] if len(self.turns) > n else self.turns
    
    def get_context_for_prompt(self) -> str:
        """Format conversation history for LLM prompt.
        
        Returns:
            Formatted context string.
        """
        recent_turns = self.get_recent_turns()
        
        if not recent_turns:
            return ""
        
        lines = []
        for turn in recent_turns:
            role_label = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{role_label}: {turn.content}")
        
        return "\n".join(lines)
    
    def get_messages_for_llm(self) -> List[Message]:
        """Get conversation history as LLM messages.
        
        Returns:
            List of Message objects for LLM chat.
        """
        recent_turns = self.get_recent_turns()
        messages = []
        
        for turn in recent_turns:
            messages.append(Message(role=turn.role, content=turn.content))
        
        return messages
    
    def rewrite_query(self, query: str) -> str:
        """Rewrite query to resolve pronouns and references.
        
        Uses LLM to expand pronouns like "it", "这个", "它" based on
        conversation context.
        
        Args:
            query: Original user query (possibly with pronouns)
            
        Returns:
            Rewritten query with resolved references.
        """
        if not self.turns:
            return query
        
        # Check if query contains pronouns
        pronouns = ["它", "这个", "那个", "这类", "这种", "it", "this", "that", "these"]
        has_pronoun = any(p in query.lower() for p in pronouns)
        
        if not has_pronoun:
            return query
        
        try:
            return self._llm_rewrite(query)
        except Exception:
            # If LLM fails, return original query
            return query
    
    def _llm_rewrite(self, query: str) -> str:
        """Use LLM to rewrite query with context."""
        context = self.get_context_for_prompt()
        
        prompt = f"""Given the conversation history and the follow-up question, rewrite the question to be self-contained and clear.

Conversation History:
{context}

Follow-up Question: {query}

Rewrite the follow-up question to replace pronouns (it, this, that, 它, 这个) with the actual subject from the conversation history. If no pronouns need replacement, return the original question.

Rewritten Question:"""

        messages = [Message(role="user", content=prompt)]
        response = self.llm.chat(messages, temperature=0.1)
        
        rewritten = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        
        # Remove quotes if present
        rewritten = rewritten.strip('"\'')
        
        return rewritten if rewritten else query
    
    def clear(self) -> None:
        """Clear all conversation history."""
        self.turns.clear()
        self.summary = None
    
    def is_empty(self) -> bool:
        """Check if memory is empty."""
        return len(self.turns) == 0
    
    def get_turn_count(self) -> int:
        """Get total number of turns."""
        return len(self.turns)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize memory to dictionary."""
        return {
            "turns": [
                {
                    "role": turn.role,
                    "content": turn.content,
                    "intent": turn.intent,
                    "tool_called": turn.tool_called,
                    "metadata": turn.metadata
                }
                for turn in self.turns
            ],
            "summary": self.summary
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], settings: Settings) -> "ConversationMemory":
        """Deserialize memory from dictionary."""
        memory = cls(settings)
        memory.summary = data.get("summary")
        
        for turn_data in data.get("turns", []):
            turn = ConversationTurn(
                role=turn_data["role"],
                content=turn_data["content"],
                intent=turn_data.get("intent"),
                tool_called=turn_data.get("tool_called"),
                metadata=turn_data.get("metadata", {})
            )
            memory.turns.append(turn)
        
        return memory
