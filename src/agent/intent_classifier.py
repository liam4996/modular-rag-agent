"""Intent Classification Module.

Uses LLM to classify user input into predefined intents with context awareness.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.base_llm import Message
from src.core.settings import Settings


class IntentType(Enum):
    """Supported intent types."""
    QUERY = "query"                    # 查询知识库
    LIST_COLLECTIONS = "list_collections"  # 列出所有集合
    GET_SUMMARY = "get_summary"        # 获取文档摘要
    CHAT = "chat"                      # 普通对话
    UNKNOWN = "unknown"                # 未知意图


@dataclass
class IntentResult:
    """Intent classification result."""
    intent: IntentType
    confidence: float
    parameters: Dict[str, Any]
    reasoning: str


class IntentClassifier:
    """LLM-based intent classifier with context awareness.
    
    Uses few-shot prompting to classify user input into intents.
    Supports multi-turn conversation context for better understanding.
    
    Example:
        >>> classifier = IntentClassifier(settings)
        >>> result = classifier.classify("查询关于RAG的论文")
        >>> print(result.intent)  # IntentType.QUERY
        
        >>> # With conversation context
        >>> context = [{"role": "user", "content": "什么是RAG?"}, {"role": "assistant", "content": "RAG是..."}]
        >>> result = classifier.classify("它有什么优势?", context)
        >>> print(result.intent)  # IntentType.QUERY (understands "它" refers to RAG)
    """
    
    SYSTEM_PROMPT = """You are an intent classifier for a RAG (Retrieval-Augmented Generation) system.
Your task is to classify user input into one of the following intents:

1. QUERY - User wants to search/query the knowledge base
   Examples: "查询论文结论", "什么是RAG", "find documents about AI", "搜索相关资料", "它有什么优势?"
   Note: Also includes follow-up questions referring to previous topics (e.g., "它是什么", "详细介绍下")
   Parameters: query (the search query, resolve pronouns like "it", "这个", "它" based on context)

2. LIST_COLLECTIONS - User wants to see all document collections
   Examples: "列出所有集合", "有哪些文档", "show collections", "list all documents"
   Parameters: none

3. GET_SUMMARY - User wants a summary of a specific document
   Examples: "总结这篇论文", "文档摘要", "summarize this paper", "给我这篇文档的总结"
   Parameters: doc_id or document name (extract from context if mentioned before)

4. CHAT - User wants general conversation
   Examples: "你好", "hello", "how are you", "讲个笑话", "谢谢"
   Parameters: none

Important:
- Consider the conversation context to resolve pronouns and references
- If user refers to something mentioned before (using "it", "这个", "它", "那篇"), classify as QUERY
- Respond ONLY in JSON format

Response format:
{
    "intent": "QUERY|LIST_COLLECTIONS|GET_SUMMARY|CHAT",
    "confidence": 0.95,
    "parameters": {"query": "..."},
    "reasoning": "brief explanation of why this intent was chosen"
}"""
    
    def __init__(self, settings: Settings):
        """Initialize the intent classifier.
        
        Args:
            settings: Application settings.
        """
        self.settings = settings
        self.llm = LLMFactory.create(settings)
    
    def classify(
        self, 
        user_input: str, 
        context: Optional[List[Dict[str, str]]] = None
    ) -> IntentResult:
        """Classify user input into an intent with optional conversation context.
        
        Args:
            user_input: The user's input text.
            context: Optional conversation history for context-aware classification.
                    Format: [{"role": "user|assistant", "content": "..."}, ...]
            
        Returns:
            IntentResult containing intent type, confidence, and parameters.
        """
        try:
            return self._llm_classify(user_input, context)
        except Exception as e:
            # Fallback to keyword matching if LLM fails
            return self._fallback_classify(user_input)
    
    def _llm_classify(
        self, 
        user_input: str, 
        context: Optional[List[Dict[str, str]]] = None
    ) -> IntentResult:
        """Use LLM for intent classification.
        
        Args:
            user_input: The user's input text.
            context: Optional conversation history.
            
        Returns:
            IntentResult from LLM classification.
        """
        # Build messages
        messages = [Message(role="system", content=self.SYSTEM_PROMPT)]
        
        # Add conversation context if provided
        if context:
            context_str = self._format_context(context)
            messages.append(Message(
                role="user", 
                content=f"Conversation history:\n{context_str}\n\nNow classify this input: {user_input}"
            ))
        else:
            messages.append(Message(role="user", content=user_input))
        
        # Call LLM
        response = self.llm.chat(messages, temperature=0.1)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Parse JSON response
        return self._parse_llm_response(content, user_input)
    
    def _format_context(self, context: List[Dict[str, str]]) -> str:
        """Format conversation context for the prompt.
        
        Args:
            context: Conversation history.
            
        Returns:
            Formatted context string.
        """
        # Keep last 4 messages for context window efficiency
        recent_context = context[-4:] if len(context) > 4 else context
        
        formatted = []
        for msg in recent_context:
            role = "User" if msg.get("role") == "user" else "Assistant"
            formatted.append(f"{role}: {msg.get('content', '')}")
        
        return "\n".join(formatted)
    
    def _parse_llm_response(self, content: str, user_input: str) -> IntentResult:
        """Parse LLM response into IntentResult.
        
        Args:
            content: Raw LLM response content.
            user_input: Original user input for fallback.
            
        Returns:
            Parsed IntentResult.
        """
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group()
            
            data = json.loads(content)
            
            # Map intent string to enum
            intent_str = data.get("intent", "UNKNOWN").upper()
            try:
                intent = IntentType[intent_str]
            except KeyError:
                intent = IntentType.UNKNOWN
            
            # Extract parameters
            parameters = data.get("parameters", {})
            if "query" not in parameters and intent == IntentType.QUERY:
                # If no query parameter, use user input
                parameters["query"] = user_input
            
            return IntentResult(
                intent=intent,
                confidence=float(data.get("confidence", 0.8)),
                parameters=parameters,
                reasoning=data.get("reasoning", "LLM classification")
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # If parsing fails, fallback to keyword matching
            return self._fallback_classify(user_input)
    
    def _fallback_classify(self, user_input: str) -> IntentResult:
        """Fallback classification using keyword matching.
        
        Args:
            user_input: The user's input text.
            
        Returns:
            IntentResult based on keyword matching.
        """
        user_input_lower = user_input.lower()
        
        # Query keywords
        query_keywords = ["查询", "搜索", "find", "search", "query", "什么是", "how to", "what is", "介绍", "explain"]
        if any(kw in user_input_lower for kw in query_keywords):
            return IntentResult(
                intent=IntentType.QUERY,
                confidence=0.7,
                parameters={"query": user_input},
                reasoning="Keyword match for query"
            )
        
        # List keywords
        list_keywords = ["列出", "所有", "list", "show", "有哪些", "有那些", "collections"]
        if any(kw in user_input_lower for kw in list_keywords) or (
            "文档" in user_input_lower and ("有哪些" in user_input_lower or "有那些" in user_input_lower)
        ):
            return IntentResult(
                intent=IntentType.LIST_COLLECTIONS,
                confidence=0.7,
                parameters={},
                reasoning="Keyword match for list collections"
            )
        
        # Summary keywords
        summary_keywords = ["总结", "摘要", "summarize", "summary"]
        if any(kw in user_input_lower for kw in summary_keywords):
            return IntentResult(
                intent=IntentType.GET_SUMMARY,
                confidence=0.7,
                parameters={},
                reasoning="Keyword match for summary"
            )
        
        # Default to chat
        return IntentResult(
            intent=IntentType.CHAT,
            confidence=0.5,
            parameters={},
            reasoning="Default to chat"
        )
