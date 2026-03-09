"""ReAct Agent Implementation.

Implements the ReAct (Reasoning + Acting) pattern for multi-step tool use.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.agent.intent_classifier import IntentClassifier, IntentResult, IntentType
from src.agent.tool_caller import ToolRegistry, ToolName, ToolResult
from src.agent.memory import ConversationMemory
from src.agent.tool_chain import ToolChainExecutor, ChainStep, ChainStepType
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.base_llm import Message
from src.core.settings import Settings
import logging


@dataclass
class ReActStep:
    """Single step in ReAct reasoning loop."""
    step: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    is_final: bool = False


@dataclass
class ReActResponse:
    """Response from ReAct Agent."""
    content: str
    intent: IntentType
    steps: List[ReActStep] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""


class ReActAgent:
    """ReAct (Reasoning + Acting) Agent.
    
    Implements a multi-step reasoning loop where the agent:
    1. Thinks about the current state
    2. Chooses an action (tool call)
    3. Observes the result
    4. Repeats until it has enough information to answer
    
    Example:
        >>> agent = ReActAgent(settings)
        >>> response = agent.run("总结这份文档的核心观点")
        >>> 
        >>> for step in response.steps:
        >>>     print(f"Step {step.step}: {step.thought}")
        >>>     if step.action:
        >>>         print(f"  Action: {step.action}")
        >>>     if step.observation:
        >>>         print(f"  Observation: {step.observation[:100]}...")
    """
    
    SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.

Available tools:
- query_knowledge_hub: Search the knowledge base for relevant documents
  Input: {"query": "search text", "top_k": 5}
- list_collections: List all document collections
  Input: {}
- get_document_summary: Get summary of a specific document
  Input: {"source_path": "document path"}

You must respond in the following format for each step:

Thought: [Your reasoning about what to do next]
Action: [Tool name or "Final Answer"]
Action Input: [JSON object with tool parameters or "N/A"]
Observation: [Result from tool or "N/A"]

If you have enough information to answer the user's question, respond with:
Thought: [Final reasoning]
Action: Final Answer
Action Input: N/A
Observation: N/A

Begin!"""

    def __init__(
        self, 
        settings: Settings,
        max_iterations: int = 5,
        enable_logging: bool = True
    ):
        """Initialize ReAct Agent.
        
        Args:
            settings: Application settings
            max_iterations: Maximum ReAct loop iterations
            enable_logging: Whether to enable execution logging
        """
        self.settings = settings
        self.llm = LLMFactory.create(settings)
        self.tool_registry = ToolRegistry(settings)
        self.memory = ConversationMemory(settings)
        self.chain_executor = ToolChainExecutor(settings)
        self.max_iterations = max_iterations
        
        self.intent_classifier = IntentClassifier(settings)
        
        # Setup logging
        self.logger = logging.getLogger(__name__) if enable_logging else None
        if self.logger:
            logging.basicConfig(level=logging.INFO)
    
    def run(self, user_input: str) -> ReActResponse:
        """Run the ReAct agent on user input.
        
        Args:
            user_input: User's question or request
            
        Returns:
            ReActResponse with reasoning steps and final answer
        """
        # Initial intent classification
        rewritten_query = self.memory.rewrite_query(user_input)
        intent_result = self.intent_classifier.classify(rewritten_query)
        
        # Save to memory
        self.memory.add_user_message(user_input, intent=intent_result.intent.value)
        
        # For simple intents, use direct approach
        if intent_result.intent in (IntentType.CHAT, IntentType.UNKNOWN):
            return self._handle_chat(user_input, intent_result)
        
        # For complex tasks, use ReAct loop
        return self._react_loop(user_input, intent_result)
    
    def _handle_chat(
        self, 
        user_input: str, 
        intent_result: IntentResult
    ) -> ReActResponse:
        """Handle simple chat without tool use."""
        # Simple rule-based response for now
        content = self._simple_chat(user_input)
        
        self.memory.add_assistant_message(content, intent="chat")
        
        return ReActResponse(
            content=content,
            intent=intent_result.intent,
            steps=[
                ReActStep(
                    step=1,
                    thought="This is a simple chat request, no tool needed.",
                    action="Final Answer",
                    action_input="N/A",
                    observation="N/A",
                    is_final=True
                )
            ],
            final_answer=content
        )
    
    def _react_loop(
        self, 
        user_input: str, 
        intent_result: IntentResult
    ) -> ReActResponse:
        """Execute the ReAct reasoning loop.
        
        Args:
            user_input: Original user input
            intent_result: Initial intent classification
            
        Returns:
            ReActResponse with all reasoning steps
        """
        steps: List[ReActStep] = []
        tool_calls: List[Dict[str, Any]] = []
        
        # Build initial context
        context = self._build_context(user_input)
        
        # Initial thought
        current_thought = self._generate_initial_thought(user_input, intent_result)
        
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            
            # Ask LLM what to do next
            action, action_input, is_final = self._decide_action(
                context, current_thought, steps
            )
            
            step = ReActStep(step=iteration, thought=current_thought)
            
            if is_final or action == "Final Answer":
                # Generate final answer
                final_answer = self._generate_final_answer(context, steps, user_input)
                
                step.action = "Final Answer"
                step.action_input = "N/A"
                step.observation = "N/A"
                step.is_final = True
                steps.append(step)
                
                # Save to memory
                self.memory.add_assistant_message(final_answer, intent=intent_result.intent.value)
                
                return ReActResponse(
                    content=final_answer,
                    intent=intent_result.intent,
                    steps=steps,
                    tool_calls=tool_calls,
                    final_answer=final_answer
                )
            
            # Execute tool
            step.action = action
            step.action_input = action_input
            
            observation = self._execute_tool(action, action_input)
            step.observation = observation
            
            steps.append(step)
            tool_calls.append({
                "tool": action,
                "input": action_input,
                "result": observation[:200] + "..." if len(observation) > 200 else observation
            })
            
            # Update context for next iteration
            context += f"\n\nObservation: {observation}"
            
            # Generate next thought based on observation
            current_thought = self._generate_next_thought(
                context, action, observation, user_input
            )
        
        # Max iterations reached
        final_answer = self._generate_final_answer(context, steps, user_input)
        
        return ReActResponse(
            content=final_answer,
            intent=intent_result.intent,
            steps=steps,
            tool_calls=tool_calls,
            final_answer=final_answer
        )
    
    def _build_context(self, user_input: str) -> str:
        """Build initial context for ReAct prompts."""
        memory_context = self.memory.get_context_for_prompt()
        
        context = f"User question: {user_input}"
        
        if memory_context:
            context += f"\n\nConversation history:\n{memory_context}"
        
        return context
    
    def _generate_initial_thought(
        self, 
        user_input: str, 
        intent_result: IntentResult
    ) -> str:
        """Generate initial thought based on intent."""
        if intent_result.intent == IntentType.QUERY:
            return f"用户想要查询知识库。我需要使用 query_knowledge_hub 工具搜索相关信息。"
        elif intent_result.intent == IntentType.LIST_COLLECTIONS:
            return f"用户想要查看文档集合。我需要使用 list_collections 工具。"
        elif intent_result.intent == IntentType.GET_SUMMARY:
            return f"用户想要获取文档摘要。我需要使用 get_document_summary 工具。"
        else:
            return "需要分析用户意图并决定下一步行动。"
    
    def _decide_action(
        self, 
        context: str, 
        current_thought: str,
        steps: List[ReActStep]
    ) -> tuple[str, Dict[str, Any], bool]:
        """Ask LLM to decide the next action.
        
        Returns:
            Tuple of (action_name, action_input, is_final)
        """
        # Check if we have enough information to answer
        if len(steps) > 0:
            last_observation = steps[-1].observation
            if last_observation and last_observation != "N/A":
                # Have results, can generate final answer
                return "Final Answer", {}, True
        
        # Build prompt
        prompt = f"""{self.SYSTEM_PROMPT}

Current context:
{context}

Previous steps:
{self._format_steps(steps)}

Now continue:

Thought:"""
        
        try:
            messages = [Message(role="user", content=prompt)]
            response = self.llm.chat(messages, temperature=0.1)
            content = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            
            # Parse response
            return self._parse_action_response(content)
            
        except Exception as e:
            # Fallback: if LLM fails, use intent-based default
            return self._fallback_action(context)
    
    def _format_steps(self, steps: List[ReActStep]) -> str:
        """Format previous steps for prompt."""
        if not steps:
            return "No previous steps."
        
        lines = []
        for step in steps:
            lines.append(f"Step {step.step}:")
            lines.append(f"  Thought: {step.thought}")
            if step.action:
                lines.append(f"  Action: {step.action}")
            if step.action_input:
                lines.append(f"  Action Input: {step.action_input}")
            if step.observation and step.observation != "N/A":
                obs = step.observation[:300] + "..." if len(step.observation) > 300 else step.observation
                lines.append(f"  Observation: {obs}")
        
        return "\n".join(lines)
    
    def _parse_action_response(self, response: str) -> tuple[str, Dict[str, Any], bool]:
        """Parse LLM response to extract action.
        
        Returns:
            Tuple of (action_name, action_input, is_final)
        """
        lines = response.split("\n")
        
        action = None
        action_input_str = None
        
        for line in lines:
            line = line.strip()
            if line.startswith("Action:"):
                action = line.replace("Action:", "").strip()
            elif line.startswith("Action Input:"):
                action_input_str = line.replace("Action Input:", "").strip()
        
        # Parse action input
        action_input = {}
        if action_input_str and action_input_str != "N/A":
            try:
                # Handle JSON-like input
                action_input_str = action_input_str.strip()
                if action_input_str.startswith("{") and action_input_str.endswith("}"):
                    action_input = json.loads(action_input_str)
                else:
                    # Simple key=value parsing
                    for part in action_input_str.split(","):
                        if ":" in part:
                            key, value = part.split(":", 1)
                            action_input[key.strip().strip('"')] = value.strip().strip('"')
            except:
                action_input = {}
        
        is_final = action == "Final Answer"
        
        # Default to query if no action specified
        if not action:
            action = "query_knowledge_hub"
            action_input = {"query": " ", "top_k": 5}
        
        return action, action_input, is_final
    
    def _fallback_action(self, context: str) -> tuple[str, Dict[str, Any], bool]:
        """Fallback action when LLM fails."""
        return "query_knowledge_hub", {"query": "information", "top_k": 5}, False
    
    def _execute_tool(self, action: str, action_input: Dict[str, Any]) -> str:
        """Execute a tool and return observation."""
        try:
            # Map action name to tool
            tool_name = action
            if action == "query_knowledge_hub":
                tool_name = ToolName.QUERY_KNOWLEDGE_HUB.value
            elif action == "list_collections":
                tool_name = ToolName.LIST_COLLECTIONS.value
            elif action == "get_document_summary":
                tool_name = ToolName.GET_DOCUMENT_SUMMARY.value
            
            result = self.tool_registry.execute(tool_name, **action_input)
            
            if result.success:
                return self._format_tool_result(result.data)
            else:
                return f"Error: {result.error}"
                
        except Exception as e:
            return f"Error executing {action}: {str(e)}"
    
    def _format_tool_result(self, data: Any) -> str:
        """Format tool result as observation string."""
        if not data:
            return "No results returned."
        
        if isinstance(data, dict):
            # Format query results
            if "results" in data:
                results = data["results"]
                if not results:
                    return "No relevant documents found."
                
                formatted = []
                for i, r in enumerate(results[:3], 1):
                    content = r.get("content", "")[:200]
                    score = r.get("score", 0)
                    source = r.get("source", "unknown")
                    formatted.append(f"{i}. [{score:.3f}] {content}... (source: {source})")
                
                return "\n".join(formatted)
            
            # Format collection list
            if "collections" in data:
                collections = data["collections"]
                if not collections:
                    return "No collections found."
                
                formatted = [f"Found {len(collections)} collection(s):"]
                for name, info in collections.items():
                    chunks = info.get("total_chunks", 0)
                    formatted.append(f"- {name}: {chunks} chunks")
                
                return "\n".join(formatted)
            
            # Generic dict
            return str(data)
        
        return str(data)
    
    def _generate_next_thought(
        self, 
        context: str, 
        action: str, 
        observation: str,
        original_question: str
    ) -> str:
        """Generate the next thought based on observation."""
        prompt = f"""Based on the conversation so far, think about what to do next.

Original question: {original_question}
Last action: {action}
Observation: {observation[:500]}

Should we continue with more tool calls or provide a final answer? Think about this step by step.

Thought:"""
        
        try:
            messages = [Message(role="user", content=prompt)]
            response = self.llm.chat(messages, temperature=0.1)
            thought = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            return thought[:200] if len(thought) > 200 else thought
        except:
            return f"Executed {action} and got observation. Now I should analyze the results."
    
    def _generate_final_answer(
        self, 
        context: str, 
        steps: List[ReActStep],
        original_question: str
    ) -> str:
        """Generate final answer based on all observations."""
        # Collect all observations
        observations = []
        for step in steps:
            if step.observation and step.observation != "N/A":
                observations.append(step.observation)
        
        if not observations:
            return "抱歉，我无法找到相关信息来回答你的问题。"
        
        # Use the last relevant observation as the answer base
        last_obs = observations[-1] if observations else ""
        
        # Format as natural response
        if "No relevant documents found" in last_obs or "No results" in last_obs:
            return f"抱歉，目前知识库中没有找到与「{original_question}」相关的内容。你可以尝试用不同的关键词搜索。"
        
        # Format search results as answer
        if "[" in last_obs:  # Has search results format
            lines = [f"根据搜索结果，以下是与「{original_question}」相关的内容：\n"]
            lines.append(last_obs)
            lines.append("\n如果你需要更详细的解释或总结，请告诉我。")
            return "\n".join(lines)
        
        return last_obs
    
    def _simple_chat(self, user_input: str) -> str:
        """Simple chat handler for non-tool requests."""
        text = user_input.lower()
        
        if any(kw in text for kw in ["你好", "hello", "hi", "嗨"]):
            return "你好！我是基于 Modular RAG 的智能助手。我可以帮你查询知识库、查看文档集合或获取文档摘要。请直接问我问题吧！"
        
        if any(kw in text for kw in ["帮助", "help", "能做什么"]):
            return """我支持以下功能：
1. 🔍 查询知识库 - 搜索相关文档
2. 📁 查看文档集合 - 列出所有文档
3. 📄 文档摘要 - 获取特定文档信息

你可以用自然语言问我问题！"""
        
        if any(kw in text for kw in ["谢谢", "thanks"]):
            return "不客气！有问题随时问我。"
        
        return "我理解你的问题。如果你希望我搜索知识库，可以尝试用「查询xxx」或「搜索xxx」的方式提问。"
    
    def clear_history(self) -> None:
        """Clear conversation memory."""
        self.memory.clear()
    
    def get_memory(self) -> ConversationMemory:
        """Get conversation memory."""
        return self.memory
