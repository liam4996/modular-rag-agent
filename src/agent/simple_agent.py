"""Simple planning Agent built on top of Modular RAG tools.

This module implements a **single-agent, single-loop** workflow：

1. 使用 `IntentClassifier` 分析用户意图，给出结构化结果
2. 基于意图生成一个简单的「计划步骤」列表（Planning）
3. 通过 `ToolRegistry` 调用底层 RAG / 元数据工具（Act）
4. 把中间结果整理成自然语言回答，同时返回可观察的执行轨迹（Trace）

保留了原有 `SimpleAgent` / `AgentResponse` 接口，方便与
`scripts/run_agent.py` 和 `examples/agent_demo.py` 兼容。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.agent.intent_classifier import IntentClassifier, IntentResult, IntentType
from src.agent.tool_caller import ToolRegistry, ToolName
from src.agent.memory import ConversationMemory
from src.core.settings import Settings, load_settings


@dataclass
class AgentStep:
    """Single step in the agent's internal reasoning / execution trace."""

    step: int
    thought: str
    action: Optional[str] = None
    observation: Optional[str] = None


@dataclass
class AgentResponse:
    """Structured agent response returned给上层调用者."""

    content: str
    intent: IntentType
    tool_called: Optional[str]
    tool_result: Optional[Dict[str, Any]]
    confidence: float
    steps: List[AgentStep] = field(default_factory=list)


class SimpleAgent:
    """Minimal but complete Agent on top of Modular RAG.

    Workflow（单轮）：

    1. **Plan**：用 `IntentClassifier` 识别意图，生成 2‑3 条自然语言步骤
    2. **Act**：基于意图调用对应工具（Hybrid Search / List Collections / Doc Summary）
    3. **Summarize**：把工具结果格式化成用户可读回答，同时记录执行轨迹
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize the agent with settings, classifier, tool registry and memory."""
        self.settings = settings or load_settings()
        self.intent_classifier = IntentClassifier(self.settings)
        self.tool_registry = ToolRegistry(self.settings)
        # 使用新的 ConversationMemory 管理对话历史
        self.memory = ConversationMemory(self.settings)
        # 保留 history 属性以兼容旧代码
        self.history: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, user_input: str) -> AgentResponse:
        """Main entry：处理用户输入并返回结构化结果.
        
        支持上下文感知的意图分类，利用对话历史理解指代和隐含意图。
        """
        # 1. 添加上下文感知的查询改写（指代消解）
        rewritten_query = self.memory.rewrite_query(user_input)
        
        # 2. 意图识别（传入对话历史以支持上下文理解）
        memory_context = self.memory.to_dict().get("turns", [])
        intent_result = self.intent_classifier.classify(rewritten_query, context=memory_context)

        # 3. 规划步骤（纯规则，便于观察 Agent 行为）
        steps = self._plan(rewritten_query, intent_result)

        # 4. 执行计划（可能涉及 Tool 调用）
        content, tool_name, tool_result, steps = self._act_and_summarize(
            user_input=rewritten_query,
            intent_result=intent_result,
            steps=steps,
        )

        # 5. 保存到新的 Memory 系统
        self.memory.add_user_message(user_input, intent=intent_result.intent.value)
        self.memory.add_assistant_message(
            content, 
            intent=intent_result.intent.value,
            tool_called=tool_name
        )
        
        # 6. 同时保留旧 history 以兼容
        self.history.append(
            {
                "user": user_input,
                "agent": content,
                "intent": intent_result.intent.value,
            }
        )

        return AgentResponse(
            content=content,
            intent=intent_result.intent,
            tool_called=tool_name,
            tool_result=tool_result.data if tool_result else None,
            confidence=intent_result.confidence,
            steps=steps,
        )

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history.clear()
        self.memory.clear()

    def get_history(self) -> List[Dict[str, str]]:
        """Return shallow copy of conversation history."""
        return list(self.history)
    
    def get_memory(self) -> ConversationMemory:
        """Return the conversation memory instance."""
        return self.memory

    # ------------------------------------------------------------------
    # Internal helpers：Plan → Act → Summarize
    # ------------------------------------------------------------------
    def _plan(self, user_input: str, intent_result: IntentResult) -> List[AgentStep]:
        """Generate a small, human-readable plan for current turn."""
        intent = intent_result.intent
        steps: List[AgentStep] = []

        if intent == IntentType.QUERY:
            steps.append(
                AgentStep(
                    step=1,
                    thought="分析用户问题，确定需要在知识库中检索相关内容。",
                    action="准备调用 query_knowledge_hub 进行混合检索。",
                )
            )
            steps.append(
                AgentStep(
                    step=2,
                    thought="从检索结果中挑选最相关的片段，组织成回答。",
                )
            )
        elif intent == IntentType.LIST_COLLECTIONS:
            steps.append(
                AgentStep(
                    step=1,
                    thought="用户想了解当前有哪些文档集合。",
                    action="调用 list_collections 获取集合列表。",
                )
            )
        elif intent == IntentType.GET_SUMMARY:
            steps.append(
                AgentStep(
                    step=1,
                    thought="用户想查看某个文档的结构化信息。",
                    action="调用 get_document_summary 按 source_path 聚合元数据。",
                )
            )
        elif intent == IntentType.CHAT:
            steps.append(
                AgentStep(
                    step=1,
                    thought="判断为普通闲聊或帮助类问题，直接给出自然语言回复。",
                )
            )
        else:
            steps.append(
                AgentStep(
                    step=1,
                    thought="无法明确识别意图，先以普通对话方式回应。",
                )
            )

        return steps

    def _act_and_summarize(
        self,
        user_input: str,
        intent_result: IntentResult,
        steps: List[AgentStep],
    ):
        """Execute tools if needed and build final natural-language response."""
        intent = intent_result.intent
        tool_result = None
        tool_name: Optional[str] = None

        # ---- CHAT / UNKNOWN：不调用工具 ----
        if intent in (IntentType.CHAT, IntentType.UNKNOWN):
            content = self._handle_chat(user_input)
            if steps:
                steps[-1].observation = "已直接给出自然语言回复（未调用工具）。"
            return content, tool_name, tool_result, steps

        # ---- QUERY / LIST_COLLECTIONS / GET_SUMMARY：调用工具 ----
        try:
            if intent == IntentType.QUERY:
                tool_name = ToolName.QUERY_KNOWLEDGE_HUB.value
                query = intent_result.parameters.get("query", user_input)
                tool_result = self.tool_registry.execute(
                    tool_name,
                    query=query,
                    top_k=5,
                    collection="default",
                )
                if steps:
                    steps[0].observation = (
                        "已完成混合检索，获得若干相关文档片段。"
                    )
                content = self._format_query_response(
                    query=query,
                    data=tool_result.data if tool_result and tool_result.success else None,
                    error=tool_result.error if tool_result and not tool_result.success else None,
                )

            elif intent == IntentType.LIST_COLLECTIONS:
                tool_name = ToolName.LIST_COLLECTIONS.value
                tool_result = self.tool_registry.execute(tool_name)
                if steps:
                    steps[0].observation = "已获取当前向量库中的集合列表。"
                content = self._format_collections_response(
                    data=tool_result.data if tool_result and tool_result.success else None,
                    error=tool_result.error if tool_result and not tool_result.success else None,
                )

            elif intent == IntentType.GET_SUMMARY:
                tool_name = ToolName.GET_DOCUMENT_SUMMARY.value
                doc_id = intent_result.parameters.get("doc_id") or user_input
                tool_result = self.tool_registry.execute(
                    tool_name,
                    source_path=doc_id,
                )
                if steps:
                    steps[0].observation = "已根据 source_path 聚合该文档在向量库中的信息。"
                content = self._format_summary_response(
                    source_hint=doc_id,
                    data=tool_result.data if tool_result and tool_result.success else None,
                    error=tool_result.error if tool_result and not tool_result.success else None,
                )
            else:
                # 理论上不会走到这里，留一个兜底
                content = "我已收到你的请求，但当前 Agent 配置还不支持这个操作。"

        except Exception as e:  # 避免 Agent 抛异常到最外层
            content = f"执行内部步骤时发生错误：{e}"

        return content, tool_name, tool_result, steps

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------
    def _format_query_response(
        self,
        query: str,
        data: Optional[Dict[str, Any]],
        error: Optional[str],
    ) -> str:
        """Format query results into user-facing answer."""
        if error:
            return f"抱歉，在查询「{query}」时出错了：{error}"

        if not data:
            return f"抱歉，目前无法获取到与「{query}」相关的检索结果。"

        results = data.get("results", [])
        if not results:
            return f"抱歉，没有找到与「{query}」相关的文档。"

        response_lines: List[str] = []
        response_lines.append(
            f"为你找到 {len(results)} 条与「{query}」相关的内容，这里是最相关的几条：\n"
        )

        for idx, item in enumerate(results[:3], start=1):
            content = item.get("content", "")
            score = item.get("score", 0.0)
            source = item.get("source", "unknown")
            response_lines.append(f"**{idx}. 片段预览**")
            response_lines.append(content.strip()[:200] + "...\n")
            response_lines.append(
                f"   📄 来源: {source} | 相关度: {score:.3f}\n"
            )

        if len(results) > 3:
            response_lines.append(f"... 另外还有 {len(results) - 3} 条结果未展示。")

        response_lines.append(
            "\n如果你需要，我可以帮你围绕这些结果继续总结、对比或写一个小结。"
        )
        return "\n".join(response_lines)

    def _format_collections_response(
        self,
        data: Optional[Dict[str, Any]],
        error: Optional[str],
    ) -> str:
        """Format list-collections result."""
        if error:
            return f"抱歉，获取文档集合列表时出错了：{error}"

        if not data:
            return "当前还没有可用的文档集合，建议先运行摄取（ingestion）流程。"

        collections = data.get("collections", {})
        total = data.get("total_collections", 0)
        if not collections:
            return "当前知识库中还没有任何文档集合，请先上传并摄取文档。"

        lines: List[str] = []
        lines.append(f"知识库中共有 **{total}** 个文档集合：\n")
        for name, info in collections.items():
            doc_count = info.get("document_count", 0)
            chunk_count = info.get("total_chunks", 0)
            image_count = info.get("total_images", 0)
            lines.append(f"📁 **{name}**")
            lines.append(f"   - 文档数: {doc_count}")
            lines.append(f"   - 总分块数: {chunk_count}")
            lines.append(f"   - 图片数: {image_count}\n")

        lines.append("你可以指定其中某个集合，让我只在该集合内进行检索。")
        return "\n".join(lines)

    def _format_summary_response(
        self,
        source_hint: str,
        data: Optional[Dict[str, Any]],
        error: Optional[str],
    ) -> str:
        """Format get-document-summary result."""
        if error:
            return f"抱歉，没有在知识库中找到与「{source_hint}」匹配的文档：{error}"

        if not data:
            return f"暂时无法获取到「{source_hint}」的摘要信息，请确认该文档是否已经完成摄取。"

        source_path = data.get("source_path", source_hint)
        collection = data.get("collection", "default")
        chunk_count = data.get("chunk_count", 0)
        image_count = data.get("image_count", 0)
        matching_sources = data.get("matching_sources", [])

        lines: List[str] = []
        lines.append("📄 **文档索引信息**\n")
        lines.append(f"- 路径 / 标识: {source_path}")
        lines.append(f"- 所在集合: {collection}")
        lines.append(f"- 分块数量: {chunk_count}")
        lines.append(f"- 图片数量: {image_count}")
        if matching_sources:
            lines.append(f"- 命中的底层 source 字段: {', '.join(matching_sources)}")

        lines.append(
            "\n如果你愿意，我可以基于该文档相关的分块，再帮你做更高层次的总结。"
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Very simple chat handler（不依赖 LLM，保证 Demo 稳定）
    # ------------------------------------------------------------------
    def _handle_chat(self, user_input: str) -> str:
        """Simple rule-based chat for non-RAG intents."""
        text = user_input.lower()

        # Greetings
        if any(kw in text for kw in ["你好", "hello", "hi", "嗨"]):
            return (
                "你好，我是基于 Modular RAG MCP Server 的小助手。\n"
                "我可以帮你：查询知识库、查看文档集合，或者根据你上传的文档做总结和说明。"
            )

        # Help
        if any(kw in text for kw in ["帮助", "help", "能做什么", "功能"]):
            return (
                "我目前支持这些能力：\n"
                "1. 🔍 查询知识库：例如「查询论文结论」「什么是电子舌」\n"
                "2. 📁 查看文档集合：例如「列出所有集合」「有哪些文档」\n"
                "3. 📄 查看单个文档索引信息：例如「总结这篇论文」+ 指定路径\n"
                "你可以直接用自然语言告诉我你的问题。"
            )

        # Thanks
        if any(kw in text for kw in ["谢谢", "thanks", "thank you"]):
            return "不客气！如果你还有其他关于知识库或文档的问题，也可以继续问我。"

        # Default
        return (
            "我大致理解了你的问题。如果你希望我结合知识库来回答，"
            "可以尝试用类似「查询 xxx」的方式提问，或者先问「有哪些文档」。"
        )


