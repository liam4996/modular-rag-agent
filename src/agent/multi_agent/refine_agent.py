"""
多智能体 RAG 系统 - Refine Agent

职责：
- 分析为什么检索结果不理想
- 改写查询以提高检索质量
- 记录重试次数
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from langchain_core.language_models import BaseLLM
from langchain_core.prompts import ChatPromptTemplate
import json

from .eval_agent import EvaluationResult


@dataclass
class RefinementResult:
    """优化结果"""
    refined_query: str  # 优化后的查询
    changes_made: List[str]  # 做了哪些改动
    reasoning: str  # 优化理由


class RefineAgent:
    """
    查询优化智能体
    
    职责：
    - 分析 Eval Agent 的反馈
    - 改写查询使其更具体、更有效
    - 添加时间限定、详细程度要求等
    
    优化策略：
    1. 添加时间限定（如"2025 年最新进展"）
    2. 添加详细程度（如"详细说明"、"原理解析"）
    3. 添加同义词扩展
    4. 添加上下文信息
    
    示例：
    - Original: "RAG 技术"
      Refined: "RAG 检索增强生成 技术原理 2025 年最新进展"
    
    - Original: "公司战略"
      Refined: "公司 2026 年战略规划文档 详细内容"
    """
    
    SYSTEM_PROMPT = """You are a query refinement specialist for a multi-agent RAG system.

Your task:
1. Analyze why previous search results were inadequate (based on evaluation feedback)
2. Rewrite the query to be more specific and effective
3. Add context, time constraints, or detailed requirements if needed

Optimization strategies:
- Add time constraints: "2025 年最新进展", "2026 年规划"
- Add detail level: "详细说明", "原理解析", "深度分析"
- Add synonyms: expand technical terms
- Add context: provide background information

Examples:

Example 1:
- Original: "RAG 技术"
- Evaluation: "结果太泛泛，没有具体原理说明"
- Refined: "RAG 检索增强生成 技术原理 2025 年最新进展 详细说明"
- Changes: ["添加了完整术语", "添加了时间限定", "添加了详细程度要求"]

Example 2:
- Original: "公司战略"
- Evaluation: "未找到具体文档"
- Refined: "公司 2026 年战略规划文档 详细内容 业务方向"
- Changes: ["添加了年份", "指定文档类型", "添加了业务方向"]

Example 3:
- Original: "AI 发展"
- Evaluation: "结果不够新"
- Refined: "AI 人工智能 2025 年 2026 年 最新进展 发展趋势 行业分析"
- Changes: ["添加了完整术语", "添加了最新年份", "添加了多个维度"]

Respond in JSON format:
```json
{{
    "refined_query": "优化后的查询",
    "changes_made": ["添加了时间限定", "添加了详细程度要求"],
    "reasoning": "为什么这样改写，基于评估反馈的哪些点"
}}
```"""

    def __init__(self, llm: BaseLLM):
        """
        初始化 Refine Agent
        
        Args:
            llm: 语言模型实例
        """
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("user", "{context}")
        ])
        self.chain = self.prompt | self.llm
    
    def refine(
        self,
        original_query: str,
        evaluation: EvaluationResult,
        retry_count: int = 0
    ) -> RefinementResult:
        """
        优化查询
        
        Args:
            original_query: 原始查询
            evaluation: Eval Agent 的评估结果
            retry_count: 当前重试次数（用于记录）
        
        Returns:
            RefinementResult 优化结果
        """
        # 构建优化上下文
        context = self._build_context(
            original_query=original_query,
            evaluation=evaluation,
            retry_count=retry_count
        )
        
        # 调用 LLM 优化
        response = self.chain.invoke({"context": context})
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        # 解析 JSON
        try:
            result = self._parse_json_response(response_content)
        except Exception as e:
            # 解析失败，返回简单的优化
            return RefinementResult(
                refined_query=f"{original_query} 详细说明 最新进展",
                changes_made=["添加了详细程度要求", "添加了时间限定"],
                reasoning=f"LLM 响应解析失败，使用默认优化策略：{str(e)}"
            )
        
        return RefinementResult(
            refined_query=result.get('refined_query', original_query),
            changes_made=result.get('changes_made', []),
            reasoning=result.get('reasoning', '')
        )
    
    def _build_context(
        self,
        original_query: str,
        evaluation: EvaluationResult,
        retry_count: int
    ) -> str:
        """构建优化上下文"""
        context_parts = [
            f"Original Query: {original_query}",
            f"Retry Count: {retry_count}",
            "",
            "=== Evaluation Feedback ===",
            f"Relevance: {evaluation.relevance:.2f}",
            f"Diversity: {evaluation.diversity:.2f}",
            f"Coverage: {evaluation.coverage:.2f}",
            f"Confidence: {evaluation.confidence:.2f}",
            f"Need Refinement: {evaluation.need_refinement}",
            f"Fallback Suggested: {evaluation.fallback_suggested}",
            "",
            f"Reason: {evaluation.reason}",
        ]
        
        return "\n".join(context_parts)
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """解析 LLM 的 JSON 响应"""
        import re
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        else:
            return json.loads(response)
