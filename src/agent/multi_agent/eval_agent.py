"""
多智能体 RAG 系统 - Eval Agent

职责：
- 评估检索结果质量
- 控制重试逻辑
- 判断是否触发兜底
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from langchain_core.language_models import BaseLLM
from langchain_core.prompts import ChatPromptTemplate
import json


@dataclass
class EvaluationResult:
    """评估结果"""
    relevance: float  # 相关性 (0-1)
    diversity: float  # 多样性 (0-1)
    coverage: float  # 覆盖度 (0-1)
    confidence: float  # 置信度 (0-1)
    need_refinement: bool  # 是否需要优化
    fallback_suggested: bool = False  # 是否建议兜底
    reason: str = ""  # 评估理由


class EvalAgent:
    """
    评估智能体
    
    职责：
    - 评估检索结果的相关性、多样性、覆盖度
    - 判断是否需要优化查询重新检索
    - 判断是否应该触发兜底回复
    
    评估维度：
    1. Relevance: 结果与查询的相关性
    2. Diversity: 结果是否覆盖不同角度
    3. Coverage: 是否回答了查询的所有部分
    4. Confidence: 整体置信度
    
    特殊处理：
    - 如果结果完全不相关 → 建议兜底
    - 如果查询不可能有答案（如"我昨天晚饭吃了什么"）→ 建议兜底
    - 如果 retry_count >= max_retries → 建议兜底
    """
    
    SYSTEM_PROMPT = """You are an evaluator for search results in a multi-agent RAG system.

Evaluate the search results based on:

1. **Relevance** (0-1): How relevant are the results to the query?
   - 1.0: Perfect match, all results directly address the query
   - 0.5: Some results are relevant, others are tangential
   - 0.0: Completely irrelevant

2. **Diversity** (0-1): Do the results cover different aspects?
   - 1.0: Results cover multiple perspectives and aspects
   - 0.5: Results are somewhat repetitive
   - 0.0: All results say the same thing

3. **Coverage** (0-1): Are all parts of the query addressed?
   - 1.0: All aspects of the query are thoroughly covered
   - 0.5: Some parts are covered, others missed
   - 0.0: Query is not addressed at all

4. **Confidence** (0-1): Overall confidence in the results
   - Based on above three metrics

Special cases - MUST suggest fallback:
- Results are completely irrelevant (relevance < 0.2)
- Query asks for impossible information (e.g., "我昨天晚饭吃了什么", personal privacy)
- retry_count >= max_retries (no point in refining further)

Respond in JSON format:
```json
{{
    "relevance": 0.85,
    "diversity": 0.80,
    "coverage": 0.90,
    "confidence": 0.85,
    "need_refinement": true,
    "fallback_suggested": false,
    "reason": "详细说明评估理由"
}}
```

Rules:
- need_refinement = true when confidence < 0.7
- fallback_suggested = true when:
  - relevance < 0.2 (completely irrelevant)
  - query is impossible to answer
  - retry_count >= max_retries"""

    def __init__(self, llm: BaseLLM):
        """
        初始化 Eval Agent
        
        Args:
            llm: 语言模型实例
        """
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("user", "{context}")
        ])
        self.chain = self.prompt | self.llm
    
    def evaluate(
        self,
        local_results: List[Dict[str, Any]],
        web_results: List[Dict[str, Any]],
        query: str,
        retry_count: int = 0,
        max_retries: int = 2
    ) -> EvaluationResult:
        """
        评估检索结果
        
        Args:
            local_results: 本地知识库检索结果
            web_results: 联网搜索结果
            query: 用户查询
            retry_count: 当前重试次数
            max_retries: 最大重试次数
        
        Returns:
            EvaluationResult 评估结果
        """
        # 构建评估上下文
        context = self._build_context(
            local_results=local_results,
            web_results=web_results,
            query=query,
            retry_count=retry_count,
            max_retries=max_retries
        )
        
        # 调用 LLM 评估
        response = self.chain.invoke({"context": context})
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        # 解析 JSON
        try:
            result = self._parse_json_response(response_content)
        except Exception as e:
            # 解析失败，返回默认评估
            return EvaluationResult(
                relevance=0.5,
                diversity=0.5,
                coverage=0.5,
                confidence=0.5,
                need_refinement=True,
                fallback_suggested=False,
                reason=f"评估结果解析失败：{str(e)}"
            )
        
        # 强制规则检查
        evaluation = self._apply_rules(
            result=result,
            query=query,
            retry_count=retry_count,
            max_retries=max_retries
        )
        
        return evaluation
    
    def _build_context(
        self,
        local_results: List[Dict],
        web_results: List[Dict],
        query: str,
        retry_count: int,
        max_retries: int
    ) -> str:
        """构建评估上下文"""
        context_parts = [
            f"Query: {query}",
            f"Retry Count: {retry_count}/{max_retries}",
            "",
            "=== Local Knowledge Base Results ===",
        ]
        
        if local_results:
            for i, result in enumerate(local_results, 1):
                content = result.get('content', result.get('text', str(result)))
                source = result.get('source', result.get('title', f'Doc{i}'))
                context_parts.append(f"[Local {i}] Source: {source}")
                context_parts.append(f"Content: {content[:500]}...")  # 限制长度
                context_parts.append("")
        else:
            context_parts.append("No local results found")
        
        context_parts.append("")
        context_parts.append("=== Web Search Results ===")
        
        if web_results:
            for i, result in enumerate(web_results, 1):
                content = result.get('snippet', result.get('content', str(result)))
                source = result.get('source', result.get('title', f'Web{i}'))
                url = result.get('url', '')
                context_parts.append(f"[Web {i}] Source: {source}")
                context_parts.append(f"URL: {url}")
                context_parts.append(f"Content: {content[:500]}...")
                context_parts.append("")
        else:
            context_parts.append("No web results found")
        
        return "\n".join(context_parts)
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """解析 LLM 的 JSON 响应"""
        # 尝试提取 JSON
        import re
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        else:
            # 尝试直接解析
            return json.loads(response)
    
    def _apply_rules(
        self,
        result: Dict[str, Any],
        query: str,
        retry_count: int,
        max_retries: int
    ) -> EvaluationResult:
        """应用强制规则"""
        relevance = float(result.get('relevance', 0.5))
        diversity = float(result.get('diversity', 0.5))
        coverage = float(result.get('coverage', 0.5))
        confidence = float(result.get('confidence', 0.5))
        need_refinement = result.get('need_refinement', False)
        fallback_suggested = result.get('fallback_suggested', False)
        reason = result.get('reason', '')
        
        # 规则 1: 相关性太低 → 兜底
        if relevance < 0.2:
            fallback_suggested = True
            reason += " [规则触发：相关性过低]"
        
        # 规则 2: 达到最大重试次数 → 兜底
        if retry_count >= max_retries:
            fallback_suggested = True
            reason += f" [规则触发：已达到最大重试次数 {max_retries}]"
        
        # 规则 3: 查询本身不可能有答案 → 兜底
        if self._is_impossible_query(query):
            fallback_suggested = True
            reason += " [规则触发：查询涉及无法获取的信息]"
        
        # 规则 4: 置信度 < 0.7 → 需要优化
        if confidence < 0.7 and not fallback_suggested:
            need_refinement = True
        
        return EvaluationResult(
            relevance=relevance,
            diversity=diversity,
            coverage=coverage,
            confidence=confidence,
            need_refinement=need_refinement,
            fallback_suggested=fallback_suggested,
            reason=reason
        )
    
    def _is_impossible_query(self, query: str) -> bool:
        """判断查询是否涉及系统无法获取的个人隐私信息。

        Only triggers for queries that *start* with personal-temporal
        patterns (e.g. "我昨天吃了什么"), avoiding false positives on
        legitimate queries like "我想了解..." or "我感觉这篇论文...".
        """
        q = query.strip()
        starts_with_patterns = [
            "我昨天", "我前天", "我上周", "我上个月",
            "我的隐私", "我的秘密", "我心里",
        ]
        contains_patterns = [
            "我家地址", "我的密码",
        ]
        return (
            any(q.startswith(p) for p in starts_with_patterns)
            or any(p in q for p in contains_patterns)
        )
