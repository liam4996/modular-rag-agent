"""
多智能体 RAG 系统 - Citation 溯源模块

提供：
- Citation 数据类（引用来源）
- CitationManager（引用管理器）
- 忠实度检查工具
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum


class CitationType(Enum):
    """引用类型"""
    LOCAL = "local"  # 本地知识库
    WEB = "web"  # 互联网


@dataclass
class Citation:
    """
    引用来源
    
    Attributes:
        type: 引用类型（local 或 web）
        source: 来源名称（文档名或网站名）
        url: URL（仅 web 类型）
        content: 引用的具体内容
        confidence: 置信度 (0-1)
        relevance: 相关性 (0-1)
        page: 页码或段落号（可选）
        metadata: 额外元数据
    """
    type: CitationType
    source: str
    content: str
    confidence: float = 1.0
    relevance: float = 1.0
    url: Optional[str] = None
    page: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def format_citation(self) -> str:
        """
        格式化引用标记
        
        Returns:
            格式化的引用字符串，如 [Local: 文档名] 或 [Web: URL]
        """
        if self.type == CitationType.LOCAL:
            if self.page:
                # 避免重复 p. 前缀，处理 int 和 str 类型
                page_str = str(self.page)
                if not page_str.startswith("p."):
                    page_str = f"p.{page_str}"
                return f"[Local: {self.source}，{page_str}]"
            else:
                return f"[Local: {self.source}]"
        else:  # WEB
            if self.url:
                # 简化 URL 显示
                from urllib.parse import urlparse
                try:
                    parsed = urlparse(self.url)
                    domain = parsed.netloc.replace('www.', '')
                    return f"[Web: {domain}]"
                except:
                    return f"[Web: {self.url}]"
            else:
                return f"[Web: {self.source}]"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "type": self.type.value,
            "source": self.source,
            "url": self.url,
            "content": self.content,
            "confidence": self.confidence,
            "relevance": self.relevance,
            "page": self.page,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Citation":
        """从字典创建"""
        return cls(
            type=CitationType(data.get("type", "local")),
            source=data.get("source", "unknown"),
            content=data.get("content", ""),
            confidence=data.get("confidence", 1.0),
            relevance=data.get("relevance", 1.0),
            url=data.get("url"),
            page=data.get("page"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class FaithfulnessCheck:
    """
    忠实度检查结果
    
    Attributes:
        is_faithful: 是否忠实于检索内容
        hallucination_detected: 是否检测到臆造
        unsupported_claims: 不支持的陈述列表
        confidence: 忠实度置信度
        suggestions: 改进建议
    """
    is_faithful: bool = True
    hallucination_detected: bool = False
    unsupported_claims: List[str] = field(default_factory=list)
    confidence: float = 1.0
    suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "is_faithful": self.is_faithful,
            "hallucination_detected": self.hallucination_detected,
            "unsupported_claims": self.unsupported_claims,
            "confidence": self.confidence,
            "suggestions": self.suggestions,
        }


class CitationManager:
    """
    引用管理器
    
    职责：
    - 从检索结果生成 Citation
    - 管理引用列表
    - 格式化引用标记
    - 检查忠实度
    """
    
    def __init__(self):
        """初始化引用管理器"""
        self.citations: List[Citation] = []
    
    def add_citation(self, citation: Citation):
        """
        添加引用
        
        Args:
            citation: Citation 对象
        """
        self.citations.append(citation)
    
    def add_citations(self, citations: List[Citation]):
        """批量添加引用"""
        self.citations.extend(citations)
    
    def clear(self):
        """清空引用列表"""
        self.citations = []
    
    def get_citations_by_type(self, citation_type: CitationType) -> List[Citation]:
        """按类型获取引用"""
        return [c for c in self.citations if c.type == citation_type]
    
    def get_local_citations(self) -> List[Citation]:
        """获取本地引用"""
        return self.get_citations_by_type(CitationType.LOCAL)
    
    def get_web_citations(self) -> List[Citation]:
        """获取联网引用"""
        return self.get_citations_by_type(CitationType.WEB)
    
    def format_all_citations(self) -> str:
        """
        格式化所有引用
        
        Returns:
            格式化的引用列表字符串
        """
        if not self.citations:
            return "无引用来源"
        
        formatted = []
        for i, citation in enumerate(self.citations, 1):
            formatted.append(f"{i}. {citation.format_citation()}")
        
        return "\n".join(formatted)
    
    @staticmethod
    def create_citation_from_local_result(
        result: Dict[str, Any],
        confidence: float = 1.0,
        relevance: float = 1.0
    ) -> Citation:
        """
        从本地检索结果创建 Citation
        
        Args:
            result: 本地检索结果字典
            confidence: 置信度
            relevance: 相关性
        
        Returns:
            Citation 对象
        """
        return Citation(
            type=CitationType.LOCAL,
            source=result.get("source", result.get("title", "unknown")),
            content=result.get("content", result.get("text", "")),
            confidence=confidence,
            relevance=relevance,
            page=result.get("page", result.get("chunk_id")),
            metadata=result.get("metadata", {}),
        )
    
    @staticmethod
    def create_citation_from_web_result(
        result: Dict[str, Any],
        confidence: float = 1.0,
        relevance: float = 1.0
    ) -> Citation:
        """
        从联网搜索结果创建 Citation
        
        Args:
            result: 联网搜索结果字典
            confidence: 置信度
            relevance: 相关性
        
        Returns:
            Citation 对象
        """
        return Citation(
            type=CitationType.WEB,
            source=result.get("title", "unknown"),
            content=result.get("snippet", result.get("content", "")),
            confidence=confidence,
            relevance=relevance,
            url=result.get("url"),
            metadata={
                "engine": result.get("engine", ""),
                "time_range": result.get("time_range", ""),
            },
        )
    
    @staticmethod
    def create_citations_from_results(
        local_results: List[Dict[str, Any]],
        web_results: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Citation]:
        """
        从检索结果批量创建 Citation
        
        Args:
            local_results: 本地检索结果列表
            web_results: 联网搜索结果列表
            top_k: 每个来源最多返回的引用数
        
        Returns:
            Citation 列表
        """
        citations = []
        
        # 本地引用
        for i, result in enumerate(local_results[:top_k]):
            # 根据排名递减置信度和相关性
            confidence = max(0.5, 1.0 - (i * 0.1))
            relevance = max(0.5, 1.0 - (i * 0.1))
            
            citation = CitationManager.create_citation_from_local_result(
                result=result,
                confidence=confidence,
                relevance=relevance
            )
            citations.append(citation)
        
        # 联网引用
        for i, result in enumerate(web_results[:top_k]):
            confidence = max(0.5, 1.0 - (i * 0.1))
            relevance = max(0.5, 1.0 - (i * 0.1))
            
            citation = CitationManager.create_citation_from_web_result(
                result=result,
                confidence=confidence,
                relevance=relevance
            )
            citations.append(citation)
        
        return citations
    
    def check_faithfulness(
        self,
        generated_text: str,
        threshold: float = 0.7,
    ) -> FaithfulnessCheck:
        """Check whether *generated_text* is grounded in the managed citations.

        Strategy (rule-based, no extra LLM call):
        1. **Citation presence** – does the answer contain any reference markers?
           Supports ``[Local: …]``, ``[Web: …]``, and numbered ``[1]``…``[N]``.
        2. **Content overlap** – for each citation, what fraction of its key
           n-grams appear in the generated text?  A high overlap means the
           answer is likely paraphrasing the source rather than hallucinating.
        3. **Coverage** – what fraction of citations are actually "used"
           (i.e. their content overlaps with the generated text)?

        Returns:
            FaithfulnessCheck with an aggregate confidence score.
        """
        import re

        if not self.citations:
            return FaithfulnessCheck(
                is_faithful=False,
                hallucination_detected=True,
                unsupported_claims=["生成文本没有任何引用支持"],
                confidence=0.0,
                suggestions=["请确保回答基于检索结果"],
            )

        # --- 1. Citation-marker check -----------------------------------
        marker_patterns = [
            r"\[Local:",       # [Local: source]
            r"\[Web:",         # [Web: domain]
            r"\[\d+\]",       # [1], [2], ...
        ]
        has_markers = any(
            re.search(pat, generated_text) for pat in marker_patterns
        )

        # --- 2. Content-overlap check -----------------------------------
        def _extract_ngrams(text: str, n: int = 4) -> set:
            """Extract character n-grams from *text* (lowercased, whitespace-collapsed)."""
            t = re.sub(r"\s+", " ", text.lower().strip())
            if len(t) < n:
                return {t}
            return {t[i:i + n] for i in range(len(t) - n + 1)}

        gen_ngrams = _extract_ngrams(generated_text)
        if not gen_ngrams:
            return FaithfulnessCheck(
                is_faithful=False,
                hallucination_detected=True,
                unsupported_claims=["生成文本为空"],
                confidence=0.0,
            )

        used_citations = 0
        overlap_scores: list[float] = []

        for citation in self.citations:
            cit_ngrams = _extract_ngrams(citation.content)
            if not cit_ngrams:
                continue
            overlap = len(cit_ngrams & gen_ngrams) / len(cit_ngrams)
            overlap_scores.append(overlap)
            if overlap > 0.15:
                used_citations += 1

        # --- 3. Aggregate scores ----------------------------------------
        coverage = used_citations / len(self.citations) if self.citations else 0.0
        avg_overlap = (
            sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0.0
        )

        # Weighted confidence: marker presence (20%) + overlap (50%) + coverage (30%)
        marker_score = 1.0 if has_markers else 0.3
        confidence = 0.2 * marker_score + 0.5 * min(avg_overlap * 3, 1.0) + 0.3 * coverage
        confidence = round(min(confidence, 1.0), 2)

        is_faithful = confidence >= threshold
        unsupported: list[str] = []
        suggestions: list[str] = []

        if not has_markers:
            suggestions.append("建议在回答中标注引用来源编号")
        if coverage < 0.3:
            unsupported.append(
                f"仅 {used_citations}/{len(self.citations)} 条引用的内容在回答中有体现"
            )
            suggestions.append("建议更多地引用检索结果中的原文信息")
        if avg_overlap < 0.1:
            unsupported.append("回答内容与检索结果的文本重叠度很低")
            suggestions.append("注意是否存在脱离原文的臆造内容")

        return FaithfulnessCheck(
            is_faithful=is_faithful,
            hallucination_detected=not is_faithful,
            unsupported_claims=unsupported,
            confidence=confidence,
            suggestions=suggestions,
        )


def format_answer_with_citations(
    answer: str,
    citations: List[Citation],
    include_reference_list: bool = True
) -> str:
    """
    格式化带引用的回答
    
    Args:
        answer: 原始回答
        citations: 引用列表
        include_reference_list: 是否在末尾包含引用列表
    
    Returns:
        格式化后的回答
    """
    if not citations:
        return answer
    
    # 如果回答中已经有引用标记，直接返回
    if "[Local:" in answer or "[Web:" in answer:
        if include_reference_list:
            reference_list = "\n\n## 引用来源\n" + "\n".join([
                f"{i}. {citation.format_citation()}"
                for i, citation in enumerate(citations, 1)
            ])
            return answer + reference_list
        return answer
    
    # 简单地在末尾添加引用列表
    if include_reference_list:
        reference_list = "\n\n## 引用来源\n" + "\n".join([
            f"{i}. {citation.format_citation()}"
            for i, citation in enumerate(citations, 1)
        ])
        return answer + reference_list
    
    return answer
