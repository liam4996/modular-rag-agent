"""
Phase 3 集成测试 - 完整多智能体系统带溯源功能

测试场景：
1. 本地知识检索 + 溯源
2. 联网搜索 + 溯源
3. 混合搜索 + 溯源
4. 忠实度检查
"""

import sys
from pathlib import Path
from typing import Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel

from src.agent.multi_agent import (
    MultiAgentRAG,
    AgentState,
    Citation,
    CitationType,
    CitationManager,
    FaithfulnessCheck,
)


class MockLLM(BaseChatModel):
    """简单的 Mock LLM 用于测试"""
    
    def _generate(self, messages, **kwargs):
        # 返回一个简单的回复
        return AIMessage(content="这是一个测试回复。")
    
    def _llm_type(self) -> str:
        return "mock_llm"


def test_local_search_with_citations():
    """测试本地检索带溯源（使用模拟状态）"""
    print("\n" + "="*80)
    print("测试 1: 本地检索带溯源（模拟）")
    print("="*80)
    
    # 模拟状态
    state = AgentState(user_input="什么是 RAG 技术？")
    
    # 模拟检索结果
    local_results = [
        {
            "content": "RAG（Retrieval-Augmented Generation）是一种检索增强生成技术",
            "source": "RAG 原理文档",
            "page": 5,
        },
    ]
    
    # 添加到黑板
    state.add_to_blackboard("local_results", local_results, "search")
    
    # 模拟创建引用
    citations = CitationManager.create_citations_from_results(
        local_results=local_results,
        web_results=[],
        top_k=5
    )
    
    # 添加引用
    citation_manager = CitationManager()
    citation_manager.add_citations(citations)
    state.add_to_blackboard("citations", [c.to_dict() for c in citations], "generate")
    
    # 模拟生成回答
    answer_parts = []
    for i, result in enumerate(local_results[:3], 1):
        content = result.get("content", "")
        source = result.get("source", "unknown")
        answer_parts.append(f"{i}. {content} [Local: {source}]")
    
    state.final_answer = "\n".join(answer_parts)
    
    # 忠实度检查
    faithfulness = citation_manager.check_faithfulness(state.final_answer)
    state.add_to_blackboard("faithfulness_check", faithfulness.to_dict(), "generate")
    
    # 添加指标
    state.add_metric("generation_mode", "normal_with_citations")
    state.add_metric("citation_count", len(citations))
    state.add_metric("faithfulness_score", faithfulness.confidence)
    
    # 输出
    print(f"\n用户查询：{state.user_input}")
    print(f"\n最终回答:\n{state.final_answer}")
    print(f"\n引用数量：{len(state.blackboard.get('citations', []))}")
    print(f"忠实度检查：{state.blackboard.get('faithfulness_check', {})}")
    print(f"生成模式：{state.metrics.get('generation_mode', 'unknown')}")
    
    # 验证
    assert hasattr(state, 'final_answer')
    assert 'citations' in state.blackboard
    assert state.metrics.get('generation_mode') == 'normal_with_citations'
    assert state.metrics.get('citation_count') == 1
    
    print("\n✅ 本地检索带溯源测试通过")


def test_citation_manager_integration():
    """测试引用管理器集成"""
    print("\n" + "="*80)
    print("测试 2: 引用管理器集成")
    print("="*80)
    
    # 模拟检索结果
    local_results = [
        {
            "content": "RAG（Retrieval-Augmented Generation）是一种检索增强生成技术",
            "source": "RAG 原理文档",
            "page": 5,
            "confidence": 0.95,
            "relevance": 0.90,
        },
        {
            "content": "向量检索是 RAG 系统的核心组件",
            "source": "检索技术手册",
            "page": 10,
            "confidence": 0.85,
            "relevance": 0.80,
        },
    ]
    
    web_results = [
        {
            "title": "Understanding RAG - Towards Data Science",
            "snippet": "RAG combines retrieval and generation for better AI responses",
            "url": "https://towardsdatascience.com/rag-explained",
            "confidence": 0.90,
            "relevance": 0.85,
        },
    ]
    
    # 创建引用
    print("\n从检索结果创建引用...")
    citations = CitationManager.create_citations_from_results(
        local_results=local_results,
        web_results=web_results,
        top_k=5
    )
    
    print(f"创建引用数：{len(citations)}")
    for i, citation in enumerate(citations, 1):
        print(f"  {i}. {citation.format_citation()} (confidence={citation.confidence:.2f})")
    
    # 验证引用
    assert len(citations) == 3
    assert citations[0].type == CitationType.LOCAL
    assert citations[0].source == "RAG 原理文档"
    assert citations[2].type == CitationType.WEB
    assert "towardsdatascience.com" in citations[2].format_citation()
    
    print("\n✅ 引用管理器集成测试通过")


def test_faithfulness_integration():
    """测试忠实度检查集成"""
    print("\n" + "="*80)
    print("测试 3: 忠实度检查集成")
    print("="*80)
    
    # 场景 1: 有引用的回答
    print("\n场景 1: 有引用的回答")
    manager = CitationManager()
    manager.add_citation(Citation(
        type=CitationType.LOCAL,
        source="RAG 文档",
        content="RAG 原理",
        confidence=0.95
    ))
    
    answer_with_citation = "RAG 技术包含检索和生成 [Local: RAG 文档]。"
    check = manager.check_faithfulness(answer_with_citation)
    
    print(f"  回答：{answer_with_citation}")
    print(f"  是否忠实：{check.is_faithful}")
    print(f"  置信度：{check.confidence}")
    print(f"  检测到臆造：{check.hallucination_detected}")
    
    assert check.is_faithful == True
    assert check.hallucination_detected == False
    
    # 场景 2: 无引用的回答
    print("\n场景 2: 无引用的回答")
    manager.clear()
    manager.add_citation(Citation(
        type=CitationType.LOCAL,
        source="RAG 文档",
        content="RAG 原理"
    ))
    
    answer_without_citation = "RAG 技术很重要，但没有引用支持。"
    check = manager.check_faithfulness(answer_without_citation)
    
    print(f"  回答：{answer_without_citation}")
    print(f"  是否忠实：{check.is_faithful}")
    print(f"  置信度：{check.confidence}")
    print(f"  检测到臆造：{check.hallucination_detected}")
    print(f"  不支持的陈述：{check.unsupported_claims}")
    
    assert check.is_faithful == False
    assert check.hallucination_detected == True
    assert len(check.unsupported_claims) > 0
    
    # 场景 3: 缺少引用标记
    print("\n场景 3: 缺少引用标记")
    answer_missing_citation = "RAG 技术包含检索和生成。"
    check = manager.check_faithfulness(answer_missing_citation)
    
    print(f"  回答：{answer_missing_citation}")
    print(f"  是否忠实：{check.is_faithful}")
    print(f"  置信度：{check.confidence}")
    print(f"  检测到臆造：{check.hallucination_detected}")
    
    assert check.is_faithful == False
    assert check.hallucination_detected == True
    
    print("\n✅ 忠实度检查集成测试通过")


def test_format_answer_with_citations():
    """测试带引用的回答格式化"""
    print("\n" + "="*80)
    print("测试 4: 带引用的回答格式化")
    print("="*80)
    
    from src.agent.multi_agent import format_answer_with_citations
    
    # 创建引用
    citations = [
        Citation(
            type=CitationType.LOCAL,
            source="RAG 原理文档",
            content="RAG 技术介绍",
            confidence=0.95,
            page=5
        ),
        Citation(
            type=CitationType.WEB,
            source="Towards Data Science",
            content="RAG explanation",
            url="https://towardsdatascience.com/rag",
            confidence=0.90
        ),
    ]
    
    # 带引用标记的回答
    answer = """
根据检索到的信息：

RAG（Retrieval-Augmented Generation）是一种检索增强生成技术 [Local: RAG 原理文档，p.5]。
它结合了检索模型和生成模型的优势 [Web: towardsdatascience.com]。

RAG 系统的工作流程：
1. 接收用户查询
2. 从知识库中检索相关文档
3. 将检索结果作为上下文输入给生成模型
4. 生成最终回答
"""
    
    # 格式化回答
    formatted = format_answer_with_citations(
        answer=answer,
        citations=citations,
        include_reference_list=True
    )
    
    print("格式化后的回答:")
    print("-" * 80)
    print(formatted)
    print("-" * 80)
    
    # 验证
    assert "## 引用来源" in formatted
    assert "[Local: RAG 原理文档，p.5]" in formatted
    assert "[Web: towardsdatascience.com]" in formatted
    assert "1." in formatted
    assert "2." in formatted
    
    print("\n✅ 带引用的回答格式化测试通过")


def test_citation_confidence_metrics():
    """测试引用置信度指标"""
    print("\n" + "="*80)
    print("测试 5: 引用置信度指标")
    print("="*80)
    
    # 模拟多个检索结果
    local_results = [
        {"content": "结果 1", "source": "文档 1", "confidence": 0.95},
        {"content": "结果 2", "source": "文档 2", "confidence": 0.85},
        {"content": "结果 3", "source": "文档 3", "confidence": 0.75},
        {"content": "结果 4", "source": "文档 4", "confidence": 0.65},
        {"content": "结果 5", "source": "文档 5", "confidence": 0.55},
    ]
    
    citations = CitationManager.create_citations_from_results(
        local_results=local_results,
        web_results=[],
        top_k=5
    )
    
    print("\n引用置信度（按排名递减）:")
    for i, citation in enumerate(citations, 1):
        print(f"  排名{i}: confidence={citation.confidence:.2f}, relevance={citation.relevance:.2f}")
    
    # 验证置信度递减
    expected_confidences = [1.0, 0.9, 0.8, 0.7, 0.6]
    for i, (citation, expected) in enumerate(zip(citations, expected_confidences), 1):
        assert abs(citation.confidence - expected) < 0.01, f"排名{i}的置信度不正确"
    
    print("\n✅ 引用置信度指标测试通过")


def test_state_serialization():
    """测试状态序列化（包含引用信息）"""
    print("\n" + "="*80)
    print("测试 6: 状态序列化")
    print("="*80)
    
    # 创建状态
    state = AgentState(user_input="测试查询")
    
    # 添加引用信息
    citations = [
        Citation(
            type=CitationType.LOCAL,
            source="测试文档",
            content="测试内容",
            confidence=0.9
        ),
    ]
    
    # 添加到黑板
    state.add_to_blackboard("citations", [c.to_dict() for c in citations], "test")
    
    # 添加忠实度检查
    faithfulness = FaithfulnessCheck(
        is_faithful=True,
        hallucination_detected=False,
        confidence=0.9
    )
    state.add_to_blackboard("faithfulness_check", faithfulness.to_dict(), "test")
    
    # 验证
    assert "citations" in state.blackboard
    assert len(state.blackboard["citations"]) == 1
    assert state.blackboard["faithfulness_check"]["is_faithful"] == True
    
    print(f"  引用数量：{len(state.blackboard['citations'])}")
    print(f"  忠实度：{state.blackboard['faithfulness_check']['is_faithful']}")
    print(f"  置信度：{state.blackboard['faithfulness_check']['confidence']}")
    
    print("\n✅ 状态序列化测试通过")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🧪 Phase 3 溯源与忠实度 - 集成测试")
    print("="*80)
    
    try:
        # 运行所有测试
        test_local_search_with_citations()
        test_citation_manager_integration()
        test_faithfulness_integration()
        test_format_answer_with_citations()
        test_citation_confidence_metrics()
        test_state_serialization()
        
        print("\n" + "="*80)
        print("✅ 所有 Phase 3 集成测试通过！")
        print("="*80)
        print("\n测试覆盖:")
        print("  ✅ 本地检索带溯源")
        print("  ✅ 引用管理器集成")
        print("  ✅ 忠实度检查集成")
        print("  ✅ 带引用的回答格式化")
        print("  ✅ 引用置信度指标")
        print("  ✅ 状态序列化")
        print("="*80)
        
    except AssertionError as e:
        print(f"\n❌ 测试失败：{str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 意外错误：{str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
