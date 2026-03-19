"""
Phase 3 溯源与忠实度 - 简单单元测试

测试场景：
1. Citation 数据类功能
2. CitationManager 引用管理
3. 忠实度检查
4. 引用格式化
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.multi_agent import (
    Citation,
    CitationType,
    CitationManager,
    FaithfulnessCheck,
    format_answer_with_citations,
)


def test_citation_dataclass():
    """测试 Citation 数据类"""
    print("\n" + "="*80)
    print("测试 1: Citation 数据类")
    print("="*80)
    
    # 测试本地引用
    print("\n场景 1: 本地引用")
    local_citation = Citation(
        type=CitationType.LOCAL,
        source="RAG 原理文档",
        content="RAG 是一种检索增强生成技术",
        confidence=0.95,
        relevance=0.90,
        page="p.5"
    )
    
    print(f"  类型：{local_citation.type.value}")
    print(f"  来源：{local_citation.source}")
    print(f"  内容：{local_citation.content[:50]}...")
    print(f"  置信度：{local_citation.confidence}")
    print(f"  相关性：{local_citation.relevance}")
    print(f"  页码：{local_citation.page}")
    print(f"  引用标记：{local_citation.format_citation()}")
    
    assert local_citation.type == CitationType.LOCAL
    assert local_citation.format_citation() == "[Local: RAG 原理文档，p.5]"
    print("  ✅ 通过")
    
    # 测试联网引用
    print("\n场景 2: 联网引用")
    web_citation = Citation(
        type=CitationType.WEB,
        source="Wikipedia",
        content="Retrieval-Augmented Generation is a technique",
        confidence=0.90,
        relevance=0.85,
        url="https://en.wikipedia.org/wiki/RAG"
    )
    
    print(f"  类型：{web_citation.type.value}")
    print(f"  来源：{web_citation.source}")
    print(f"  URL: {web_citation.url}")
    print(f"  引用标记：{web_citation.format_citation()}")
    
    assert web_citation.type == CitationType.WEB
    assert "[Web:" in web_citation.format_citation()
    print("  ✅ 通过")
    
    # 测试序列化/反序列化
    print("\n场景 3: 序列化/反序列化")
    citation_dict = local_citation.to_dict()
    restored_citation = Citation.from_dict(citation_dict)
    
    print(f"  原始引用：{local_citation.format_citation()}")
    print(f"  恢复引用：{restored_citation.format_citation()}")
    
    assert restored_citation.source == local_citation.source
    assert restored_citation.type == local_citation.type
    print("  ✅ 通过")
    
    print("\n✅ Citation 数据类测试全部通过")


def test_citation_manager():
    """测试 CitationManager"""
    print("\n" + "="*80)
    print("测试 2: CitationManager 引用管理")
    print("="*80)
    
    # 创建引用管理器
    manager = CitationManager()
    
    # 测试添加引用
    print("\n场景 1: 添加引用")
    local_citation = Citation(
        type=CitationType.LOCAL,
        source="文档 1",
        content="内容 1"
    )
    web_citation = Citation(
        type=CitationType.WEB,
        source="网站 1",
        content="内容 2",
        url="https://example.com"
    )
    
    manager.add_citation(local_citation)
    manager.add_citation(web_citation)
    
    print(f"  总引用数：{len(manager.citations)}")
    print(f"  本地引用数：{len(manager.get_local_citations())}")
    print(f"  联网引用数：{len(manager.get_web_citations())}")
    
    assert len(manager.citations) == 2
    assert len(manager.get_local_citations()) == 1
    assert len(manager.get_web_citations()) == 1
    print("  ✅ 通过")
    
    # 测试从检索结果创建引用
    print("\n场景 2: 从检索结果创建引用")
    local_results = [
        {"content": "RAG 原理", "source": "RAG 文档", "page": 5},
        {"content": "向量检索", "source": "检索技术", "page": 10},
    ]
    web_results = [
        {"title": "AI News", "snippet": "AI latest news", "url": "https://ainews.com/1"},
    ]
    
    citations = CitationManager.create_citations_from_results(
        local_results=local_results,
        web_results=web_results,
        top_k=5
    )
    
    print(f"  创建引用数：{len(citations)}")
    print(f"  本地引用：{len([c for c in citations if c.type == CitationType.LOCAL])}")
    print(f"  联网引用：{len([c for c in citations if c.type == CitationType.WEB])}")
    
    assert len(citations) == 3
    assert citations[0].source == "RAG 文档"
    assert citations[2].url == "https://ainews.com/1"
    print("  ✅ 通过")
    
    # 测试格式化所有引用
    print("\n场景 3: 格式化所有引用")
    manager.clear()
    manager.add_citations(citations)
    
    formatted = manager.format_all_citations()
    print(f"  格式化结果:\n{formatted}")
    
    assert "1." in formatted
    assert "2." in formatted
    assert "3." in formatted
    print("  ✅ 通过")
    
    print("\n✅ CitationManager 引用管理测试全部通过")


def test_faithfulness_check():
    """测试忠实度检查"""
    print("\n" + "="*80)
    print("测试 3: 忠实度检查")
    print("="*80)
    
    # 创建引用管理器
    manager = CitationManager()
    
    # 场景 1: 有引用支持
    print("\n场景 1: 有引用支持")
    manager.add_citation(Citation(
        type=CitationType.LOCAL,
        source="文档 1",
        content="内容 1"
    ))
    
    generated_text = "根据检索结果 [Local: 文档 1]，RAG 技术很重要。"
    check = manager.check_faithfulness(generated_text)
    
    print(f"  生成文本：{generated_text}")
    print(f"  是否忠实：{check.is_faithful}")
    print(f"  置信度：{check.confidence}")
    print(f"  检测到臆造：{check.hallucination_detected}")
    
    assert check.is_faithful == True
    assert check.hallucination_detected == False
    print("  ✅ 通过")
    
    # 场景 2: 无引用支持
    print("\n场景 2: 无引用支持")
    manager.clear()
    
    generated_text = "RAG 技术很重要，但是没有任何引用。"
    check = manager.check_faithfulness(generated_text)
    
    print(f"  生成文本：{generated_text}")
    print(f"  是否忠实：{check.is_faithful}")
    print(f"  置信度：{check.confidence}")
    print(f"  检测到臆造：{check.hallucination_detected}")
    print(f"  不支持的陈述：{check.unsupported_claims}")
    
    assert check.is_faithful == False
    assert check.hallucination_detected == True
    assert len(check.unsupported_claims) > 0
    print("  ✅ 通过")
    
    # 场景 3: 有内容但无引用标记
    print("\n场景 3: 有内容但无引用标记")
    manager.add_citation(Citation(
        type=CitationType.LOCAL,
        source="文档 1",
        content="内容 1"
    ))
    
    generated_text = "RAG 技术很重要，但是缺少引用标记。"
    check = manager.check_faithfulness(generated_text)
    
    print(f"  生成文本：{generated_text}")
    print(f"  是否忠实：{check.is_faithful}")
    print(f"  置信度：{check.confidence}")
    print(f"  检测到臆造：{check.hallucination_detected}")
    
    assert check.is_faithful == False
    assert check.hallucination_detected == True
    print("  ✅ 通过")
    
    # 测试 FaithfulnessCheck 数据类
    print("\n场景 4: FaithfulnessCheck 数据类")
    check = FaithfulnessCheck(
        is_faithful=False,
        hallucination_detected=True,
        unsupported_claims=["陈述 1", "陈述 2"],
        confidence=0.5,
        suggestions=["建议 1", "建议 2"]
    )
    
    check_dict = check.to_dict()
    print(f"  序列化：{check_dict}")
    
    assert check_dict["is_faithful"] == False
    assert len(check_dict["unsupported_claims"]) == 2
    assert len(check_dict["suggestions"]) == 2
    print("  ✅ 通过")
    
    print("\n✅ 忠实度检查测试全部通过")


def test_format_answer_with_citations():
    """测试带引用的回答格式化"""
    print("\n" + "="*80)
    print("测试 4: 带引用的回答格式化")
    print("="*80)
    
    # 场景 1: 回答中已有引用标记
    print("\n场景 1: 回答中已有引用标记")
    answer = "RAG 技术很重要 [Local: 文档 1]。它包含检索和生成 [Web: wikipedia.org]。"
    citations = [
        Citation(type=CitationType.LOCAL, source="文档 1", content="内容 1"),
        Citation(type=CitationType.WEB, source="Wikipedia", content="内容 2", url="https://wikipedia.org"),
    ]
    
    formatted = format_answer_with_citations(answer, citations, include_reference_list=True)
    print(f"  格式化结果:\n{formatted}")
    
    assert "## 引用来源" in formatted
    assert "[Local: 文档 1]" in formatted
    print("  ✅ 通过")
    
    # 场景 2: 回答中无引用标记
    print("\n场景 2: 回答中无引用标记")
    answer = "RAG 技术很重要。"
    citations = [
        Citation(type=CitationType.LOCAL, source="文档 1", content="内容 1"),
    ]
    
    formatted = format_answer_with_citations(answer, citations, include_reference_list=True)
    print(f"  格式化结果:\n{formatted}")
    
    assert "## 引用来源" in formatted
    assert "[Local: 文档 1]" in formatted
    print("  ✅ 通过")
    
    # 场景 3: 不包含引用列表
    print("\n场景 3: 不包含引用列表")
    formatted = format_answer_with_citations(answer, citations, include_reference_list=False)
    print(f"  格式化结果:\n{formatted}")
    
    assert "## 引用来源" not in formatted
    print("  ✅ 通过")
    
    # 场景 4: 无引用
    print("\n场景 4: 无引用")
    answer = "没有找到相关信息。"
    citations = []
    
    formatted = format_answer_with_citations(answer, citations, include_reference_list=True)
    print(f"  格式化结果:\n{formatted}")
    
    assert formatted == answer
    print("  ✅ 通过")
    
    print("\n✅ 带引用的回答格式化测试全部通过")


def test_citation_confidence_decay():
    """测试引用置信度递减"""
    print("\n" + "="*80)
    print("测试 5: 引用置信度递减")
    print("="*80)
    
    local_results = [
        {"content": "结果 1", "source": "文档 1"},
        {"content": "结果 2", "source": "文档 2"},
        {"content": "结果 3", "source": "文档 3"},
        {"content": "结果 4", "source": "文档 4"},
        {"content": "结果 5", "source": "文档 5"},
    ]
    
    citations = CitationManager.create_citations_from_results(
        local_results=local_results,
        web_results=[],
        top_k=5
    )
    
    print("\n引用置信度:")
    for i, citation in enumerate(citations, 1):
        print(f"  {i}. {citation.source}: confidence={citation.confidence:.2f}, relevance={citation.relevance:.2f}")
    
    # 验证置信度递减
    assert citations[0].confidence == 1.0
    assert citations[1].confidence == 0.9
    assert citations[2].confidence == 0.8
    assert citations[3].confidence == 0.7
    assert citations[4].confidence == 0.6  # max(0.5, 1.0 - 4*0.1)
    print("  ✅ 通过 - 置信度正确递减")
    
    print("\n✅ 引用置信度递减测试全部通过")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🧪 Phase 3 溯源与忠实度 - 单元测试")
    print("="*80)
    
    try:
        test_citation_dataclass()
        test_citation_manager()
        test_faithfulness_check()
        test_format_answer_with_citations()
        test_citation_confidence_decay()
        
        print("\n" + "="*80)
        print("✅ 所有 Phase 3 单元测试通过！")
        print("="*80)
        print("\n测试覆盖:")
        print("  ✅ Citation 数据类（本地/联网引用、序列化）")
        print("  ✅ CitationManager 引用管理（创建、格式化）")
        print("  ✅ 忠实度检查（有引用、无引用、缺少标记）")
        print("  ✅ 带引用的回答格式化")
        print("  ✅ 引用置信度递减")
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
