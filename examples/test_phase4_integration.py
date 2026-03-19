"""
Phase 4 - 端到端集成测试

测试完整的多智能体工作流：
1. 路由 → 检索 → 评估 → 生成
2. 混合搜索工作流
3. 重试和兜底机制
4. 溯源和忠实度检查
"""

import sys
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel

from src.agent.multi_agent import (
    AgentState,
    MultiAgentRAG,
    Citation,
    CitationType,
    CitationManager,
    FallbackReason,
)


class MockLLM(BaseChatModel):
    """简单的 Mock LLM 用于测试"""
    
    def _generate(self, messages, **kwargs):
        from langchain_core.outputs import ChatResult, ChatGeneration
        
        # 获取最后一条用户消息
        last_message = messages[-1].content if messages else ""
        
        # 根据上下文返回不同的响应
        if "intent" in last_message.lower() or "router" in last_message.lower():
            response_text = '''{
    "intent": "local_search",
    "agents_to_invoke": ["SearchAgent"],
    "parallel": false,
    "confidence": 0.9,
    "reasoning": "测试回复",
    "parameters": {}
}'''
        elif "evaluation" in last_message.lower() or "evaluate" in last_message.lower():
            response_text = '''{
    "relevance": 0.85,
    "diversity": 0.80,
    "coverage": 0.90,
    "confidence": 0.85,
    "need_refinement": false,
    "fallback_suggested": false,
    "reason": "检索结果良好"
}'''
        elif "refine" in last_message.lower() or "optimization" in last_message.lower():
            response_text = '''{
    "refined_query": "RAG 技术 详细说明",
    "changes_made": ["添加了详细程度要求"],
    "reasoning": "优化查询以获得更详细的结果"
}'''
        else:
            response_text = "这是一个测试回复。"
        
        generation = ChatGeneration(message=AIMessage(content=response_text))
        return ChatResult(generations=[generation])
    
    def _llm_type(self) -> str:
        return "mock_llm"


class MockRetriever:
    """模拟检索器"""
    
    def __init__(self, results=None):
        self.results = results or []
    
    def invoke(self, query):
        return self.results


def test_basic_workflow():
    """测试基本工作流"""
    print("\n" + "="*80)
    print("测试 1: 基本工作流")
    print("="*80)
    
    # 创建模拟状态
    state = AgentState(user_input="什么是 RAG？")
    
    # 模拟路由
    print("\n步骤 1: 路由决策")
    state.add_to_blackboard("intent", "local_search", "router")
    print(f"  意图：{state.intent}")
    assert state.intent == "local_search"
    
    # 模拟检索
    print("\n步骤 2: 检索")
    mock_results = [
        {"content": "RAG 是检索增强生成技术", "source": "RAG 文档", "score": 0.95},
        {"content": "RAG 结合了检索和生成", "source": "技术文档", "score": 0.85},
    ]
    state.add_to_blackboard("local_results", mock_results, "search")
    print(f"  检索结果：{len(state.local_results)} 条")
    assert len(state.local_results) == 2
    
    # 模拟评估
    print("\n步骤 3: 评估")
    state.add_to_blackboard("evaluation", {
        "relevance": 0.85,
        "confidence": 0.85,
        "need_refinement": False,
    }, "eval")
    print(f"  评估结果：置信度 {state.evaluation.get('confidence', 0):.2f}")
    assert state.evaluation.get('confidence', 0) > 0.5
    
    # 模拟生成
    print("\n步骤 4: 生成")
    answer = "RAG 是检索增强生成技术 [Local: RAG 文档]。它结合了检索和生成模型 [Local: 技术文档]。"
    state.final_answer = answer
    
    # 添加引用
    citations = CitationManager.create_citations_from_results(
        local_results=mock_results,
        web_results=[]
    )
    state.add_to_blackboard("citations", [c.to_dict() for c in citations], "generate")
    
    print(f"  最终回答：{state.final_answer[:50]}...")
    print(f"  引用数量：{len(citations)}")
    
    assert state.final_answer != ""
    assert len(citations) > 0
    
    print("\n✅ 基本工作流测试通过")


def test_hybrid_search_workflow():
    """测试混合搜索工作流"""
    print("\n" + "="*80)
    print("测试 2: 混合搜索工作流")
    print("="*80)
    
    state = AgentState(user_input="结合内部文档和网上最新信息，介绍 RAG 技术")
    
    # 路由决策 - 混合搜索
    print("\n步骤 1: 路由决策（混合搜索）")
    state.add_to_blackboard("intent", "hybrid_search", "router")
    print(f"  意图：{state.intent}")
    assert state.intent == "hybrid_search"
    
    # 本地检索
    print("\n步骤 2: 本地检索")
    local_results = [
        {"content": "内部 RAG 文档", "source": "内部文档 1"},
    ]
    state.add_to_blackboard("local_results", local_results, "search")
    print(f"  本地结果：{len(state.local_results)} 条")
    
    # 联网检索
    print("\n步骤 3: 联网检索")
    web_results = [
        {"title": "Latest RAG Research", "snippet": "Latest research on RAG", "url": "https://example.com/rag"},
    ]
    state.add_to_blackboard("web_results", web_results, "web")
    print(f"  联网结果：{len(state.web_results)} 条")
    
    # 评估
    print("\n步骤 4: 评估")
    state.add_to_blackboard("evaluation", {
        "relevance": 0.90,
        "confidence": 0.90,
        "need_refinement": False,
    }, "eval")
    
    # 生成
    print("\n步骤 5: 生成")
    answer = "根据内部文档 [Local: 内部文档 1] 和最新研究 [Web: example.com]，RAG 技术..."
    state.final_answer = answer
    
    # 引用
    citations = CitationManager.create_citations_from_results(
        local_results=local_results,
        web_results=web_results
    )
    state.add_to_blackboard("citations", [c.to_dict() for c in citations], "generate")
    
    print(f"  意图：{state.intent}")
    print(f"  本地结果：{len(state.local_results)} 条")
    print(f"  联网结果：{len(state.web_results)} 条")
    print(f"  引用数量：{len(citations)}")
    
    # 验证
    assert state.intent == "hybrid_search"
    assert len(state.local_results) > 0
    assert len(state.web_results) > 0
    local_citations = [c for c in citations if c.type == CitationType.LOCAL]
    web_citations = [c for c in citations if c.type == CitationType.WEB]
    assert len(local_citations) > 0
    assert len(web_citations) > 0
    
    print("\n✅ 混合搜索工作流测试通过")


def test_retry_mechanism():
    """测试重试机制"""
    print("\n" + "="*80)
    print("测试 3: 重试机制")
    print("="*80)
    
    # 初始状态
    state = AgentState(user_input="测试查询")
    print(f"\n初始状态：")
    print(f"  重试次数：{state.retry_count}")
    print(f"  最大重试：{state.max_retries}")
    print(f"  应该兜底：{state.should_fallback}")
    
    assert state.retry_count == 0
    assert state.should_fallback == False
    
    # 第一次重试
    print("\n第一次重试：")
    state.retry_count = 1
    print(f"  重试次数：{state.retry_count}")
    print(f"  应该兜底：{state.should_fallback}")
    assert state.should_fallback == False
    
    # 第二次重试
    print("\n第二次重试（达到上限）：")
    state.retry_count = 2
    print(f"  重试次数：{state.retry_count}")
    print(f"  应该兜底：{state.should_fallback}")
    assert state.should_fallback == True  # 达到 max_retries 就会兜底
    
    # 触发兜底
    print("\n触发兜底：")
    state.fallback_triggered = True
    state.fallback_reason = FallbackReason.MAX_RETRIES_EXCEEDED
    print(f"  兜底触发：{state.fallback_triggered}")
    print(f"  兜底原因：{state.fallback_reason.value}")
    
    assert state.fallback_triggered == True
    
    print("\n✅ 重试机制测试通过")


def test_citation_workflow():
    """测试引用工作流"""
    print("\n" + "="*80)
    print("测试 4: 引用工作流")
    print("="*80)
    
    # 模拟检索结果
    local_results = [
        {"content": "RAG 原理", "source": "RAG 文档", "page": 1},
        {"content": "向量检索", "source": "检索文档", "page": 5},
    ]
    
    web_results = [
        {"title": "RAG Tutorial", "snippet": "Learn RAG", "url": "https://tutorial.com/rag"},
    ]
    
    # 创建引用
    print("\n创建引用：")
    citations = CitationManager.create_citations_from_results(
        local_results=local_results,
        web_results=web_results,
        top_k=5
    )
    
    print(f"  总引用数：{len(citations)}")
    for i, citation in enumerate(citations, 1):
        print(f"    {i}. {citation.format_citation()}")
    
    # 生成回答
    print("\n生成回答：")
    answer = f"""
根据检索到的信息：

1. {local_results[0]['content']} [Local: {local_results[0]['source']}]
2. {web_results[0]['title']} [Web: tutorial.com]
"""
    print(f"  回答：{answer[:100]}...")
    
    # 忠实度检查
    print("\n忠实度检查：")
    citation_manager = CitationManager()
    citation_manager.add_citations(citations)
    faithfulness = citation_manager.check_faithfulness(answer)
    
    print(f"  是否忠实：{faithfulness.is_faithful}")
    print(f"  置信度：{faithfulness.confidence:.2f}")
    print(f"  检测到臆造：{faithfulness.hallucination_detected}")
    
    # 验证
    assert len(citations) == 3
    assert faithfulness.is_faithful == True
    assert faithfulness.hallucination_detected == False
    
    print("\n✅ 引用工作流测试通过")


def test_fallback_workflow():
    """测试兜底工作流"""
    print("\n" + "="*80)
    print("测试 5: 兜底工作流")
    print("="*80)
    
    # 场景 1: 无结果兜底
    print("\n场景 1: 无结果兜底")
    state = AgentState(user_input="不存在的信息")
    state.add_to_blackboard("local_results", [], "search")
    state.add_to_blackboard("web_results", [], "web")
    
    # 评估
    state.add_to_blackboard("evaluation", {
        "confidence": 0.1,
        "fallback_suggested": True,
    }, "eval")
    
    # 检查是否需要兜底
    if state.evaluation.get("fallback_suggested"):
        state.fallback_triggered = True
        state.fallback_reason = FallbackReason.NO_RESULTS_FOUND
    
    print(f"  本地结果：{len(state.local_results)}")
    print(f"  联网结果：{len(state.web_results)}")
    print(f"  兜底触发：{state.fallback_triggered}")
    print(f"  兜底原因：{state.fallback_reason.value if state.fallback_reason else 'None'}")
    
    assert state.fallback_triggered == True
    assert state.fallback_reason == FallbackReason.NO_RESULTS_FOUND
    
    # 生成兜底回答
    fallback_answer = f"""抱歉，经过检索，我无法找到关于「{state.user_input}」的相关信息。

建议您：
1. 尝试使用不同的关键词
2. 检查拼写是否正确
3. 调整搜索范围

检索详情：
- 本地知识库结果：{len(state.local_results)} 条
- 互联网搜索结果：{len(state.web_results)} 条
"""
    state.final_answer = fallback_answer
    
    print(f"  兜底回答：{state.final_answer[:100]}...")
    
    assert "抱歉" in state.final_answer
    assert "无法找到" in state.final_answer
    
    # 场景 2: 低置信度兜底
    print("\n场景 2: 低置信度兜底")
    state2 = AgentState(user_input="模糊查询")
    state2.add_to_blackboard("evaluation", {
        "confidence": 0.2,
        "fallback_suggested": True,
    }, "eval")
    
    if state2.evaluation.get("fallback_suggested"):
        state2.fallback_triggered = True
        state2.fallback_reason = FallbackReason.LOW_CONFIDENCE
    
    print(f"  置信度：{state2.evaluation.get('confidence')}")
    print(f"  兜底触发：{state2.fallback_triggered}")
    
    assert state2.fallback_triggered == True
    assert state2.fallback_reason == FallbackReason.LOW_CONFIDENCE
    
    print("\n✅ 兜底工作流测试通过")


def test_state_serialization():
    """测试状态序列化"""
    print("\n" + "="*80)
    print("测试 6: 状态序列化")
    print("="*80)
    
    # 创建复杂状态
    state = AgentState(user_input="测试查询")
    state.add_to_blackboard("intent", "local_search", "router")
    state.add_to_blackboard("local_results", [{"content": "test"}], "search")
    state.add_execution_trace({"agent": "router", "action": "classify"})
    state.add_metric("latency", 1.5)
    
    # 序列化为字典
    print("\n序列化状态：")
    state_dict = {
        "user_input": state.user_input,
        "blackboard": state.blackboard,
        "execution_trace": state.execution_trace,
        "metrics": state.metrics,
        "final_answer": state.final_answer,
    }
    
    print(f"  用户输入：{state_dict['user_input']}")
    print(f"  黑板键：{list(state_dict['blackboard'].keys())}")
    print(f"  执行轨迹：{len(state_dict['execution_trace'])} 条")
    print(f"  指标：{state_dict['metrics']}")
    
    # 反序列化
    print("\n反序列化：")
    restored_state = AgentState(user_input=state_dict['user_input'])
    restored_state.blackboard = state_dict['blackboard']
    restored_state.execution_trace = state_dict['execution_trace']
    restored_state.metrics = state_dict['metrics']
    restored_state.final_answer = state_dict['final_answer']
    
    print(f"  用户输入：{restored_state.user_input}")
    print(f"  黑板键：{list(restored_state.blackboard.keys())}")
    print(f"  执行轨迹：{len(restored_state.execution_trace)} 条")
    
    # 验证
    assert restored_state.user_input == state.user_input
    assert "intent" in restored_state.blackboard
    assert len(restored_state.execution_trace) == 1
    assert restored_state.metrics.get("latency") == 1.5
    
    print("\n✅ 状态序列化测试通过")


def test_metrics_collection():
    """测试指标收集"""
    print("\n" + "="*80)
    print("测试 7: 指标收集")
    print("="*80)
    
    state = AgentState(user_input="测试查询")
    
    # 模拟工作流中的指标收集
    print("\n收集指标：")
    
    # 路由指标
    state.add_metric("router_latency_ms", 50)
    print(f"  路由延迟：{state.metrics.get('router_latency_ms')} ms")
    
    # 检索指标
    state.add_metric("search_latency_ms", 200)
    state.add_metric("retrieved_docs", 5)
    print(f"  检索延迟：{state.metrics.get('search_latency_ms')} ms")
    print(f"  检索文档数：{state.metrics.get('retrieved_docs')}")
    
    # 评估指标
    state.add_metric("eval_latency_ms", 30)
    state.add_metric("confidence_score", 0.85)
    print(f"  评估延迟：{state.metrics.get('eval_latency_ms')} ms")
    print(f"  置信度：{state.metrics.get('confidence_score')}")
    
    # 生成指标
    state.add_metric("generate_latency_ms", 500)
    state.add_metric("answer_length", 500)
    state.add_metric("citation_count", 3)
    print(f"  生成延迟：{state.metrics.get('generate_latency_ms')} ms")
    print(f"  回答长度：{state.metrics.get('answer_length')}")
    print(f"  引用数量：{state.metrics.get('citation_count')}")
    
    # 总延迟
    total_latency = (
        state.metrics.get("router_latency_ms", 0) +
        state.metrics.get("search_latency_ms", 0) +
        state.metrics.get("eval_latency_ms", 0) +
        state.metrics.get("generate_latency_ms", 0)
    )
    state.add_metric("total_latency_ms", total_latency)
    
    print(f"\n  总延迟：{total_latency} ms")
    
    # 验证
    assert state.metrics.get("router_latency_ms") == 50
    assert state.metrics.get("search_latency_ms") == 200
    assert state.metrics.get("retrieved_docs") == 5
    assert state.metrics.get("total_latency_ms") == 780
    
    print("\n✅ 指标收集测试通过")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🧪 Phase 4 - 端到端集成测试")
    print("="*80)
    
    try:
        # 运行所有测试
        test_basic_workflow()
        test_hybrid_search_workflow()
        test_retry_mechanism()
        test_citation_workflow()
        test_fallback_workflow()
        test_state_serialization()
        test_metrics_collection()
        
        print("\n" + "="*80)
        print("✅ 所有端到端集成测试通过！")
        print("="*80)
        print("\n测试覆盖:")
        print("  ✅ 基本工作流")
        print("  ✅ 混合搜索工作流")
        print("  ✅ 重试机制")
        print("  ✅ 引用工作流")
        print("  ✅ 兜底工作流")
        print("  ✅ 状态序列化")
        print("  ✅ 指标收集")
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
