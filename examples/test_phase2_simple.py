"""
Phase 2 容错机制 - 简单单元测试

测试场景：
1. Eval Agent 的强制规则应用
2. Refine Agent 的基本功能
3. AgentState 的重试控制
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.multi_agent import (
    AgentState,
    EvalAgent,
    RefineAgent,
    EvaluationResult,
    FallbackReason,
)


def test_eval_agent_rules():
    """测试 Eval Agent 的强制规则"""
    print("\n" + "="*80)
    print("测试 1: Eval Agent - 强制规则应用")
    print("="*80)
    
    # 初始化（使用 Mock LLM）
    from unittest.mock import Mock
    llm = Mock()
    eval_agent = EvalAgent(llm)
    
    # 测试场景 1: 相关性太低 → 自动触发兜底
    print("\n场景 1: 相关性太低 (< 0.2)")
    result = {
        "relevance": 0.15,
        "diversity": 0.5,
        "coverage": 0.5,
        "confidence": 0.4,
        "need_refinement": True,
        "fallback_suggested": False,
        "reason": "结果不相关"
    }
    
    evaluation = eval_agent._apply_rules(
        result=result,
        query="测试查询",
        retry_count=0,
        max_retries=2
    )
    
    print(f"  输入相关性：0.15")
    print(f"  输出 fallback_suggested: {evaluation.fallback_suggested}")
    print(f"  理由：{evaluation.reason}")
    assert evaluation.fallback_suggested == True, "相关性 < 0.2 应该触发兜底"
    assert "[规则触发：相关性过低]" in evaluation.reason
    print("  ✅ 通过")
    
    # 测试场景 2: 达到最大重试次数 → 自动触发兜底
    print("\n场景 2: 达到最大重试次数")
    result = {
        "relevance": 0.5,
        "diversity": 0.5,
        "coverage": 0.5,
        "confidence": 0.5,
        "need_refinement": True,
        "fallback_suggested": False,
        "reason": "需要优化"
    }
    
    evaluation = eval_agent._apply_rules(
        result=result,
        query="测试查询",
        retry_count=2,  # 已达到最大重试次数
        max_retries=2
    )
    
    print(f"  retry_count: 2/2")
    print(f"  输出 fallback_suggested: {evaluation.fallback_suggested}")
    print(f"  理由：{evaluation.reason}")
    assert evaluation.fallback_suggested == True, "达到最大重试次数应该触发兜底"
    assert "[规则触发：已达到最大重试次数 2]" in evaluation.reason
    print("  ✅ 通过")
    
    # 测试场景 3: 不可能回答的问题 → 自动触发兜底
    print("\n场景 3: 不可能回答的问题")
    result = {
        "relevance": 0.5,
        "diversity": 0.5,
        "coverage": 0.5,
        "confidence": 0.5,
        "need_refinement": False,
        "fallback_suggested": False,
        "reason": "正常"
    }
    
    evaluation = eval_agent._apply_rules(
        result=result,
        query="我昨天晚饭吃了什么",
        retry_count=0,
        max_retries=2
    )
    
    print(f"  查询：我昨天晚饭吃了什么")
    print(f"  输出 fallback_suggested: {evaluation.fallback_suggested}")
    print(f"  理由：{evaluation.reason}")
    assert evaluation.fallback_suggested == True, "不可能回答的问题应该触发兜底"
    assert "[规则触发：查询涉及无法获取的信息]" in evaluation.reason
    print("  ✅ 通过")
    
    # 测试场景 4: 置信度 < 0.7 → 需要优化
    print("\n场景 4: 置信度 < 0.7")
    result = {
        "relevance": 0.6,
        "diversity": 0.6,
        "coverage": 0.6,
        "confidence": 0.65,
        "need_refinement": False,
        "fallback_suggested": False,
        "reason": "一般"
    }
    
    evaluation = eval_agent._apply_rules(
        result=result,
        query="测试查询",
        retry_count=0,
        max_retries=2
    )
    
    print(f"  输入置信度：0.65")
    print(f"  输出 need_refinement: {evaluation.need_refinement}")
    assert evaluation.need_refinement == True, "置信度 < 0.7 需要优化"
    print("  ✅ 通过")
    
    print("\n✅ Eval Agent 规则测试全部通过")


def test_refine_agent_basic():
    """测试 Refine Agent 基本功能"""
    print("\n" + "="*80)
    print("测试 2: Refine Agent - 基本功能")
    print("="*80)
    
    # 初始化
    from unittest.mock import Mock
    llm = Mock()
    refine_agent = RefineAgent(llm)
    
    # 测试场景：解析失败时的默认优化
    print("\n场景：LLM 响应解析失败")
    
    # Mock LLM 返回无效响应
    llm.invoke = Mock(return_value=Mock(content="无效的 JSON 响应"))
    
    evaluation = EvaluationResult(
        relevance=0.5,
        diversity=0.5,
        coverage=0.5,
        confidence=0.5,
        need_refinement=True,
        fallback_suggested=False,
        reason="需要优化"
    )
    
    refinement = refine_agent.refine(
        original_query="RAG 技术",
        evaluation=evaluation,
        retry_count=0
    )
    
    print(f"  原始查询：RAG 技术")
    print(f"  优化查询：{refinement.refined_query}")
    print(f"  改动：{refinement.changes_made}")
    print(f"  理由：{refinement.reasoning}")
    
    # 解析失败时应该有默认优化
    assert refinement.refined_query is not None
    assert len(refinement.changes_made) > 0
    print("  ✅ 通过 - 解析失败时使用默认优化策略")
    
    print("\n✅ Refine Agent 基本功能测试通过")


def test_state_retry_control():
    """测试 AgentState 的重试控制"""
    print("\n" + "="*80)
    print("测试 3: AgentState - 重试控制")
    print("="*80)
    
    # 初始化状态
    state = AgentState(
        user_input="测试查询",
        max_retries=2
    )
    
    print(f"\n初始状态:")
    print(f"  retry_count: {state.retry_count}")
    print(f"  max_retries: {state.max_retries}")
    print(f"  should_fallback: {state.should_fallback}")
    assert state.retry_count == 0
    assert state.should_fallback == False
    print("  ✅ 初始状态正确")
    
    # 第一次重试
    print(f"\n第一次重试:")
    state.increment_retry("refine")
    print(f"  retry_count: {state.retry_count}")
    print(f"  should_fallback: {state.should_fallback}")
    assert state.retry_count == 1
    assert state.should_fallback == False
    print("  ✅ 第一次重试正确")
    
    # 第二次重试
    print(f"\n第二次重试:")
    state.increment_retry("refine")
    print(f"  retry_count: {state.retry_count}")
    print(f"  should_fallback: {state.should_fallback}")
    assert state.retry_count == 2
    assert state.should_fallback == True  # 达到最大重试次数
    print("  ✅ 第二次重试正确 - 触发 should_fallback")
    
    # 触发兜底
    print(f"\n触发兜底:")
    state.trigger_fallback(FallbackReason.MAX_RETRIES_EXCEEDED, "eval")
    print(f"  fallback_triggered: {state.fallback_triggered}")
    print(f"  fallback_reason: {state.fallback_reason}")
    print(f"  should_fallback: {state.should_fallback}")
    assert state.fallback_triggered == True
    assert state.fallback_reason == FallbackReason.MAX_RETRIES_EXCEEDED
    print("  ✅ 兜底触发正确")
    
    print("\n✅ AgentState 重试控制测试完成")


def test_impossible_query_patterns():
    """测试不可能回答的问题模式识别"""
    print("\n" + "="*80)
    print("测试 4: 不可能回答的问题模式识别")
    print("="*80)
    
    from unittest.mock import Mock
    llm = Mock()
    eval_agent = EvalAgent(llm)
    
    # 测试各种不可能回答的模式
    impossible_queries = [
        "我昨天晚饭吃了什么",
        "我前天去了哪里",
        "我上周做了什么",
        "我的隐私信息",
        "我的秘密",
        "我心里在想什么",
        "我猜我今天会遇到谁",
        "我想买什么",
        "我感觉如何",
        "我家住哪里",
        "我老婆喜欢什么",
        "我老公的工作",
        "我孩子的学校",
        "我父母的生日",
    ]
    
    print("\n测试不可能回答的问题模式:")
    for query in impossible_queries:
        is_impossible = eval_agent._is_impossible_query(query)
        print(f"  - '{query}': {'❌ 不可能' if is_impossible else '✅ 可能'}")
        assert is_impossible == True, f"'{query}' 应该被识别为不可能回答的问题"
    
    # 测试正常问题
    normal_queries = [
        "什么是 RAG 技术",
        "公司 2026 年战略",
        "AI 发展趋势",
        "如何优化检索",
    ]
    
    print("\n测试正常问题:")
    for query in normal_queries:
        is_impossible = eval_agent._is_impossible_query(query)
        print(f"  - '{query}': {'❌ 不可能' if is_impossible else '✅ 可能'}")
        assert is_impossible == False, f"'{query}' 应该是正常问题"
    
    print("\n✅ 模式识别测试全部通过")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🧪 Phase 2 容错机制 - 单元测试")
    print("="*80)
    
    try:
        test_eval_agent_rules()
        test_refine_agent_basic()
        test_state_retry_control()
        test_impossible_query_patterns()
        
        print("\n" + "="*80)
        print("✅ 所有 Phase 2 单元测试通过！")
        print("="*80)
        print("\n测试覆盖:")
        print("  ✅ Eval Agent 强制规则")
        print("  ✅ Refine Agent 基本功能")
        print("  ✅ AgentState 重试控制")
        print("  ✅ 不可能回答的问题模式识别")
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
