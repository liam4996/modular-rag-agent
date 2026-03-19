"""
Phase 2 容错机制测试脚本

测试场景：
1. 正常检索（无需优化）
2. 需要优化的检索（触发 Refine）
3. 无法回答的查询（触发兜底）

注意：本测试使用 Mock LLM，不需要 API key
"""

import sys
from pathlib import Path
from typing import Any

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.multi_agent import (
    AgentState,
    EvalAgent,
    RefineAgent,
    EvaluationResult,
)

# Mock LLM for testing
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage


class MockLLM(BaseChatModel):
    """Mock LLM for testing without API key"""
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Mock generate that returns predefined responses"""
        from langchain_core.outputs import ChatGeneration, ChatResult
        
        # 获取 prompt 内容
        prompt_str = str(messages)
        
        response_content = self._get_mock_response(prompt_str)
        
        message = AIMessage(content=response_content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
    
    def _llm_type(self):
        return "mock_llm"
    
    def _get_mock_response(self, prompt_str: str) -> str:
        """根据 prompt 内容返回不同的 mock 响应"""
        
        if "Query: 什么是 RAG 技术" in prompt_str and "Local Results" in prompt_str:
            # 好结果的评估
            return """{
                "relevance": 0.95,
                "diversity": 0.85,
                "coverage": 0.90,
                "confidence": 0.90,
                "need_refinement": false,
                "fallback_suggested": false,
                "reason": "结果相关性很高，覆盖了查询的主要方面"
            }"""
        
        elif "Query: 公司 2099 年的战略" in prompt_str:
            # 无结果的评估
            return """{
                "relevance": 0.1,
                "diversity": 0.0,
                "coverage": 0.0,
                "confidence": 0.05,
                "need_refinement": true,
                "fallback_suggested": true,
                "reason": "未找到任何相关结果"
            }"""
        
        elif "Query: 我昨天晚饭吃了什么" in prompt_str:
            # 不可能回答的问题
            return """{
                "relevance": 0.0,
                "diversity": 0.0,
                "coverage": 0.0,
                "confidence": 0.0,
                "need_refinement": false,
                "fallback_suggested": true,
                "reason": "查询涉及用户隐私，系统无法获取"
            }"""
        
        elif "Query: 一个很难的问题" in prompt_str and "Retry Count: 2" in prompt_str:
            # 达到最大重试次数
            return """{
                "relevance": 0.3,
                "diversity": 0.3,
                "coverage": 0.3,
                "confidence": 0.3,
                "need_refinement": true,
                "fallback_suggested": true,
                "reason": "已达到最大重试次数"
            }"""
        
        elif "Original Query: RAG 技术" in prompt_str:
            # 优化查询
            return """{
                "refined_query": "RAG 检索增强生成 技术原理 2025 年最新进展 详细说明",
                "changes_made": ["添加了完整术语", "添加了时间限定", "添加了详细程度要求"],
                "reasoning": "基于评估反馈，添加了更具体的限定词"
            }"""
        
        elif "Original Query: AI 发展" in prompt_str:
            return """{
                "refined_query": "AI 人工智能 2025 年 2026 年 最新进展 发展趋势 行业分析",
                "changes_made": ["添加了完整术语", "添加了最新年份", "添加了多个维度"],
                "reasoning": "基于评估反馈，添加时间限定和多角度分析"
            }"""
        
        elif "Original Query: 公司战略" in prompt_str:
            return """{
                "refined_query": "公司 2026 年战略规划文档 详细内容 业务方向",
                "changes_made": ["添加了年份", "指定文档类型", "添加了业务方向"],
                "reasoning": "基于评估反馈，具体化查询"
            }"""
        
        # 默认响应
        return """{
            "relevance": 0.5,
            "diversity": 0.5,
            "coverage": 0.5,
            "confidence": 0.5,
            "need_refinement": false,
            "fallback_suggested": false,
            "reason": "默认评估结果"
        }"""


def create_mock_llm():
    """创建 Mock LLM 用于测试"""
    return MockLLM()


def test_eval_agent():
    """测试 Eval Agent"""
    print("\n" + "="*80)
    print("测试 1: Eval Agent - 评估检索结果")
    print("="*80)
    
    # 初始化
    llm = create_mock_llm()
    eval_agent = EvalAgent(llm)
    
    # 测试场景 1: 有好结果
    print("\n场景 1: 有好结果")
    local_results = [
        {
            "content": "RAG（Retrieval-Augmented Generation）是一种检索增强生成技术，"
                      "通过结合检索器和生成器来提高语言模型的性能。",
            "source": "RAG 原理文档",
        }
    ]
    web_results = []
    
    evaluation = eval_agent.evaluate(
        local_results=local_results,
        web_results=web_results,
        query="什么是 RAG 技术？",
        retry_count=0,
        max_retries=2
    )
    
    print(f"  相关性：{evaluation.relevance:.2f}")
    print(f"  多样性：{evaluation.diversity:.2f}")
    print(f"  覆盖度：{evaluation.coverage:.2f}")
    print(f"  置信度：{evaluation.confidence:.2f}")
    print(f"  需要优化：{evaluation.need_refinement}")
    print(f"  建议兜底：{evaluation.fallback_suggested}")
    print(f"  评估理由：{evaluation.reason}")
    
    # 测试场景 2: 无结果
    print("\n场景 2: 无结果")
    local_results = []
    web_results = []
    
    evaluation = eval_agent.evaluate(
        local_results=local_results,
        web_results=web_results,
        query="公司 2099 年的战略是什么？",
        retry_count=0,
        max_retries=2
    )
    
    print(f"  相关性：{evaluation.relevance:.2f}")
    print(f"  多样性：{evaluation.diversity:.2f}")
    print(f"  覆盖度：{evaluation.coverage:.2f}")
    print(f"  置信度：{evaluation.confidence:.2f}")
    print(f"  需要优化：{evaluation.need_refinement}")
    print(f"  建议兜底：{evaluation.fallback_suggested}")
    print(f"  评估理由：{evaluation.reason}")
    
    # 测试场景 3: 不可能回答的问题
    print("\n场景 3: 不可能回答的问题")
    evaluation = eval_agent.evaluate(
        local_results=[],
        web_results=[],
        query="我昨天晚饭吃了什么？",
        retry_count=0,
        max_retries=2
    )
    
    print(f"  相关性：{evaluation.relevance:.2f}")
    print(f"  多样性：{evaluation.diversity:.2f}")
    print(f"  覆盖度：{evaluation.coverage:.2f}")
    print(f"  置信度：{evaluation.confidence:.2f}")
    print(f"  需要优化：{evaluation.need_refinement}")
    print(f"  建议兜底：{evaluation.fallback_suggested}")
    print(f"  评估理由：{evaluation.reason}")
    
    # 测试场景 4: 达到最大重试次数
    print("\n场景 4: 达到最大重试次数")
    evaluation = eval_agent.evaluate(
        local_results=[],
        web_results=[],
        query="一个很难的问题",
        retry_count=2,  # 已达到最大重试次数
        max_retries=2
    )
    
    print(f"  相关性：{evaluation.relevance:.2f}")
    print(f"  多样性：{evaluation.diversity:.2f}")
    print(f"  覆盖度：{evaluation.coverage:.2f}")
    print(f"  置信度：{evaluation.confidence:.2f}")
    print(f"  需要优化：{evaluation.need_refinement}")
    print(f"  建议兜底：{evaluation.fallback_suggested}")
    print(f"  评估理由：{evaluation.reason}")
    
    print("\n✅ Eval Agent 测试完成")


def test_refine_agent():
    """测试 Refine Agent"""
    print("\n" + "="*80)
    print("测试 2: Refine Agent - 查询优化")
    print("="*80)
    
    # 初始化
    llm = create_mock_llm()
    refine_agent = RefineAgent(llm)
    
    # 测试场景 1: 查询太泛
    print("\n场景 1: 查询太泛")
    evaluation = EvaluationResult(
        relevance=0.5,
        diversity=0.4,
        coverage=0.5,
        confidence=0.45,
        need_refinement=True,
        fallback_suggested=False,
        reason="结果太泛泛，没有具体原理说明"
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
    
    # 测试场景 2: 结果不够新
    print("\n场景 2: 结果不够新")
    evaluation = EvaluationResult(
        relevance=0.7,
        diversity=0.6,
        coverage=0.6,
        confidence=0.5,
        need_refinement=True,
        fallback_suggested=False,
        reason="结果不够新，需要最新进展"
    )
    
    refinement = refine_agent.refine(
        original_query="AI 发展",
        evaluation=evaluation,
        retry_count=1
    )
    
    print(f"  原始查询：AI 发展")
    print(f"  优化查询：{refinement.refined_query}")
    print(f"  改动：{refinement.changes_made}")
    print(f"  理由：{refinement.reasoning}")
    
    # 测试场景 3: 未找到具体文档
    print("\n场景 3: 未找到具体文档")
    evaluation = EvaluationResult(
        relevance=0.3,
        diversity=0.3,
        coverage=0.4,
        confidence=0.35,
        need_refinement=True,
        fallback_suggested=False,
        reason="未找到具体文档"
    )
    
    refinement = refine_agent.refine(
        original_query="公司战略",
        evaluation=evaluation,
        retry_count=0
    )
    
    print(f"  原始查询：公司战略")
    print(f"  优化查询：{refinement.refined_query}")
    print(f"  改动：{refinement.changes_made}")
    print(f"  理由：{refinement.reasoning}")
    
    print("\n✅ Refine Agent 测试完成")


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
    
    # 第一次重试
    print(f"\n第一次重试:")
    state.increment_retry("refine")
    print(f"  retry_count: {state.retry_count}")
    print(f"  should_fallback: {state.should_fallback}")
    
    # 第二次重试
    print(f"\n第二次重试:")
    state.increment_retry("refine")
    print(f"  retry_count: {state.retry_count}")
    print(f"  should_fallback: {state.should_fallback}")
    
    # 触发兜底
    print(f"\n触发兜底:")
    from src.agent.multi_agent import FallbackReason
    state.trigger_fallback(FallbackReason.MAX_RETRIES_EXCEEDED, "eval")
    print(f"  fallback_triggered: {state.fallback_triggered}")
    print(f"  fallback_reason: {state.fallback_reason}")
    print(f"  should_fallback: {state.should_fallback}")
    
    print("\n✅ AgentState 重试控制测试完成")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🧪 Phase 2 容错机制测试")
    print("="*80)
    
    try:
        test_eval_agent()
        test_refine_agent()
        test_state_retry_control()
        
        print("\n" + "="*80)
        print("✅ 所有 Phase 2 测试通过！")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ 测试失败：{str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
