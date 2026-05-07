"""E2E 测试 run_stream() 全流程 — Mock LLM，无需真实 API。

用法: python examples/test_run_stream_mock.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import MagicMock, patch
from src.agent.multi_agent import MultiAgentRAG


def test_classify_and_plan_prompt():
    """测试 PLAN_PROMPT 模板变量解析正确性"""
    from src.agent.multi_agent.supervisor_agent import SupervisorAgent
    from langchain_core.prompts import ChatPromptTemplate

    s = SupervisorAgent.__new__(SupervisorAgent)
    prompt = s.PLAN_PROMPT

    # 验证模板能正常创建
    try:
        ct = ChatPromptTemplate.from_messages([
            ("system", prompt + "\n\nCurrent datetime: {current_datetime}"),
            ("user", "{query}"),
        ])
        # 如果模板变量解析正常，input_variables 应该只包含 current_datetime 和 query
        vars_set = set(ct.input_variables)
        expected = {"current_datetime", "query"}
        if vars_set == expected:
            print(f"✅ PLAN_PROMPT template OK: input_variables={vars_set}")
            return True
        else:
            print(f"❌ PLAN_PROMPT has unexpected variables: {vars_set}")
            return False
    except Exception as e:
        print(f"❌ PLAN_PROMPT template ERROR: {e}")
        return False


def test_run_stream_mock():
    """Mock LLM 测试 run_stream 事件流"""
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_core.outputs import ChatResult, ChatGeneration

    # 构造 mock LLM
    class MockLLM(BaseChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            import json
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=json.dumps({
                "intent": "financial_analysis",
                "needs_local": True,
                "needs_web": False,
                "complexity": "complex",
                "confidence": 0.95,
                "reasoning": "user asks financial analysis with charts",
                "retrieve_plan": "financial",
                "task_plan": {
                    "subtasks": [
                        {"id": "task_1", "type": "document_search",
                         "description": "检索本地财报文档",
                         "query": "宁德时代 2024 Q3 财报 营收 利润",
                         "depends_on": []},
                        {"id": "task_2", "type": "financial_market",
                         "description": "获取行情数据",
                         "query": "300750 行情 PE PB ROE",
                         "symbols": ["300750.SZ"],
                         "depends_on": []},
                        {"id": "task_3", "type": "financial_computation",
                         "description": "计算财务指标并生成图表",
                         "query": "计算指标并对比行业",
                         "operation": "calculate_metrics",
                         "depends_on": ["task_1", "task_2"]},
                    ]
                }
            }, ensure_ascii=False)))])
        def _llm_type(self):
            return "mock"
        def _stream(self, messages, stop=None, run_manager=None, **kwargs):
            yield AIMessage(content="mock")

    mock_llm = MockLLM()

    # Mock SearchAgent, WebAgent, FinanceDataAgent 避免依赖外部服务
    with patch.object(MultiAgentRAG, '__init__', lambda self, *a, **kw: None):
        agent = MultiAgentRAG.__new__(MultiAgentRAG)
        agent.llm = mock_llm
        agent.supervisor_agent = None  # 动态创建

        from src.agent.multi_agent.supervisor_agent import SupervisorAgent
        agent.supervisor_agent = SupervisorAgent(mock_llm)

        # mock 节点方法
        agent._retrieve_node = lambda s: s
        agent._web_node = lambda s: s
        agent._read_node = lambda s: s
        agent._eval_node = lambda s: s
        agent._refine_node = lambda s: s
        agent._generate_node = lambda s: (
            setattr(s, 'final_answer', '# 宁德时代 Q3 财报分析报告\n\nMock report content.'),
            setattr(s, 'blackboard', s.blackboard),  # no-op
            s
        )[-1]
        agent._finance_data_node = lambda s: (
            s.add_execution_trace({"agent": "finance_data", "action": "mock_done", "status": "done"}),
            s
        )[-1]
        agent._business_compute_node = lambda s: (
            s.add_execution_trace({"agent": "business_compute", "action": "mock_done", "status": "done"}),
            s
        )[-1]
        agent._aggregate_node = lambda s: s
        agent._global_eval_node = lambda s: s

    print("=" * 60)
    print("Test: run_stream() with mock LLM — financial_analysis")
    print("=" * 60)

    gen = agent.run_stream(user_input="分析宁德时代2024Q3财报，出图表")
    events = []
    state_result = None

    try:
        while True:
            try:
                event = next(gen)
                events.append(event)
                et = event.type.value if hasattr(event.type, 'value') else str(event.type)
                agent_name = event.agent_name or "??"

                if et in ("agent_start", "agent_step", "agent_result", "done", "thinking"):
                    content = (event.content or "")[:100]
                    details = event.details or {}
                    if details.get("subtasks"):
                        print(f"[{et}] {agent_name}: {content[:60]} | {len(details['subtasks'])} subtasks")
                    elif details.get("task_index"):
                        print(f"[{et}] {agent_name}: 子任务 {details['task_index']}/{details['total_tasks']} | {content[:60]}")
                    else:
                        print(f"[{et}] {agent_name}: {content[:60]}")
                elif et == "conflict":
                    print(f"[{et}] {agent_name}: {event.content[:60]}")
                elif et == "error":
                    print(f"[{et}] {agent_name}: {event.content[:60]}")
                else:
                    print(f"[{et}] {agent_name}: {event.content[:60] if event.content else ''}")
            except StopIteration as e:
                state_result = e.value
                break
    except Exception as e:
        print(f"\n❌ Generator exception: {e}")
        import traceback
        traceback.print_exc()

    print("-" * 50)
    print(f"📊 Events yielded: {len(events)}")

    # 验证关键事件
    agent_starts = [e for e in events if getattr(e, 'type', None) and e.type.value == 'agent_start']
    agent_steps = [e for e in events if getattr(e, 'type', None) and e.type.value == 'agent_step']
    agent_results = [e for e in events if getattr(e, 'type', None) and e.type.value == 'agent_result']
    done_events = [e for e in events if getattr(e, 'type', None) and e.type.value == 'done']

    print(f"  AGENT_START: {len(agent_starts)}")
    print(f"  AGENT_STEP: {len(agent_steps)}")
    print(f"  AGENT_RESULT: {len(agent_results)}")
    print(f"  DONE: {len(done_events)}")

    # 检查 state
    if state_result:
        if isinstance(state_result, dict):
            answer_len = len(state_result.get("final_answer", ""))
        else:
            answer_len = len(getattr(state_result, 'final_answer', ""))
        print(f"  Final answer: {answer_len} chars")

    errors = [e for e in events if getattr(e, 'type', None) and e.type.value == 'error']
    if errors:
        print(f"❌ ERRORS: {len(errors)}")
        for e in errors:
            print(f"   - {e.content}")
    else:
        print("✅ No errors")

    # 验证有完整的流程
    assert len(agent_starts) >= 2, f"Expected >=2 AGENT_START, got {len(agent_starts)}"
    assert len(done_events) == 1, f"Expected 1 DONE, got {len(done_events)}"
    if state_result:
        assert answer_len > 0, "Expected non-empty final_answer"
    print("\n✅ All assertions passed!")


if __name__ == "__main__":
    print("=== Test 1: PLAN_PROMPT template validation ===")
    test_classify_and_plan_prompt()
    print()
    print("=== Test 2: run_stream mock full flow ===")
    test_run_stream_mock()
