"""E2E 测试 run_stream() 全流程 — 无需 Streamlit，直接验证事件流。

用法: python examples/test_run_stream_e2e.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.multi_agent import MultiAgentRAG, AgentState
from src.agent.multi_agent.stream_events import AgentEvent, EventType


def test_run_stream_financial():
    """测试金融意图 run_stream 全流程"""
    print("=" * 60)
    print("Test: run_stream() with financial_analysis intent")
    print("=" * 60)

    # 使用真实 LLM（必须能初始化才能跑）
    try:
        from src.core.settings import load_settings
        from langchain_openai import ChatOpenAI
        settings = load_settings()
        llm = ChatOpenAI(
            model=settings.llm.model,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url if settings.llm.base_url else None,
        )
        agent = MultiAgentRAG(llm=llm, settings=settings, enable_logging=True)
        print(f"✅ Agent initialized: model={settings.llm.model}")
    except Exception as e:
        print(f"❌ Agent init failed: {e}")
        print("Trying mock test instead...")
        test_via_agentstate()
        return

    query = "分析宁德时代2024Q3财报，对比行业平均，出图表"
    print(f"\n📝 Query: {query}")
    print("-" * 50)

    gen = agent.run_stream(user_input=query, conversation_history=[])
    event_count = 0

    for event in gen:
        event_count += 1
        et = event.type.value if hasattr(event.type, 'value') else str(event.type)
        agent_name = event.agent_name or "??"
        content = event.content or ""
        details = event.details or {}

        # 简洁打印
        if et == "agent_start":
            print(f"\n🚀 [{agent_name}] START: {content}")
        elif et == "agent_step":
            print(f"   📌 [{agent_name}] STEP: {content}")
            if details.get("task_index"):
                print(f"      ↳ 任务 {details['task_index']}/{details['total_tasks']}")
        elif et == "agent_result":
            print(f"   ✅ [{agent_name}] RESULT: {content}")
        elif et == "thinking":
            print(f"   🧠 [{agent_name}] THINK: {content[:120]}")
            if details.get("subtasks"):
                for t in details["subtasks"]:
                    print(f"      ↳ {t['id']}: {t['type']} — {t['description']}")
        elif et == "conflict":
            print(f"   ⚠️ [{agent_name}] CONFLICT: {content}")
        elif et == "done":
            print(f"\n🏁 [{agent_name}] DONE: {content}")
        elif et == "partial_text":
            print(f"   ✍️ [{agent_name}] PARTIAL: {content[:100]}...")
        elif et == "error":
            print(f"   ❌ [{agent_name}] ERROR: {content}")
        else:
            print(f"   ▪ [{agent_name}] {et}: {content[:100]}")

    print("-" * 50)
    print(f"📊 Total events yielded: {event_count}")
    print("=" * 60)


def test_via_agentstate():
    """Fallback: 通过 AgentState 测试 run() 路径"""
    print("\n--- Testing via agent.run() (sync path) ---")
    try:
        from src.core.settings import load_settings
        from langchain_openai import ChatOpenAI
        settings = load_settings()
        llm = ChatOpenAI(
            model=settings.llm.model,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url if settings.llm.base_url else None,
        )
        agent = MultiAgentRAG(llm=llm, settings=settings, enable_logging=True)
        state = agent.run("分析宁德时代2024Q3财报", conversation_history=[])
        if isinstance(state, dict):
            answer_len = len(state.get("final_answer", ""))
            trace_len = len(state.get("execution_trace", []))
        else:
            answer_len = len(state.final_answer)
            trace_len = len(state.execution_trace)
        print(f"✅ Sync run OK: answer={answer_len} chars, trace={trace_len} steps")
    except Exception as e:
        print(f"❌ Sync run failed: {e}")


def test_classify_single():
    """仅测试 classify_and_plan，不执行后续节点"""
    print("\n--- Testing classify_and_plan 独立调用 ---")
    try:
        from src.core.settings import load_settings
        from langchain_openai import ChatOpenAI
        from src.agent.multi_agent.supervisor_agent import SupervisorAgent
        from src.agent.multi_agent.state import AgentState

        settings = load_settings()
        llm = ChatOpenAI(
            model=settings.llm.model,
            temperature=0.3,
            max_tokens=2048,
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url if settings.llm.base_url else None,
        )

        supervisor = SupervisorAgent(llm)
        state = AgentState(user_input="分析宁德时代2024Q3财报")

        state = supervisor.classify_and_plan(state)
        print(f"  intent = {state.intent}")
        print(f"  retrieve_plan = {state.blackboard.get('retrieve_plan')}")
        print(f"  analysis_depth = {state.blackboard.get('analysis_depth')}")

        task_plan = state.blackboard.get("task_plan")
        if task_plan:
            subtasks = task_plan.get("subtasks", [])
            print(f"  subtasks: {len(subtasks)}")
            for t in subtasks:
                print(f"    - {t['id']}: {t['type']} | {t.get('description','')[:60]}")
        else:
            print(f"  task_plan: None")

        # 测试子任务派发
        print(f"\n  get_next_subtasks: {supervisor.get_next_subtasks(state)}")
        print(f"  all_completed: {supervisor.all_completed(state)}")
        print("✅ classify_and_plan OK")
    except Exception as e:
        print(f"❌ classify_and_plan failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_classify_single()
    print("\n" + "=" * 60)
    test_run_stream_financial()
