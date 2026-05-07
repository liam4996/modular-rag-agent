"""测试金融后端管线 — 逐步验证每个组件。

用法: .venv\Scripts\python.exe scripts/test_finance_backend.py
"""

import os
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 加载 .env 文件中的环境变量（项目没有自动加载）
_dotenv_path = PROJECT_ROOT / ".env"
if _dotenv_path.exists():
    with open(_dotenv_path, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _value = _line.partition("=")
                _value = _value.strip().strip('"').strip("'")
                if _key.strip() and _value and not os.getenv(_key.strip()):
                    os.environ[_key.strip()] = _value
    print(f"[INFO] 已从 {_dotenv_path} 加载环境变量")
else:
    print(f"[WARN] 未找到 .env 文件: {_dotenv_path}")


def test_step(name, fn):
    """运行测试步骤并打印结果。"""
    print(f"\n{'='*60}")
    print(f"[TEST] {name}")
    print(f"{'='*60}")
    try:
        result = fn()
        if result is not None:
            print(f"[PASS] {name} — 返回值类型: {type(result).__name__}")
        else:
            print(f"[PASS] {name} (无返回值)")
        return True, result
    except Exception as e:
        print(f"[FAIL] {name}")
        print(f"  异常类型: {type(e).__name__}")
        print(f"  异常消息: {e}")
        traceback.print_exc()
        return False, None


def step1_load_settings():
    from src.core.settings import load_settings
    s = load_settings()
    print(f"  LLM provider: {s.llm.provider}")
    print(f"  LLM model: {s.llm.model}")
    print(f"  LLM base_url: {s.llm.base_url}")
    api_key = s.llm.api_key
    is_set = api_key and api_key != '__OPENAI_API_KEY__'
    print(f"  LLM api_key: {'已设置 (' + api_key[:8] + '...)' if is_set else '未设置/占位符'}")
    print(f"  Embedding model: {s.embedding.model}")
    return s


def step2_init_llm(settings):
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=settings.llm.model,
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens,
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url if settings.llm.base_url else None,
    )
    resp = llm.invoke("请用一句话回答：你是谁？")
    ans = resp.content if hasattr(resp, "content") else str(resp)
    print(f"  LLM 响应: {ans[:150]}")
    return llm


def step3_init_agent(settings, llm):
    from src.agent.multi_agent import MultiAgentRAG
    agent = MultiAgentRAG(llm=llm, settings=settings, enable_logging=True)
    node_count = len(agent.workflow.nodes) if hasattr(agent.workflow, 'nodes') else 'N/A'
    print(f"  Agent 工作流节点数: {node_count}")
    print(f"  Supervisor: {type(agent.supervisor_agent).__name__}")
    print(f"  FinanceData: {type(agent.finance_data_agent).__name__}")
    print(f"  BusinessCompute: {type(agent.business_compute_agent).__name__}")
    return agent


def step4_test_supervisor(agent):
    from src.agent.multi_agent.state import AgentState
    state = AgentState(user_input="快看看宁德时代", conversation_history=[],)
    agent.supervisor_agent.classify_and_plan(state)
    intent = state.intent or "unknown"
    plan = state.blackboard.get("task_plan", {})
    tasks = plan.get("subtasks", []) if plan else []
    depth = state.blackboard.get("analysis_depth", "standard")
    print(f"  意图分类: {intent}")
    print(f"  分析深度: {depth}")
    print(f"  子任务数: {len(tasks)}")
    for i, t in enumerate(tasks):
        print(f"    子任务 [{i}]: type={t.get('type')}, desc={t.get('description', '?')}")
    return state


def step5_test_finance_data(agent):
    from src.agent.multi_agent.state import AgentState
    state = AgentState(user_input="查询宁德时代行情", conversation_history=[],)
    state.add_to_blackboard("current_task", {
        "type": "financial_market",
        "symbols": ["000001.SZ"],
        "query": "查询宁德时代行情",
    }, "test")
    state.add_to_blackboard("active_sub_question", "查询宁德时代行情", "test")
    state.add_to_blackboard("analysis_depth", "quick", "test")
    agent._finance_data_node(state)
    market_data = state.blackboard.get("market_data", {})
    print(f"  market_data keys: {list(market_data.keys()) if market_data else '[]'}")
    quotes = market_data.get("quote", []) if market_data else []
    fundamentals = market_data.get("fundamentals", []) if market_data else []
    print(f"  行情数据条数: {len(quotes)}")
    print(f"  基本面数据条数: {len(fundamentals)}")
    if quotes:
        q = quotes[0]
        print(f"  行情示例: {str(q)[:200]}")
    return state


def step6_test_business_compute(agent):
    from src.agent.multi_agent.state import AgentState
    state = AgentState(user_input="分析宁德时代财报", conversation_history=[],)
    state.add_to_blackboard("current_task", {
        "type": "financial_computation",
        "operation": "calculate_metrics",
        "query": "分析宁德时代财报",
    }, "test")
    state.add_to_blackboard("active_sub_question", "分析宁德时代财报", "test")
    state.add_to_blackboard("analysis_depth", "quick", "test")
    state.add_to_blackboard("market_data", {
        "quote": [{"symbol": "000001.SZ", "price": 15.5, "change_pct": 2.3}],
        "fundamentals": [{"symbol": "000001.SZ", "pe": 12.5, "pb": 1.8, "roe": 0.15}],
    }, "test")
    agent.business_compute_agent.compute(state=state, operation="calculate_metrics", task_params={})
    computed = state.blackboard.get("computed_results", {})
    print(f"  computed_results keys: {list(computed.keys()) if computed else '[]'}")
    metrics = computed.get("metrics", {}) if computed else {}
    print(f"  指标内容: {str(metrics)[:300]}")
    return state


def step7_test_run_stream_quick(agent):
    gen = agent.run_stream(user_input="快看看宁德时代", conversation_history=[])
    event_count = 0
    state = None
    event_types = set()
    for event in gen:
        event_count += 1
        if event:
            et = event.type.value if hasattr(event.type, 'value') else str(event.type)
            event_types.add(et)
            content_preview = (event.content or "")[:80]
            print(f"  #{event_count}: [{et}] {content_preview}")
            if isinstance(event, object) and hasattr(event, '__class__'):
                pass
    print(f"\n  总事件数: {event_count}")
    print(f"  事件类型: {event_types}")
    # 最终 state 需要从生成器的 return 获取
    return gen


# ── 主流程 ──
if __name__ == "__main__":
    print("=" * 60)
    print("金融后端管线测试")
    print("=" * 60)

    # Step 1: Settings
    ok, settings = test_step("Step 1: 加载 Settings + 环境变量", step1_load_settings)
    if not ok or settings is None:
        print("\n[X] Settings 加载失败，终止测试")
        sys.exit(1)

    # Step 2: LLM
    ok, llm = test_step("Step 2: 初始化 LLM（测试 API 连通性）", lambda: step2_init_llm(settings))
    if not ok:
        print("\n[X] LLM 初始化失败，终止测试")
        sys.exit(1)

    # Step 3: Agent
    ok, agent = test_step("Step 3: 初始化 MultiAgentRAG", lambda: step3_init_agent(settings, llm))
    if not ok:
        print("\n[X] Agent 初始化失败，终止测试")
        sys.exit(1)

    # Step 4: Supervisor
    test_step("Step 4: Supervisor 意图识别", lambda: step4_test_supervisor(agent))

    # Step 5: FinanceData
    test_step("Step 5: FinanceDataAgent 行情查询", lambda: step5_test_finance_data(agent))

    # Step 6: BusinessCompute
    test_step("Step 6: BusinessComputeAgent 指标计算", lambda: step6_test_business_compute(agent))

    # Step 7: End-to-end run_stream
    ok, gen = test_step("Step 7: run_stream 端到端 (quick 模式)", lambda: step7_test_run_stream_quick(agent))

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
