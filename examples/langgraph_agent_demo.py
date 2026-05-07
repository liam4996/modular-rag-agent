"""Multi-Agent Demo - 展示多智能体系统的执行轨迹.

对比不同 intent 下的 Agent 编排路径。

Usage:
    python examples/langgraph_agent_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.multi_agent import MultiAgentRAG, AgentState
from src.core.settings import load_settings


def demo_multi_agent():
    print("=" * 60)
    print("📝 Demo: 多智能体 RAG 执行轨迹")
    print("=" * 60)

    settings = load_settings()
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=settings.llm.model,
        temperature=0.0,
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
    )

    agent = MultiAgentRAG(llm=llm, settings=settings, enable_logging=True)

    queries = [
        "你好，今天天气怎么样？",
        "宁德时代2024年第三季度财报怎么样？",
    ]

    for query in queries:
        print(f"\n{'─' * 60}")
        print(f"查询: {query}")
        print(f"{'─' * 60}")

        state = agent.run(user_input=query)

        answer = state.final_answer if hasattr(state, "final_answer") else state.get("final_answer", "")
        trace = state.execution_trace if hasattr(state, "execution_trace") else state.get("execution_trace", [])

        print(f"\n执行轨迹 ({len(trace)} 步):")
        for step in trace:
            agent_name = step.get("agent", "?")
            action = step.get("action", "?")
            print(f"  ▶ {agent_name} → {action}")

        print(f"\n回答预览: {answer[:200]}...\n")


if __name__ == "__main__":
    demo_multi_agent()
