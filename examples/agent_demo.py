"""Multi-Agent Demo - 展示多智能体 RAG 的基本用法.

Usage:
    python examples/agent_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.multi_agent import MultiAgentRAG
from src.core.settings import load_settings


def demo_query():
    print("=" * 60)
    print("📝 Demo: 多智能体 RAG 查询")
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

    query = "分布式系统中的 CAP 定理是什么？"
    print(f"\n查询: {query}\n")

    result = agent.run(user_input=query)
    answer = result.final_answer if hasattr(result, "final_answer") else result.get("final_answer", "")
    print(f"回答:\n{answer}\n")


if __name__ == "__main__":
    demo_query()
