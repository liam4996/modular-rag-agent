"""Agent Demo - 展示 Simple Agent 的基本用法.

这个示例展示了如何使用 Simple Agent 进行:
1. 知识库查询
2. 文档集合浏览
3. 普通对话

Usage:
    python examples/agent_demo.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.simple_agent import SimpleAgent
from src.core.settings import load_settings


def demo_query():
    """Demo: 查询知识库"""
    print("=" * 60)
    print("📝 Demo 1: 知识库查询")
    print("=" * 60)
    
    queries = [
        "查询论文结论",
        "什么是电子舌",
        "电子舌有什么应用",
    ]
    
    settings = load_settings()
    agent = SimpleAgent(settings)
    
    for query in queries:
        print(f"\n👤 User: {query}")
        print("🤖 Agent: ", end="", flush=True)
        
        response = agent.run(query)
        print(response.content)
        print(f"\n   [意图: {response.intent.value}, 置信度: {response.confidence:.2f}, 工具: {response.tool_called or '无'}]")


def demo_list_collections():
    """Demo: 列出文档集合"""
    print("\n" + "=" * 60)
    print("📁 Demo 2: 列出文档集合")
    print("=" * 60)
    
    settings = load_settings()
    agent = SimpleAgent(settings)
    
    queries = [
        "列出所有集合",
        "有哪些文档",
    ]
    
    for query in queries:
        print(f"\n👤 User: {query}")
        print("🤖 Agent: ", end="", flush=True)
        
        response = agent.run(query)
        print(response.content)
        print(f"\n   [意图: {response.intent.value}, 置信度: {response.confidence:.2f}]")


def demo_chat():
    """Demo: 普通对话"""
    print("\n" + "=" * 60)
    print("💬 Demo 3: 普通对话")
    print("=" * 60)
    
    settings = load_settings()
    agent = SimpleAgent(settings)
    
    queries = [
        "你好",
        "你能做什么",
    ]
    
    for query in queries:
        print(f"\n👤 User: {query}")
        print("🤖 Agent: ", end="", flush=True)
        
        response = agent.run(query)
        print(response.content)
        print(f"\n   [意图: {response.intent.value}, 置信度: {response.confidence:.2f}]")


def main():
    """Run all demos."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     🤖 Simple Agent Demo                                      ║
║                                                               ║
║     展示 Intent Classification + Tool Calling 能力           ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
""")
    
    try:
        demo_query()
        demo_list_collections()
        demo_chat()
        
        print("\n" + "=" * 60)
        print("✅ 所有 Demo 完成!")
        print("=" * 60)
        print("\n💡 提示: 使用 `python scripts/run_agent.py` 启动交互模式")
        
    except Exception as e:
        print(f"\n❌ Demo 失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
