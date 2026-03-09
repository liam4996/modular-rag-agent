#!/usr/bin/env python
"""Interactive Agent CLI for Modular RAG MCP Server.

This script provides an interactive command-line interface for the Simple Agent,
allowing users to chat with the knowledge base using natural language.

Usage:
    # Start interactive mode
    python scripts/run_agent.py
    
    # Run a single query
    python scripts/run_agent.py --query "查询论文结论"
    
    # Run with verbose output
    python scripts/run_agent.py --verbose
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from src.agent.simple_agent import SimpleAgent
from src.core.settings import load_settings


def print_banner():
    """Print welcome banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     🤖 Modular RAG Agent                                      ║
║                                                               ║
║     支持: 知识查询 | 文档浏览 | 智能对话                      ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

输入 'quit' 或 'exit' 退出
输入 'help' 查看帮助
"""
    print(banner)


def print_help():
    """Print help message."""
    help_text = """
📖 使用帮助:

1️⃣  查询知识库:
    > 查询论文结论
    > 什么是电子舌
    > find documents about AI

2️⃣  浏览文档集合:
    > 列出所有集合
    > 有哪些文档
    > show collections

3️⃣  获取文档信息:
    > 总结这篇论文
    > 文档摘要

4️⃣  普通对话:
    > 你好
    > 介绍一下这个系统

5️⃣  其他命令:
    > help  - 显示帮助
    > clear - 清空历史
    > quit  - 退出程序
"""
    print(help_text)


def interactive_mode(agent: SimpleAgent, verbose: bool = False):
    """Run agent in interactive mode.
    
    Args:
        agent: SimpleAgent instance.
        verbose: Whether to show verbose output.
    """
    print_banner()
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 再见!")
                break
            
            if user_input.lower() == "help":
                print_help()
                continue
            
            if user_input.lower() == "clear":
                agent.clear_history()
                print("\n✅ 历史记录已清空")
                continue
            
            # Process user input
            print("\n🤖 Agent: ", end="", flush=True)
            
            response = agent.run(user_input)
            
            # Print response
            print(response.content)
            
            # Print verbose info if requested
            if verbose:
                print(f"\n📊 [调试信息]")
                print(f"   意图: {response.intent.value}")
                print(f"   置信度: {response.confidence:.2f}")
                print(f"   调用工具: {response.tool_called or '无'}")
                
        except KeyboardInterrupt:
            print("\n\n👋 再见!")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")


def single_query(agent: SimpleAgent, query: str, verbose: bool = False):
    """Run a single query.
    
    Args:
        agent: SimpleAgent instance.
        query: User query.
        verbose: Whether to show verbose output.
    """
    print(f"\n👤 Query: {query}\n")
    print("🤖 Agent: ", end="", flush=True)
    
    response = agent.run(query)
    print(response.content)
    
    if verbose:
        print(f"\n📊 [调试信息]")
        print(f"   意图: {response.intent.value}")
        print(f"   置信度: {response.confidence:.2f}")
        print(f"   调用工具: {response.tool_called or '无'}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive Agent for Modular RAG MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive mode
    python scripts/run_agent.py
    
    # Single query
    python scripts/run_agent.py -q "查询论文结论"
    
    # Verbose mode
    python scripts/run_agent.py -v
        """
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Single query mode (non-interactive)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Load settings and initialize agent
    try:
        print("🚀 正在初始化 Agent...")
        settings = load_settings()
        agent = SimpleAgent(settings)
        print("✅ Agent 初始化完成\n")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        sys.exit(1)
    
    # Run in appropriate mode
    if args.query:
        single_query(agent, args.query, args.verbose)
    else:
        interactive_mode(agent, args.verbose)


if __name__ == "__main__":
    main()
