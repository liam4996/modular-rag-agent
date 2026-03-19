"""LangGraph Agent Demo - 对比新旧 Agent 实现.

这个示例展示了：
1. SimpleAgent（基于规则的实现）vs LangGraphAgent（状态机实现）
2. LangGraph 的工作流可视化能力
3. 执行轨迹的详细观察

Usage:
    python examples/langgraph_agent_demo.py
"""

import sys
from pathlib import Path
from typing import List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.simple_agent import SimpleAgent
from src.agent.langgraph_agent import LangGraphAgent
from src.core.settings import load_settings


def compare_agents(queries: List[str]):
    """对比 SimpleAgent 和 LangGraphAgent 的执行过程。
    
    Args:
        queries: 测试查询列表
    """
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     🤖 SimpleAgent vs LangGraphAgent 对比演示                  ║
║                                                               ║
║     左边：传统实现 (if-else 路由)                              ║
║     右边：LangGraph 实现 (状态机路由)                          ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
""")
    
    settings = load_settings()
    
    # 初始化两个 Agent
    print("🔧 初始化 Agent...")
    simple_agent = SimpleAgent(settings)
    langgraph_agent = LangGraphAgent(settings, enable_logging=True)
    print("✅ Agent 初始化完成\n")
    
    for idx, query in enumerate(queries, start=1):
        print(f"\n{'='*80}")
        print(f"📝 查询 {idx}: {query}")
        print(f"{'='*80}\n")
        
        # 运行 SimpleAgent
        print("【SimpleAgent】")
        print("-" * 40)
        simple_response = simple_agent.run(query)
        print(f"意图：{simple_response.intent.value}")
        print(f"置信度：{simple_response.confidence:.2f}")
        print(f"工具调用：{simple_response.tool_called or '无'}")
        print(f"执行步骤：{len(simple_response.steps)}")
        for step in simple_response.steps:
            print(f"  {step.step}. {step.thought[:60]}...")
            if step.action:
                print(f"     动作：{step.action}")
            if step.observation:
                print(f"     观察：{step.observation[:60]}...")
        print(f"\n回复预览：{simple_response.content[:150]}...\n")
        
        # 运行 LangGraphAgent
        print("\n【LangGraphAgent】")
        print("-" * 40)
        langgraph_response = langgraph_agent.run(query)
        print(f"意图：{langgraph_response.intent.value}")
        print(f"置信度：{langgraph_response.confidence:.2f}")
        print(f"工具调用：{langgraph_response.tool_called or '无'}")
        print(f"执行步骤：{len(langgraph_response.steps)}")
        for step in langgraph_response.steps:
            print(f"  {step.step}. [{step.node}] {step.thought[:50]}...")
            if step.action:
                print(f"     动作：{step.action}")
            if step.observation:
                print(f"     观察：{step.observation[:60]}...")
        print(f"\n回复预览：{langgraph_response.content[:150]}...\n")
        
        # 对比分析
        print("\n【对比分析】")
        print("-" * 40)
        print(f"意图一致性：{'✅' if simple_response.intent == langgraph_response.intent else '❌'}")
        print(f"步骤数对比：Simple={len(simple_response.steps)} vs LangGraph={len(langgraph_response.steps)}")
        
        # 提取节点类型
        langgraph_nodes = [step.node for step in langgraph_response.steps]
        print(f"LangGraph 路径：{' → '.join(langgraph_nodes)}")


def demo_multi_turn():
    """演示多轮对话能力（LangGraph 的优势场景）"""
    print("\n\n" + "="*80)
    print("💬 多轮对话演示 - LangGraphAgent")
    print("="*80)
    
    settings = load_settings()
    agent = LangGraphAgent(settings, enable_logging=True)
    
    conversation = [
        "什么是 RAG？",
        "它有什么优势？",  # 指代消解："它" = RAG
        "有哪些相关的应用场景？",  # 继续讨论 RAG
    ]
    
    for query in conversation:
        print(f"\n👤 User: {query}")
        response = agent.run(query)
        print(f"🤖 Agent: {response.content[:200]}...")
        print(f"   [意图：{response.intent.value}, 步骤：{len(response.steps)}]")
    
    # 显示完整的执行轨迹
    print("\n📊 完整执行轨迹:")
    for turn in agent.memory.turns:
        print(f"\n{turn.role.upper()}: {turn.content[:100]}")
        if turn.intent:
            print(f"   意图：{turn.intent}")
        if turn.tool_called:
            print(f"   工具：{turn.tool_called}")


def demo_workflow_visualization():
    """演示 LangGraph 工作流可视化（打印状态图）"""
    print("\n\n" + "="*80)
    print("🔍 LangGraph 工作流结构")
    print("="*80)
    
    settings = load_settings()
    agent = LangGraphAgent(settings)
    
    # 访问编译后的图
    graph = agent.workflow
    
    print("\n工作流节点:")
    for node_name in graph.nodes.keys():
        print(f"  - {node_name}")
    
    print("\n边（Edges）:")
    # 获取边的信息
    if hasattr(graph, 'edges'):
        for edge in graph.edges:
            print(f"  {edge}")
    
    print("\n条件边（Conditional Edges）:")
    if hasattr(graph, 'get_conditional_edges'):
        for cond_edge in graph.get_conditional_edges():
            print(f"  {cond_edge}")
    
    print("\n入口点:")
    print(f"  {graph.get_entry_point()}")
    
    print("\n结束点:")
    print(f"  END")
    
    print("\n\n工作流程图:")
    print("""
    [START]
       ↓
    classify_intent
       ↓
    route (条件路由)
       ↓
    ┌──────────────┬──────────────┬──────────────┬──────────────┐
    ↓              ↓              ↓              ↓              ↓
handle_chat   search_knowledge  get_summary  list_collections  unknown
    ↓              ↓              ↓              ↓              ↓
    │          rerank_results     │              │              │
    ↓              ↓              │              │              │
    │         generate_response ←─┴──────────────┘              │
    ↓              ↓                                            │
   [END] ←─────────┴────────────────────────────────────────────┘
    """)


def demo_execution_trace():
    """演示详细的执行轨迹观察"""
    print("\n\n" + "="*80)
    print("📋 详细执行轨迹示例")
    print("="*80)
    
    settings = load_settings()
    agent = LangGraphAgent(settings, enable_rerank=False)
    
    query = "查询关于电子舌的论文"
    print(f"\n查询：{query}\n")
    
    response = agent.run(query)
    
    print("执行轨迹详情:")
    print("-" * 80)
    for idx, step in enumerate(response.steps, start=1):
        print(f"\n【步骤 {idx}】")
        print(f"节点：{step.node}")
        print(f"思考：{step.thought}")
        print(f"动作：{step.action or '无'}")
        print(f"观察：{step.observation or '无'}")
        if step.metadata:
            print(f"元数据：{step.metadata}")
    
    print("\n" + "-" * 80)
    print(f"\n最终回复:\n{response.content}")
    print(f"\n工具调用：{response.tool_called}")
    print(f"总步骤数：{len(response.steps)}")


def demo_parallel_retrieval():
    """演示并行检索的详细过程"""
    print("\n\n" + "="*80)
    print("⚡ 并行检索流程演示 (Dense + Sparse → RRF)")
    print("="*80)
    
    settings = load_settings()
    agent = LangGraphAgent(settings, enable_rerank=False, enable_logging=True)
    
    query = "RAG 检索机制"
    print(f"\n查询：{query}\n")
    
    response = agent.run(query)
    
    # 找到 search 节点
    search_step = None
    for step in response.steps:
        if step.node == "search_knowledge":
            search_step = step
            break
    
    if search_step and search_step.metadata:
        print("【并行检索统计】")
        print("-" * 80)
        print(f"Dense 检索结果数：{search_step.metadata.get('dense_count', 0)}")
        print(f"Sparse 检索结果数：{search_step.metadata.get('sparse_count', 0)}")
        print(f"融合方法：{search_step.metadata.get('fusion_method', 'N/A')}")
        print(f"并行执行：{search_step.metadata.get('parallel_execution', False)}")
        print(f"最终结果数：{search_step.metadata.get('result_count', 0)}")
        
        print("\n【执行流程】")
        print("""
        query
          ↓
    ┌─────┴──────────────────────┐
    │                            │
    ↓                            ↓
┌─────────┐              ┌──────────┐
│  Dense  │              │  Sparse  │
│ (语义)  │              │ (BM25)   │
│ ThreadPoolExecutor (并行)        │
└────┬────┘              └────┬─────┘
     │                        │
     └──────────┬─────────────┘
                ↓
        ┌───────────────┐
        │  RRF Fusion   │  ← 倒数排名融合
        └───────┬───────┘
                ↓
        ┌───────────────┐
        │ Post-Filter   │  ← 元数据过滤
        └───────┬───────┘
                ↓
        [最终结果返回]
        """)
    
    print("\n" + "-" * 80)
    print(f"\n最终回复预览:\n{response.content[:300]}...")


def main():
    """运行所有演示"""
    test_queries = [
        "你好",  # Chat
        "查询关于 RAG 的论文",  # Query
        "列出所有集合",  # List collections
        "总结这篇文档",  # Get summary
    ]
    
    try:
        # 1. 对比两种实现
        compare_agents(test_queries)
        
        # 2. 多轮对话演示
        demo_multi_turn()
        
        # 3. 工作流可视化
        demo_workflow_visualization()
        
        # 4. 详细执行轨迹
        demo_execution_trace()
        
        # 5. 并行检索演示
        demo_parallel_retrieval()
        
        print("\n\n" + "="*80)
        print("✅ 所有演示完成!")
        print("="*80)
        print("\n💡 关键优势总结:")
        print("1. ✅ 状态机可视化：清晰的工作流图")
        print("2. ✅ 条件路由：替代 if-else 分支")
        print("3. ✅ 执行轨迹：每个节点的详细观察")
        print("4. ✅ 可扩展性：轻松添加新节点和边")
        print("5. ✅ 接口兼容：与 SimpleAgent 相同的 API")
        print("6. ✅ 并行检索：Dense + Sparse → RRF (已有!)")
        print("\n📚 下一步:")
        print("- 添加 rerank 节点")
        print("- 添加自我反思循环")
        print("- 集成 LangChain callbacks")
        
    except Exception as e:
        print(f"\n❌ 演示失败：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
