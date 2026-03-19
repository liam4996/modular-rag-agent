"""
多智能体 RAG 系统演示脚本

展示 Phase 1 功能：
- 简单查询（单次检索）
- 复杂查询（并行融合检索）
- 执行日志和指标展示
"""

from src.agent.multi_agent import MultiAgentRAG
from src.core.settings import Settings
from langchain_openai import ChatOpenAI


def load_settings() -> Settings:
    """加载系统配置"""
    return Settings()


def create_llm():
    """创建语言模型"""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
    )


def demo_simple_query():
    """演示简单查询（单次检索）"""
    print("\n" + "="*80)
    print("📝 场景 1：简单查询（本地知识库）")
    print("="*80)
    
    # 初始化
    settings = load_settings()
    llm = create_llm()
    agent = MultiAgentRAG(llm=llm, settings=settings)
    
    # 查询
    query = "公司文档里关于 RAG 的说明"
    print(f"\n用户查询：{query}\n")
    
    # 运行
    response = agent.run(query)
    
    # 输出结果
    print("📄 回答:")
    print("-" * 80)
    print(response.final_answer)
    print("-" * 80)
    
    # 输出指标
    print("\n📊 执行指标:")
    print(f"  - 意图：{response.intent}")
    print(f"  - 本地结果数：{len(response.local_results)}")
    print(f"  - 联网结果数：{len(response.web_results)}")
    print(f"  - 重试次数：{response.retry_count}")
    print(f"  - 兜底触发：{response.fallback_triggered}")
    
    # 输出执行轨迹
    print("\n📝 执行轨迹:")
    for trace in response.execution_trace:
        print(f"  - {trace['agent']}: {trace['action']}")


def demo_hybrid_query():
    """演示复杂查询（并行融合检索）"""
    print("\n" + "="*80)
    print("📝 场景 2：复杂查询（并行融合检索）")
    print("="*80)
    
    # 初始化
    settings = load_settings()
    llm = create_llm()
    agent = MultiAgentRAG(llm=llm, settings=settings)
    
    # 查询
    query = "结合我们内部的文档和网上最新的 AI 发展，写一份总结"
    print(f"\n用户查询：{query}\n")
    
    # 运行
    response = agent.run(query)
    
    # 输出结果
    print("📄 回答:")
    print("-" * 80)
    print(response.final_answer)
    print("-" * 80)
    
    # 输出指标
    print("\n📊 执行指标:")
    print(f"  - 意图：{response.intent}")
    print(f"  - 本地结果数：{len(response.local_results)}")
    print(f"  - 联网结果数：{len(response.web_results)}")
    print(f"  - 并行执行：{response.metrics.get('parallel_execution', False)}")
    print(f"  - 重试次数：{response.retry_count}")
    
    # 输出执行轨迹
    print("\n📝 执行轨迹:")
    for trace in response.execution_trace:
        print(f"  - {trace['agent']}: {trace['action']}")


def demo_chat():
    """演示闲聊"""
    print("\n" + "="*80)
    print("📝 场景 3：闲聊")
    print("="*80)
    
    # 初始化
    settings = load_settings()
    llm = create_llm()
    agent = MultiAgentRAG(llm=llm, settings=settings)
    
    # 查询
    query = "你好，介绍一下你自己"
    print(f"\n用户查询：{query}\n")
    
    # 运行
    response = agent.run(query)
    
    # 输出结果
    print("📄 回答:")
    print("-" * 80)
    print(response.final_answer)
    print("-" * 80)
    
    # 输出指标
    print("\n📊 执行指标:")
    print(f"  - 意图：{response.intent}")
    print(f"  - 本地结果数：{len(response.local_results)}")
    print(f"  - 联网结果数：{len(response.web_results)}")
    print(f"  - 重试次数：{response.retry_count}")


def demo_fallback():
    """演示兜底场景"""
    print("\n" + "="*80)
    print("📝 场景 4：兜底场景（无答案查询）")
    print("="*80)
    
    # 初始化
    settings = load_settings()
    llm = create_llm()
    agent = MultiAgentRAG(llm=llm, settings=settings, enable_logging=True)
    
    # 查询
    query = "我昨天晚饭吃了什么"
    print(f"\n用户查询：{query}\n")
    
    # 运行
    response = agent.run(query)
    
    # 输出结果
    print("📄 回答:")
    print("-" * 80)
    print(response.final_answer)
    print("-" * 80)
    
    # 输出指标
    print("\n📊 执行指标:")
    print(f"  - 意图：{response.intent}")
    print(f"  - 本地结果数：{len(response.local_results)}")
    print(f"  - 联网结果数：{len(response.web_results)}")
    print(f"  - 重试次数：{response.retry_count}")
    print(f"  - 兜底触发：{response.fallback_triggered}")
    print(f"  - 兜底原因：{response.fallback_reason}")


def demo_execution_details():
    """演示详细的执行信息"""
    print("\n" + "="*80)
    print("📝 场景 5：详细执行信息展示")
    print("="*80)
    
    # 初始化
    settings = load_settings()
    llm = create_llm()
    agent = MultiAgentRAG(llm=llm, settings=settings)
    
    # 查询
    query = "RAG 技术原理"
    print(f"\n用户查询：{query}\n")
    
    # 运行
    response = agent.run(query)
    
    # 输出完整的状态
    print("📄 完整状态:")
    print("-" * 80)
    print(f"用户输入：{response.user_input}")
    print(f"意图：{response.intent}")
    print(f"最终回答：{response.final_answer[:200]}...")
    print("-" * 80)
    
    # 输出黑板内容
    print("\n📊 黑板内容:")
    for key, value in response.blackboard.items():
        if isinstance(value, list):
            print(f"  - {key}: {len(value)} 条结果")
        elif isinstance(value, dict):
            print(f"  - {key}: {list(value.keys())}")
        else:
            print(f"  - {key}: {value}")
    
    # 输出执行日志
    print("\n📝 执行日志:")
    for log in response.execution_log:
        print(f"  - {log}")


def main():
    """主函数"""
    print("\n" + "="*80)
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║                                                               ║")
    print("║     🤖 多智能体 RAG 系统 - Phase 1 演示                        ║")
    print("║                                                               ║")
    print("║     功能展示：                                                ║")
    print("║     - 简单查询（单次检索）                                    ║")
    print("║     - 复杂查询（并行融合检索）                                ║")
    print("║     - 闲聊                                                    ║")
    print("║     - 兜底场景                                                ║")
    print("║     - 详细执行信息                                            ║")
    print("║                                                               ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print("="*80)
    
    # 运行所有演示
    try:
        demo_simple_query()
        demo_hybrid_query()
        demo_chat()
        demo_fallback()
        demo_execution_details()
        
        print("\n" + "="*80)
        print("✅ 所有演示完成！")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
