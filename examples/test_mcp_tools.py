"""示例：直接调用 MCP Tools（不通过 MCP 协议）

用于测试和开发调试。
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mcp_server.tools.query_knowledge_hub import QueryKnowledgeHubTool
from src.mcp_server.tools.list_collections import ListCollectionsTool
from src.mcp_server.tools.get_document_summary import GetDocumentSummaryTool
from src.core.settings import load_settings


async def test_list_collections():
    """测试 list_collections 工具"""
    print("=" * 50)
    print("测试: list_collections")
    print("=" * 50)
    
    settings = load_settings()
    tool = ListCollectionsTool(settings)
    
    result = await tool.execute({"include_stats": True})
    print(f"结果: {result}")
    print()


async def test_query_knowledge_hub():
    """测试 query_knowledge_hub 工具"""
    print("=" * 50)
    print("测试: query_knowledge_hub")
    print("=" * 50)
    
    settings = load_settings()
    tool = QueryKnowledgeHubTool(settings)
    
    # 示例查询（需要有已摄取的文档才能返回结果）
    result = await tool.execute({
        "query": "什么是 RAG？",
        "top_k": 5,
        "collection": "default"
    })
    print(f"结果: {result}")
    print()


async def test_get_document_summary():
    """测试 get_document_summary 工具"""
    print("=" * 50)
    print("测试: get_document_summary")
    print("=" * 50)
    
    settings = load_settings()
    tool = GetDocumentSummaryTool(settings)
    
    # 示例 doc_id（需要替换为实际的 doc_id）
    result = await tool.execute({
        "doc_id": "doc_example",
        "collection": "default"
    })
    print(f"结果: {result}")
    print()


async def main():
    """运行所有测试"""
    print("MCP Tools 测试示例")
    print("注意：需要先摄取文档才能看到查询结果\n")
    
    try:
        await test_list_collections()
    except Exception as e:
        print(f"list_collections 错误: {e}\n")
    
    try:
        await test_query_knowledge_hub()
    except Exception as e:
        print(f"query_knowledge_hub 错误: {e}\n")
    
    try:
        await test_get_document_summary()
    except Exception as e:
        print(f"get_document_summary 错误: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
