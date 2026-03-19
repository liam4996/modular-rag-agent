"""
Phase 4 - 性能测试和边界场景测试

测试内容：
1. 并发性能
2. 延迟测试
3. 内存使用
4. 边界场景（空输入、超长输入、特殊字符等）
5. 异常处理
"""

import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.multi_agent import (
    AgentState,
    Citation,
    CitationType,
    CitationManager,
    FaithfulnessCheck,
    FallbackReason,
)


# ========== 性能测试 ==========

def test_concurrent_state_creation():
    """测试并发创建状态"""
    print("\n" + "="*80)
    print("测试 1: 并发创建状态")
    print("="*80)
    
    def create_state(i):
        state = AgentState(user_input=f"查询{i}")
        state.add_to_blackboard("test", f"value{i}", "test")
        return state
    
    # 并发创建 100 个状态
    print("\n并发创建 100 个状态...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(create_state, i) for i in range(100)]
        results = [f.result() for f in as_completed(futures)]
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"  创建数量：{len(results)}")
    print(f"  耗时：{elapsed:.3f} 秒")
    print(f"  平均速度：{len(results) / elapsed:.1f} 个/秒")
    
    assert len(results) == 100
    assert elapsed < 1.0  # 应该在 1 秒内完成
    
    print("\n✅ 并发创建状态测试通过")


def test_citation_manager_performance():
    """测试引用管理器性能"""
    print("\n" + "="*80)
    print("测试 2: 引用管理器性能")
    print("="*80)
    
    # 创建大量引用
    print("\n创建 1000 个引用...")
    start_time = time.time()
    
    citations = []
    for i in range(1000):
        citation = Citation(
            type=CitationType.LOCAL if i % 2 == 0 else CitationType.WEB,
            source=f"文档{i}",
            content=f"内容{i}" * 10,  # 较长的内容
            confidence=0.9 - (i * 0.0001),
            relevance=0.85 - (i * 0.0001),
            page=i % 100
        )
        citations.append(citation)
    
    creation_time = time.time() - start_time
    print(f"  创建时间：{creation_time:.3f} 秒")
    print(f"  平均速度：{1000 / creation_time:.1f} 个/秒")
    
    # 批量添加
    print("\n批量添加引用...")
    start_time = time.time()
    
    manager = CitationManager()
    manager.add_citations(citations)
    
    add_time = time.time() - start_time
    print(f"  添加时间：{add_time:.3f} 秒")
    
    # 格式化所有引用
    print("\n格式化所有引用...")
    start_time = time.time()
    
    formatted = manager.format_all_citations()
    
    format_time = time.time() - start_time
    print(f"  格式化时间：{format_time:.3f} 秒")
    print(f"  格式化长度：{len(formatted)} 字符")
    
    # 验证
    assert len(manager.citations) == 1000
    assert creation_time < 1.0
    assert add_time < 0.1
    assert format_time < 1.0
    
    print("\n✅ 引用管理器性能测试通过")


def test_large_state_performance():
    """测试大型状态对象性能"""
    print("\n" + "="*80)
    print("测试 3: 大型状态对象性能")
    print("="*80)
    
    # 创建包含大量数据的状态
    print("\n创建大型状态...")
    start_time = time.time()
    
    state = AgentState(user_input="复杂查询")
    
    # 添加大量检索结果
    local_results = [
        {"content": f"内容{i}" * 100, "source": f"文档{i}", "score": 0.95 - i * 0.001}
        for i in range(100)
    ]
    
    web_results = [
        {"title": f"标题{i}", "snippet": f"摘要{i}" * 50, "url": f"https://example.com/{i}"}
        for i in range(100)
    ]
    
    state.add_to_blackboard("local_results", local_results, "search")
    state.add_to_blackboard("web_results", web_results, "web")
    
    # 添加大量执行轨迹
    for i in range(50):
        state.add_execution_trace({
            "agent": f"agent{i % 5}",
            "action": f"action{i}",
            "timestamp": time.time() - i
        })
    
    # 添加大量指标
    for i in range(20):
        state.add_metric(f"metric_{i}", i * 0.1)
    
    creation_time = time.time() - start_time
    print(f"  创建时间：{creation_time:.3f} 秒")
    print(f"  本地结果：{len(state.local_results)} 条")
    print(f"  联网结果：{len(state.web_results)} 条")
    print(f"  执行轨迹：{len(state.execution_trace)} 条")
    print(f"  指标数量：{len(state.metrics)} 个")
    
    # 序列化
    print("\n序列化状态...")
    start_time = time.time()
    
    state_dict = {
        "user_input": state.user_input,
        "blackboard": state.blackboard,
        "execution_trace": state.execution_trace,
        "metrics": state.metrics,
    }
    
    serialize_time = time.time() - start_time
    print(f"  序列化时间：{serialize_time:.3f} 秒")
    
    # 验证
    assert creation_time < 1.0
    assert serialize_time < 0.5
    
    print("\n✅ 大型状态对象性能测试通过")


# ========== 边界场景测试 ==========

def test_empty_input():
    """测试空输入"""
    print("\n" + "="*80)
    print("测试 4: 空输入")
    print("="*80)
    
    # 空字符串
    print("\n测试空字符串...")
    state = AgentState(user_input="")
    print(f"  用户输入：'{state.user_input}'")
    assert state.user_input == ""
    
    # 空白字符
    print("\n测试空白字符...")
    state = AgentState(user_input="   ")
    print(f"  用户输入：'{state.user_input}'")
    assert state.user_input.strip() == ""
    
    print("\n✅ 空输入测试通过")


def test_very_long_input():
    """测试超长输入"""
    print("\n" + "="*80)
    print("测试 5: 超长输入")
    print("="*80)
    
    # 创建超长输入（10000 字符）
    long_input = "测试内容 " * 1000
    print(f"\n创建超长输入：{len(long_input)} 字符")
    
    state = AgentState(user_input=long_input)
    print(f"  状态创建成功")
    print(f"  输入长度：{len(state.user_input)}")
    
    assert len(state.user_input) == len(long_input)
    
    # 添加长内容到黑板
    state.add_to_blackboard("long_content", long_input * 10, "test")
    print(f"  黑板内容长度：{len(state.blackboard.get('long_content', ''))}")
    
    print("\n✅ 超长输入测试通过")


def test_special_characters():
    """测试特殊字符"""
    print("\n" + "="*80)
    print("测试 6: 特殊字符")
    print("="*80)
    
    # 各种特殊字符
    special_inputs = [
        "测试\n换行",
        "测试\t制表符",
        "测试\r回车",
        "测试\"引号\"",
        "测试'单引号'",
        "测试\\反斜杠",
        "测试/斜杠",
        "测试|管道符",
        "测试<大于>小于",
        "测试&和符",
        "测试$美元",
        "测试#井号",
        "测试@at 符号",
        "测试!感叹号",
        "测试？问号",
        "测试 emoji 😀🎉🚀",
        "测试中文 中文 中文",
        "测试 日本語 日本語",
        "测试 한국어 한국어",
    ]
    
    print("\n测试特殊字符输入:")
    for i, special_input in enumerate(special_inputs, 1):
        state = AgentState(user_input=special_input)
        print(f"  {i}. '{special_input[:20]}...' - OK")
        assert state.user_input == special_input
    
    print("\n✅ 特殊字符测试通过")


def test_unicode_input():
    """测试 Unicode 输入"""
    print("\n" + "="*80)
    print("测试 7: Unicode 输入")
    print("="*80)
    
    unicode_inputs = [
        "你好世界",  # 中文
        "こんにちは",  # 日文
        "안녕하세요",  # 韩文
        "مرحبا",  # 阿拉伯文
        "שלום",  # 希伯来文
        "Γειά σου",  # 希腊文
        "Привет",  # 俄文
        "สวัสดี",  # 泰文
        "🎉🚀💻🌟",  # Emoji
        "∑∏∫∞≈≠",  # 数学符号
    ]
    
    print("\n测试 Unicode 输入:")
    for i, unicode_input in enumerate(unicode_inputs, 1):
        state = AgentState(user_input=unicode_input)
        print(f"  {i}. {unicode_input} - OK")
        assert state.user_input == unicode_input
    
    print("\n✅ Unicode 输入测试通过")


def test_citation_edge_cases():
    """测试引用边界情况"""
    print("\n" + "="*80)
    print("测试 8: 引用边界情况")
    print("="*80)
    
    # 空引用
    print("\n测试空引用...")
    try:
        citation = Citation(
            type=CitationType.LOCAL,
            source="",
            content=""
        )
        print(f"  空引用创建成功：{citation.format_citation()}")
    except Exception as e:
        print(f"  空引用创建失败：{e}")
    
    # 超长内容
    print("\n测试超长内容...")
    long_citation = Citation(
        type=CitationType.LOCAL,
        source="长文档",
        content="测试内容 " * 10000,
        confidence=0.95
    )
    print(f"  超长引用创建成功，内容长度：{len(long_citation.content)}")
    
    # 极端置信度
    print("\n测试极端置信度...")
    citation_high = Citation(
        type=CitationType.LOCAL,
        source="测试",
        content="内容",
        confidence=1.0
    )
    citation_low = Citation(
        type=CitationType.LOCAL,
        source="测试",
        content="内容",
        confidence=0.0
    )
    print(f"  高置信度：{citation_high.confidence}")
    print(f"  低置信度：{citation_low.confidence}")
    
    # 负数页码
    print("\n测试负数页码...")
    citation = Citation(
        type=CitationType.LOCAL,
        source="测试",
        content="内容",
        page=-1
    )
    print(f"  负数页码引用：{citation.format_citation()}")
    
    # 超大页码
    print("\n测试超大页码...")
    citation = Citation(
        type=CitationType.LOCAL,
        source="测试",
        content="内容",
        page=999999
    )
    print(f"  超大页码引用：{citation.format_citation()}")
    
    print("\n✅ 引用边界情况测试通过")


def test_faithfulness_edge_cases():
    """测试忠实度边界情况"""
    print("\n" + "="*80)
    print("测试 9: 忠实度边界情况")
    print("="*80)
    
    manager = CitationManager()
    
    # 空文本
    print("\n测试空文本...")
    check = manager.check_faithfulness("")
    print(f"  空文本：is_faithful={check.is_faithful}, confidence={check.confidence}")
    
    # 超长文本
    print("\n测试超长文本...")
    long_text = "测试内容 " * 10000
    check = manager.check_faithfulness(long_text)
    print(f"  超长文本：is_faithful={check.is_faithful}")
    
    # 只有引用标记
    print("\n测试只有引用标记...")
    text_only_citations = "[Local: 文档 1] [Local: 文档 2]"
    check = manager.check_faithfulness(text_only_citations)
    print(f"  只有引用标记：is_faithful={check.is_faithful}")
    
    # 大量引用标记
    print("\n测试大量引用标记...")
    many_citations = " ".join([f"[Local: 文档{i}]" for i in range(100)])
    check = manager.check_faithfulness(many_citations)
    print(f"  大量引用标记：is_faithful={check.is_faithful}")
    
    print("\n✅ 忠实度边界情况测试通过")


def test_state_edge_cases():
    """测试状态边界情况"""
    print("\n" + "="*80)
    print("测试 10: 状态边界情况")
    print("="*80)
    
    # 空状态
    print("\n测试空状态...")
    state = AgentState()
    print(f"  默认用户输入：'{state.user_input}'")
    print(f"  默认重试次数：{state.retry_count}")
    print(f"  默认最大重试：{state.max_retries}")
    
    # 极端重试次数
    print("\n测试极端重试次数...")
    state.retry_count = 1000
    print(f"  重试 1000 次：should_fallback={state.should_fallback}")
    assert state.should_fallback == True
    
    # 负数重试
    print("\n测试负数重试...")
    state.retry_count = -1
    print(f"  重试 -1 次：should_fallback={state.should_fallback}")
    
    # 大量执行轨迹
    print("\n测试大量执行轨迹...")
    state = AgentState(user_input="测试")
    for i in range(1000):
        state.add_execution_trace({"agent": f"agent{i}", "action": f"action{i}"})
    print(f"  执行轨迹数量：{len(state.execution_trace)}")
    assert len(state.execution_trace) == 1000
    
    # 大量指标
    print("\n测试大量指标...")
    state = AgentState(user_input="测试")
    for i in range(100):
        state.add_metric(f"metric_{i}", i)
    print(f"  指标数量：{len(state.metrics)}")
    assert len(state.metrics) == 100
    
    print("\n✅ 状态边界情况测试通过")


def test_concurrent_citation_creation():
    """测试并发创建引用"""
    print("\n" + "="*80)
    print("测试 11: 并发创建引用")
    print("="*80)
    
    def create_citation(i):
        return Citation(
            type=CitationType.LOCAL if i % 2 == 0 else CitationType.WEB,
            source=f"文档{i}",
            content=f"内容{i}",
            confidence=0.9
        )
    
    # 并发创建 100 个引用
    print("\n并发创建 100 个引用...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(create_citation, i) for i in range(100)]
        citations = [f.result() for f in as_completed(futures)]
    
    elapsed = time.time() - start_time
    print(f"  创建数量：{len(citations)}")
    print(f"  耗时：{elapsed:.3f} 秒")
    print(f"  平均速度：{len(citations) / elapsed:.1f} 个/秒")
    
    assert len(citations) == 100
    assert elapsed < 1.0
    
    print("\n✅ 并发创建引用测试通过")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🧪 Phase 4 - 性能测试和边界场景测试")
    print("="*80)
    
    try:
        # 性能测试
        test_concurrent_state_creation()
        test_citation_manager_performance()
        test_large_state_performance()
        
        # 边界场景测试
        test_empty_input()
        test_very_long_input()
        test_special_characters()
        test_unicode_input()
        test_citation_edge_cases()
        test_faithfulness_edge_cases()
        test_state_edge_cases()
        test_concurrent_citation_creation()
        
        print("\n" + "="*80)
        print("✅ 所有性能和边界场景测试通过！")
        print("="*80)
        print("\n测试覆盖:")
        print("  ✅ 并发性能")
        print("  ✅ 引用管理器性能")
        print("  ✅ 大型状态对象性能")
        print("  ✅ 空输入")
        print("  ✅ 超长输入")
        print("  ✅ 特殊字符")
        print("  ✅ Unicode 输入")
        print("  ✅ 引用边界情况")
        print("  ✅ 忠实度边界情况")
        print("  ✅ 状态边界情况")
        print("  ✅ 并发创建引用")
        print("="*80)
        
    except AssertionError as e:
        print(f"\n❌ 测试失败：{str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 意外错误：{str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
