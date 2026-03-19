"""Test Panel page – Run and view Phase 4 tests from the dashboard.

Layout:
1. Test suite selector (Unit, Integration, Performance, All)
2. Run test button
3. Real-time test output
4. Test results summary with metrics
5. Historical test runs
"""

from __future__ import annotations

import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

logger = logging.getLogger(__name__)

# Test file paths
TEST_FILES = {
    "单元测试": "examples/test_phase4_comprehensive.py",
    "集成测试": "examples/test_phase4_integration.py",
    "性能测试": "examples/test_phase4_performance.py",
}

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def run_tests(test_suite: str) -> Dict[str, Any]:
    """
    Run tests and capture output.
    
    Args:
        test_suite: Test suite name ("单元测试", "集成测试", "性能测试", or "全部")
    
    Returns:
        Dictionary with test results
    """
    if test_suite == "全部":
        # Run all test files
        test_files = list(TEST_FILES.values())
    else:
        test_files = [TEST_FILES[test_suite]]
    
    results = {
        "success": True,
        "output": "",
        "error": "",
        "start_time": datetime.now(),
        "end_time": None,
        "duration": None,
        "test_count": 0,
        "passed_count": 0,
        "failed_count": 0,
    }
    
    try:
        # Run tests
        all_output = []
        for test_file in test_files:
            test_path = PROJECT_ROOT / test_file
            if not test_path.exists():
                results["success"] = False
                results["error"] = f"Test file not found: {test_file}"
                return results
            
            cmd = [sys.executable, str(test_path)]
            logger.info(f"Running tests: {' '.join(cmd)}")
            
            try:
                process = subprocess.run(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                )
                
                all_output.append(process.stdout)
                if process.stderr:
                    all_output.append("\nSTDERR:\n" + process.stderr)
                
                if process.returncode != 0:
                    results["success"] = False
                
            except subprocess.TimeoutExpired:
                results["success"] = False
                results["error"] = f"Test timeout after 300 seconds: {test_file}"
                all_output.append(f"\n⚠️ TIMEOUT: {test_file}")
        
        results["output"] = "\n".join(all_output)
        
        # Parse test results
        output_text = results["output"]
        
        # Count test cases
        test_markers = output_text.count("测试 ")
        results["test_count"] = max(test_markers, len(test_files))
        
        # Count passed tests
        passed_markers = output_text.count("✅")
        results["passed_count"] = passed_markers
        
        # Count failed tests
        failed_markers = output_text.count("❌")
        results["failed_count"] = failed_markers
        
        # Check for assertion errors
        if "AssertionError" in output_text or "AssertionError" in results["error"]:
            results["success"] = False
        
    except Exception as e:
        results["success"] = False
        results["error"] = str(e)
        logger.exception(f"Error running tests: {e}")
    
    finally:
        results["end_time"] = datetime.now()
        results["duration"] = (results["end_time"] - results["start_time"]).total_seconds()
    
    return results


def format_test_output(output: str) -> str:
    """Format test output for display."""
    # Remove excessive whitespace
    lines = output.split('\n')
    formatted_lines = []
    
    for line in lines:
        # Keep important lines
        if any(marker in line for marker in ['测试', '✅', '❌', '=', '耗时', '通过']):
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)


def render() -> None:
    """Render the Test Panel page."""
    st.header("🧪 测试面板")
    st.markdown("运行和查看 Phase 4 测试结果")
    
    # Initialize session state for test results
    if "test_results" not in st.session_state:
        st.session_state.test_results = []
    
    # ── Test Suite Selector ─────────────────────────────────────
    st.subheader("1️⃣ 选择测试套件")
    
    test_suite = st.selectbox(
        "选择要运行的测试:",
        options=list(TEST_FILES.keys()) + ["全部"],
        format_func=lambda x: f"📋 {x}",
        help="选择要运行的测试套件",
    )
    
    # Display test files
    if test_suite == "全部":
        st.info("将运行所有测试套件（单元测试、集成测试、性能测试）")
    else:
        st.info(f"将运行：`{TEST_FILES[test_suite]}`")
    
    # ── Run Button ───────────────────────────────────────────────
    st.subheader("2️⃣ 运行测试")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        run_button = st.button(
            "▶️ 运行测试",
            type="primary",
            use_container_width=True,
        )
    
    if run_button:
        with st.spinner(f"正在运行 {test_suite}..."):
            # Run tests
            results = run_tests(test_suite)
            
            # Store results
            st.session_state.test_results.insert(0, {
                "timestamp": results["start_time"],
                "suite": test_suite,
                "results": results,
            })
            
            # Keep only last 10 runs
            if len(st.session_state.test_results) > 10:
                st.session_state.test_results = st.session_state.test_results[:10]
            
            # Display results
            st.divider()
            display_test_results(results)
    
    # ── Historical Results ──────────────────────────────────────
    if st.session_state.test_results:
        st.divider()
        st.subheader("📊 历史测试记录")
        display_history()


def display_test_results(results: Dict[str, Any]) -> None:
    """Display test results."""
    
    # Summary metrics
    st.subheader("测试结果概览")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_icon = "✅" if results["success"] else "❌"
        st.metric("状态", f"{status_icon} {'成功' if results['success'] else '失败'}")
    
    with col2:
        st.metric("测试数量", results["test_count"])
    
    with col3:
        st.metric("通过", f"✅ {results['passed_count']}")
    
    with col4:
        st.metric("失败", f"❌ {results['failed_count']}")
    
    # Duration
    if results["duration"]:
        st.metric("耗时", f"⏱️ {results['duration']:.2f} 秒")
    
    # Error message
    if results["error"]:
        st.error(f"**错误**: {results['error']}")
    
    # Test output
    if results["output"]:
        st.subheader("📝 测试输出")
        
        # Use expander for long output
        with st.expander("查看详细输出", expanded=True):
            st.code(results["output"], language="text")
    
    # Success message
    if results["success"]:
        st.success("🎉 所有测试通过！")
    else:
        st.error("❌ 部分测试失败，请查看详细输出")


def display_history() -> None:
    """Display test history."""
    
    history = st.session_state.test_results
    
    for idx, run in enumerate(history, 1):
        timestamp = run["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        suite = run["suite"]
        results = run["results"]
        
        status_icon = "✅" if results["success"] else "❌"
        duration = f"{results['duration']:.2f}s" if results["duration"] else "N/A"
        
        # Create a compact summary
        summary = (
            f"**{idx}. {status_icon} {suite}**  ·  "
            f"测试：{results['test_count']}  ·  "
            f"通过：{results['passed_count']}  ·  "
            f"失败：{results['failed_count']}  ·  "
            f"耗时：{duration}  ·  "
            f"{timestamp}"
        )
        
        # Expandable details
        with st.expander(summary, expanded=(idx == 1)):
            # Detailed metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("测试数量", results["test_count"])
            with col2:
                st.metric("通过率", f"{results['passed_count']}/{results['test_count']}")
            with col3:
                st.metric("耗时", f"{results['duration']:.2f}s" if results["duration"] else "N/A")
            
            # Output snippet
            if results["output"]:
                st.markdown("**输出预览:**")
                # Show first 20 lines
                output_lines = results["output"].split('\n')[:20]
                st.code('\n'.join(output_lines), language="text")
                
                if len(results["output"].split('\n')) > 20:
                    st.info(f"... 还有 {len(results['output'].split(chr(10))) - 20} 行输出")


def main() -> None:
    """Main entry point."""
    render()


if __name__ == "__main__":
    main()
