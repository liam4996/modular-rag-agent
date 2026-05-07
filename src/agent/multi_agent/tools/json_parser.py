"""安全的 JSON 解析工具 — 处理 LLM 输出的各种不稳定情况。

LLM 输出的 JSON 可能存在的问题：
1. 标准 JSON → 直接 json.loads()
2. Markdown 围栏 → ```json ... ```
3. 单引号 → 替换为双引号
4. 尾随逗号 → 移除最后一个 , 
5. 被截断 → 只取最后一个完整 } 或 ]
6. 字符串内未转义换行 → 替换为 \\n
"""

import json
import re
from typing import Any, Dict, Optional


def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """
    多层级安全的 JSON 解析器。
    
    按优先级尝试多种修复策略，全部失败返回 None。
    外部调用者根据 None 做业务层面的兜底。
    
    Args:
        text: LLM 返回的文本（可能包含 JSON）
    
    Returns:
        解析成功的字典，或 None（所有策略都失败）
    """
    if not text or not text.strip():
        return None

    # 尝试策略 1-5，按复杂度递增
    strategies = [
        _try_standard,           # 1. 标准 JSON
        _try_strip_fences,       # 2. 去掉 markdown ```json 围栏
        _try_fix_single_quotes,  # 3. 修复单引号
        _try_fix_trailing_comma, # 4. 修复尾随逗号
        _try_truncated,          # 5. 修复被截断的 JSON
    ]

    for strategy in strategies:
        result = strategy(text)
        if result is not None:
            return result

    return None


def safe_parse_json_with_default(text: str, default: Dict[str, Any]) -> Dict[str, Any]:
    """
    带默认值的 JSON 解析器。
    
    用法:
        result = safe_parse_json_with_default(llm_output, {"name": "fallback"})
    """
    parsed = safe_parse_json(text)
    if parsed is not None:
        return parsed
    return default


# ── 各层策略实现 ──


def _try_standard(text: str) -> Optional[Dict[str, Any]]:
    """策略 1: 标准 JSON 解析"""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _try_strip_fences(text: str) -> Optional[Dict[str, Any]]:
    """策略 2: 去掉 markdown 围栏后解析"""
    # 匹配 ```json ... ```, ``` ... ```, ``...``
    patterns = [
        r'```(?:json)?\s*\n?(.*?)\n?```',  # 多行围栏
        r'```(?:json)?\s*(.*?)\s*```',      # 行内围栏
    ]
    for p in patterns:
        match = re.search(p, text, re.DOTALL)
        if match:
            candidate = match.group(1).strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    return None


def _try_fix_single_quotes(text: str) -> Optional[Dict[str, Any]]:
    """策略 3: 修复单引号 (Python dict 风格) → 标准 JSON"""
    # 只有确认是单引号风格才处理
    if "'" not in text:
        return None
    
    # 提取最有可能的 JSON 对象
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return None
    
    candidate = match.group(0)
    
    # 替换单引号为双引号（但保留已在双引号内的内容）
    # 策略: 将 ' 替换为 "，但要跳过已转义的和在字符串中的"
    # 简化版: 先替换 key 和 value 周围的单引号
    try:
        # 将属性名和字符串值周围的单引号替换为双引号
        fixed = re.sub(r"(?<!\\)'", '"', candidate)
        # 修复可能由此产生的 "" 问题
        fixed = fixed.replace('""', '"')
        return json.loads(fixed)
    except (json.JSONDecodeError, ValueError):
        return None


def _try_fix_trailing_comma(text: str) -> Optional[Dict[str, Any]]:
    """策略 4: 修复尾随逗号"""
    # 提取 JSON
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return None
    
    candidate = match.group(0)
    
    # 修复尾随逗号: ,] → ]  ,} → }
    fixed = re.sub(r',\s*\}', '}', candidate)
    fixed = re.sub(r',\s*\]', ']', fixed)
    
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        return None


def _try_truncated(text: str) -> Optional[Dict[str, Any]]:
    """策略 5: 修复被截断的 JSON（找到最后一个完整对象）"""
    # 从右向左找到第一个完整的 }
    brace_count = 0
    last_complete = -1
    
    for i in range(len(text) - 1, -1, -1):
        ch = text[i]
        if ch == '}':
            brace_count += 1
        elif ch == '{':
            brace_count -= 1
            if brace_count == 0:
                last_complete = i
                break
    
    if last_complete >= 0:
        candidate = text[last_complete:]
        # 找到配对的结束 }
        open_count = 0
        for j, ch in enumerate(candidate):
            if ch == '{':
                open_count += 1
            elif ch == '}':
                open_count -= 1
                if open_count == 0:
                    try:
                        return json.loads(candidate[:j + 1])
                    except json.JSONDecodeError:
                        # 修补未转义的换行符
                        fixed = candidate[:j + 1].replace('\n', '\\n').replace('\r', '\\r')
                        try:
                            return json.loads(fixed)
                        except json.JSONDecodeError:
                            return None
    return None
