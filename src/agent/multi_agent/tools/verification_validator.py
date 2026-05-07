"""数字校验器 + 引用覆盖率检查 — 三层溯源校验的 Layer 3。

校验 LLM 输出中的所有数字与 FixedNumbers 一致。
偏差 > 1% 时自动修正并标记 ⚠️。
Phase 5: 目标价改用区间校验（±30% 容忍），不再强制修正为单点值。
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from .verification_calculator import FixedNumbers


class VerificationValidator:
    """
    校验 LLM 输出中的所有数字与 FixedNumbers 一致。

    策略:
    1. 正则提取 LLM 输出中的 "XX 元" / "XX%" / "PE=XX" 等模式
    2. 与 FixedNumbers 对比
    3. 目标价区间校验: 提取 LLM 输出的区间 [low, high]，与计算区间对比（±30% 容忍）
    4. 其他数字: 偏差 > 1% → 自动修正 + ⚠️ 标记
    5. 统计引用覆盖率
    """

    def validate(self, report: str, fixed: FixedNumbers) -> Tuple[str, List[str]]:
        warnings = []

        # ── 目标价区间校验 ──
        report, range_warnings = self._validate_target_range(report, fixed)
        warnings.extend(range_warnings)

        # ── 其他数字精确校验 ──
        patterns = self._get_number_patterns(fixed)
        for label, expected, pattern in patterns:
            if expected is None:
                continue
            matches = re.findall(pattern, report)
            for match in matches:
                try:
                    actual = float(match)
                except ValueError:
                    continue
                if abs(expected) < 0.001:
                    continue
                deviation = abs(actual - expected) / abs(expected)
                if deviation > 0.01:
                    old = f"{label}: {actual}"
                    new = f"{label}: {expected:.2f} ⚠️"
                    report = report.replace(old, new)
                    warnings.append(f"{label}: {actual} → {expected:.2f}（自动修正）")

        return report, warnings

    def _validate_target_range(self, report: str, fixed: FixedNumbers) -> Tuple[str, List[str]]:
        warnings: List[str] = []
        calc_low = fixed.target_price_low
        calc_high = fixed.target_price_high

        if calc_low is None or calc_high is None:
            return report, warnings

        range_pattern = re.compile(
            r'目标价(?:\s*(?:区间|范围))?[：:\s]*'
            r'([\d]+(?:\.[\d]+)?)'
            r'\s*[-–—~到至]\s*'
            r'([\d]+(?:\.[\d]+)?)'
            r'\s*元?'
        )
        match = range_pattern.search(report)
        if match:
            try:
                llm_low = float(match.group(1))
                llm_high = float(match.group(2))
            except ValueError:
                return report, warnings

            if llm_low > llm_high:
                llm_low, llm_high = llm_high, llm_low

            low_dev = abs(llm_low - calc_low) / calc_low if calc_low > 0 else 0
            high_dev = abs(llm_high - calc_high) / calc_high if calc_high > 0 else 0

            if low_dev > 0.30 or high_dev > 0.30:
                warnings.append(
                    f"目标价区间偏差过大: LLM输出 [{llm_low:.2f}, {llm_high:.2f}] vs "
                    f"计算参考 [{calc_low:.2f}, {calc_high:.2f}]（区间验证仅作提示，不强制修正）"
                )
        else:
            # 检测旧格式单点目标价
            single_pattern = re.compile(r'目标价[：:\s]*([\d]+(?:\.[\d]+)?)\s*元?')
            single_match = single_pattern.search(report)
            if single_match:
                try:
                    single_val = float(single_match.group(1))
                    if single_val < calc_low or single_val > calc_high:
                        warnings.append(
                            f"目标价 {single_val:.2f} 不在计算参考区间 [{calc_low:.2f}, {calc_high:.2f}] "
                            f"内（区间验证仅作提示，不强制修正）"
                        )
                except ValueError:
                    pass

        return report, warnings

    def check_citation_coverage(self, report: str, source_docs: Optional[List[Dict]] = None) -> float:
        paragraphs = [p for p in report.split("\n\n") if len(p.strip()) > 20]
        if not paragraphs:
            return 1.0
        cited = sum(
            1 for p in paragraphs
            if "📖" in p or "[来源" in p or "[1]" in p or "[2]" in p
        )
        return cited / len(paragraphs)

    def _get_number_patterns(self, fixed: FixedNumbers) -> List[Tuple[str, Optional[float], str]]:
        return [
            ("预期收益", fixed.expected_return, r'(?:预期收益|expected return)[：:\s]*([+-]?[\d]+(?:\.[\d]+)?)'),
            ("营收增长", fixed.revenue_growth, r'(?:营收增长|revenue growth)[：:\s]*([+-]?[\d]+(?:\.[\d]+)?)'),
            ("利润增长", fixed.profit_growth, r'(?:利润增长|profit growth)[：:\s]*([+-]?[\d]+(?:\.[\d]+)?)'),
            ("PE偏离", fixed.pe_vs_industry, r'(?:PE[ 　]*(?:偏离|vs industry))[：:\s]*([+-]?[\d]+(?:\.[\d]+)?)'),
        ]
