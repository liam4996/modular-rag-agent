"""数据校验器（数字一致性检查）。

检查 LLM 生成的财务数字与实际计算值是否一致。
不一致时自动修正并用 ⚠️ 标记。
"""

import re
from typing import Any, Dict, List, Tuple


class DataVerifier:
    """
    数据校验器。

    例:
      LLM: "ROE 为 19.5%"
      实际: compute_all_metrics() 返回 roe=18.2
      → 修正: "ROE 为 18.2% ⚠️原述19.5%已校正"
    """

    def verify(self, report: str, computed: Dict) -> Tuple[str, List[str]]:
        warnings = []
        metrics = computed.get("metrics", {})

        for symbol, m in metrics.items():
            if not isinstance(m, dict):
                continue
            corrections = self._check_metric_category(report, m, "profitability", warnings)
            if corrections:
                report = corrections
            corrections = self._check_metric_category(report, m, "valuation", warnings)
            if corrections:
                report = corrections
            corrections = self._check_metric_category(report, m, "growth", warnings)
            if corrections:
                report = corrections

        return report, warnings

    def _check_metric_category(self, report: str, metrics: Dict, category: str,
                                warnings: List[str]) -> str:
        category_data = metrics.get(category, {})
        if not isinstance(category_data, dict):
            return report

        for metric_name, actual_value in category_data.items():
            if actual_value is None:
                continue
            abs_val = abs(actual_value)
            if abs_val < 0.01:
                continue
            pattern = rf'{metric_name}\s*[:：]?\s*([\d]+(?:\.[\d]+)?)'
            matches = re.findall(pattern, report, re.IGNORECASE)
            for match in matches:
                try:
                    reported_val = float(match)
                except ValueError:
                    continue
                if abs_val > 0:
                    deviation = abs(reported_val - abs_val) / abs_val
                    if deviation > 0.01:
                        old_str = f"{metric_name}: {reported_val}"
                        new_str = f"{metric_name}: {abs_val:.2f} ⚠️"
                        report = report.replace(old_str, new_str)
                        warnings.append(
                            f"{metric_name}: {reported_val} → {abs_val:.2f}（自动修正）"
                        )
        return report
