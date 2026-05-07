"""
Report Renderer — Jinja2-based financial report generation.

Renders structured Markdown reports from computed metrics and charts.
Pure Python, no LLM. LLM only selects template type.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, BaseLoader, Template


class ReportRenderer:
    """Renders financial analysis reports from structured data.

    Uses inline Jinja2 templates (no external template files needed).
    """

    TEMPLATES: Dict[str, str] = {}

    def __init__(self, template_dir: str = "templates/finance"):
        self.template_dir = Path(template_dir)
        self._env = Environment(loader=BaseLoader())

    def _get_template(self, name: str) -> Template:
        key = f"report_{name}"
        if key not in self.TEMPLATES:
            return self._env.from_string(self._default_template())
        return self._env.from_string(self.TEMPLATES[key])

    @staticmethod
    def _default_template() -> str:
        return """# {{ title }}

**生成时间**: {{ generated_at }}

{% if metrics %}
## 核心指标摘要

| 指标 | 数值 |
|------|------|
{% for k, v in metrics.items() %}
| {{ k }} | {{ v }} |
{% endfor %}
{% endif %}

{% if valuation %}
## 估值分析

{% for k, v in valuation.items() %}
- **{{ k }}**: {{ v }}
{% endfor %}
{% endif %}

{% if profitability %}
## 盈利能力

{% for k, v in profitability.items() %}
- **{{ k }}**: {{ v }}%
{% endfor %}
{% endif %}

{% if growth %}
## 成长性

{% for k, v in growth.items() %}
- **{{ k }}**: {{ v }}%
{% endfor %}
{% endif %}

{% if solvency %}
## 偿债能力

{% for k, v in solvency.items() %}
- **{{ k }}**: {{ v }}%
{% endfor %}
{% endif %}

{% if comparison_table %}
## 对比分析

{{ comparison_table }}
{% endif %}

{% if key_findings %}
## 关键发现

{% for f in key_findings %}
- {{ f }}
{% endfor %}
{% endif %}

{% if charts %}
## 图表

{% for c in charts %}
![{{ c.title }}]({{ c.path }})
*{{ c.description }}*
{% endfor %}
{% endif %}

{% if risks %}
## 风险提示

{% for r in risks %}
- {{ r }}
{% endfor %}
{% endif %}

---
*报告由 Business Compute Agent 自动生成*
"""

    def render_earnings_review(
        self,
        symbol: str = "",
        metrics: Optional[Dict[str, Any]] = None,
        valuation: Optional[Dict[str, Any]] = None,
        profitability: Optional[Dict[str, Any]] = None,
        growth: Optional[Dict[str, Any]] = None,
        solvency: Optional[Dict[str, Any]] = None,
        comparison_table: str = "",
        key_findings: Optional[List[str]] = None,
        charts: Optional[List[Dict[str, str]]] = None,
        risks: Optional[List[str]] = None,
    ) -> str:
        """Render an earnings review report."""
        from datetime import datetime
        template = self._get_template("earnings_review")
        return template.render(
            title=f"{symbol} 财报分析" if symbol else "财报分析报告",
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            metrics=self._flatten_metrics(metrics or {}),
            valuation=valuation,
            profitability=profitability,
            growth=growth,
            solvency=solvency,
            comparison_table=comparison_table,
            key_findings=key_findings or [],
            charts=charts or [],
            risks=risks or [],
        )

    def render_industry_comparison(
        self,
        symbols: Optional[List[str]] = None,
        comparison_table: str = "",
        key_findings: Optional[List[str]] = None,
        charts: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Render an industry comparison report."""
        from datetime import datetime
        template = self._get_template("industry_comparison")
        return template.render(
            title=f"行业对比分析: {', '.join(symbols or ['—'])}",
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            metrics={},
            comparison_table=comparison_table,
            key_findings=key_findings or [],
            charts=charts or [],
            valuation={},
            profitability={},
            growth={},
            solvency={},
            risks=[],
        )

    @staticmethod
    def _flatten_metrics(metrics: Dict[str, Any]) -> Dict[str, str]:
        result = {}
        for category, values in metrics.items():
            if isinstance(values, dict):
                for k, v in values.items():
                    if v is not None:
                        result[f"{category}.{k}"] = f"{v:.2f}"
        return result
