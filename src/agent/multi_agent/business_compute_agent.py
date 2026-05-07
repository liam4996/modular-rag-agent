"""
Business Compute Agent — orchestrates financial computation + visualization + reporting.

Receives structured data from RAG + Finance Data + market APIs.
Outputs metrics, charts, tables, and rendered reports.
Pure Python — no LLM calls here.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from .tools.financial_calculator import (
    FinancialMetrics,
    compute_all_metrics,
)
from .tools.data_processor import (
    clean_financial_data,
    build_comparison_table,
    rank_companies,
    detect_outliers,
    to_markdown_table,
    aggregate_market_to_table,
)
from .tools.data_visualizer import DataVisualizer
from .tools.report_renderer import ReportRenderer


class BusinessComputeAgent:
    """
    Business Compute Agent.

    No LLM. Input is structured data from other agents' blackboard output.
    Output is computed results + chart paths + rendered reports.

    Modes:
    - calculate_metrics: compute financial ratios from market data
    - compare_companies: cross-company comparison table + chart
    - visualize: generate charts from data
    - generate_report: render a full financial report
    """

    def __init__(self, output_dir: str = "data/charts"):
        self.calculator_module = None
        self.processor_module = None
        self.visualizer = DataVisualizer(output_dir=output_dir)
        self.renderer = ReportRenderer()
        self.chart_critic = None

    def _get_chart_critic(self, state=None):
        if self.chart_critic is not None:
            return self.chart_critic
        if state is None:
            self.chart_critic = __import__("src.agent.multi_agent.tools.chart_critic", fromlist=["ChartCritic"]).ChartCritic()
            return self.chart_critic
        from .tools.chart_critic import ChartCritic
        vlm = getattr(state, "vlm_llm", None) or getattr(state, "_vlm", None)
        self.chart_critic = ChartCritic(vlm_llm=vlm)
        return self.chart_critic

    def compute(
        self,
        state: Any,
        operation: str = "calculate_metrics",
        task_params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Main entry point. Executes the requested operation on state data.

        Args:
            state: AgentState instance with blackboard data
            operation: "calculate_metrics" | "compare_companies" | "visualize" | "generate_report"
            task_params: optional task-specific parameters

        Returns:
            Updated AgentState with results in blackboard
        """
        params = task_params or {}
        session_id = str(uuid.uuid4())[:8]

        if operation == "calculate_metrics":
            self._calc_metrics(state, params)
        elif operation == "compare_companies":
            self._compare(state, params, session_id)
        elif operation == "visualize":
            self._visualize(state, params, session_id)
        elif operation == "generate_report":
            self._render_report(state, params, session_id)

        return state

    def _calc_metrics(self, state: Any, params: Dict) -> None:
        market_data = state.blackboard.get("market_data", {})
        all_metrics = {}
        for symbol_data in market_data.get("fundamentals", []):
            sym = symbol_data.get("symbol", "unknown")
            m = compute_all_metrics(financial_data=symbol_data, market_data=symbol_data)
            all_metrics[sym] = m.to_dict()

        if not all_metrics and market_data.get("quote"):
            for q in market_data.get("quote", []):
                sym = q.get("symbol", "unknown")
                m = compute_all_metrics(market_data=q)
                if any(v for v in m.valuation.values() if v is not None) or \
                   any(v for v in m.profitability.values() if v is not None):
                    all_metrics[sym] = m.to_dict()

        state.add_to_blackboard("computed_results", {
            **state.blackboard.get("computed_results", {}),
            "metrics": all_metrics,
            "type": "calculate_metrics",
        }, "business_compute")

        state.add_execution_trace({
            "agent": "business_compute",
            "action": "calculate_metrics",
            "symbols_processed": len(all_metrics),
        })

    def _compare(self, state: Any, params: Dict, session_id: str) -> None:
        market_data = state.blackboard.get("market_data", {})
        computed = state.blackboard.get("computed_results", {}).get("metrics", {})

        compare_data = {}
        for sym, metrics in computed.items():
            flat = {}
            for cat, vals in metrics.items():
                if isinstance(vals, dict):
                    for k, v in vals.items():
                        if v is not None:
                            flat[f"{k}"] = v
            if flat:
                compare_data[sym] = flat

        metrics_to_compare = params.get("metrics", ["PE", "PB", "ROE", "gross_margin", "net_margin"])
        df = build_comparison_table(compare_data, metrics_to_compare)

        table_md = to_markdown_table(df, "行业指标对比")

        outliers = {}
        for col in df.columns:
            idx = detect_outliers(df, col, method="iqr")
            if idx:
                outliers[col] = idx

        charts = []
        if len(compare_data) > 1 and not df.empty:
            import pandas as pd
            chart_data = {}
            for sym, d in compare_data.items():
                sub = pd.DataFrame([d])
                chart_data[sym] = sub

            for metric in metrics_to_compare[:3]:
                if metric in df.columns:
                    path = self.visualizer.comparison_chart(
                        data=chart_data, metric=metric,
                        title=f"行业对比 — {metric}",
                        session_id=session_id, name=f"compare_{metric.lower()}")
                    if path:
                        charts.append({"title": f"{metric} 对比", "path": path,
                            "description": f"各公司 {metric} 指标横向对比"})

        state.add_to_blackboard("computed_results", {
            **state.blackboard.get("computed_results", {}),
            "comparison_table": table_md,
            "comparison_data": compare_data,
            "outliers": outliers,
            "type": "compare_companies",
        }, "business_compute")

        state.add_to_blackboard("chart_paths", state.chart_paths + charts, "business_compute")

        state.add_execution_trace({
            "agent": "business_compute",
            "action": "compare_companies",
            "companies": len(compare_data),
            "charts_generated": len(charts),
        })

    def _visualize(self, state: Any, params: Dict, session_id: str) -> None:
        market_data = state.blackboard.get("market_data", {})
        computed = state.blackboard.get("computed_results", {}).get("metrics", {})
        chart_type = params.get("chart_type", "trend")
        enable_critique = params.get("enable_chart_critique", False)
        charts = []

        for h in market_data.get("history", []):
            data = h.get("data", [])
            if not data:
                continue
            import pandas as pd
            df = pd.DataFrame(data)
            sym = h.get("symbol", "unknown")

            if chart_type in ("kline", "trend") and "close" in df.columns:
                path = self.visualizer.kline_chart(
                    df, title=f"{sym} K线图",
                    session_id=session_id, name=f"kline_{sym.lower()}")
                if path:
                    charts.append({"title": f"{sym} K线", "path": path,
                        "description": f"{sym} 历史K线走势"})
                    if enable_critique:
                        self._apply_chart_critique(state, path, session_id, f"kline_{sym.lower()}")

        for sym, metrics in computed.items():
            flat = {}
            for cat, vals in metrics.items():
                if isinstance(vals, dict):
                    for k, v in vals.items():
                        if v is not None:
                            flat[k] = v
            if not flat:
                continue

            labels = [k for k, v in flat.items() if v is not None and v > 0]
            values = [flat[k] for k in labels]

            if len(labels) >= 2 and chart_type in ("pie", "trend"):
                path = self.visualizer.pie_chart(
                    labels=labels[:6], values=values[:6],
                    title=f"{sym} 指标分布",
                    session_id=session_id, name=f"pie_{sym.lower()}")
                if path:
                    charts.append({"title": f"{sym} 指标分布", "path": path,
                        "description": "关键财务指标占比"})

        state.add_to_blackboard("chart_paths", state.chart_paths + charts, "business_compute")
        state.add_to_blackboard("computed_results", {
            **state.blackboard.get("computed_results", {}),
            "charts_generated": len(charts),
        }, "business_compute")

        state.add_execution_trace({
            "agent": "business_compute",
            "action": "visualize",
            "charts_generated": len(charts),
        })

    def _render_report(self, state: Any, params: Dict, session_id: str) -> None:
        computed = state.blackboard.get("computed_results", {})
        metrics = computed.get("metrics", {})
        comparison_table = computed.get("comparison_table", "")
        charts = state.chart_paths
        report_type = params.get("report_type", "earnings_review")

        symbols = list(metrics.keys()) if metrics else []

        chart_refs = []
        for c in charts:
            if isinstance(c, dict):
                chart_refs.append(c)

        all_valuation = {}
        all_profitability = {}
        all_growth = {}
        all_solvency = {}

        for sym, m in metrics.items():
            for k, v in m.get("valuation", {}).items():
                all_valuation[f"{sym} {k}"] = f"{v:.2f}" if v else "—"
            for k, v in m.get("profitability", {}).items():
                all_profitability[f"{sym} {k}"] = f"{v:.2f}%" if v else "—"
            for k, v in m.get("growth", {}).items():
                all_growth[f"{sym} {k}"] = f"{v:+.2f}%" if v else "—"
            for k, v in m.get("solvency", {}).items():
                all_solvency[f"{sym} {k}"] = f"{v:.2f}%" if v else "—"

        if report_type == "industry_comparison":
            report = self.renderer.render_industry_comparison(
                symbols=symbols,
                comparison_table=comparison_table,
                charts=chart_refs,
            )
        else:
            report = self.renderer.render_earnings_review(
                symbol=symbols[0] if len(symbols) == 1 else ", ".join(symbols),
                metrics=metrics.get(symbols[0] if symbols else "unknown", {}),
                valuation=all_valuation,
                profitability=all_profitability,
                growth=all_growth,
                solvency=all_solvency,
                comparison_table=comparison_table,
                charts=chart_refs,
            )

        state.add_to_blackboard("generated_report", report, "business_compute")

        state.add_execution_trace({
            "agent": "business_compute",
            "action": "generate_report",
            "report_type": report_type,
            "report_length": len(report),
        })

    def _apply_chart_critique(self, state: Any, chart_path: str, session_id: str, name: str) -> None:
        critic = self._get_chart_critic(state)
        max_iterations = 3
        for i in range(max_iterations):
            verdict, feedback = critic.review(chart_path)
            state.add_execution_trace({
                "agent": "chart_critic", "iteration": i + 1,
                "overall": feedback.get("overall", 1.0),
                "verdict": verdict,
            })
            if verdict == "PASS":
                break
            if i < max_iterations - 1:
                state.add_execution_trace({
                    "agent": "chart_critic", "action": "refine",
                    "suggestions": feedback.get("suggestions", []),
                })
