"""
Business Compute Agent — orchestrates financial computation + visualization + reporting.
Pure Python — no LLM.
"""

from __future__ import annotations
import uuid
from typing import Any, Dict, List, Optional
from .tools.financial_calculator import compute_all_metrics
from .tools.data_processor import build_comparison_table, detect_outliers, to_markdown_table
from .tools.data_visualizer import DataVisualizer
from .tools.report_renderer import ReportRenderer


class BusinessComputeAgent:
    def __init__(self, output_dir="data/charts"):
        self.visualizer = DataVisualizer(output_dir=output_dir)
        self.renderer = ReportRenderer()

    def compute(self, state, operation="calculate_metrics", task_params=None):
        params = task_params or {}
        sid = str(uuid.uuid4())[:8]
        if operation == "calculate_metrics": self._calc_metrics(state, params)
        elif operation == "compare_companies": self._compare(state, params, sid)
        elif operation == "visualize": self._visualize(state, params, sid)
        elif operation == "generate_report": self._render_report(state, params, sid)
        return state

    def _calc_metrics(self, state, params):
        md = state.blackboard.get("market_data", {})
        all_m = {}
        for sd in md.get("fundamentals", []):
            sym = sd.get("symbol", "unknown")
            m = compute_all_metrics(financial_data=sd, market_data=sd)
            all_m[sym] = m.to_dict()
        if not all_m and md.get("quote"):
            for q in md.get("quote", []):
                sym = q.get("symbol", "unknown")
                m = compute_all_metrics(market_data=q)
                if any(v for v in m.valuation.values() if v is not None):
                    all_m[sym] = m.to_dict()
        state.add_to_blackboard("computed_results", {
            **state.blackboard.get("computed_results", {}),
            "metrics": all_m, "type": "calculate_metrics",
        }, "business_compute")
        state.add_execution_trace({"agent": "business_compute", "action": "calculate_metrics", "symbols_processed": len(all_m)})

    def _compare(self, state, params, sid):
        computed = state.blackboard.get("computed_results", {}).get("metrics", {})
        cd = {}
        for sym, metrics in computed.items():
            flat = {}
            for cat, vals in metrics.items():
                if isinstance(vals, dict):
                    for k, v in vals.items():
                        if v is not None: flat[k] = v
            if flat: cd[sym] = flat
        mtc = params.get("metrics", ["PE", "PB", "ROE", "gross_margin", "net_margin"])
        df = build_comparison_table(cd, mtc)
        tmd = to_markdown_table(df, "行业指标对比")
        outliers = {}
        for col in df.columns:
            idx = detect_outliers(df, col, method="iqr")
            if idx: outliers[col] = idx
        charts = []
        if len(cd) > 1 and not df.empty:
            import pandas as pd
            chd = {}
            for sym, d in cd.items(): chd[sym] = pd.DataFrame([d])
            for metric in mtc[:3]:
                if metric in df.columns:
                    p = self.visualizer.comparison_chart(data=chd, metric=metric,
                        title=f"行业对比 — {metric}", session_id=sid, name=f"compare_{metric.lower()}")
                    if p: charts.append({"title": f"{metric} 对比", "path": p})
        state.add_to_blackboard("computed_results", {
            **state.blackboard.get("computed_results", {}),
            "comparison_table": tmd, "comparison_data": cd,
            "outliers": outliers, "type": "compare_companies",
        }, "business_compute")
        state.add_to_blackboard("chart_paths", state.chart_paths + charts, "business_compute")

    def _visualize(self, state, params, sid):
        md = state.blackboard.get("market_data", {})
        computed = state.blackboard.get("computed_results", {}).get("metrics", {})
        ct = params.get("chart_type", "trend")
        charts = []
        import pandas as pd
        for h in md.get("history", []):
            data = h.get("data", [])
            if not data: continue
            df = pd.DataFrame(data)
            sym = h.get("symbol", "unknown")
            if ct in ("kline", "trend") and "close" in df.columns:
                p = self.visualizer.kline_chart(df, title=f"{sym} K线图", session_id=sid, name=f"kline_{sym.lower()}")
                if p: charts.append({"title": f"{sym} K线", "path": p})
        for sym, metrics in computed.items():
            flat = {}
            for cat, vals in metrics.items():
                if isinstance(vals, dict):
                    for k, v in vals.items():
                        if v is not None: flat[k] = v
            if not flat: continue
            labels = [k for k, v in flat.items() if v is not None and v > 0]
            values = [flat[k] for k in labels]
            if len(labels) >= 2 and ct in ("pie", "trend"):
                p = self.visualizer.pie_chart(labels=labels[:6], values=values[:6],
                    title=f"{sym} 指标分布", session_id=sid, name=f"pie_{sym.lower()}")
                if p: charts.append({"title": f"{sym} 指标分布", "path": p})
        state.add_to_blackboard("chart_paths", state.chart_paths + charts, "business_compute")
        state.add_to_blackboard("computed_results", {
            **state.blackboard.get("computed_results", {}),
            "charts_generated": len(charts),
        }, "business_compute")

    def _render_report(self, state, params, sid):
        computed = state.blackboard.get("computed_results", {})
        metrics = computed.get("metrics", {})
        ct = computed.get("comparison_table", "")
        charts = state.chart_paths
        rt = params.get("report_type", "earnings_review")
        symbols = list(metrics.keys()) if metrics else []
        cr = [c for c in charts if isinstance(c, dict)]
        av, ap, ag, aks = {}, {}, {}, {}
        for sym, m in metrics.items():
            for k, v in m.get("valuation", {}).items():
                av[f"{sym} {k}"] = f"{v:.2f}" if v else "—"
            for k, v in m.get("profitability", {}).items():
                ap[f"{sym} {k}"] = f"{v:.2f}%" if v else "—"
            for k, v in m.get("growth", {}).items():
                ag[f"{sym} {k}"] = f"{v:+.2f}%" if v else "—"
            for k, v in m.get("solvency", {}).items():
                aks[f"{sym} {k}"] = f"{v:.2f}%" if v else "—"
        if rt == "industry_comparison":
            report = self.renderer.render_industry_comparison(symbols=symbols, comparison_table=ct, charts=cr)
        else:
            report = self.renderer.render_earnings_review(
                symbol=symbols[0] if len(symbols) == 1 else ", ".join(symbols),
                metrics=metrics.get(symbols[0] if symbols else "unknown", {}),
                valuation=av, profitability=ap, growth=ag, solvency=aks,
                comparison_table=ct, charts=cr)
        state.add_to_blackboard("generated_report", report, "business_compute")
        state.add_execution_trace({"agent": "business_compute", "action": "generate_report", "report_type": rt, "report_length": len(report)})
