"""行业基准数据管理。

存储行业平均估值区间、财务指标均值，
供 Business Compute Agent 做行业对标时使用。

Phase 1: 硬编码典型值
Phase 2: 通过 FinanceDataAgent 实时拉取行业指数
Phase 3: 从 RAG 文档中自动提取
"""

from typing import Dict, Optional


class IndustryBenchmark:
    """行业基准数据查询与对比。"""

    DEFAULT_BENCHMARKS = {
        "新能源电池": {"pe_avg": 25.0, "pb_avg": 3.5, "roe_avg": 15.0, "gross_margin_avg": 25.0},
        "光伏":       {"pe_avg": 20.0, "pb_avg": 2.5, "roe_avg": 12.0, "gross_margin_avg": 20.0},
        "半导体":     {"pe_avg": 45.0, "pb_avg": 5.0, "roe_avg": 10.0, "gross_margin_avg": 35.0},
        "白酒":       {"pe_avg": 30.0, "pb_avg": 8.0, "roe_avg": 25.0, "gross_margin_avg": 75.0},
        "银行":       {"pe_avg": 6.0,  "pb_avg": 0.6, "roe_avg": 10.0, "gross_margin_avg": None},
        "医药":       {"pe_avg": 35.0, "pb_avg": 4.0, "roe_avg": 12.0, "gross_margin_avg": 60.0},
        "消费电子":   {"pe_avg": 28.0, "pb_avg": 4.5, "roe_avg": 14.0, "gross_margin_avg": 30.0},
        "互联网":     {"pe_avg": 30.0, "pb_avg": 5.0, "roe_avg": 18.0, "gross_margin_avg": 45.0},
        "房地产":     {"pe_avg": 8.0,  "pb_avg": 0.8, "roe_avg": 8.0,  "gross_margin_avg": None},
        "食品饮料":   {"pe_avg": 35.0, "pb_avg": 6.0, "roe_avg": 20.0, "gross_margin_avg": 50.0},
    }

    def get_benchmark(self, industry: str) -> Dict:
        return self.DEFAULT_BENCHMARKS.get(industry, {})

    def compare_to_benchmark(self, metrics: Dict, industry: str) -> Dict:
        benchmark = self.get_benchmark(industry)
        if not benchmark:
            return {}

        result = {}

        pe = metrics.get("pe")
        pe_avg = benchmark.get("pe_avg")
        if pe is not None and pe_avg is not None and pe_avg > 0:
            delta = (pe - pe_avg) / pe_avg * 100
            result["pe"] = {
                "company": round(pe, 2),
                "industry": pe_avg,
                "delta_pct": round(delta, 1),
                "assessment": "偏高" if delta > 30 else ("偏低" if delta < -30 else "合理"),
            }

        roe = metrics.get("roe")
        roe_avg = benchmark.get("roe_avg")
        if roe is not None and roe_avg is not None and roe_avg > 0:
            delta = roe - roe_avg
            result["roe"] = {
                "company": round(roe, 2),
                "industry": roe_avg,
                "delta": round(delta, 2),
                "assessment": "领先" if delta > 3 else ("落后" if delta < -3 else "持平"),
            }

        pb = metrics.get("pb")
        pb_avg = benchmark.get("pb_avg")
        if pb is not None and pb_avg is not None and pb_avg > 0:
            delta = (pb - pb_avg) / pb_avg * 100
            result["pb"] = {
                "company": round(pb, 2),
                "industry": pb_avg,
                "delta_pct": round(delta, 1),
                "assessment": "偏高" if delta > 30 else ("偏低" if delta < -30 else "合理"),
            }

        gm = metrics.get("gross_margin")
        gm_avg = benchmark.get("gross_margin_avg")
        if gm is not None and gm_avg is not None and gm_avg > 0:
            delta = gm - gm_avg
            result["gross_margin"] = {
                "company": round(gm, 2),
                "industry": gm_avg,
                "delta": round(delta, 2),
                "assessment": "领先" if delta > 5 else ("落后" if delta < -5 else "持平"),
            }

        return result
