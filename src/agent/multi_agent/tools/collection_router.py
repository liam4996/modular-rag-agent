"""
Collection Router — 根据用户 query + intent 决定应该检索哪些 collection。

在现有架构中等价于「同一个 SearchAgent + 不同的 collection 参数」。
"""

from typing import Any, Dict, List, Tuple


class CollectionRouter:
    """
    输入：用户 query + intent
    输出：要检索的 collection 列表 + primary collection
    """

    RULES: List[Tuple[List[str], float, str]] = [
        (["财报", "利润", "营收", "净利润", "毛利率", "ROE", "Q1", "Q2", "Q3", "Q4", "半年报", "年报"], 0.7, "financial_reports"),
        (["公告", "重大", "提示", "停牌", "复牌", "减持", "增持", "分红", "股权"], 0.7, "announcements"),
        (["行业", "景气", "产业链", "上下游", "渗透率", "市场规模"], 0.6, "industry_data"),
        (["研报", "券商", "推荐", "评级", "目标价", "买入", "增持"], 0.6, "research_reports"),
        (["对比", "对标", "竞品", "行业平均", "同行", "同业", "排名"], 0.7, "peer_comparison"),
        (["估值", "PE", "PB", "PS", "DCF", "目标价", "估值区间", "贵", "便宜"], 0.7, "financial_reports"),
        (["内部", "调研", "纪要", "私有", "尽调"], 0.8, "private_research"),
    ]

    INTENT_COLLECTION_MAP = {
        "financial_analysis": ["financial_reports", "industry_data", "peer_comparison"],
        "financial_market": ["research_reports"],
        "financial_report": ["financial_reports", "announcements", "industry_data"],
    }

    def route(self, query: str, intent: str) -> Dict[str, Any]:
        matched = []
        for keywords, weight, collection in self.RULES:
            score = sum(1 for kw in keywords if kw in query) / max(len(keywords), 1)
            if score >= weight:
                matched.append((score, collection))

        intent_collections = self.INTENT_COLLECTION_MAP.get(intent, [])

        all_collections = list(set(
            [c for _, c in sorted(matched, reverse=True)] + intent_collections
        ))
        return {
            "collections": all_collections,
            "primary": all_collections[0] if all_collections else "financial_reports",
        }
