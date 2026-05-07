"""跨 Agent 输出矛盾检测器。

在 aggregate 节点后运行，检测 4 类冲突:
1. 数据冲突: RAG 提取的财报数据 vs FinanceData 的行情数据不一致
2. 逻辑冲突: 营收增长但股价暴跌 → 需要 LLM 给出解释
3. 情绪冲突: 新闻分析师看多但基本面指标看空
4. 基准冲突: LLM 判断"估值合理"但 PE 远高于行业均值
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ConflictFlag:
    """一条冲突标记"""
    dimension: str = ""           # "data" / "logic" / "sentiment" / "benchmark"
    severity: str = ""            # "high" / "medium" / "low"
    source_a: str = ""            # 冲突方 A
    source_b: str = ""            # 冲突方 B
    claim_a: str = ""             # A 的表述
    claim_b: str = ""             # B 的表述
    explanation: str = ""         # 可能的解释


class ConflictDetector:
    """
    跨 Agent 输出矛盾检测器。

    用法:
      detector = ConflictDetector()
      flags = detector.detect(state.blackboard)
      state.blackboard["conflict_flags"] = flags
    """

    def detect(self, blackboard: Dict[str, Any]) -> List[ConflictFlag]:
        flags = []
        flags.extend(self._check_revenue_vs_price(blackboard))
        flags.extend(self._check_pe_vs_benchmark(blackboard))
        flags.extend(self._check_sentiment_vs_fundamental(blackboard))
        flags.extend(self._check_growth_vs_valuation(blackboard))
        return flags

    def _check_revenue_vs_price(self, blackboard: Dict) -> List[ConflictFlag]:
        """营收增长 vs 股价表现 — 高优先级"""
        flags = []
        local = blackboard.get("local_results", [])
        market = blackboard.get("market_data", {})

        revenue_growth = self._extract_revenue_growth(local)
        price_change = self._extract_price_change(market)

        if revenue_growth is not None and price_change is not None:
            if revenue_growth > 10 and price_change < -10:
                flags.append(ConflictFlag(
                    dimension="logic",
                    severity="high",
                    source_a="RAG Agent (财报)",
                    source_b="Finance Data Agent (行情)",
                    claim_a=f"营收同比增长 {revenue_growth:+.1f}%",
                    claim_b=f"股价同期下跌 {price_change:+.1f}%",
                    explanation="营收增长但股价下跌，可能原因：行业估值中枢下移、"
                                "市场提前消化预期、财报存在一次性收益、"
                                "市场关注非财务风险因素。建议研报中重点分析。"
                ))
        return flags

    def _check_pe_vs_benchmark(self, blackboard: Dict) -> List[ConflictFlag]:
        """实际 PE 与行业均值 — 中优先级"""
        flags = []
        computed = blackboard.get("computed_results", {})
        valuation = computed.get("valuation", {})

        for symbol, val in valuation.items():
            pe = val.get("pe") if isinstance(val, dict) else None
            industry_pe = val.get("industry_pe_avg") if isinstance(val, dict) else None

            if pe is not None and industry_pe is not None and industry_pe > 0:
                if pe > industry_pe * 1.3:
                    flags.append(ConflictFlag(
                        dimension="data",
                        severity="medium",
                        source_a="Business Compute Agent (计算)",
                        source_b="IndustryBenchmark (行业基准)",
                        claim_a=f"{symbol} PE = {pe:.1f}x",
                        claim_b=f"行业均值 PE = {industry_pe:.1f}x",
                        explanation=f"PE 超出行业均值 {((pe/industry_pe)-1)*100:.0f}%，"
                                    f"需要 LLM 判断是否合理。"
                    ))
        return flags

    def _check_sentiment_vs_fundamental(self, blackboard: Dict) -> List[ConflictFlag]:
        """新闻情绪 vs 基本面 — 中优先级"""
        flags = []
        computed = blackboard.get("computed_results", {})
        sentiment = computed.get("sentiment", {})
        metrics = computed.get("metrics", {})

        sentiment_label = sentiment.get("label") if isinstance(sentiment, dict) else None

        for symbol, m in metrics.items():
            if not isinstance(m, dict):
                continue
            profitability = m.get("profitability", {})
            roe = profitability.get("roe") if isinstance(profitability, dict) else None

            if roe is not None and sentiment_label == "利好":
                if roe < 5:
                    flags.append(ConflictFlag(
                        dimension="sentiment",
                        severity="medium",
                        source_a="SentimentAnalyzer (新闻情绪)",
                        source_b="Business Compute Agent (基本面)",
                        claim_a=f"新闻情绪: {sentiment_label}",
                        claim_b=f"ROE = {roe:.1f}% (低于5%)",
                        explanation="新闻面利好但基本面弱势，建议关注是否短期炒作。"
                    ))
        return flags

    def _check_growth_vs_valuation(self, blackboard: Dict) -> List[ConflictFlag]:
        """成长性 vs 估值 (PEG) — 低优先级"""
        flags = []
        computed = blackboard.get("computed_results", {})
        metrics = computed.get("metrics", {})

        for symbol, m in metrics.items():
            if not isinstance(m, dict):
                continue
            valuation = m.get("valuation", {})
            growth = m.get("growth", {})
            if not isinstance(valuation, dict) or not isinstance(growth, dict):
                continue
            pe = valuation.get("pe")
            revenue_growth = growth.get("revenue_yoy")

            if pe is not None and revenue_growth is not None and revenue_growth > 0:
                peg = pe / revenue_growth
                if peg > 2:
                    flags.append(ConflictFlag(
                        dimension="benchmark",
                        severity="low",
                        source_a="PEG 指标",
                        source_b="估值合理性",
                        claim_a=f"PEG = {peg:.2f} (> 2.0)",
                        claim_b=f"PE = {pe:.1f}x, 增长率 = {revenue_growth:.1f}%",
                        explanation="PEG > 2 表明估值偏高，需 LLM 判断是否有特殊溢价逻辑。"
                    ))
        return flags

    @staticmethod
    def _extract_revenue_growth(local_results: List[Dict]) -> Optional[float]:
        for doc in local_results:
            content = doc.get("content", "")
            match = re.search(r"营收(?:同比)?(?:增长|下降|增加|减少)?[：:]?\s*([+-]?\d+\.?\d*)%", content)
            if match:
                val = float(match.group(1))
                if "下降" in content or "减少" in content:
                    val = -val
                return val
        return None

    @staticmethod
    def _extract_price_change(market_data: Dict) -> Optional[float]:
        for q in market_data.get("quote", []):
            if isinstance(q, dict) and q.get("change_pct") is not None:
                return q["change_pct"]
        return None
