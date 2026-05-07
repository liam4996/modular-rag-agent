"""纯数学计算引擎 — 三层溯源校验的 Layer 1。

输入：market_data + computed_results
输出：FixedNumbers（不可变数据结构）
不调 LLM，所有计算是确定性的。
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class FixedNumbers:
    """不可变数字集合 — 所有计算值只在这里产生，LLM 无权修改"""
    target_price: Optional[float] = None
    target_price_low: Optional[float] = None
    target_price_high: Optional[float] = None
    target_price_3m: Optional[float] = None
    target_price_6m: Optional[float] = None
    target_price_12m: Optional[float] = None
    expected_return: Optional[float] = None
    pe_vs_industry: Optional[float] = None
    peg_ratio: Optional[float] = None
    revenue_growth: Optional[float] = None
    profit_growth: Optional[float] = None
    valuation_gap: Optional[float] = None
    deprecated_reference: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for k, v in asdict(self).items():
            if k == "deprecated_reference":
                continue
            if v is not None:
                result[k] = v
        return result


class VerificationCalculator:
    """
    纯数学计算引擎。
    
    输入：market_data + computed_results
    输出：FixedNumbers（不可变）
    不调 LLM，所有计算是确定性的。
    """

    def compute(self, market_data: Dict, computed: Dict) -> FixedNumbers:
        numbers = FixedNumbers()
        metrics = computed.get("metrics", {})

        for symbol, m in metrics.items():
            if not isinstance(m, dict):
                continue

            valuation = m.get("valuation", {})
            growth = m.get("growth", {})
            if not isinstance(valuation, dict) or not isinstance(growth, dict):
                continue

            pe = valuation.get("PE")
            pb = valuation.get("PB")
            industry_pe = valuation.get("industry_pe_avg")
            revenue_growth = growth.get("revenue_yoy")
            profit_growth = growth.get("profit_yoy")

            self._calc_target_price(numbers, pe, pb, industry_pe, revenue_growth, market_data)
            self._calc_peg(numbers, pe, revenue_growth)
            self._calc_growth(numbers, revenue_growth, profit_growth)
            self._calc_valuation_gap(numbers, pe, industry_pe)
            break

        return numbers

    def _calc_target_price(self, numbers: FixedNumbers, pe: Optional[float],
                            pb: Optional[float], industry_pe: Optional[float],
                            growth: Optional[float], market_data: Dict) -> None:
        quotes = market_data.get("quote", [])
        price = None
        for q in quotes:
            if isinstance(q, dict) and q.get("price") is not None:
                price = q["price"]
                break

        if price is not None and industry_pe is not None and industry_pe > 0 and pe is not None and pe > 0:
            eps = price / pe
            numbers.target_price = round(eps * industry_pe, 2)
            numbers.target_price_3m = round(numbers.target_price * 0.95, 2)
            numbers.target_price_6m = round(numbers.target_price * 1.02, 2)
            numbers.target_price_12m = round(numbers.target_price * 1.08, 2)
            numbers.target_price_low = round(numbers.target_price * 0.85, 2)
            numbers.target_price_high = round(numbers.target_price * 1.15, 2)
            numbers.deprecated_reference = "target_price is deprecated; use target_price_low/target_price_high range"
            if price > 0:
                numbers.expected_return = round((numbers.target_price - price) / price * 100, 2)

    def _calc_peg(self, numbers: FixedNumbers, pe: Optional[float],
                  growth: Optional[float]) -> None:
        if pe is not None and growth is not None and growth > 0:
            numbers.peg_ratio = round(pe / growth, 2)

    def _calc_growth(self, numbers: FixedNumbers, revenue_growth: Optional[float],
                     profit_growth: Optional[float]) -> None:
        if revenue_growth is not None:
            numbers.revenue_growth = round(revenue_growth, 2)
        if profit_growth is not None:
            numbers.profit_growth = round(profit_growth, 2)

    def _calc_valuation_gap(self, numbers: FixedNumbers, pe: Optional[float],
                            industry_pe: Optional[float]) -> None:
        if pe is not None and industry_pe is not None and industry_pe > 0:
            numbers.pe_vs_industry = round((pe - industry_pe) / industry_pe * 100, 2)
