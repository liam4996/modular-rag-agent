"""
Financial Calculator — pure Python computation engine.

Executes financial metric calculations locally without LLM or API calls.
Input: structured dicts from RAG extraction or market data APIs.
Output: FinancialMetrics dataclass with categorized metrics.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import math


@dataclass
class FinancialMetrics:
    profitability: Dict[str, Optional[float]] = field(default_factory=dict)
    solvency: Dict[str, Optional[float]] = field(default_factory=dict)
    efficiency: Dict[str, Optional[float]] = field(default_factory=dict)
    valuation: Dict[str, Optional[float]] = field(default_factory=dict)
    growth: Dict[str, Optional[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profitability": {k: _round4(v) for k, v in self.profitability.items()},
            "solvency": {k: _round4(v) for k, v in self.solvency.items()},
            "efficiency": {k: _round4(v) for k, v in self.efficiency.items()},
            "valuation": {k: _round4(v) for k, v in self.valuation.items()},
            "growth": {k: _round4(v) for k, v in self.growth.items()},
        }

    def summary_text(self, symbol: str = "") -> str:
        lines = [f"## 财务指标分析 {symbol}".strip()]
        if self.valuation:
            lines.append("### 估值指标")
            for k, v in self.valuation.items():
                if v is not None:
                    lines.append(f"- {k}: {v:.2f}")
        if self.profitability:
            lines.append("### 盈利能力")
            for k, v in self.profitability.items():
                if v is not None:
                    lines.append(f"- {k}: {v:.2f}%")
        if self.solvency:
            lines.append("### 偿债能力")
            for k, v in self.solvency.items():
                if v is not None:
                    lines.append(f"- {k}: {v:.2f}%")
        if self.growth:
            lines.append("### 成长性")
            for k, v in self.growth.items():
                if v is not None:
                    lines.append(f"- {k}: {v:+.2f}%")
        return "\n".join(lines)


def _safe_div(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None:
        return None
    try:
        if math.isnan(numerator) or math.isnan(denominator):
            return None
    except TypeError:
        pass
    if denominator == 0:
        return None
    return numerator / denominator


def _round4(val: Optional[float]) -> Optional[float]:
    if val is None:
        return None
    return round(val, 4)


def calculate_roe(net_income: Optional[float], equity: Optional[float]) -> Optional[float]:
    result = _safe_div(net_income, equity)
    return result * 100 if result is not None else None


def calculate_roa(net_income: Optional[float], total_assets: Optional[float]) -> Optional[float]:
    result = _safe_div(net_income, total_assets)
    return result * 100 if result is not None else None


def calculate_gross_margin(revenue: Optional[float], cost_of_revenue: Optional[float]) -> Optional[float]:
    return _safe_div(revenue - cost_of_revenue, revenue) * 100 if revenue is not None and cost_of_revenue is not None else None


def calculate_net_margin(net_income: Optional[float], revenue: Optional[float]) -> Optional[float]:
    return _safe_div(net_income, revenue) * 100 if net_income is not None and revenue is not None else None


def calculate_debt_to_asset(total_liabilities: Optional[float], total_assets: Optional[float]) -> Optional[float]:
    return _safe_div(total_liabilities, total_assets) * 100 if total_liabilities is not None and total_assets is not None else None


def calculate_current_ratio(current_assets: Optional[float], current_liabilities: Optional[float]) -> Optional[float]:
    return _safe_div(current_assets, current_liabilities) if current_assets is not None and current_liabilities is not None else None


def calculate_pe(price: Optional[float], eps: Optional[float]) -> Optional[float]:
    return _safe_div(price, eps) if price is not None and eps is not None else None


def calculate_pb(price: Optional[float], bvps: Optional[float]) -> Optional[float]:
    return _safe_div(price, bvps) if price is not None and bvps is not None else None


def calculate_ps(price: Optional[float], revenue_per_share: Optional[float]) -> Optional[float]:
    return _safe_div(price, revenue_per_share) if price is not None and revenue_per_share is not None else None


def calculate_yoy(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    if current is None or previous is None:
        return None
    if previous == 0:
        return None
    return (current - previous) / abs(previous) * 100


def calculate_qoq(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    return calculate_yoy(current, previous)


def compute_all_metrics(
    financial_data: Optional[Dict[str, Any]] = None,
    market_data: Optional[Dict[str, Any]] = None,
) -> FinancialMetrics:
    """
    Batch-compute all financial metrics from available data.

    financial_data: dict with keys like net_income, revenue, total_assets, total_equity, etc.
    market_data: dict with keys like price, eps, bvps, pe, pb, etc.
    """
    fd = financial_data or {}
    md = market_data or {}

    net_income = fd.get("net_income") or fd.get("net_profit")
    revenue = fd.get("revenue")
    total_assets = fd.get("total_assets")
    total_equity = fd.get("total_equity") or fd.get("equity")
    total_liabilities = fd.get("total_liabilities")
    cost_of_revenue = fd.get("cost_of_revenue") or fd.get("operating_cost")
    current_assets = fd.get("current_assets")
    current_liabilities = fd.get("current_liabilities")

    price = md.get("price")
    eps = md.get("eps") or fd.get("eps")
    bvps = md.get("bvps") or fd.get("bvps")
    revenue_per_share = md.get("revenue_per_share")

    pe_val = md.get("pe")
    pb_val = md.get("pb")
    ps_val = md.get("ps")
    roe_val = md.get("roe")
    roa_val = md.get("roa")
    gross_margin_val = md.get("gross_margin")
    net_margin_val = md.get("net_margin")

    metrics = FinancialMetrics()

    metrics.valuation["PE"] = pe_val if pe_val is not None else calculate_pe(price, eps)
    metrics.valuation["PB"] = pb_val if pb_val is not None else calculate_pb(price, bvps)
    metrics.valuation["PS"] = ps_val if ps_val is not None else calculate_ps(price, revenue_per_share)

    metrics.profitability["ROE"] = roe_val if roe_val is not None else calculate_roe(net_income, total_equity)
    metrics.profitability["ROA"] = roa_val if roa_val is not None else calculate_roa(net_income, total_assets)
    metrics.profitability["gross_margin"] = gross_margin_val if gross_margin_val is not None else calculate_gross_margin(revenue, cost_of_revenue)
    metrics.profitability["net_margin"] = net_margin_val if net_margin_val is not None else calculate_net_margin(net_income, revenue)

    metrics.solvency["debt_to_asset"] = calculate_debt_to_asset(total_liabilities, total_assets)
    metrics.solvency["current_ratio"] = calculate_current_ratio(current_assets, current_liabilities)

    prev_net_income = fd.get("prev_net_income") or fd.get("prev_net_profit")
    prev_revenue = fd.get("prev_revenue")
    prev_eps = fd.get("prev_eps")

    metrics.growth["revenue_yoy"] = calculate_yoy(revenue, prev_revenue)
    metrics.growth["profit_yoy"] = calculate_yoy(net_income, prev_net_income)
    metrics.growth["eps_yoy"] = calculate_yoy(eps, prev_eps)

    return metrics


@dataclass
class ValuationResult:
    """估值分析结果"""
    pe: Optional[float] = None
    pb: Optional[float] = None
    ps: Optional[float] = None
    ev_ebitda: Optional[float] = None
    peg: Optional[float] = None
    industry_pe_avg: Optional[float] = None
    pe_percentile: Optional[float] = None
    valuation_assessment: str = ""
    target_price: Optional[float] = None
    target_price_lower: Optional[float] = None
    target_price_upper: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pe": _round4(self.pe),
            "pb": _round4(self.pb),
            "ps": _round4(self.ps),
            "ev_ebitda": _round4(self.ev_ebitda),
            "peg": _round4(self.peg),
            "industry_pe_avg": _round4(self.industry_pe_avg),
            "pe_percentile": _round4(self.pe_percentile),
            "valuation_assessment": self.valuation_assessment,
            "target_price": _round4(self.target_price),
            "target_price_lower": _round4(self.target_price_lower),
            "target_price_upper": _round4(self.target_price_upper),
        }


def assess_valuation(
    pe: float,
    pb: float,
    industry_pe_avg: float,
    historical_pe_low: float = 0,
    historical_pe_high: float = 100,
    growth_rate: float = 0,
    eps: float = 0,
) -> ValuationResult:
    """
    综合估值评估。

    判断逻辑:
      - PE < 行业均值 × 0.7 → "低估"
      - PE 行业均值 × 0.7 ~ 1.3 → "合理"
      - PE > 行业均值 × 1.3 → "高估"

    PEG 维度:
      - PEG < 1 → 低估
      - PEG 1~2 → 合理
      - PEG > 2 → 高估
    """
    result = ValuationResult(pe=pe, pb=pb, industry_pe_avg=industry_pe_avg)

    if industry_pe_avg > 0:
        if pe < industry_pe_avg * 0.7:
            result.valuation_assessment = "低估"
        elif pe > industry_pe_avg * 1.3:
            result.valuation_assessment = "高估"
        else:
            result.valuation_assessment = "合理"

    if historical_pe_high > historical_pe_low:
        result.pe_percentile = (pe - historical_pe_low) / (historical_pe_high - historical_pe_low)

    if growth_rate > 0 and eps > 0:
        result.peg = pe / growth_rate
        result.target_price = eps * industry_pe_avg if industry_pe_avg > 0 else None
        if result.target_price:
            result.target_price_lower = result.target_price * 0.9
            result.target_price_upper = result.target_price * 1.1

    return result


def sensitivity_analysis(
    base_eps: float,
    growth_rates: List[float],
    discount_rates: List[float],
) -> Dict:
    """
    DCF 敏感性分析矩阵。

    输入:
      base_eps: 基准 EPS
      growth_rates: 增长率假设列表 [0.05, 0.10, 0.15, 0.20]
      discount_rates: 折现率假设列表 [0.08, 0.10, 0.12]

    输出:
      敏感性分析矩阵
    """
    matrix = {}
    for g in growth_rates:
        row = {}
        for r in discount_rates:
            if r > g:
                value = base_eps * (1 + g) / (r - g)
                row[f"{r*100:.0f}%"] = round(value, 2)
            else:
                row[f"{r*100:.0f}%"] = None
        matrix[f"{g*100:.0f}%"] = row
    return matrix
