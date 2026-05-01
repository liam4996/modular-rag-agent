"""
Multi-Agent internal tools.

Pure Python functions and utilities used by sub-agents.
Not MCP tools — these are called directly, not via function calling.
"""

from .financial_calculator import (
    FinancialMetrics,
    calculate_roe,
    calculate_roa,
    calculate_gross_margin,
    calculate_net_margin,
    calculate_debt_to_asset,
    calculate_current_ratio,
    calculate_pe,
    calculate_pb,
    calculate_yoy,
    calculate_qoq,
    compute_all_metrics,
)

__all__ = [
    "FinancialMetrics",
    "calculate_roe",
    "calculate_roa",
    "calculate_gross_margin",
    "calculate_net_margin",
    "calculate_debt_to_asset",
    "calculate_current_ratio",
    "calculate_pe",
    "calculate_pb",
    "calculate_yoy",
    "calculate_qoq",
    "compute_all_metrics",
]
