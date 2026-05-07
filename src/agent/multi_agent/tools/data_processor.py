"""
Data Processor — Pandas-based financial data cleaning and analysis.

Pure Python functions. Input: structured dicts from market data or RAG extraction.
Output: cleaned DataFrames, comparison tables, rankings, outlier lists.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import pandas as pd


def clean_financial_data(raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert heterogeneous Agent outputs into a unified DataFrame."""
    if not raw_data:
        return pd.DataFrame()
    df = pd.DataFrame(raw_data)
    for col in df.columns:
        if col in ("symbol", "name", "currency", "report_period", "source"):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_comparison_table(
    data_by_symbol: Dict[str, Dict[str, Any]],
    metrics: List[str],
) -> pd.DataFrame:
    """Build a multi-dimensional comparison table.

    Args:
        data_by_symbol: {"300750.SZ": {"roe": 18.2, "pe": 23.4}, ...}
        metrics: list of metric names to compare

    Returns:
        DataFrame with symbols as index and metrics as columns
    """
    rows = []
    for symbol, d in data_by_symbol.items():
        row = {"symbol": symbol}
        for m in metrics:
            row[m] = d.get(m)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def rank_companies(
    df: pd.DataFrame,
    by_metric: str,
    ascending: bool = False,
) -> pd.DataFrame:
    """Rank companies by a specific metric. Higher is better by default."""
    if df.empty or by_metric not in df.columns:
        return df
    return df.sort_values(by_metric, ascending=ascending)


def detect_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = "iqr",
) -> List[int]:
    """Detect outlier row indices using IQR or Z-score method."""
    if df.empty or column not in df.columns:
        return []
    series = df[column].dropna()
    if len(series) < 4:
        return []

    if method == "iqr":
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (series < lower) | (series > upper)
        return series[mask].index.tolist()

    if method == "zscore":
        mean = series.mean()
        std = series.std()
        if std == 0:
            return []
        z = (series - mean).abs() / std
        return z[z > 3].index.tolist()

    return []


def to_markdown_table(df: pd.DataFrame, title: str = "") -> str:
    """Convert DataFrame to Markdown table string."""
    if df.empty:
        return f"**{title}**\n\n(empty)\n" if title else "(empty)\n"
    lines = []
    if title:
        lines.append(f"### {title}\n")
    lines.append("| " + " | ".join(df.columns) + " |")
    lines.append("|" + "|".join("------" for _ in df.columns) + "|")
    for _, row in df.iterrows():
        vals = []
        for v in row.values:
            if isinstance(v, float):
                vals.append(f"{v:.2f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def aggregate_market_to_table(
    market_data: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """Convert raw market_data blackboard entry into typed DataFrames."""
    result = {}

    quotes = market_data.get("quote", [])
    if quotes:
        result["quote"] = clean_financial_data(quotes)

    fundamentals = market_data.get("fundamentals", [])
    if fundamentals:
        result["fundamentals"] = clean_financial_data(fundamentals)

    return result
