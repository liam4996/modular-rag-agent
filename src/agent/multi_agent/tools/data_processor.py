"""
Data Processor — Pandas-based financial data cleaning and analysis.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
import pandas as pd


def clean_financial_data(raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
    if not raw_data:
        return pd.DataFrame()
    df = pd.DataFrame(raw_data)
    for col in df.columns:
        if col in ("symbol", "name", "currency", "report_period", "source"):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_comparison_table(data_by_symbol, metrics):
    rows = []
    for symbol, d in data_by_symbol.items():
        row = {"symbol": symbol}
        for m in metrics:
            row[m] = d.get(m)
        rows.append(row)
    return pd.DataFrame(rows)


def rank_companies(df, by_metric, ascending=False):
    if df.empty or by_metric not in df.columns:
        return df
    return df.sort_values(by_metric, ascending=ascending)


def detect_outliers(df, column, method="iqr"):
    if df.empty or column not in df.columns:
        return []
    series = df[column].dropna()
    if len(series) < 4:
        return []
    if method == "iqr":
        q1 = series.quantile(0.25); q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr; upper = q3 + 1.5 * iqr
        mask = (series < lower) | (series > upper)
        return series[mask].index.tolist()
    if method == "zscore":
        mean = series.mean(); std = series.std()
        if std == 0: return []
        z = (series - mean).abs() / std
        return z[z > 3].index.tolist()
    return []


def to_markdown_table(df, title=""):
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


def aggregate_market_to_table(market_data):
    result = {}
    quotes = market_data.get("quote", [])
    if quotes: result["quote"] = clean_financial_data(quotes)
    fundamentals = market_data.get("fundamentals", [])
    if fundamentals: result["fundamentals"] = clean_financial_data(fundamentals)
    return result
