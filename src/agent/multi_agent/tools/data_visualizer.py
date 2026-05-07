"""
Data Visualizer — chart generation with Matplotlib.

Generates PNG charts from financial data: K-line, trend, comparison, pie.
Pure Python — no LLM involved. LLM only decides chart_type and passes data.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

import numpy as np
import pandas as pd


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class DataVisualizer:
    """Generate financial charts as PNG files.

    Output: data/charts/{session_id}/{chart_name}.png

    Each method returns the file path of the generated chart.
    """

    def __init__(self, output_dir: str = "data/charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def kline_chart(
        self,
        df: pd.DataFrame,
        title: str = "K线图",
        session_id: str = "default",
        name: str = "kline",
    ) -> str:
        """Candlestick-style OHLC chart using matplotlib bars.

        Args:
            df: columns [date, open, high, low, close, volume] (optional volume)
            title: chart title
            session_id: session subdirectory
            name: base filename

        Returns:
            Absolute path to the generated PNG.
        """
        if df.empty:
            return ""

        os.makedirs(self.output_dir / session_id, exist_ok=True)
        save_path = self.output_dir / session_id / f"{name}.png"

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        has_vol = "volume" in df.columns

        ratio = 0.6 if has_vol else 0.95
        fig_height = 6 if has_vol else 5

        if has_vol:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, fig_height),
                gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
        else:
            fig, ax1 = plt.subplots(figsize=(12, fig_height))
            ax2 = None

        colors = ["#e74c3c" if row["close"] >= row["open"] else "#2ecc71"
            for _, row in df.iterrows()]
        bar_width = np.timedelta64(12, "h") if len(df) > 1 else 0.8

        if len(df) > 1:
            delta = (df["date"].iloc[1] - df["date"].iloc[0])
            if isinstance(delta, pd.Timedelta):
                bar_width = delta * 0.6

        ax1.bar(df["date"], df["high"] - df["low"], bottom=df["low"],
            width=bar_width, color=colors, edgecolor="black", linewidth=0.5)
        ax1.bar(df["date"], abs(df["close"] - df["open"]),
            bottom=df[["open", "close"]].min(axis=1),
            width=bar_width * 0.8, color=colors, edgecolor="black", linewidth=0.5)

        ax1.set_title(title, fontsize=14, fontweight="bold")
        ax1.set_ylabel("价格")
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()

        if has_vol and ax2:
            vol_colors = ["#e74c3c" if row["close"] >= row["open"] else "#2ecc71"
                for _, row in df.iterrows()]
            ax2.bar(df["date"], df["volume"], width=bar_width, color=vol_colors,
                alpha=0.6, edgecolor="black", linewidth=0.3)
            ax2.set_ylabel("成交量")
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(save_path.resolve())

    def trend_chart(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_cols: Union[str, List[str]],
        chart_type: str = "line",
        title: str = "趋势图",
        session_id: str = "default",
        name: str = "trend",
    ) -> str:
        """Line/bar/area trend chart.

        Args:
            df: DataFrame with x_col as x-axis and y_cols as y-axis values
            x_col: column for x-axis
            y_cols: single column name or list of columns for y-axis
            chart_type: "line", "bar", or "area"
        """
        if df.empty:
            return ""

        os.makedirs(self.output_dir / session_id, exist_ok=True)
        save_path = self.output_dir / session_id / f"{name}.png"

        fig, ax = plt.subplots(figsize=(10, 5))
        y_list = [y_cols] if isinstance(y_cols, str) else y_cols
        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

        if chart_type == "bar":
            x = range(len(df))
            w = 0.8 / len(y_list)
            for i, ycol in enumerate(y_list):
                if ycol in df.columns:
                    ax.bar([v + i * w for v in x], df[ycol].values, w,
                        label=ycol, color=colors[i % len(colors)], alpha=0.85)
            ax.set_xticks([v + w * (len(y_list) - 1) / 2 for v in x])
            ax.set_xticklabels(df[x_col].values, rotation=45)
        elif chart_type == "area":
            for i, ycol in enumerate(y_list):
                if ycol in df.columns:
                    ax.fill_between(range(len(df)), df[ycol].values, alpha=0.3,
                        color=colors[i % len(colors)], label=ycol)
                    ax.plot(range(len(df)), df[ycol].values,
                        color=colors[i % len(colors)], linewidth=2)
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(df[x_col].values, rotation=45)
        else:
            for i, ycol in enumerate(y_list):
                if ycol in df.columns:
                    ax.plot(df[x_col], df[ycol], marker="o", linewidth=2,
                        color=colors[i % len(colors)], label=ycol, markersize=4)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            fig.autofmt_xdate()

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(save_path.resolve())

    def comparison_chart(
        self,
        data: Dict[str, pd.DataFrame],
        metric: str,
        chart_type: str = "grouped_bar",
        title: str = "对比图",
        session_id: str = "default",
        name: str = "comparison",
    ) -> str:
        """Multi-company grouped bar comparison chart.

        Args:
            data: {"CompanyA": DataFrame, "CompanyB": DataFrame}
            metric: column name to compare
        """
        if not data:
            return ""

        os.makedirs(self.output_dir / session_id, exist_ok=True)
        save_path = self.output_dir / session_id / f"{name}.png"

        fig, ax = plt.subplots(figsize=(10, 5))
        names = list(data.keys())
        values = []
        for name_k in names:
            d = data[name_k]
            val = d[metric].values[0] if metric in d.columns and not d.empty else 0
            values.append(float(val))

        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
        bars = ax.bar(names, values, color=colors[:len(names)], alpha=0.85, edgecolor="white")

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontweight="bold")

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(save_path.resolve())

    def pie_chart(
        self,
        labels: List[str],
        values: List[float],
        title: str = "占比图",
        session_id: str = "default",
        name: str = "pie",
    ) -> str:
        """Pie / donut chart.

        Args:
            labels: category names
            values: corresponding values (auto-normalized to sum=100%)
        """
        if not labels or not values:
            return ""

        os.makedirs(self.output_dir / session_id, exist_ok=True)
        save_path = self.output_dir / session_id / f"{name}.png"

        fig, ax = plt.subplots(figsize=(8, 8))
        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c",
            "#34495e", "#e67e22"]
        total = sum(values) or 1

        wedges, texts, autotexts = ax.pie(
            values, labels=labels, autopct=lambda pct: f"{pct:.1f}%",
            colors=colors[:len(labels)], startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2})

        for t in autotexts:
            t.set_fontsize(10)
            t.set_fontweight("bold")

        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(save_path.resolve())
