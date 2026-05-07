"""
Multi-Agent internal tools.

Pure Python functions and utilities used by sub-agents.
Not MCP tools — these are called directly, not via function calling.
"""

from .financial_calculator import (
    FinancialMetrics,
    ValuationResult,
    assess_valuation,
    sensitivity_analysis,
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
from .data_processor import (
    clean_financial_data,
    build_comparison_table,
    rank_companies,
    detect_outliers,
    to_markdown_table,
    aggregate_market_to_table,
)
from .data_visualizer import DataVisualizer
from .report_renderer import ReportRenderer
from .collection_router import CollectionRouter
from .citation_footnoter import CitationFootnoter
from .data_verifier import DataVerifier
from .conflict_detector import ConflictDetector
from .tushare_client import TushareClient
from .sentiment_analyzer import SentimentAnalyzer
from .industry_benchmark import IndustryBenchmark
from .chart_critic import ChartCritic
from .verification_calculator import VerificationCalculator, FixedNumbers
from .verification_validator import VerificationValidator
from .chain_of_analysis import ChainOfAnalysis
from .section_assembler import SectionAssembler
from .pdf_exporter import PDFExporter
from .json_parser import safe_parse_json, safe_parse_json_with_default
from .financial_document_fetcher import FinancialDocumentFetcher
from .report_downloader import ReportDownloader

__all__ = [
    "FinancialMetrics",
    "ValuationResult",
    "assess_valuation",
    "sensitivity_analysis",
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
    "clean_financial_data",
    "build_comparison_table",
    "rank_companies",
    "detect_outliers",
    "to_markdown_table",
    "aggregate_market_to_table",
    "DataVisualizer",
    "ReportRenderer",
    "CollectionRouter",
    "CitationFootnoter",
    "DataVerifier",
    "ConflictDetector",
    "TushareClient",
    "SentimentAnalyzer",
    "IndustryBenchmark",
    "ChartCritic",
    "VerificationCalculator",
    "FixedNumbers",
    "VerificationValidator",
    "ChainOfAnalysis",
    "SectionAssembler",
    "PDFExporter",
    "safe_parse_json",
    "safe_parse_json_with_default",
    "FinancialDocumentFetcher",
    "ReportDownloader",
]
