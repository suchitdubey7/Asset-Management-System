"""
Asset Management Intelligence System — Core Data Models
========================================================
Shared data structures used across all agents.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


# ─── Enumerations ───────────────────────────────────────────────────────────

class Sector(str, Enum):
    TECHNOLOGY      = "Technology"
    HEALTHCARE      = "Healthcare"
    FINANCIALS      = "Financials"
    ENERGY          = "Energy"
    CONSUMER_DISC   = "Consumer Discretionary"
    CONSUMER_STAPLES = "Consumer Staples"
    INDUSTRIALS     = "Industrials"
    MATERIALS       = "Materials"
    UTILITIES       = "Utilities"
    REAL_ESTATE     = "Real Estate"
    COMMUNICATION   = "Communication Services"

class MacroRegime(str, Enum):
    EXPANSION       = "Expansion"
    PEAK            = "Peak"
    CONTRACTION     = "Contraction"
    TROUGH          = "Trough"
    STAGFLATION     = "Stagflation"
    RECOVERY        = "Recovery"

class RiskLevel(str, Enum):
    LOW             = "Low"
    MODERATE        = "Moderate"
    HIGH            = "High"
    CRITICAL        = "Critical"

class AlertType(str, Enum):
    EARNINGS_DOWNGRADE   = "Earnings Downgrade"
    MACRO_REGIME_SHIFT   = "Macro Regime Shift"
    SECTOR_BREACH        = "Sector Exposure Breach"
    LIQUIDITY_RISK       = "Liquidity Deterioration"
    VOLATILITY_SPIKE     = "Volatility Spike"
    DRAWDOWN_BREACH      = "Drawdown Threshold Breach"

class Sentiment(str, Enum):
    VERY_POSITIVE   = "Very Positive"
    POSITIVE        = "Positive"
    NEUTRAL         = "Neutral"
    NEGATIVE        = "Negative"
    VERY_NEGATIVE   = "Very Negative"


# ─── Market & Company Data ───────────────────────────────────────────────────

@dataclass
class PriceData:
    ticker: str
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: float
    returns_1d: float = 0.0
    returns_1w: float = 0.0
    returns_1m: float = 0.0
    returns_3m: float = 0.0
    returns_ytd: float = 0.0
    volatility_30d: float = 0.0

@dataclass
class FinancialMetrics:
    ticker: str
    company_name: str
    sector: Sector
    fiscal_period: str
    # Income Statement
    revenue: float
    revenue_growth_yoy: float
    gross_margin: float
    ebitda_margin: float
    net_income: float
    eps: float
    eps_growth_yoy: float
    # Balance Sheet
    total_assets: float
    total_debt: float
    cash_and_equivalents: float
    net_debt: float
    book_value_per_share: float
    # Cash Flow
    operating_cash_flow: float
    free_cash_flow: float
    capex: float
    # Valuation
    market_cap: float
    pe_ratio: float
    ev_ebitda: float
    price_to_book: float
    price_to_sales: float
    dividend_yield: float
    # Quality Metrics
    roe: float
    roa: float
    roic: float
    debt_to_equity: float
    current_ratio: float
    interest_coverage: float

@dataclass
class EarningsTranscript:
    ticker: str
    company_name: str
    date: datetime
    quarter: str
    management_tone: Sentiment
    key_topics: List[str]
    guidance_revision: str          # "raised" | "maintained" | "lowered" | "withdrawn"
    revenue_beat_miss: float        # % vs consensus
    eps_beat_miss: float
    full_text_summary: str
    key_quotes: List[str]

@dataclass
class NewsItem:
    headline: str
    source: str
    published_at: datetime
    tickers_mentioned: List[str]
    sectors_mentioned: List[str]
    sentiment: Sentiment
    relevance_score: float
    summary: str


# ─── Macro Data ──────────────────────────────────────────────────────────────

@dataclass
class MacroIndicator:
    indicator_name: str
    date: datetime
    value: float
    prior_value: float
    change: float
    change_pct: float
    frequency: str          # "monthly" | "quarterly" | "daily"
    source: str

@dataclass
class MacroSnapshot:
    timestamp: datetime
    # Inflation
    cpi_yoy: float
    pce_yoy: float
    # Interest Rates
    fed_funds_rate: float
    us_10y_yield: float
    us_2y_yield: float
    yield_curve_spread: float       # 10Y - 2Y
    # Growth
    gdp_growth_qoq: float
    gdp_growth_yoy: float
    unemployment_rate: float
    # Commodities
    oil_price_wti: float
    gold_price: float
    copper_price: float
    # Credit
    hy_spread: float                # High-yield credit spread
    ig_spread: float                # Investment-grade credit spread
    # Currency
    dxy_index: float
    eurusd: float
    usdjpy: float
    # Sentiment
    vix: float
    consumer_confidence: float
    ism_manufacturing: float
    ism_services: float


# ─── Portfolio Data ──────────────────────────────────────────────────────────

@dataclass
class Holding:
    ticker: str
    company_name: str
    sector: Sector
    shares: int
    cost_basis: float
    current_price: float
    market_value: float
    weight: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    beta: float
    liquidity_score: float          # 0–1 (1 = highly liquid)

@dataclass
class Portfolio:
    portfolio_id: str
    portfolio_name: str
    portfolio_manager: str
    as_of_date: datetime
    total_nav: float
    holdings: List[Holding]
    benchmark: str
    cash_weight: float
    inception_date: datetime

@dataclass
class PortfolioAllocation:
    ticker: str
    company_name: str
    sector: Sector
    target_weight: float
    current_weight: float
    signal_score: float             # -1 to +1
    expected_return: float
    conviction: str                 # "High" | "Medium" | "Low"
    rationale: str


# ─── Intelligence Outputs ────────────────────────────────────────────────────

@dataclass
class ResearchSummary:
    ticker: str
    company_name: str
    sector: Sector
    generated_at: datetime
    analyst_rating: str             # "Strong Buy" | "Buy" | "Hold" | "Sell" | "Strong Sell"
    price_target: float
    upside_downside: float
    revenue_trend: str
    margin_trend: str
    strategic_developments: List[str]
    management_outlook: Sentiment
    key_risks: List[str]
    key_catalysts: List[str]
    summary: str
    earnings_summary: Optional[str] = None
    confidence_score: float = 0.0

@dataclass
class MacroRegimeReport:
    generated_at: datetime
    current_regime: MacroRegime
    regime_confidence: float        # 0–1
    inflation_trend: str
    interest_rate_outlook: str
    growth_outlook: str
    sector_implications: Dict[str, str]     # Sector -> "Overweight" | "Neutral" | "Underweight"
    risk_factors: List[str]
    investment_signals: Dict[str, float]    # Asset class -> signal score
    narrative: str

@dataclass
class RiskReport:
    portfolio_id: str
    generated_at: datetime
    overall_risk_level: RiskLevel
    portfolio_var_1d: float         # 1-day VaR at 95%
    portfolio_var_5d: float
    max_drawdown_ytd: float
    sharpe_ratio: float
    sortino_ratio: float
    beta_to_benchmark: float
    tracking_error: float
    # Concentrations
    sector_exposures: Dict[str, float]
    top_10_holdings_weight: float
    # Factor Exposures
    factor_exposures: Dict[str, float]      # Factor -> exposure
    # Liquidity
    liquidity_score: float
    days_to_liquidate_90pct: float
    # Stress Tests
    stress_test_results: Dict[str, float]   # Scenario -> impact %
    # Vulnerabilities
    vulnerabilities: List[str]
    recommendations: List[str]

@dataclass
class Alert:
    alert_id: str
    generated_at: datetime
    alert_type: AlertType
    severity: RiskLevel
    title: str
    description: str
    affected_tickers: List[str]
    affected_sectors: List[str]
    suggested_action: str
    auto_resolved: bool = False

@dataclass
class ScenarioResult:
    scenario_name: str
    description: str
    portfolio_impact_pct: float
    sector_impacts: Dict[str, float]
    holding_impacts: Dict[str, float]
    var_change: float
    recovery_estimate_months: int
    key_drivers: List[str]

@dataclass
class InvestorReport:
    report_id: str
    generated_at: datetime
    reporting_period: str
    portfolio_name: str
    portfolio_manager: str
    # Performance
    portfolio_return: float
    benchmark_return: float
    alpha: float
    # Sections
    portfolio_overview: str
    performance_attribution: str
    risk_commentary: str
    market_outlook: str
    recommended_actions: List[str]
    # Data Tables
    top_contributors: List[Dict]
    top_detractors: List[Dict]
    allocation_table: List[Dict]
    scenario_table: List[Dict]


# ─── Unified Dataset ─────────────────────────────────────────────────────────

@dataclass
class UnifiedDataset:
    """Master dataset produced by the Data Ingestion Agent."""
    ingested_at: datetime
    price_data: Dict[str, PriceData]            # ticker -> PriceData
    financial_metrics: Dict[str, FinancialMetrics]
    earnings_transcripts: Dict[str, EarningsTranscript]
    news_feed: List[NewsItem]
    macro_snapshot: MacroSnapshot
    macro_indicators: List[MacroIndicator]
    sentiment_scores: Dict[str, float]          # ticker -> aggregate sentiment -1 to +1
    data_quality_score: float
    missing_tickers: List[str]
