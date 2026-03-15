"""
Sample Data Generator
=====================
Generates realistic simulated financial data for demonstration purposes.
"""

import random
import math
from datetime import datetime, timedelta
from typing import Dict, List
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.data_models import (
    PriceData, FinancialMetrics, EarningsTranscript, NewsItem,
    MacroSnapshot, MacroIndicator, Holding, Portfolio, UnifiedDataset,
    Sector, Sentiment
)

random.seed(42)

# ─── Universe of Stocks ──────────────────────────────────────────────────────

STOCK_UNIVERSE = {
    # Technology
    "AAPL":  {"name": "Apple Inc.",                  "sector": Sector.TECHNOLOGY,       "beta": 1.15, "mcap": 2800},
    "MSFT":  {"name": "Microsoft Corporation",       "sector": Sector.TECHNOLOGY,       "beta": 0.95, "mcap": 2600},
    "NVDA":  {"name": "NVIDIA Corporation",          "sector": Sector.TECHNOLOGY,       "beta": 1.85, "mcap": 1800},
    "GOOGL": {"name": "Alphabet Inc.",               "sector": Sector.COMMUNICATION,    "beta": 1.05, "mcap": 1700},
    "META":  {"name": "Meta Platforms Inc.",         "sector": Sector.COMMUNICATION,    "beta": 1.25, "mcap": 1100},
    # Healthcare
    "JNJ":   {"name": "Johnson & Johnson",           "sector": Sector.HEALTHCARE,       "beta": 0.55, "mcap": 410},
    "UNH":   {"name": "UnitedHealth Group",          "sector": Sector.HEALTHCARE,       "beta": 0.65, "mcap": 490},
    "LLY":   {"name": "Eli Lilly & Co.",             "sector": Sector.HEALTHCARE,       "beta": 0.45, "mcap": 720},
    # Financials
    "JPM":   {"name": "JPMorgan Chase",              "sector": Sector.FINANCIALS,       "beta": 1.10, "mcap": 540},
    "BAC":   {"name": "Bank of America",             "sector": Sector.FINANCIALS,       "beta": 1.35, "mcap": 290},
    "GS":    {"name": "Goldman Sachs",               "sector": Sector.FINANCIALS,       "beta": 1.45, "mcap": 140},
    # Energy
    "XOM":   {"name": "ExxonMobil Corporation",      "sector": Sector.ENERGY,           "beta": 0.85, "mcap": 450},
    "CVX":   {"name": "Chevron Corporation",         "sector": Sector.ENERGY,           "beta": 0.80, "mcap": 290},
    # Consumer Discretionary
    "AMZN":  {"name": "Amazon.com Inc.",             "sector": Sector.CONSUMER_DISC,    "beta": 1.20, "mcap": 1900},
    "TSLA":  {"name": "Tesla Inc.",                  "sector": Sector.CONSUMER_DISC,    "beta": 1.95, "mcap": 800},
    # Industrials
    "CAT":   {"name": "Caterpillar Inc.",            "sector": Sector.INDUSTRIALS,      "beta": 1.05, "mcap": 175},
    "HON":   {"name": "Honeywell International",     "sector": Sector.INDUSTRIALS,      "beta": 0.90, "mcap": 145},
    # Consumer Staples
    "PG":    {"name": "Procter & Gamble",            "sector": Sector.CONSUMER_STAPLES, "beta": 0.55, "mcap": 365},
    "KO":    {"name": "Coca-Cola Company",           "sector": Sector.CONSUMER_STAPLES, "beta": 0.60, "mcap": 260},
    # Materials
    "LIN":   {"name": "Linde PLC",                   "sector": Sector.MATERIALS,        "beta": 0.85, "mcap": 220},
}

def _rand_pct(base: float, variance: float) -> float:
    """Return base ± variance%."""
    return round(base + random.uniform(-variance, variance), 4)

def _rand_val(base: float, variance_pct: float = 0.1) -> float:
    """Return base ± variance_pct of base."""
    return round(base * (1 + random.uniform(-variance_pct, variance_pct)), 2)


# ─── Price Data ──────────────────────────────────────────────────────────────

BASE_PRICES = {
    "AAPL": 195, "MSFT": 415, "NVDA": 875, "GOOGL": 170, "META": 510,
    "JNJ": 158, "UNH": 555, "LLY": 810, "JPM": 215, "BAC": 42,
    "GS": 490, "XOM": 112, "CVX": 158, "AMZN": 185, "TSLA": 250,
    "CAT": 365, "HON": 215, "PG": 168, "KO": 63, "LIN": 455,
}

def generate_price_data(as_of: datetime = None) -> Dict[str, PriceData]:
    if as_of is None:
        as_of = datetime.now()
    data = {}
    for ticker, base in BASE_PRICES.items():
        close = _rand_val(base, 0.05)
        data[ticker] = PriceData(
            ticker=ticker,
            date=as_of,
            open=_rand_val(base, 0.04),
            high=close * random.uniform(1.005, 1.025),
            low=close * random.uniform(0.975, 0.995),
            close=close,
            volume=int(random.uniform(5_000_000, 80_000_000)),
            adjusted_close=close,
            returns_1d=_rand_pct(0.0, 0.015),
            returns_1w=_rand_pct(0.008, 0.03),
            returns_1m=_rand_pct(0.025, 0.06),
            returns_3m=_rand_pct(0.06, 0.12),
            returns_ytd=_rand_pct(0.08, 0.15),
            volatility_30d=random.uniform(0.18, 0.45),
        )
    return data


# ─── Financial Metrics ───────────────────────────────────────────────────────

FINANCIAL_TEMPLATES = {
    "AAPL":  dict(rev=383, rev_g=0.06, gm=0.445, ebitdam=0.33, pe=29, eveb=22, pb=45),
    "MSFT":  dict(rev=245, rev_g=0.16, gm=0.695, ebitdam=0.50, pe=34, eveb=26, pb=12),
    "NVDA":  dict(rev=80,  rev_g=1.22, gm=0.740, ebitdam=0.56, pe=58, eveb=42, pb=38),
    "GOOGL": dict(rev=307, rev_g=0.09, gm=0.560, ebitdam=0.28, pe=23, eveb=16, pb=6),
    "META":  dict(rev=135, rev_g=0.22, gm=0.810, ebitdam=0.42, pe=25, eveb=18, pb=8),
    "JNJ":   dict(rev=97,  rev_g=0.04, gm=0.685, ebitdam=0.30, pe=14, eveb=12, pb=4),
    "UNH":   dict(rev=372, rev_g=0.12, gm=0.235, ebitdam=0.09, pe=20, eveb=13, pb=5),
    "LLY":   dict(rev=34,  rev_g=0.28, gm=0.790, ebitdam=0.38, pe=55, eveb=45, pb=58),
    "JPM":   dict(rev=162, rev_g=0.10, gm=0.520, ebitdam=0.38, pe=12, eveb=9,  pb=2),
    "BAC":   dict(rev=98,  rev_g=0.06, gm=0.480, ebitdam=0.32, pe=11, eveb=8,  pb=1),
    "GS":    dict(rev=46,  rev_g=0.08, gm=0.620, ebitdam=0.29, pe=13, eveb=10, pb=1),
    "XOM":   dict(rev=398, rev_g=-0.04, gm=0.210, ebitdam=0.14, pe=13, eveb=8, pb=2),
    "CVX":   dict(rev=201, rev_g=-0.06, gm=0.185, ebitdam=0.13, pe=14, eveb=9, pb=2),
    "AMZN":  dict(rev=575, rev_g=0.12, gm=0.480, ebitdam=0.16, pe=42, eveb=24, pb=8),
    "TSLA":  dict(rev=97,  rev_g=0.02, gm=0.175, ebitdam=0.11, pe=65, eveb=48, pb=12),
    "CAT":   dict(rev=64,  rev_g=0.03, gm=0.385, ebitdam=0.18, pe=16, eveb=10, pb=5),
    "HON":   dict(rev=36,  rev_g=0.05, gm=0.355, ebitdam=0.21, pe=20, eveb=14, pb=7),
    "PG":    dict(rev=84,  rev_g=0.04, gm=0.530, ebitdam=0.25, pe=24, eveb=18, pb=7),
    "KO":    dict(rev=46,  rev_g=0.03, gm=0.600, ebitdam=0.31, pe=21, eveb=17, pb=10),
    "LIN":   dict(rev=33,  rev_g=0.07, gm=0.545, ebitdam=0.40, pe=28, eveb=20, pb=4),
}

def generate_financial_metrics(price_data: Dict[str, PriceData]) -> Dict[str, FinancialMetrics]:
    metrics = {}
    for ticker, info in STOCK_UNIVERSE.items():
        t = FINANCIAL_TEMPLATES.get(ticker, dict(rev=50, rev_g=0.05, gm=0.40, ebitdam=0.20, pe=20, eveb=15, pb=3))
        rev = _rand_val(t["rev"] * 1e9, 0.03)
        gm  = _rand_pct(t["gm"], 0.02)
        em  = _rand_pct(t["ebitdam"], 0.02)
        ni  = rev * em * 0.72
        price = price_data[ticker].close if ticker in price_data else 100
        shares = (info["mcap"] * 1e9) / price
        metrics[ticker] = FinancialMetrics(
            ticker=ticker,
            company_name=info["name"],
            sector=info["sector"],
            fiscal_period="FY2024",
            revenue=rev,
            revenue_growth_yoy=_rand_pct(t["rev_g"], 0.03),
            gross_margin=gm,
            ebitda_margin=em,
            net_income=ni,
            eps=round(ni / shares, 2),
            eps_growth_yoy=_rand_pct(t["rev_g"] * 1.1, 0.04),
            total_assets=rev * 1.5,
            total_debt=rev * 0.35,
            cash_and_equivalents=rev * 0.12,
            net_debt=rev * 0.23,
            book_value_per_share=round((rev * 0.45) / shares, 2),
            operating_cash_flow=rev * em * 0.95,
            free_cash_flow=rev * em * 0.78,
            capex=rev * 0.06,
            market_cap=info["mcap"] * 1e9,
            pe_ratio=_rand_pct(t["pe"], 3),
            ev_ebitda=_rand_pct(t["eveb"], 2),
            price_to_book=_rand_pct(t["pb"], 0.5),
            price_to_sales=round(info["mcap"] * 1e9 / rev, 2),
            dividend_yield=_rand_pct(0.015, 0.008) if info["sector"] not in [Sector.TECHNOLOGY] else 0.0,
            roe=round(ni / (rev * 0.45), 4),
            roa=round(ni / (rev * 1.5), 4),
            roic=round(ni / (rev * 0.55), 4),
            debt_to_equity=round((rev * 0.35) / (rev * 0.45), 2),
            current_ratio=_rand_pct(1.4, 0.3),
            interest_coverage=_rand_pct(12.0, 4.0),
        )
    return metrics


# ─── Earnings Transcripts ────────────────────────────────────────────────────

MGMT_TOPICS = {
    Sector.TECHNOLOGY:       ["AI investment", "cloud growth", "margin expansion", "buybacks"],
    Sector.HEALTHCARE:       ["pipeline progress", "FDA approvals", "cost savings", "generics"],
    Sector.FINANCIALS:       ["net interest income", "loan growth", "credit quality", "capital"],
    Sector.ENERGY:           ["production volumes", "capital discipline", "dividend sustainability"],
    Sector.CONSUMER_DISC:    ["consumer demand", "inventory levels", "pricing power", "margin"],
    Sector.CONSUMER_STAPLES: ["pricing", "volume trends", "private label competition", "input costs"],
    Sector.INDUSTRIALS:      ["order backlog", "pricing power", "supply chain", "margin"],
    Sector.COMMUNICATION:    ["user growth", "advertising trends", "content investment", "regulation"],
    Sector.MATERIALS:        ["raw material costs", "demand outlook", "capacity utilization"],
    Sector.UTILITIES:        ["regulatory environment", "capex plan", "renewable transition"],
    Sector.REAL_ESTATE:      ["occupancy rates", "rent growth", "debt refinancing"],
}

def generate_earnings_transcripts(metrics: Dict[str, FinancialMetrics]) -> Dict[str, EarningsTranscript]:
    transcripts = {}
    sentiments = [Sentiment.VERY_POSITIVE, Sentiment.POSITIVE, Sentiment.NEUTRAL,
                  Sentiment.NEGATIVE, Sentiment.VERY_NEGATIVE]
    weights     = [0.20, 0.40, 0.25, 0.12, 0.03]
    guidance = ["raised", "maintained", "maintained", "lowered", "withdrawn"]
    for ticker, m in metrics.items():
        tone = random.choices(sentiments, weights=weights)[0]
        rev_beat = round(random.uniform(-0.03, 0.06), 4)
        eps_beat = round(random.uniform(-0.04, 0.08), 4)
        topics = MGMT_TOPICS.get(m.sector, ["growth outlook", "cost management"])
        transcripts[ticker] = EarningsTranscript(
            ticker=ticker,
            company_name=m.company_name,
            date=datetime.now() - timedelta(days=random.randint(5, 45)),
            quarter="Q4 2024",
            management_tone=tone,
            key_topics=random.sample(topics, k=min(3, len(topics))),
            guidance_revision=random.choices(guidance, weights=[0.25, 0.40, 0.25, 0.08, 0.02])[0],
            revenue_beat_miss=rev_beat,
            eps_beat_miss=eps_beat,
            full_text_summary=(
                f"{m.company_name} reported Q4 2024 results with revenue "
                f"{'beating' if rev_beat > 0 else 'missing'} consensus by "
                f"{abs(rev_beat)*100:.1f}%. EPS came in {abs(eps_beat)*100:.1f}% "
                f"{'above' if eps_beat > 0 else 'below'} expectations. "
                f"Management struck a {tone.value.lower()} tone regarding the outlook."
            ),
            key_quotes=[
                f"\"We are {'confident' if tone in [Sentiment.POSITIVE, Sentiment.VERY_POSITIVE] else 'cautious'} "
                f"about our trajectory in {random.choice(topics)}.\"",
                f"\"{random.choice(['Demand signals remain robust', 'We see near-term headwinds', 'Our pipeline is strong', 'Margins are under pressure'])}.\"",
            ],
        )
    return transcripts


# ─── Macro Snapshot ──────────────────────────────────────────────────────────

def generate_macro_snapshot() -> MacroSnapshot:
    return MacroSnapshot(
        timestamp=datetime.now(),
        cpi_yoy=_rand_pct(3.2, 0.3),
        pce_yoy=_rand_pct(2.8, 0.3),
        fed_funds_rate=_rand_pct(5.25, 0.25),
        us_10y_yield=_rand_pct(4.45, 0.25),
        us_2y_yield=_rand_pct(4.85, 0.20),
        yield_curve_spread=_rand_pct(-0.40, 0.15),
        gdp_growth_qoq=_rand_pct(0.0075, 0.003),
        gdp_growth_yoy=_rand_pct(0.028, 0.005),
        unemployment_rate=_rand_pct(0.038, 0.003),
        oil_price_wti=_rand_val(78.0, 0.08),
        gold_price=_rand_val(2050.0, 0.05),
        copper_price=_rand_val(3.85, 0.06),
        hy_spread=_rand_pct(350, 40),
        ig_spread=_rand_pct(110, 15),
        dxy_index=_rand_val(104.5, 0.03),
        eurusd=_rand_val(1.082, 0.02),
        usdjpy=_rand_val(149.5, 0.02),
        vix=_rand_val(18.5, 0.15),
        consumer_confidence=_rand_val(98.5, 0.05),
        ism_manufacturing=_rand_val(48.5, 0.04),
        ism_services=_rand_val(53.2, 0.03),
    )


# ─── Macro Indicators ────────────────────────────────────────────────────────

def generate_macro_indicators() -> List[MacroIndicator]:
    indicators = []
    specs = [
        ("CPI (YoY)", 3.2, 3.4, "monthly", "BLS"),
        ("Core PCE (YoY)", 2.8, 2.9, "monthly", "BEA"),
        ("Fed Funds Rate", 5.25, 5.50, "periodic", "Federal Reserve"),
        ("10Y Treasury Yield", 4.45, 4.60, "daily", "US Treasury"),
        ("GDP Growth (QoQ Ann.)", 2.8, 3.1, "quarterly", "BEA"),
        ("Unemployment Rate", 3.8, 3.9, "monthly", "BLS"),
        ("ISM Manufacturing", 48.5, 47.9, "monthly", "ISM"),
        ("ISM Services", 53.2, 52.7, "monthly", "ISM"),
        ("WTI Oil Price", 78.0, 74.5, "daily", "EIA"),
        ("VIX", 18.5, 16.2, "daily", "CBOE"),
    ]
    for name, val, prior, freq, src in specs:
        v = _rand_val(val, 0.02)
        p = _rand_val(prior, 0.02)
        indicators.append(MacroIndicator(
            indicator_name=name, date=datetime.now(),
            value=v, prior_value=p,
            change=round(v - p, 4),
            change_pct=round((v - p) / p * 100, 4) if p != 0 else 0.0,
            frequency=freq, source=src,
        ))
    return indicators


# ─── News Feed ───────────────────────────────────────────────────────────────

NEWS_TEMPLATES = [
    ("Fed signals patience on rate cuts amid sticky inflation", ["JPM","BAC","GS"], [Sector.FINANCIALS], Sentiment.NEGATIVE, 0.85),
    ("NVIDIA reports record data-center revenue on AI demand surge", ["NVDA"], [Sector.TECHNOLOGY], Sentiment.VERY_POSITIVE, 0.95),
    ("Oil prices slip as OPEC+ considers production increase", ["XOM","CVX"], [Sector.ENERGY], Sentiment.NEGATIVE, 0.80),
    ("Apple launches new AI features across product lineup", ["AAPL"], [Sector.TECHNOLOGY], Sentiment.POSITIVE, 0.88),
    ("Healthcare sector rallies on FDA approval news", ["LLY","JNJ"], [Sector.HEALTHCARE], Sentiment.POSITIVE, 0.82),
    ("Amazon Web Services growth accelerates in Q4", ["AMZN"], [Sector.CONSUMER_DISC], Sentiment.VERY_POSITIVE, 0.90),
    ("Tesla misses delivery targets, shares under pressure", ["TSLA"], [Sector.CONSUMER_DISC], Sentiment.NEGATIVE, 0.87),
    ("JPMorgan raises dividend, announces buyback program", ["JPM"], [Sector.FINANCIALS], Sentiment.POSITIVE, 0.84),
    ("Microsoft Azure revenue growth beats estimates", ["MSFT"], [Sector.TECHNOLOGY], Sentiment.POSITIVE, 0.91),
    ("Macro headwinds weigh on industrial sector outlook", ["CAT","HON"], [Sector.INDUSTRIALS], Sentiment.NEGATIVE, 0.75),
    ("Consumer staples resilient despite margin pressure", ["PG","KO"], [Sector.CONSUMER_STAPLES], Sentiment.NEUTRAL, 0.70),
    ("Alphabet beats on search revenue, cloud growth strong", ["GOOGL"], [Sector.COMMUNICATION], Sentiment.POSITIVE, 0.89),
]

def generate_news_feed() -> List[NewsItem]:
    news = []
    for headline, tickers, sectors, sentiment, relevance in NEWS_TEMPLATES:
        summary_map = {
            Sentiment.VERY_POSITIVE: "Strong positive developments signal potential upside.",
            Sentiment.POSITIVE: "Broadly positive implications for the mentioned securities.",
            Sentiment.NEUTRAL: "Mixed signals; impact on portfolio likely muted.",
            Sentiment.NEGATIVE: "Headwinds identified; may pressure near-term performance.",
            Sentiment.VERY_NEGATIVE: "Significant negative catalyst; immediate review warranted.",
        }
        news.append(NewsItem(
            headline=headline,
            source=random.choice(["Bloomberg", "Reuters", "WSJ", "FT", "CNBC"]),
            published_at=datetime.now() - timedelta(hours=random.randint(1, 48)),
            tickers_mentioned=tickers,
            sectors_mentioned=[s.value for s in sectors],
            sentiment=sentiment,
            relevance_score=_rand_val(relevance, 0.05),
            summary=summary_map[sentiment],
        ))
    return news


# ─── Portfolio ───────────────────────────────────────────────────────────────

PORTFOLIO_WEIGHTS = {
    "AAPL": 0.085, "MSFT": 0.080, "NVDA": 0.070, "GOOGL": 0.065, "META": 0.050,
    "JNJ":  0.045, "UNH":  0.040, "LLY":  0.055, "JPM":   0.055, "BAC":  0.025,
    "GS":   0.020, "XOM":  0.035, "CVX":  0.025, "AMZN":  0.075, "TSLA": 0.030,
    "CAT":  0.025, "HON":  0.020, "PG":   0.030, "KO":    0.020, "LIN":  0.025,
}

def generate_portfolio(price_data: Dict[str, PriceData], nav: float = 500_000_000) -> Portfolio:
    holdings = []
    cash_weight = 0.045
    equity_nav = nav * (1 - cash_weight)

    for ticker, weight in PORTFOLIO_WEIGHTS.items():
        info = STOCK_UNIVERSE[ticker]
        price = price_data[ticker].close
        mv = equity_nav * weight
        shares = int(mv / price)
        cost = price * random.uniform(0.75, 1.05)
        pnl = (price - cost) * shares
        holdings.append(Holding(
            ticker=ticker,
            company_name=info["name"],
            sector=info["sector"],
            shares=shares,
            cost_basis=round(cost, 2),
            current_price=round(price, 2),
            market_value=round(mv, 0),
            weight=weight,
            unrealized_pnl=round(pnl, 0),
            unrealized_pnl_pct=round((price - cost) / cost, 4),
            beta=info["beta"],
            liquidity_score=random.uniform(0.72, 0.98),
        ))

    return Portfolio(
        portfolio_id="AMIS-ALPHA-001",
        portfolio_name="AMIS Alpha Equity Fund",
        portfolio_manager="Suchit Dubey",
        as_of_date=datetime.now(),
        total_nav=nav,
        holdings=holdings,
        benchmark="S&P 500",
        cash_weight=cash_weight,
        inception_date=datetime(2020, 1, 1),
    )


# ─── Full Unified Dataset ────────────────────────────────────────────────────

def generate_unified_dataset() -> UnifiedDataset:
    price_data   = generate_price_data()
    fin_metrics  = generate_financial_metrics(price_data)
    earnings     = generate_earnings_transcripts(fin_metrics)
    news         = generate_news_feed()
    macro        = generate_macro_snapshot()
    macro_inds   = generate_macro_indicators()
    sentiment    = {t: round(random.uniform(-0.5, 0.8), 3) for t in STOCK_UNIVERSE}

    return UnifiedDataset(
        ingested_at=datetime.now(),
        price_data=price_data,
        financial_metrics=fin_metrics,
        earnings_transcripts=earnings,
        news_feed=news,
        macro_snapshot=macro,
        macro_indicators=macro_inds,
        sentiment_scores=sentiment,
        data_quality_score=round(random.uniform(0.92, 0.99), 4),
        missing_tickers=[],
    )

def generate_sample_portfolio(price_data: Dict[str, PriceData] = None) -> Portfolio:
    if price_data is None:
        price_data = generate_price_data()
    return generate_portfolio(price_data)
