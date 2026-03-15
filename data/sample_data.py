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

import yfinance as yf

from models.data_models import (
    PriceData, FinancialMetrics, EarningsTranscript, NewsItem,
    MacroSnapshot, MacroIndicator, Holding, Portfolio, UnifiedDataset,
    Sector, Sentiment
)

random.seed(42)

# ─── Universe of Stocks ──────────────────────────────────────────────────────

STOCK_UNIVERSE = {
    # Technology
    "TCS.NS":      {"name": "Tata Consultancy Services",    "sector": Sector.TECHNOLOGY,       "beta": 0.85, "mcap": 1500},
    "INFY.NS":     {"name": "Infosys Limited",              "sector": Sector.TECHNOLOGY,       "beta": 0.90, "mcap": 700},
    "WIPRO.NS":    {"name": "Wipro Limited",                "sector": Sector.TECHNOLOGY,       "beta": 0.95, "mcap": 250},
    # Financials
    "HDFCBANK.NS": {"name": "HDFC Bank",                    "sector": Sector.FINANCIALS,       "beta": 1.05, "mcap": 1200},
    "ICICIBANK.NS":{"name": "ICICI Bank",                   "sector": Sector.FINANCIALS,       "beta": 1.20, "mcap": 800},
    "KOTAKBANK.NS":{"name": "Kotak Mahindra Bank",          "sector": Sector.FINANCIALS,       "beta": 1.10, "mcap": 500},
    "BAJFINANCE.NS":{"name": "Bajaj Finance",               "sector": Sector.FINANCIALS,       "beta": 1.30, "mcap": 450},
    # Energy/Consumer
    "RELIANCE.NS": {"name": "Reliance Industries",          "sector": Sector.ENERGY,           "beta": 1.15, "mcap": 2000},
    # Consumer Discretionary
    "MARUTI.NS":   {"name": "Maruti Suzuki India",          "sector": Sector.CONSUMER_DISC,    "beta": 0.80, "mcap": 400},
    # Communication
    "BHARTIARTL.NS":{"name": "Bharti Airtel",              "sector": Sector.COMMUNICATION,    "beta": 1.00, "mcap": 900},
    # Industrials
    "LT.NS":       {"name": "Larsen & Toubro",             "sector": Sector.INDUSTRIALS,      "beta": 1.05, "mcap": 500},
    "ADANIPORTS.NS":{"name": "Adani Ports and Special",    "sector": Sector.INDUSTRIALS,      "beta": 1.25, "mcap": 300},
    # Healthcare
    "SUNPHARMA.NS":{"name": "Sun Pharmaceutical",           "sector": Sector.HEALTHCARE,       "beta": 0.75, "mcap": 400},
    "DRREDDY.NS":  {"name": "Dr. Reddy's Laboratories",    "sector": Sector.HEALTHCARE,       "beta": 0.85, "mcap": 130},
    # Consumer Staples
    "HINDUNILVR.NS":{"name": "Hindustan Unilever",        "sector": Sector.CONSUMER_STAPLES, "beta": 0.60, "mcap": 600},
    "ITC.NS":      {"name": "ITC Limited",                 "sector": Sector.CONSUMER_STAPLES, "beta": 0.70, "mcap": 550},
    # Utilities
    "NTPC.NS":     {"name": "NTPC Limited",                "sector": Sector.UTILITIES,        "beta": 0.80, "mcap": 400},
    "POWERGRID.NS":{"name": "Power Grid Corporation",      "sector": Sector.UTILITIES,        "beta": 0.75, "mcap": 300},
    # Materials
    "ULTRACEMCO.NS":{"name": "UltraTech Cement",           "sector": Sector.MATERIALS,        "beta": 0.90, "mcap": 300},
    "GRASIM.NS":   {"name": "Grasim Industries",           "sector": Sector.MATERIALS,        "beta": 1.00, "mcap": 180},
}

def _rand_pct(base: float, variance: float) -> float:
    """Return base ± variance%."""
    return round(base + random.uniform(-variance, variance), 4)

def _rand_val(base: float, variance_pct: float = 0.1) -> float:
    """Return base ± variance_pct of base."""
    return round(base * (1 + random.uniform(-variance_pct, variance_pct)), 2)


# ─── Price Data ──────────────────────────────────────────────────────────────

def generate_price_data(as_of: datetime = None) -> Dict[str, PriceData]:
    if as_of is None:
        as_of = datetime.now()
    data = {}
    tickers = list(STOCK_UNIVERSE.keys())
    try:
        # Fetch data for all tickers
        hist = yf.download(tickers, period="2mo", interval="1d", group_by='ticker', threads=False)
        for ticker in tickers:
            if ticker in hist and not hist[ticker].empty:
                df = hist[ticker].dropna()
                if not df.empty:
                    latest = df.iloc[-1]
                    prev_day = df.iloc[-2] if len(df) > 1 else latest
                    close = latest['Close']
                    open_ = latest['Open']
                    high = latest['High']
                    low = latest['Low']
                    volume = latest['Volume']
                    returns_1d = (close - prev_day['Close']) / prev_day['Close'] if len(df) > 1 else 0.0
                    # Approximate other returns
                    if len(df) > 5:
                        returns_1w = (close - df.iloc[-6]['Close']) / df.iloc[-6]['Close']
                    else:
                        returns_1w = returns_1d * 5
                    if len(df) > 20:
                        returns_1m = (close - df.iloc[-21]['Close']) / df.iloc[-21]['Close']
                    else:
                        returns_1m = returns_1d * 20
                    returns_3m = returns_1m * 3  # approximate
                    returns_ytd = returns_1m * 12  # approximate
                    volatility_30d = df['Close'].pct_change().std() * math.sqrt(252) if len(df) > 1 else 0.2
                    data[ticker] = PriceData(
                        ticker=ticker,
                        date=as_of,
                        open=open_,
                        high=high,
                        low=low,
                        close=close,
                        volume=int(volume),
                        adjusted_close=close,
                        returns_1d=round(returns_1d, 4),
                        returns_1w=round(returns_1w, 4),
                        returns_1m=round(returns_1m, 4),
                        returns_3m=round(returns_3m, 4),
                        returns_ytd=round(returns_ytd, 4),
                        volatility_30d=round(volatility_30d, 4),
                    )
                else:
                    # Fallback to synthetic
                    data[ticker] = _generate_synthetic_price(ticker, as_of)
            else:
                # Fallback
                data[ticker] = _generate_synthetic_price(ticker, as_of)
    except Exception as e:
        print(f"Error fetching data from Yahoo Finance: {e}. Using synthetic data.")
        for ticker in tickers:
            data[ticker] = _generate_synthetic_price(ticker, as_of)
    return data

def _generate_synthetic_price(ticker: str, as_of: datetime) -> PriceData:
    base = 1000  # dummy base price
    close = _rand_val(base, 0.05)
    return PriceData(
        ticker=ticker,
        date=as_of,
        open=_rand_val(base, 0.04),
        high=close * random.uniform(1.005, 1.025),
        low=close * random.uniform(0.975, 0.995),
        close=close,
        volume=int(random.uniform(1_000_000, 10_000_000)),
        adjusted_close=close,
        returns_1d=_rand_pct(0.0, 0.015),
        returns_1w=_rand_pct(0.008, 0.03),
        returns_1m=_rand_pct(0.025, 0.06),
        returns_3m=_rand_pct(0.06, 0.12),
        returns_ytd=_rand_pct(0.08, 0.15),
        volatility_30d=random.uniform(0.18, 0.45),
    )


# ─── Financial Metrics ───────────────────────────────────────────────────────

FINANCIAL_TEMPLATES = {
    "TCS.NS":       dict(rev=25,  rev_g=0.08, gm=0.35, ebitdam=0.25, pe=25, eveb=18, pb=8),
    "INFY.NS":      dict(rev=18,  rev_g=0.06, gm=0.32, ebitdam=0.28, pe=22, eveb=16, pb=7),
    "WIPRO.NS":     dict(rev=11,  rev_g=0.05, gm=0.30, ebitdam=0.22, pe=20, eveb=14, pb=4),
    "HDFCBANK.NS":  dict(rev=20,  rev_g=0.10, gm=0.50, ebitdam=0.35, pe=18, eveb=12, pb=3),
    "ICICIBANK.NS": dict(rev=15,  rev_g=0.12, gm=0.45, ebitdam=0.30, pe=16, eveb=10, pb=2),
    "KOTAKBANK.NS": dict(rev=8,   rev_g=0.15, gm=0.55, ebitdam=0.40, pe=28, eveb=20, pb=4),
    "BAJFINANCE.NS":dict(rev=5,   rev_g=0.20, gm=0.60, ebitdam=0.45, pe=35, eveb=25, pb=6),
    "RELIANCE.NS":  dict(rev=100, rev_g=0.10, gm=0.25, ebitdam=0.15, pe=22, eveb=15, pb=2),
    "MARUTI.NS":    dict(rev=15,  rev_g=0.05, gm=0.20, ebitdam=0.12, pe=24, eveb=18, pb=4),
    "BHARTIARTL.NS":dict(rev=18,  rev_g=0.08, gm=0.40, ebitdam=0.30, pe=65, eveb=50, pb=8),
    "LT.NS":        dict(rev=22,  rev_g=0.06, gm=0.15, ebitdam=0.10, pe=30, eveb=20, pb=3),
    "ADANIPORTS.NS":dict(rev=6,   rev_g=0.12, gm=0.35, ebitdam=0.25, pe=40, eveb=30, pb=5),
    "SUNPHARMA.NS": dict(rev=6,   rev_g=0.07, gm=0.25, ebitdam=0.20, pe=35, eveb=25, pb=3),
    "DRREDDY.NS":   dict(rev=3,   rev_g=0.05, gm=0.30, ebitdam=0.25, pe=18, eveb=12, pb=3),
    "HINDUNILVR.NS":dict(rev=7,   rev_g=0.04, gm=0.50, ebitdam=0.35, pe=50, eveb=40, pb=12),
    "ITC.NS":       dict(rev=8,   rev_g=0.06, gm=0.35, ebitdam=0.25, pe=25, eveb=18, pb=6),
    "NTPC.NS":      dict(rev=18,  rev_g=0.03, gm=0.20, ebitdam=0.15, pe=18, eveb=12, pb=2),
    "POWERGRID.NS": dict(rev=5,   rev_g=0.04, gm=0.25, ebitdam=0.20, pe=20, eveb=15, pb=3),
    "ULTRACEMCO.NS":dict(rev=7,   rev_g=0.08, gm=0.20, ebitdam=0.15, pe=30, eveb=20, pb=4),
    "GRASIM.NS":    dict(rev=15,  rev_g=0.05, gm=0.25, ebitdam=0.18, pe=25, eveb=18, pb=2),
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
    "RELIANCE.NS": 0.10, "TCS.NS": 0.08, "HDFCBANK.NS": 0.07, "INFY.NS": 0.06, "ICICIBANK.NS": 0.05,
    "BHARTIARTL.NS": 0.05, "ITC.NS": 0.04, "HINDUNILVR.NS": 0.04, "KOTAKBANK.NS": 0.04, "BAJFINANCE.NS": 0.04,
    "MARUTI.NS": 0.03, "LT.NS": 0.03, "SUNPHARMA.NS": 0.03, "NTPC.NS": 0.03, "POWERGRID.NS": 0.03,
    "WIPRO.NS": 0.03, "ADANIPORTS.NS": 0.03, "DRREDDY.NS": 0.02, "ULTRACEMCO.NS": 0.02, "GRASIM.NS": 0.02,
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
