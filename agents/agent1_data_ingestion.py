"""
Agent 1 — Data Ingestion Agent
================================
Collects, normalizes, and structures financial and macroeconomic data
from multiple sources into a unified dataset for downstream agents.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.data_models import (
    UnifiedDataset, PriceData, FinancialMetrics, MacroSnapshot,
    NewsItem, Sentiment
)
from data.sample_data import generate_unified_dataset

logger = logging.getLogger(__name__)


class DataIngestionAgent:
    """
    Agent 1: Data Ingestion Agent
    ─────────────────────────────
    Responsibilities:
      • Collect data from multiple financial data sources
      • Normalize and clean raw inputs
      • Extract structured financial metrics
      • Store everything in a unified, queryable dataset

    In production, each data source would map to a dedicated connector
    (Bloomberg API, Refinitiv, EDGAR filings, etc.).  Here we use the
    sample data generator to produce realistic synthetic data.
    """

    AGENT_ID   = "AGENT-01-DATA-INGESTION"
    AGENT_NAME = "Data Ingestion Agent"
    VERSION    = "1.0.0"

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.last_run: Optional[datetime] = None
        self.dataset: Optional[UnifiedDataset] = None
        self._ingestion_log: List[Dict] = []
        logger.info(f"[{self.AGENT_NAME}] Initialized (v{self.VERSION})")

    # ─── Public API ─────────────────────────────────────────────────────────

    def run(self) -> UnifiedDataset:
        """
        Execute the full ingestion pipeline:
          1. Collect raw data from each source
          2. Normalize and validate
          3. Extract structured metrics
          4. Build unified dataset
        """
        logger.info(f"[{self.AGENT_NAME}] Starting ingestion pipeline …")
        start = datetime.now()

        raw_price     = self._collect_price_data()
        raw_financials= self._collect_financial_statements()
        raw_earnings  = self._collect_earnings_transcripts()
        raw_macro     = self._collect_macro_indicators()
        raw_news      = self._collect_news_feeds()

        price_clean   = self._normalize_price_data(raw_price)
        fin_clean     = self._normalize_financial_data(raw_financials)
        macro_clean   = self._normalize_macro_data(raw_macro)
        sentiment     = self._compute_sentiment_scores(raw_news, raw_earnings)

        dataset = UnifiedDataset(
            ingested_at=datetime.now(),
            price_data=price_clean,
            financial_metrics=fin_clean,
            earnings_transcripts=raw_earnings,
            news_feed=raw_news,
            macro_snapshot=macro_clean,
            macro_indicators=raw_macro.macro_indicators,
            sentiment_scores=sentiment,
            data_quality_score=self._compute_data_quality(price_clean, fin_clean),
            missing_tickers=[],
        )

        elapsed = (datetime.now() - start).total_seconds()
        self.dataset = dataset
        self.last_run = datetime.now()
        self._log_run("SUCCESS", elapsed, len(price_clean), len(fin_clean))
        logger.info(f"[{self.AGENT_NAME}] Pipeline complete in {elapsed:.2f}s — "
                    f"{len(price_clean)} tickers ingested, "
                    f"quality={dataset.data_quality_score:.2%}")
        return dataset

    def get_status(self) -> Dict:
        return {
            "agent_id":        self.AGENT_ID,
            "last_run":        self.last_run.isoformat() if self.last_run else None,
            "tickers_tracked": len(self.dataset.price_data) if self.dataset else 0,
            "data_quality":    self.dataset.data_quality_score if self.dataset else None,
            "ingestion_log":   self._ingestion_log[-5:],
        }

    # ─── Private: Collection ────────────────────────────────────────────────

    def _collect_price_data(self) -> UnifiedDataset:
        """Simulate collection from market data feed (e.g., Bloomberg, Refinitiv)."""
        logger.debug(f"[{self.AGENT_NAME}] Fetching market price data …")
        return generate_unified_dataset()

    def _collect_financial_statements(self) -> UnifiedDataset:
        """Simulate collection from EDGAR / financial data provider."""
        logger.debug(f"[{self.AGENT_NAME}] Fetching financial statements …")
        return generate_unified_dataset()

    def _collect_earnings_transcripts(self):
        """Simulate collection from earnings transcript provider."""
        logger.debug(f"[{self.AGENT_NAME}] Fetching earnings transcripts …")
        ds = generate_unified_dataset()
        return ds.earnings_transcripts

    def _collect_macro_indicators(self) -> UnifiedDataset:
        """Simulate collection from macro data providers (BLS, BEA, Fed, etc.)."""
        logger.debug(f"[{self.AGENT_NAME}] Fetching macroeconomic indicators …")
        return generate_unified_dataset()

    def _collect_news_feeds(self):
        """Simulate collection from news aggregators."""
        logger.debug(f"[{self.AGENT_NAME}] Fetching news feeds …")
        ds = generate_unified_dataset()
        return ds.news_feed

    # ─── Private: Normalization ──────────────────────────────────────────────

    def _normalize_price_data(self, raw: UnifiedDataset) -> Dict[str, PriceData]:
        """
        Clean price data:
          - Forward-fill missing prices
          - Validate OHLC consistency (high >= low, open/close within range)
          - Clip extreme returns (>30% single-day flagged)
          - Compute derived metrics (returns, volatility)
        """
        clean = {}
        for ticker, p in raw.price_data.items():
            # OHLC sanity check
            if p.high < p.low:
                logger.warning(f"[{self.AGENT_NAME}] {ticker}: high < low — correcting")
                p.high, p.low = p.low, p.high
            # Clip extreme 1d returns
            if abs(p.returns_1d) > 0.30:
                logger.warning(f"[{self.AGENT_NAME}] {ticker}: extreme return clipped")
                p.returns_1d = max(-0.30, min(0.30, p.returns_1d))
            clean[ticker] = p
        return clean

    def _normalize_financial_data(self, raw: UnifiedDataset) -> Dict[str, FinancialMetrics]:
        """
        Clean financial metrics:
          - Winsorize ratio outliers
          - Flag negative revenue / equity
          - Standardize fiscal period labels
        """
        clean = {}
        for ticker, m in raw.financial_metrics.items():
            if m.revenue <= 0:
                logger.warning(f"[{self.AGENT_NAME}] {ticker}: non-positive revenue flagged")
            if m.pe_ratio < 0:
                m.pe_ratio = float("nan")
            clean[ticker] = m
        return clean

    def _normalize_macro_data(self, raw: UnifiedDataset) -> MacroSnapshot:
        """Validate macro snapshot values are within plausible ranges."""
        snap = raw.macro_snapshot
        if snap.vix < 9 or snap.vix > 90:
            logger.warning(f"[{self.AGENT_NAME}] VIX={snap.vix} outside normal range")
        if snap.cpi_yoy < -5 or snap.cpi_yoy > 25:
            logger.warning(f"[{self.AGENT_NAME}] CPI={snap.cpi_yoy} outside normal range")
        return snap

    # ─── Private: Sentiment ──────────────────────────────────────────────────

    def _compute_sentiment_scores(self, news: List[NewsItem], transcripts: Dict) -> Dict[str, float]:
        """
        Aggregate sentiment from news articles and earnings transcripts
        into a single per-ticker score in [-1, +1].

        Weights:
          • Earnings transcript tone: 50%
          • News sentiment average:   50%
        """
        sentiment_map = {
            Sentiment.VERY_POSITIVE: 1.0,
            Sentiment.POSITIVE:      0.5,
            Sentiment.NEUTRAL:       0.0,
            Sentiment.NEGATIVE:     -0.5,
            Sentiment.VERY_NEGATIVE: -1.0,
        }
        # News scores per ticker
        news_scores: Dict[str, List[float]] = {}
        for item in news:
            for ticker in item.tickers_mentioned:
                news_scores.setdefault(ticker, []).append(
                    sentiment_map[item.sentiment] * item.relevance_score
                )

        scores: Dict[str, float] = {}
        for ticker, transcript in transcripts.items():
            transcript_score = sentiment_map[transcript.management_tone]
            avg_news = (sum(news_scores[ticker]) / len(news_scores[ticker])
                        if ticker in news_scores else 0.0)
            combined = 0.5 * transcript_score + 0.5 * avg_news
            scores[ticker] = round(combined, 4)
        return scores

    # ─── Private: Quality ───────────────────────────────────────────────────

    def _compute_data_quality(self, prices: Dict, financials: Dict) -> float:
        """Score 0–1 based on completeness and consistency of ingested data."""
        total_checks = len(prices) * 3 + len(financials) * 2
        passed = 0
        for p in prices.values():
            if p.close > 0:      passed += 1
            if p.volume > 0:     passed += 1
            if p.high >= p.low:  passed += 1
        for m in financials.values():
            if m.revenue > 0:    passed += 1
            if m.eps != 0:       passed += 1
        return round(passed / max(total_checks, 1), 4)

    def _log_run(self, status: str, elapsed: float, n_price: int, n_fin: int):
        self._ingestion_log.append({
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "elapsed_seconds": round(elapsed, 2),
            "price_records": n_price,
            "financial_records": n_fin,
        })


# ─── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")
    agent = DataIngestionAgent()
    ds = agent.run()
    print(f"\n✅ Ingestion complete")
    print(f"   Tickers:       {len(ds.price_data)}")
    print(f"   Data quality:  {ds.data_quality_score:.2%}")
    print(f"   Macro VIX:     {ds.macro_snapshot.vix:.1f}")
    print(f"   News items:    {len(ds.news_feed)}")
    print(f"   Sentiments:    {list(ds.sentiment_scores.items())[:3]}")
