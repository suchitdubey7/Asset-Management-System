"""
Agent 2 — Research Intelligence Agent
========================================
Automates fundamental research by analyzing financial statements,
earnings transcripts, and news to generate actionable research summaries.
"""

import logging
import math
from datetime import datetime
from typing import Dict, List, Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.data_models import (
    UnifiedDataset, ResearchSummary, FinancialMetrics, EarningsTranscript,
    Sector, Sentiment, RiskLevel
)

logger = logging.getLogger(__name__)


class ResearchIntelligenceAgent:
    """
    Agent 2: Research Intelligence Agent
    ─────────────────────────────────────
    Responsibilities:
      • Extract key financial ratios and quality indicators
      • Detect management sentiment from earnings transcripts
      • Identify strategic developments and themes
      • Summarize earnings calls
      • Generate analyst ratings and price targets
    """

    AGENT_ID   = "AGENT-02-RESEARCH"
    AGENT_NAME = "Research Intelligence Agent"
    VERSION    = "1.0.0"

    # Scoring thresholds
    STRONG_REVENUE_GROWTH  = 0.15
    GOOD_REVENUE_GROWTH    = 0.07
    STRONG_GROSS_MARGIN    = 0.55
    STRONG_FCF_CONVERSION  = 0.70     # FCF / Net Income
    HIGH_ROIC              = 0.15
    HIGH_DEBT_EQUITY       = 1.5
    LOW_INTEREST_COVERAGE  = 4.0

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.summaries: Dict[str, ResearchSummary] = {}
        logger.info(f"[{self.AGENT_NAME}] Initialized (v{self.VERSION})")

    # ─── Public API ─────────────────────────────────────────────────────────

    def run(self, dataset: UnifiedDataset) -> Dict[str, ResearchSummary]:
        """
        Generate research summaries for every company in the dataset.
        Steps:
          1. Score financial quality
          2. Analyze earnings transcript
          3. Scan news for strategic signals
          4. Derive analyst rating and price target
          5. Compile research summary
        """
        logger.info(f"[{self.AGENT_NAME}] Generating research summaries …")
        summaries = {}

        for ticker, metrics in dataset.financial_metrics.items():
            transcript = dataset.earnings_transcripts.get(ticker)
            price_data  = dataset.price_data.get(ticker)
            sentiment   = dataset.sentiment_scores.get(ticker, 0.0)
            news_items  = [n for n in dataset.news_feed if ticker in n.tickers_mentioned]

            # Core analytical modules
            financial_score  = self._score_financials(metrics)
            momentum_score   = self._score_momentum(price_data)
            sentiment_score  = sentiment                         # already -1 to +1
            earnings_score   = self._score_earnings(transcript) if transcript else 0.0

            # Composite signal: quality + momentum + sentiment + earnings
            composite = (
                0.40 * financial_score  +
                0.25 * momentum_score   +
                0.20 * sentiment_score  +
                0.15 * earnings_score
            )

            rating, price_target, upside = self._derive_rating_and_target(
                composite, metrics, price_data
            )

            revenue_trend  = self._assess_revenue_trend(metrics)
            margin_trend   = self._assess_margin_trend(metrics)
            strategic_devs = self._extract_strategic_developments(metrics, transcript, news_items)
            mgmt_outlook   = self._infer_management_outlook(transcript, sentiment_score)
            risks          = self._identify_risks(metrics, price_data)
            catalysts      = self._identify_catalysts(metrics, transcript)
            summary_text   = self._write_summary(
                metrics, rating, revenue_trend, margin_trend,
                strategic_devs, mgmt_outlook, composite
            )

            summaries[ticker] = ResearchSummary(
                ticker=ticker,
                company_name=metrics.company_name,
                sector=metrics.sector,
                generated_at=datetime.now(),
                analyst_rating=rating,
                price_target=price_target,
                upside_downside=upside,
                revenue_trend=revenue_trend,
                margin_trend=margin_trend,
                strategic_developments=strategic_devs,
                management_outlook=mgmt_outlook,
                key_risks=risks,
                key_catalysts=catalysts,
                summary=summary_text,
                earnings_summary=self._summarize_earnings(transcript) if transcript else None,
                confidence_score=round(abs(composite), 4),
            )

        self.summaries = summaries
        logger.info(f"[{self.AGENT_NAME}] Generated {len(summaries)} research summaries")
        return summaries

    def get_top_picks(self, n: int = 5) -> List[ResearchSummary]:
        """Return top N buy-rated stocks sorted by confidence."""
        buys = [s for s in self.summaries.values()
                if s.analyst_rating in ("Strong Buy", "Buy")]
        return sorted(buys, key=lambda x: -x.confidence_score)[:n]

    def get_avoid_list(self, n: int = 3) -> List[ResearchSummary]:
        """Return sell-rated stocks."""
        sells = [s for s in self.summaries.values()
                 if s.analyst_rating in ("Strong Sell", "Sell")]
        return sorted(sells, key=lambda x: x.upside_downside)[:n]

    # ─── Financial Scoring ──────────────────────────────────────────────────

    def _score_financials(self, m: FinancialMetrics) -> float:
        """
        Multi-factor financial quality score in [-1, +1].
        Factors: growth, margins, returns, balance sheet, valuation.
        """
        scores = []

        # 1. Revenue growth
        if m.revenue_growth_yoy >= self.STRONG_REVENUE_GROWTH:
            scores.append(1.0)
        elif m.revenue_growth_yoy >= self.GOOD_REVENUE_GROWTH:
            scores.append(0.5)
        elif m.revenue_growth_yoy >= 0:
            scores.append(0.0)
        else:
            scores.append(-0.8)

        # 2. Gross margin
        if m.gross_margin >= self.STRONG_GROSS_MARGIN:
            scores.append(0.8)
        elif m.gross_margin >= 0.35:
            scores.append(0.3)
        else:
            scores.append(-0.3)

        # 3. ROIC
        if m.roic >= self.HIGH_ROIC:
            scores.append(1.0)
        elif m.roic >= 0.08:
            scores.append(0.4)
        else:
            scores.append(-0.5)

        # 4. FCF conversion
        fcf_conv = (m.free_cash_flow / m.net_income) if m.net_income > 0 else 0
        if fcf_conv >= self.STRONG_FCF_CONVERSION:
            scores.append(0.7)
        elif fcf_conv >= 0.40:
            scores.append(0.2)
        else:
            scores.append(-0.4)

        # 5. Leverage risk
        if m.debt_to_equity > self.HIGH_DEBT_EQUITY:
            scores.append(-0.6)
        elif m.debt_to_equity > 0.8:
            scores.append(0.0)
        else:
            scores.append(0.5)

        # 6. Interest coverage
        if m.interest_coverage < self.LOW_INTEREST_COVERAGE:
            scores.append(-0.8)
        elif m.interest_coverage > 10:
            scores.append(0.6)
        else:
            scores.append(0.2)

        # 7. EPS growth
        if m.eps_growth_yoy >= 0.20:
            scores.append(0.9)
        elif m.eps_growth_yoy >= 0.05:
            scores.append(0.4)
        elif m.eps_growth_yoy >= 0:
            scores.append(0.0)
        else:
            scores.append(-0.7)

        return round(sum(scores) / len(scores), 4)

    def _score_momentum(self, price_data) -> float:
        """Price momentum score based on return trend across timeframes."""
        if price_data is None:
            return 0.0
        r1m  = price_data.returns_1m
        r3m  = price_data.returns_3m
        r_ytd= price_data.returns_ytd
        vol  = price_data.volatility_30d

        # Risk-adjusted momentum
        raw_score = (0.30 * r1m + 0.35 * r3m + 0.35 * r_ytd)
        vol_adj   = raw_score / max(vol, 0.01)        # Sharpe-like
        return round(max(-1.0, min(1.0, vol_adj * 4)), 4)

    def _score_earnings(self, transcript: EarningsTranscript) -> float:
        """Score earnings quality: beat/miss, guidance, and tone."""
        sentiment_map = {
            Sentiment.VERY_POSITIVE: 1.0,
            Sentiment.POSITIVE:      0.5,
            Sentiment.NEUTRAL:       0.0,
            Sentiment.NEGATIVE:     -0.5,
            Sentiment.VERY_NEGATIVE:-1.0,
        }
        guidance_map = {"raised": 1.0, "maintained": 0.2, "lowered": -0.8, "withdrawn": -1.0}
        beat_score = min(1.0, max(-1.0, transcript.eps_beat_miss * 10))
        tone_score = sentiment_map[transcript.management_tone]
        guide_score= guidance_map.get(transcript.guidance_revision, 0.0)
        return round(0.40 * beat_score + 0.35 * tone_score + 0.25 * guide_score, 4)

    # ─── Rating & Target ────────────────────────────────────────────────────

    def _derive_rating_and_target(self, composite: float, m: FinancialMetrics, price_data):
        """Map composite score to analyst rating and compute price target."""
        if composite >= 0.55:     rating = "Strong Buy"
        elif composite >= 0.25:   rating = "Buy"
        elif composite >= -0.10:  rating = "Hold"
        elif composite >= -0.40:  rating = "Sell"
        else:                     rating = "Strong Sell"

        # Intrinsic value proxy: DCF-lite using FCF yield
        if price_data:
            current_price = price_data.close
            # Estimated intrinsic = FCF yield reversion + growth premium
            if m.market_cap > 0 and m.free_cash_flow > 0:
                base_yield   = m.free_cash_flow / m.market_cap
                growth_prem  = max(0, m.revenue_growth_yoy * 2.5)
                iv_multiple  = 1 / max(0.03, base_yield - growth_prem) * base_yield
                target_price = round(current_price * (1 + (composite * 0.25 + 0.05)), 2)
            else:
                target_price = round(current_price * 1.10, 2)
            upside = round((target_price / current_price - 1) * 100, 2)
        else:
            target_price = 0.0
            upside = 0.0

        return rating, target_price, upside

    # ─── Analysis Modules ───────────────────────────────────────────────────

    def _assess_revenue_trend(self, m: FinancialMetrics) -> str:
        g = m.revenue_growth_yoy
        if g >= 0.20:   return f"Strong acceleration (+{g*100:.1f}% YoY) — top-line momentum robust"
        if g >= 0.10:   return f"Healthy growth (+{g*100:.1f}% YoY) — demand resilient"
        if g >= 0.03:   return f"Modest growth (+{g*100:.1f}% YoY) — gradual improvement"
        if g >= 0.0:    return f"Flat revenues (+{g*100:.1f}% YoY) — limited top-line momentum"
        return f"Revenue decline ({g*100:.1f}% YoY) — demand headwinds evident"

    def _assess_margin_trend(self, m: FinancialMetrics) -> str:
        gm = m.gross_margin
        em = m.ebitda_margin
        if gm >= 0.60 and em >= 0.30:
            return f"Exceptional margins (GM {gm*100:.1f}%, EBITDA {em*100:.1f}%) — pricing power intact"
        if gm >= 0.40 and em >= 0.18:
            return f"Healthy margins (GM {gm*100:.1f}%, EBITDA {em*100:.1f}%) — above-average profitability"
        if gm >= 0.25:
            return f"Adequate margins (GM {gm*100:.1f}%, EBITDA {em*100:.1f}%) — in line with sector"
        return f"Thin margins (GM {gm*100:.1f}%, EBITDA {em*100:.1f}%) — cost pressure monitoring required"

    def _extract_strategic_developments(
        self, m: FinancialMetrics, transcript: Optional[EarningsTranscript], news: List
    ) -> List[str]:
        developments = []
        if m.capex / max(m.revenue, 1) > 0.08:
            developments.append(f"Elevated capex ({m.capex/m.revenue*100:.1f}% of revenue) signals capacity expansion")
        if m.free_cash_flow / max(m.market_cap, 1) > 0.04:
            developments.append(f"High FCF yield ({m.free_cash_flow/m.market_cap*100:.1f}%) supports buybacks/dividends")
        if m.revenue_growth_yoy > self.STRONG_REVENUE_GROWTH:
            developments.append("Rapid organic growth indicates strong competitive positioning")
        if transcript:
            for topic in transcript.key_topics[:2]:
                developments.append(f"Management focus: {topic}")
            if transcript.guidance_revision == "raised":
                developments.append("Guidance raised — management confidence in near-term outlook")
            elif transcript.guidance_revision == "lowered":
                developments.append("Guidance cut — near-term visibility deteriorating")
        for item in news[:2]:
            developments.append(f"News catalyst: {item.headline[:80]}")
        return developments[:5]

    def _infer_management_outlook(
        self, transcript: Optional[EarningsTranscript], sentiment: float
    ) -> Sentiment:
        if transcript is None:
            return Sentiment.NEUTRAL if sentiment > -0.2 else Sentiment.NEGATIVE
        if transcript.management_tone in (Sentiment.VERY_POSITIVE, Sentiment.POSITIVE):
            return Sentiment.POSITIVE if sentiment >= 0 else Sentiment.NEUTRAL
        if transcript.management_tone in (Sentiment.NEGATIVE, Sentiment.VERY_NEGATIVE):
            return Sentiment.NEGATIVE if sentiment <= 0 else Sentiment.NEUTRAL
        return Sentiment.NEUTRAL

    def _identify_risks(self, m: FinancialMetrics, price_data) -> List[str]:
        risks = []
        if m.debt_to_equity > self.HIGH_DEBT_EQUITY:
            risks.append(f"Elevated leverage (D/E {m.debt_to_equity:.1f}x) — refinancing risk in rising rate environment")
        if m.interest_coverage < self.LOW_INTEREST_COVERAGE:
            risks.append(f"Low interest coverage ({m.interest_coverage:.1f}x) — earnings sensitive to rate changes")
        if m.pe_ratio > 40:
            risks.append(f"Premium valuation (P/E {m.pe_ratio:.0f}x) — vulnerable to multiple compression")
        if price_data and price_data.volatility_30d > 0.40:
            risks.append(f"High realized volatility ({price_data.volatility_30d*100:.0f}%) — wider confidence intervals on returns")
        if m.revenue_growth_yoy < 0:
            risks.append("Negative revenue growth — structural demand headwinds")
        if m.gross_margin < 0.20:
            risks.append("Thin gross margins provide limited buffer against cost inflation")
        if not risks:
            risks.append("Standard market, sector, and company-specific risks apply")
        return risks[:4]

    def _identify_catalysts(self, m: FinancialMetrics, transcript: Optional[EarningsTranscript]) -> List[str]:
        catalysts = []
        if m.revenue_growth_yoy > self.GOOD_REVENUE_GROWTH:
            catalysts.append("Sustained revenue growth momentum above sector average")
        if m.roic > self.HIGH_ROIC:
            catalysts.append(f"High ROIC ({m.roic*100:.1f}%) indicates compounding capability")
        if m.free_cash_flow > 0 and m.free_cash_flow / m.market_cap > 0.03:
            catalysts.append("Strong FCF generation supports capital return program")
        if transcript and transcript.guidance_revision == "raised":
            catalysts.append("Upward guidance revision signals above-consensus execution")
        catalysts.append("Potential sector rotation or macro tailwind benefit")
        return catalysts[:4]

    def _summarize_earnings(self, t: EarningsTranscript) -> str:
        beat_miss = "beat" if t.eps_beat_miss > 0 else "missed"
        direction = "ahead of" if t.revenue_beat_miss > 0 else "below"
        return (
            f"{t.company_name} {t.quarter}: EPS {beat_miss} consensus by "
            f"{abs(t.eps_beat_miss)*100:.1f}%; revenue {direction} estimates by "
            f"{abs(t.revenue_beat_miss)*100:.1f}%. "
            f"Guidance {t.guidance_revision}. Management tone: {t.management_tone.value}."
        )

    def _write_summary(
        self, m: FinancialMetrics, rating: str, revenue_trend: str,
        margin_trend: str, strategic: List[str], outlook: Sentiment, score: float
    ) -> str:
        return (
            f"[{rating.upper()}] {m.company_name} ({m.ticker}) — {m.sector.value}\n\n"
            f"Revenue: {revenue_trend}\n"
            f"Margins: {margin_trend}\n"
            f"Management outlook: {outlook.value}\n"
            f"Composite AI score: {score:+.2f}\n\n"
            f"Key developments:\n" +
            "\n".join(f"  • {d}" for d in strategic)
        )


# ─── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    from data.sample_data import generate_unified_dataset
    ds = generate_unified_dataset()
    agent = ResearchIntelligenceAgent()
    summaries = agent.run(ds)
    print(f"\n✅ Research complete — {len(summaries)} summaries generated")
    tops = agent.get_top_picks(3)
    print("\n📈 Top Picks:")
    for s in tops:
        print(f"   {s.ticker} | {s.analyst_rating} | PT ${s.price_target:.2f} | {s.upside_downside:+.1f}%")
    print("\n📉 Avoid:")
    for s in agent.get_avoid_list(2):
        print(f"   {s.ticker} | {s.analyst_rating} | {s.upside_downside:+.1f}%")
