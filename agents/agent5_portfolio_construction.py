"""
Agent 5 — Portfolio Construction Agent
=========================================
Converts AI signals from research, macro, and risk agents into
optimized portfolio allocation recommendations.
"""

import logging
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.data_models import (
    UnifiedDataset, ResearchSummary, MacroRegimeReport, RiskReport,
    PortfolioAllocation, Portfolio, Sector
)

logger = logging.getLogger(__name__)


class PortfolioConstructionAgent:
    """
    Agent 5: Portfolio Construction Agent
    ──────────────────────────────────────
    Responsibilities:
      • Aggregate signals from research, macro, and risk agents
      • Estimate expected returns per security
      • Apply sector, position size, and ESG constraints
      • Compute risk-adjusted optimal weights
      • Generate a full allocation table with rationales
    """

    AGENT_ID   = "AGENT-05-PORTFOLIO"
    AGENT_NAME = "Portfolio Construction Agent"
    VERSION    = "1.0.0"

    # Constraints
    MAX_POSITION_WEIGHT    = 0.10    # single-stock max 10%
    MIN_POSITION_WEIGHT    = 0.005   # minimum meaningful position 0.5%
    MAX_SECTOR_WEIGHT      = 0.35    # single-sector max 35%
    MIN_POSITIONS          = 15      # minimum diversification
    MAX_POSITIONS          = 30
    CASH_RESERVE           = 0.03    # keep 3% cash
    HIGH_CONVICTION_THRESHOLD = 0.45

    RATING_WEIGHTS = {
        "Strong Buy":  1.00,
        "Buy":         0.60,
        "Hold":        0.05,
        "Sell":       -0.40,
        "Strong Sell":-0.80,
    }

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.latest_allocations: List[PortfolioAllocation] = []
        logger.info(f"[{self.AGENT_NAME}] Initialized (v{self.VERSION})")

    # ─── Public API ─────────────────────────────────────────────────────────

    def run(
        self,
        dataset: UnifiedDataset,
        research: Dict[str, ResearchSummary],
        macro_report: MacroRegimeReport,
        risk_report: RiskReport,
        current_portfolio: Optional[Portfolio] = None,
    ) -> List[PortfolioAllocation]:
        """
        Full portfolio construction pipeline:
          1. Score all securities using composite signal
          2. Estimate expected returns
          3. Apply macro overlay
          4. Optimize weights under constraints
          5. Generate allocation table with rationale
        """
        logger.info(f"[{self.AGENT_NAME}] Running portfolio construction …")

        # Step 1: Composite signal per security
        signals = self._build_composite_signals(dataset, research)

        # Step 2: Macro overlay adjustments
        signals = self._apply_macro_overlay(signals, macro_report, dataset)

        # Step 3: Expected returns
        expected_returns = self._estimate_expected_returns(signals, dataset, research)

        # Step 4: Risk-adjusted weight optimization
        raw_weights = self._optimize_weights(signals, expected_returns, dataset)

        # Step 5: Apply constraints and normalize
        final_weights = self._apply_constraints(raw_weights, dataset, macro_report)

        # Step 6: Build allocation objects
        allocations = self._build_allocation_table(
            final_weights, signals, expected_returns, research,
            dataset, current_portfolio
        )

        self.latest_allocations = allocations
        long_only = [a for a in allocations if a.target_weight > 0]
        logger.info(f"[{self.AGENT_NAME}] Construction complete — "
                    f"{len(long_only)} long positions, total weight "
                    f"{sum(a.target_weight for a in long_only):.1%}")
        return allocations

    def get_changes_vs_current(
        self, current: Portfolio, allocations: List[PortfolioAllocation]
    ) -> List[Dict]:
        """Return a diff of proposed vs current weights."""
        current_weights = {h.ticker: h.weight for h in current.holdings}
        changes = []
        for alloc in allocations:
            curr_w = current_weights.get(alloc.ticker, 0.0)
            delta  = alloc.target_weight - curr_w
            if abs(delta) > 0.005:
                changes.append({
                    "ticker":          alloc.ticker,
                    "current_weight":  round(curr_w, 4),
                    "target_weight":   round(alloc.target_weight, 4),
                    "delta":           round(delta, 4),
                    "action":          "BUY" if delta > 0 else "SELL",
                    "conviction":      alloc.conviction,
                })
        return sorted(changes, key=lambda x: -abs(x["delta"]))

    # ─── Signal Building ────────────────────────────────────────────────────

    def _build_composite_signals(
        self, dataset: UnifiedDataset, research: Dict[str, ResearchSummary]
    ) -> Dict[str, float]:
        """
        Composite signal = weighted average of:
          • Analyst rating score (40%)
          • Sentiment score (25%)
          • Price momentum (20%)
          • Earnings quality score (15%)
        """
        signals = {}
        for ticker, summary in research.items():
            rating_score    = self.RATING_WEIGHTS.get(summary.analyst_rating, 0.0)
            sentiment_score = dataset.sentiment_scores.get(ticker, 0.0)
            pd              = dataset.price_data.get(ticker)
            momentum_score  = pd.returns_3m if pd else 0.0
            momentum_score  = max(-1.0, min(1.0, momentum_score * 4))

            transcript = dataset.earnings_transcripts.get(ticker)
            earnings_score = 0.0
            if transcript:
                guide_bonus = {"raised": 0.3, "maintained": 0.0, "lowered": -0.3, "withdrawn": -0.5}
                earnings_score = (transcript.eps_beat_miss * 5 +
                                  guide_bonus.get(transcript.guidance_revision, 0.0))
                earnings_score = max(-1.0, min(1.0, earnings_score))

            composite = (
                0.40 * rating_score    +
                0.25 * sentiment_score +
                0.20 * momentum_score  +
                0.15 * earnings_score
            )
            signals[ticker] = round(composite, 4)
        return signals

    def _apply_macro_overlay(
        self, signals: Dict[str, float],
        macro: MacroRegimeReport, dataset: UnifiedDataset
    ) -> Dict[str, float]:
        """Adjust individual stock signals by macro sector tilts."""
        adjusted = {}
        TILT_MAP = {"Overweight": +0.15, "Neutral": 0.0, "Underweight": -0.20}

        for ticker, signal in signals.items():
            fm = dataset.financial_metrics.get(ticker)
            sector_name = fm.sector.value if fm else ""
            macro_tilt = TILT_MAP.get(
                macro.sector_implications.get(sector_name, "Neutral"), 0.0
            )
            adjusted[ticker] = round(signal + macro_tilt, 4)
        return adjusted

    # ─── Expected Returns ────────────────────────────────────────────────────

    def _estimate_expected_returns(
        self, signals: Dict[str, float],
        dataset: UnifiedDataset,
        research: Dict[str, ResearchSummary]
    ) -> Dict[str, float]:
        """
        Expected return estimate:
          • Analyst price target implied return (60%)
          • Signal-derived return premium (40%)
        """
        exp_returns = {}
        for ticker, signal in signals.items():
            summary = research.get(ticker)
            pd      = dataset.price_data.get(ticker)

            pt_return = 0.0
            if summary and pd and pd.close > 0:
                pt_return = (summary.price_target / pd.close) - 1

            # Signal-to-return mapping: assume a score of +1 → ~20% annualized alpha
            signal_return = signal * 0.20

            exp_ret = 0.60 * pt_return + 0.40 * signal_return
            exp_ret = max(-0.30, min(0.50, exp_ret))  # clip to reasonable range
            exp_returns[ticker] = round(exp_ret, 4)
        return exp_returns

    # ─── Weight Optimization ────────────────────────────────────────────────

    def _optimize_weights(
        self, signals: Dict[str, float],
        exp_returns: Dict[str, float], dataset: UnifiedDataset
    ) -> Dict[str, float]:
        """
        Simplified risk-adjusted weight optimization.
        Uses a signal-scaled inverse-volatility approach as a proxy for
        mean-variance optimization (avoids quadratic solver dependency).

        w_i ∝ max(0, signal_i) × (1/vol_i) × expected_return_i
        """
        raw_weights: Dict[str, float] = {}

        for ticker, signal in signals.items():
            if signal <= 0:          # exclude net-negative signals
                continue
            pd      = dataset.price_data.get(ticker)
            vol     = pd.volatility_30d if pd and pd.volatility_30d > 0 else 0.25
            exp_ret = max(0, exp_returns.get(ticker, 0.0))

            raw_w = signal * (1 / vol) * (1 + exp_ret)
            raw_weights[ticker] = max(0, raw_w)

        # Normalize to 1 - cash reserve
        investable = 1.0 - self.CASH_RESERVE
        total = sum(raw_weights.values())
        if total > 0:
            raw_weights = {t: (w / total) * investable for t, w in raw_weights.items()}
        return raw_weights

    # ─── Constraint Engine ──────────────────────────────────────────────────

    def _apply_constraints(
        self, weights: Dict[str, float],
        dataset: UnifiedDataset,
        macro: MacroRegimeReport
    ) -> Dict[str, float]:
        """
        Apply investment constraints iteratively:
          1. Cap individual positions at MAX_POSITION_WEIGHT
          2. Cap sector total at MAX_SECTOR_WEIGHT
          3. Remove sub-minimum positions
          4. Renormalize after each constraint pass
        """
        constrained = dict(weights)

        # Pass 1: position cap
        excess = 0.0
        for t, w in constrained.items():
            if w > self.MAX_POSITION_WEIGHT:
                excess += w - self.MAX_POSITION_WEIGHT
                constrained[t] = self.MAX_POSITION_WEIGHT

        # Redistribute excess proportionally to non-capped positions
        if excess > 0:
            non_capped = {t: w for t, w in constrained.items() if w < self.MAX_POSITION_WEIGHT}
            nc_total   = sum(non_capped.values())
            if nc_total > 0:
                for t in non_capped:
                    constrained[t] += excess * (constrained[t] / nc_total)

        # Pass 2: sector cap
        sector_totals: Dict[str, float] = {}
        for ticker, weight in constrained.items():
            fm = dataset.financial_metrics.get(ticker)
            if fm:
                s = fm.sector.value
                sector_totals[s] = sector_totals.get(s, 0.0) + weight

        for sector, total in sector_totals.items():
            if total > self.MAX_SECTOR_WEIGHT:
                scale = self.MAX_SECTOR_WEIGHT / total
                for ticker in list(constrained.keys()):
                    fm = dataset.financial_metrics.get(ticker)
                    if fm and fm.sector.value == sector:
                        constrained[ticker] *= scale

        # Pass 3: remove below minimum
        constrained = {t: w for t, w in constrained.items() if w >= self.MIN_POSITION_WEIGHT}

        # Renormalize
        investable = 1.0 - self.CASH_RESERVE
        total = sum(constrained.values())
        if total > 0:
            constrained = {t: round((w / total) * investable, 5) for t, w in constrained.items()}

        return constrained

    # ─── Allocation Table ───────────────────────────────────────────────────

    def _build_allocation_table(
        self, weights: Dict[str, float],
        signals: Dict[str, float], exp_returns: Dict[str, float],
        research: Dict[str, ResearchSummary],
        dataset: UnifiedDataset,
        current_portfolio: Optional[Portfolio]
    ) -> List[PortfolioAllocation]:
        allocations = []
        current_weights = {}
        if current_portfolio:
            current_weights = {h.ticker: h.weight for h in current_portfolio.holdings}

        for ticker, target_w in weights.items():
            fm      = dataset.financial_metrics.get(ticker)
            summary = research.get(ticker)
            signal  = signals.get(ticker, 0.0)
            exp_ret = exp_returns.get(ticker, 0.0)

            if abs(signal) >= self.HIGH_CONVICTION_THRESHOLD:
                conviction = "High"
            elif abs(signal) >= 0.20:
                conviction = "Medium"
            else:
                conviction = "Low"

            rationale_parts = []
            if summary:
                rationale_parts.append(f"{summary.analyst_rating} — {summary.revenue_trend[:50]}")
                if summary.key_catalysts:
                    rationale_parts.append(summary.key_catalysts[0])
            rationale = ". ".join(rationale_parts) if rationale_parts else "Systematic signal-based allocation."

            allocations.append(PortfolioAllocation(
                ticker=ticker,
                company_name=fm.company_name if fm else ticker,
                sector=fm.sector if fm else Sector.TECHNOLOGY,
                target_weight=target_w,
                current_weight=current_weights.get(ticker, 0.0),
                signal_score=signal,
                expected_return=exp_ret,
                conviction=conviction,
                rationale=rationale,
            ))

        return sorted(allocations, key=lambda x: -x.target_weight)


# ─── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    from data.sample_data import generate_unified_dataset, generate_sample_portfolio
    from agents.agent2_research_intelligence import ResearchIntelligenceAgent
    from agents.agent3_macro_intelligence    import MacroIntelligenceAgent
    from agents.agent4_risk_intelligence     import RiskIntelligenceAgent

    ds   = generate_unified_dataset()
    port = generate_sample_portfolio(ds.price_data)
    research = ResearchIntelligenceAgent().run(ds)
    macro    = MacroIntelligenceAgent().run(ds)
    risk     = RiskIntelligenceAgent().run(port, ds, macro)

    agent = PortfolioConstructionAgent()
    allocs = agent.run(ds, research, macro, risk, port)

    print(f"\n✅ Portfolio construction complete — {len(allocs)} positions")
    print(f"\n{'Ticker':<8} {'Company':<30} {'Sector':<25} {'Target%':>8} {'Signal':>8} {'Conviction':<10}")
    print("-" * 92)
    for a in allocs[:12]:
        print(f"{a.ticker:<8} {a.company_name[:28]:<30} {a.sector.value[:23]:<25} "
              f"{a.target_weight*100:>7.2f}% {a.signal_score:>+8.3f} {a.conviction:<10}")
    print(f"\n{'Total weight':>67}: {sum(a.target_weight for a in allocs)*100:.2f}%")
