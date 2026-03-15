"""
Agent 7 — Monitoring & Alert Agent
=====================================
Real-time surveillance of portfolio, macro, and company-level triggers.
Generates structured alerts with actionable guidance.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.data_models import (
    Alert, AlertType, RiskLevel, Portfolio, UnifiedDataset,
    MacroRegimeReport, RiskReport, ResearchSummary, Sentiment
)

logger = logging.getLogger(__name__)


class MonitoringAlertAgent:
    """
    Agent 7: Monitoring & Alert Agent
    ────────────────────────────────────
    Responsibilities:
      • Monitor earnings downgrades and guidance cuts
      • Detect macro regime shifts and threshold breaches
      • Alert on sector concentration limit violations
      • Flag liquidity deterioration
      • Monitor volatility spikes and drawdown thresholds
      • Generate ranked, actionable alerts for portfolio managers
    """

    AGENT_ID   = "AGENT-07-MONITOR"
    AGENT_NAME = "Monitoring & Alert Agent"
    VERSION    = "1.0.0"

    # Alert thresholds
    EARNINGS_DOWNGRADE_EPS_MISS   = -0.05     # > 5% EPS miss triggers alert
    GUIDANCE_CUT_THRESHOLD        = "lowered"
    MACRO_CONFIDENCE_SHIFT        = 0.20      # 20pp confidence change in regime = shift
    SECTOR_CONCENTRATION_LIMIT    = 0.30      # 30% single sector
    LIQUIDITY_ALERT_THRESHOLD     = 0.65      # below 65% = alert
    VIX_SPIKE_THRESHOLD           = 28.0
    DRAWDOWN_ALERT_THRESHOLD      = -0.10     # 10% drawdown alert
    HY_SPREAD_ALERT               = 450       # bps
    SINGLE_STOCK_LOSS_ALERT       = -0.15     # 15% loss on any holding

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        self._prior_macro_regime = None
        logger.info(f"[{self.AGENT_NAME}] Initialized (v{self.VERSION})")

    # ─── Public API ─────────────────────────────────────────────────────────

    def run(
        self,
        portfolio: Portfolio,
        dataset: UnifiedDataset,
        macro_report: MacroRegimeReport,
        risk_report: RiskReport,
        research: Dict[str, ResearchSummary],
    ) -> List[Alert]:
        """
        Execute full monitoring sweep and generate all applicable alerts.
        Alert types checked:
          • Earnings downgrades
          • Macro regime shifts
          • Sector concentration breaches
          • Liquidity deterioration
          • Volatility / VIX spikes
          • Individual stock drawdowns
          • Credit spread widening
        """
        logger.info(f"[{self.AGENT_NAME}] Running monitoring sweep …")
        new_alerts: List[Alert] = []

        # 1. Earnings & guidance alerts
        new_alerts.extend(self._check_earnings_alerts(dataset, research, portfolio))

        # 2. Macro regime alerts
        new_alerts.extend(self._check_macro_alerts(macro_report, dataset))

        # 3. Sector concentration alerts
        new_alerts.extend(self._check_concentration_alerts(risk_report))

        # 4. Liquidity alerts
        new_alerts.extend(self._check_liquidity_alerts(risk_report))

        # 5. Volatility / market stress alerts
        new_alerts.extend(self._check_market_stress_alerts(dataset))

        # 6. Individual holding drawdown alerts
        new_alerts.extend(self._check_holding_alerts(portfolio, dataset))

        # 7. Credit market alerts
        new_alerts.extend(self._check_credit_alerts(dataset))

        # Deduplicate and rank by severity
        new_alerts = self._rank_and_deduplicate(new_alerts)

        self.active_alerts = new_alerts
        self.alert_history.extend(new_alerts)

        critical = [a for a in new_alerts if a.severity == RiskLevel.CRITICAL]
        high     = [a for a in new_alerts if a.severity == RiskLevel.HIGH]
        logger.info(f"[{self.AGENT_NAME}] Generated {len(new_alerts)} alerts: "
                    f"{len(critical)} CRITICAL, {len(high)} HIGH")
        return new_alerts

    def get_alert_summary(self) -> Dict:
        return {
            "total_alerts":    len(self.active_alerts),
            "critical":        len([a for a in self.active_alerts if a.severity == RiskLevel.CRITICAL]),
            "high":            len([a for a in self.active_alerts if a.severity == RiskLevel.HIGH]),
            "moderate":        len([a for a in self.active_alerts if a.severity == RiskLevel.MODERATE]),
            "low":             len([a for a in self.active_alerts if a.severity == RiskLevel.LOW]),
            "latest":          self.active_alerts[0].title if self.active_alerts else "No alerts",
        }

    def format_alert_digest(self) -> str:
        if not self.active_alerts:
            return "✅ No active alerts — portfolio within all monitoring thresholds."
        lines = [f"🔔 ALERT DIGEST — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                 f"{'='*60}"]
        severity_order = [RiskLevel.CRITICAL, RiskLevel.HIGH, RiskLevel.MODERATE, RiskLevel.LOW]
        icons = {RiskLevel.CRITICAL: "🔴", RiskLevel.HIGH: "🟠", RiskLevel.MODERATE: "🟡", RiskLevel.LOW: "🟢"}
        for severity in severity_order:
            group = [a for a in self.active_alerts if a.severity == severity]
            if group:
                lines.append(f"\n{icons[severity]} {severity.value.upper()} ({len(group)})")
                for alert in group:
                    lines.append(f"\n  [{alert.alert_type.value}]")
                    lines.append(f"  {alert.title}")
                    lines.append(f"  {alert.description}")
                    lines.append(f"  → Action: {alert.suggested_action}")
                    if alert.affected_tickers:
                        lines.append(f"  → Tickers: {', '.join(alert.affected_tickers)}")
        return "\n".join(lines)

    # ─── Earnings Monitoring ─────────────────────────────────────────────────

    def _check_earnings_alerts(
        self, dataset: UnifiedDataset,
        research: Dict[str, ResearchSummary], portfolio: Portfolio
    ) -> List[Alert]:
        alerts = []
        held_tickers = {h.ticker for h in portfolio.holdings}
        for ticker, transcript in dataset.earnings_transcripts.items():
            if ticker not in held_tickers:
                continue
            # EPS miss alert
            if transcript.eps_beat_miss < self.EARNINGS_DOWNGRADE_EPS_MISS:
                severity = RiskLevel.HIGH if transcript.eps_beat_miss < -0.10 else RiskLevel.MODERATE
                summary  = research.get(ticker)
                alerts.append(Alert(
                    alert_id=_new_id(),
                    generated_at=datetime.now(),
                    alert_type=AlertType.EARNINGS_DOWNGRADE,
                    severity=severity,
                    title=f"{ticker}: Earnings Miss — EPS {transcript.eps_beat_miss*100:+.1f}% vs consensus",
                    description=(
                        f"{transcript.company_name} missed EPS estimates by "
                        f"{abs(transcript.eps_beat_miss)*100:.1f}% in {transcript.quarter}. "
                        f"Guidance {transcript.guidance_revision}. "
                        f"Management tone: {transcript.management_tone.value}."
                    ),
                    affected_tickers=[ticker],
                    affected_sectors=[],
                    suggested_action=(
                        "Review position sizing. Consider reducing exposure "
                        f"if guidance trajectory remains negative. Current analyst rating: "
                        f"{summary.analyst_rating if summary else 'N/A'}."
                    ),
                ))
            # Guidance cut alert
            if transcript.guidance_revision == "lowered":
                alerts.append(Alert(
                    alert_id=_new_id(),
                    generated_at=datetime.now(),
                    alert_type=AlertType.EARNINGS_DOWNGRADE,
                    severity=RiskLevel.MODERATE,
                    title=f"{ticker}: Forward Guidance Cut — Earnings outlook deteriorating",
                    description=(
                        f"{transcript.company_name} lowered guidance for the next quarter/year. "
                        f"Revenue miss: {transcript.revenue_beat_miss*100:+.1f}%."
                    ),
                    affected_tickers=[ticker],
                    affected_sectors=[],
                    suggested_action="Monitor for further guidance revisions. Consider trailing stop-loss review.",
                ))
        return alerts

    # ─── Macro Monitoring ────────────────────────────────────────────────────

    def _check_macro_alerts(
        self, macro: MacroRegimeReport, dataset: UnifiedDataset
    ) -> List[Alert]:
        alerts = []
        snap = dataset.macro_snapshot

        # Regime shift vs prior
        if (self._prior_macro_regime is not None and
                self._prior_macro_regime != macro.current_regime):
            alerts.append(Alert(
                alert_id=_new_id(),
                generated_at=datetime.now(),
                alert_type=AlertType.MACRO_REGIME_SHIFT,
                severity=RiskLevel.HIGH,
                title=f"Macro Regime Shift: {self._prior_macro_regime.value} → {macro.current_regime.value}",
                description=(
                    f"The macroeconomic regime has shifted from {self._prior_macro_regime.value} "
                    f"to {macro.current_regime.value} (confidence {macro.regime_confidence:.0%}). "
                    f"Portfolio sector allocations should be reviewed."
                ),
                affected_tickers=[],
                affected_sectors=list(macro.sector_implications.keys()),
                suggested_action=(
                    "Rebalance sector tilts per new regime. "
                    "Overweight: " + ", ".join(s for s, t in macro.sector_implications.items() if t == "Overweight") + ". "
                    "Underweight: " + ", ".join(s for s, t in macro.sector_implications.items() if t == "Underweight") + "."
                ),
            ))
        self._prior_macro_regime = macro.current_regime

        # Inverted yield curve
        if snap.yield_curve_spread < -0.50:
            alerts.append(Alert(
                alert_id=_new_id(),
                generated_at=datetime.now(),
                alert_type=AlertType.MACRO_REGIME_SHIFT,
                severity=RiskLevel.HIGH,
                title=f"Deeply Inverted Yield Curve ({snap.yield_curve_spread:+.2f}%) — Recession signal",
                description=(
                    f"The 10Y-2Y yield spread stands at {snap.yield_curve_spread:+.2f}%, "
                    f"deeply inverted. Historically, sustained inversion of this magnitude "
                    f"has preceded recessions within 12-18 months."
                ),
                affected_tickers=[],
                affected_sectors=["Financials", "Consumer Discretionary", "Industrials"],
                suggested_action=(
                    "Increase defensive exposure (Healthcare, Consumer Staples, Utilities). "
                    "Reduce beta and consider increasing cash/Treasury allocation."
                ),
            ))

        # Elevated inflation
        if snap.cpi_yoy > 4.5:
            alerts.append(Alert(
                alert_id=_new_id(),
                generated_at=datetime.now(),
                alert_type=AlertType.MACRO_REGIME_SHIFT,
                severity=RiskLevel.MODERATE,
                title=f"Inflation Re-acceleration: CPI {snap.cpi_yoy:.1f}% — Policy risk elevated",
                description=(
                    f"CPI has risen to {snap.cpi_yoy:.1f}% YoY, significantly above the Fed's "
                    f"2% target. Risk of additional rate hikes increases."
                ),
                affected_tickers=[],
                affected_sectors=["Real Estate", "Utilities", "Technology"],
                suggested_action="Review duration-sensitive positions. Consider TIPS and commodity-linked assets as inflation hedges.",
            ))
        return alerts

    # ─── Concentration Monitoring ─────────────────────────────────────────────

    def _check_concentration_alerts(self, risk: RiskReport) -> List[Alert]:
        alerts = []
        for sector, weight in risk.sector_exposures.items():
            if weight > self.SECTOR_CONCENTRATION_LIMIT:
                severity = RiskLevel.CRITICAL if weight > 0.40 else RiskLevel.HIGH
                alerts.append(Alert(
                    alert_id=_new_id(),
                    generated_at=datetime.now(),
                    alert_type=AlertType.SECTOR_BREACH,
                    severity=severity,
                    title=f"Sector Concentration Breach: {sector} at {weight*100:.1f}%",
                    description=(
                        f"Portfolio exposure to {sector} ({weight*100:.1f}%) exceeds the "
                        f"{self.SECTOR_CONCENTRATION_LIMIT*100:.0f}% limit. "
                        f"Idiosyncratic sector risk is elevated."
                    ),
                    affected_tickers=[],
                    affected_sectors=[sector],
                    suggested_action=(
                        f"Trim {sector} positions to bring exposure below "
                        f"{self.SECTOR_CONCENTRATION_LIMIT*100:.0f}%. Redeploy into "
                        f"underweighted sectors per macro regime guidance."
                    ),
                ))

        if risk.top_10_holdings_weight > 0.75:
            alerts.append(Alert(
                alert_id=_new_id(),
                generated_at=datetime.now(),
                alert_type=AlertType.SECTOR_BREACH,
                severity=RiskLevel.MODERATE,
                title=f"High Idiosyncratic Concentration: Top 10 = {risk.top_10_holdings_weight*100:.1f}%",
                description="Top 10 holdings exceed 75% of NAV. Portfolio is highly concentrated in a small number of positions.",
                affected_tickers=[],
                affected_sectors=[],
                suggested_action="Consider broadening portfolio to 20+ names to reduce single-stock risk.",
            ))
        return alerts

    # ─── Liquidity Monitoring ────────────────────────────────────────────────

    def _check_liquidity_alerts(self, risk: RiskReport) -> List[Alert]:
        alerts = []
        if risk.liquidity_score < self.LIQUIDITY_ALERT_THRESHOLD:
            severity = (RiskLevel.CRITICAL if risk.liquidity_score < 0.50
                        else RiskLevel.HIGH)
            alerts.append(Alert(
                alert_id=_new_id(),
                generated_at=datetime.now(),
                alert_type=AlertType.LIQUIDITY_RISK,
                severity=severity,
                title=f"Liquidity Deterioration: Score {risk.liquidity_score:.2f}",
                description=(
                    f"Portfolio liquidity has fallen to {risk.liquidity_score:.2f} — "
                    f"below the {self.LIQUIDITY_ALERT_THRESHOLD} alert threshold. "
                    f"Estimated days to liquidate 90% of portfolio: {risk.days_to_liquidate_90pct:.1f} days."
                ),
                affected_tickers=[],
                affected_sectors=[],
                suggested_action=(
                    "Review illiquid positions. Replace small-cap or thin-traded names with "
                    "large-cap equivalents. Ensure redemption capacity matches fund liquidity terms."
                ),
            ))
        return alerts

    # ─── Market Stress Monitoring ────────────────────────────────────────────

    def _check_market_stress_alerts(self, dataset: UnifiedDataset) -> List[Alert]:
        alerts = []
        vix = dataset.macro_snapshot.vix
        if vix >= 35:
            alerts.append(Alert(
                alert_id=_new_id(),
                generated_at=datetime.now(),
                alert_type=AlertType.VOLATILITY_SPIKE,
                severity=RiskLevel.CRITICAL,
                title=f"Extreme Volatility: VIX at {vix:.1f} — Market stress event",
                description=(
                    f"The VIX Index has spiked to {vix:.1f}, indicating extreme market fear "
                    f"and potential for large drawdowns. Risk-off positioning warranted."
                ),
                affected_tickers=[],
                affected_sectors=["Technology", "Consumer Discretionary", "Financials"],
                suggested_action="Activate contingency hedging plan. Increase cash. Review stop-losses on high-beta positions.",
            ))
        elif vix >= self.VIX_SPIKE_THRESHOLD:
            alerts.append(Alert(
                alert_id=_new_id(),
                generated_at=datetime.now(),
                alert_type=AlertType.VOLATILITY_SPIKE,
                severity=RiskLevel.HIGH,
                title=f"Volatility Spike: VIX at {vix:.1f} — Risk-off conditions",
                description=f"VIX has risen to {vix:.1f}, above the {self.VIX_SPIKE_THRESHOLD} alert threshold.",
                affected_tickers=[],
                affected_sectors=[],
                suggested_action="Review portfolio beta. Consider partial hedges. Reduce speculative positions.",
            ))
        return alerts

    # ─── Individual Holding Monitoring ──────────────────────────────────────

    def _check_holding_alerts(
        self, portfolio: Portfolio, dataset: UnifiedDataset
    ) -> List[Alert]:
        alerts = []
        for h in portfolio.holdings:
            # Unrealized loss alert
            if h.unrealized_pnl_pct < self.SINGLE_STOCK_LOSS_ALERT:
                severity = (RiskLevel.HIGH if h.unrealized_pnl_pct < -0.25
                            else RiskLevel.MODERATE)
                alerts.append(Alert(
                    alert_id=_new_id(),
                    generated_at=datetime.now(),
                    alert_type=AlertType.DRAWDOWN_BREACH,
                    severity=severity,
                    title=f"{h.ticker}: Drawdown Alert — {h.unrealized_pnl_pct*100:.1f}% unrealized loss",
                    description=(
                        f"{h.company_name} ({h.ticker}) is showing an unrealized loss of "
                        f"{h.unrealized_pnl_pct*100:.1f}% (${abs(h.unrealized_pnl):,.0f}). "
                        f"Position weight: {h.weight*100:.1f}%."
                    ),
                    affected_tickers=[h.ticker],
                    affected_sectors=[h.sector.value],
                    suggested_action=(
                        f"Review investment thesis. If fundamentals unchanged, consider averaging in. "
                        f"If thesis broken, initiate orderly exit. Position review needed."
                    ),
                ))
        return alerts

    # ─── Credit Monitoring ────────────────────────────────────────────────────

    def _check_credit_alerts(self, dataset: UnifiedDataset) -> List[Alert]:
        alerts = []
        snap = dataset.macro_snapshot
        if snap.hy_spread > self.HY_SPREAD_ALERT:
            alerts.append(Alert(
                alert_id=_new_id(),
                generated_at=datetime.now(),
                alert_type=AlertType.MACRO_REGIME_SHIFT,
                severity=RiskLevel.HIGH,
                title=f"Credit Stress: HY Spread at {snap.hy_spread:.0f}bps — Risk appetite weakening",
                description=(
                    f"High-yield credit spreads have widened to {snap.hy_spread:.0f}bps, "
                    f"above the {self.HY_SPREAD_ALERT}bps alert level. "
                    f"Investment-grade spreads at {snap.ig_spread:.0f}bps."
                ),
                affected_tickers=[],
                affected_sectors=["Financials", "Consumer Discretionary", "Industrials"],
                suggested_action=(
                    "Increase quality tilt within equity portfolio. "
                    "Favor strong balance sheets over leveraged names. "
                    "Review exposure to high-yield issuers."
                ),
            ))
        return alerts

    # ─── Utilities ──────────────────────────────────────────────────────────

    def _rank_and_deduplicate(self, alerts: List[Alert]) -> List[Alert]:
        """Sort by severity (Critical first) and remove near-duplicates."""
        seen_types: Dict[str, Alert] = {}
        severity_rank = {
            RiskLevel.CRITICAL: 0,
            RiskLevel.HIGH:     1,
            RiskLevel.MODERATE: 2,
            RiskLevel.LOW:      3,
        }
        # Deduplicate by (type, first_ticker) keeping highest severity
        for alert in alerts:
            key = f"{alert.alert_type.value}:{alert.affected_tickers[0] if alert.affected_tickers else alert.affected_sectors[0] if alert.affected_sectors else alert.title[:30]}"
            if key not in seen_types or (
                severity_rank[alert.severity] < severity_rank[seen_types[key].severity]
            ):
                seen_types[key] = alert

        return sorted(seen_types.values(), key=lambda a: severity_rank[a.severity])


def _new_id() -> str:
    return f"ALT-{uuid.uuid4().hex[:8].upper()}"


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

    agent  = MonitoringAlertAgent()
    alerts = agent.run(port, ds, macro, risk, research)
    print(agent.format_alert_digest())
    print(f"\n\nSummary: {agent.get_alert_summary()}")
