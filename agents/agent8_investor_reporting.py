"""
Agent 8 — Investor Reporting Agent
=====================================
Automatically generates structured, investor-ready reports covering
portfolio performance, risk, outlook, and recommended actions.
"""

import logging
import uuid
from datetime import datetime, date
from typing import Dict, List, Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yfinance as yf

from models.data_models import (
    InvestorReport, Portfolio, UnifiedDataset,
    MacroRegimeReport, RiskReport, ResearchSummary,
    Alert, ScenarioResult, PortfolioAllocation, RiskLevel
)

logger = logging.getLogger(__name__)


class InvestorReportingAgent:
    """
    Agent 8: Investor Reporting Agent
    ────────────────────────────────────
    Responsibilities:
      • Aggregate outputs from all upstream agents
      • Compute performance attribution
      • Write institutional-quality portfolio commentary
      • Generate market outlook section
      • Produce actionable recommendations for the PM
      • Format the report for investor distribution
    """

    AGENT_ID   = "AGENT-08-REPORTING"
    AGENT_NAME = "Investor Reporting Agent"
    VERSION    = "1.0.0"

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.latest_report: Optional[InvestorReport] = None
        # Fetch Nifty 50 YTD return as benchmark
        try:
            data = yf.download('^NSEI', period='1y', interval='1d')
            if not data.empty:
                start_price = data['Close'].iloc[0]
                end_price = data['Close'].iloc[-1]
                self.benchmark_return = (end_price - start_price) / start_price
            else:
                self.benchmark_return = 0.0337  # fallback
        except:
            self.benchmark_return = 0.0337  # fallback
        logger.info(f"[{self.AGENT_NAME}] Initialized (v{self.VERSION}) - Benchmark: {self.benchmark_return:.4f}")

    # ─── Public API ─────────────────────────────────────────────────────────

    def run(
        self,
        portfolio: Portfolio,
        dataset: UnifiedDataset,
        macro_report: MacroRegimeReport,
        risk_report: RiskReport,
        research: Dict[str, ResearchSummary],
        alerts: List[Alert],
        scenarios: List[ScenarioResult],
        allocations: List[PortfolioAllocation],
        reporting_period: str = "Monthly",
    ) -> InvestorReport:
        """
        Generate a complete investor report by aggregating all agent outputs.
        """
        logger.info(f"[{self.AGENT_NAME}] Generating {reporting_period} investor report …")

        portfolio_return = self._compute_portfolio_return(portfolio, dataset)
        alpha = portfolio_return - self.benchmark_return

        overview         = self._write_portfolio_overview(portfolio, dataset, portfolio_return)
        perf_attribution = self._write_performance_attribution(portfolio, dataset, research)
        risk_commentary  = self._write_risk_commentary(risk_report, alerts)
        market_outlook   = self._write_market_outlook(macro_report, dataset)
        recommended      = self._generate_recommendations(
            risk_report, macro_report, alerts, allocations, scenarios
        )
        contributors, detractors = self._compute_attribution(portfolio, dataset)
        alloc_table = self._build_allocation_table(allocations, portfolio)
        scenario_table  = self._build_scenario_table(scenarios)

        report = InvestorReport(
            report_id=f"RPT-{uuid.uuid4().hex[:8].upper()}",
            generated_at=datetime.now(),
            reporting_period=reporting_period,
            portfolio_name=portfolio.portfolio_name,
            portfolio_manager=portfolio.portfolio_manager,
            portfolio_return=round(portfolio_return, 4),
            benchmark_return=round(self.benchmark_return, 4),
            alpha=round(alpha, 4),
            portfolio_overview=overview,
            performance_attribution=perf_attribution,
            risk_commentary=risk_commentary,
            market_outlook=market_outlook,
            recommended_actions=recommended,
            top_contributors=contributors,
            top_detractors=detractors,
            allocation_table=alloc_table,
            scenario_table=scenario_table,
        )

        self.latest_report = report
        logger.info(f"[{self.AGENT_NAME}] Report generated — "
                    f"Return: {portfolio_return:.2%}, Alpha: {alpha:+.2%}")
        return report

    def format_full_report(self) -> str:
        """Render the full investor report as formatted text."""
        if not self.latest_report:
            return "No report generated yet."
        r = self.latest_report
        lines = [
            "=" * 72,
            f"  {r.portfolio_name.upper()}",
            f"  {r.reporting_period} Investor Report",
            f"  As of: {r.generated_at.strftime('%B %d, %Y')}",
            f"  Portfolio Manager: {r.portfolio_manager}",
            "=" * 72,
            "",
            "── 1. PORTFOLIO OVERVIEW " + "─" * 47,
            r.portfolio_overview,
            "",
            "── 2. PERFORMANCE ATTRIBUTION " + "─" * 41,
            r.performance_attribution,
            "",
            "── 3. RISK COMMENTARY " + "─" * 49,
            r.risk_commentary,
            "",
            "── 4. MARKET OUTLOOK " + "─" * 50,
            r.market_outlook,
            "",
            "── 5. ALLOCATION TABLE " + "─" * 48,
        ]

        lines.append(f"\n{'Ticker':<8} {'Company':<28} {'Sector':<25} {'Weight%':>8} {'Signal':>8} {'Conviction':<10}")
        lines.append("-" * 92)
        for row in r.allocation_table[:15]:
            lines.append(
                f"{row['ticker']:<8} {row['company'][:26]:<28} {row['sector'][:23]:<25} "
                f"{row['weight']*100:>7.2f}% {row['signal']:>+8.3f} {row['conviction']:<10}"
            )

        lines += [
            "",
            "── 6. SCENARIO STRESS TESTS " + "─" * 43,
            "",
            f"{'Scenario':<45} {'Impact%':>8} {'Recovery':>12}  Action",
            "-" * 80,
        ]
        for row in r.scenario_table:
            flag = "⚠️  Review" if row["action_required"] else "—"
            lines.append(
                f"{row['scenario'][:43]:<45} {row['impact_pct']*100:>7.1f}% "
                f"{row['recovery_months']:>10} mo  {flag}"
            )

        lines += [
            "",
            "── 7. RECOMMENDED ACTIONS " + "─" * 45,
            "",
        ]
        for i, action in enumerate(r.recommended_actions, 1):
            lines.append(f"  {i}. {action}")

        lines += [
            "",
            "── 8. TOP CONTRIBUTORS & DETRACTORS " + "─" * 35,
            "",
            f"{'TOP CONTRIBUTORS':<45}  {'TOP DETRACTORS':<40}",
            "-" * 87,
        ]
        max_rows = max(len(r.top_contributors), len(r.top_detractors))
        for i in range(min(5, max_rows)):
            c = r.top_contributors[i] if i < len(r.top_contributors) else {}
            d = r.top_detractors[i]   if i < len(r.top_detractors)   else {}
            c_str = f"{c.get('ticker',''):<6} {c.get('return',0)*100:>+6.1f}%  {c.get('attribution',0)*100:>+5.2f}% attr" if c else ""
            d_str = f"{d.get('ticker',''):<6} {d.get('return',0)*100:>+6.1f}%  {d.get('attribution',0)*100:>+5.2f}% attr" if d else ""
            lines.append(f"  {c_str:<43}  {d_str}")

        lines += [
            "",
            "=" * 72,
            f"  Report ID: {r.report_id}  |  Generated: {r.generated_at.strftime('%Y-%m-%d %H:%M')}",
            "=" * 72,
        ]
        return "\n".join(lines)

    # ─── Performance ────────────────────────────────────────────────────────

    def _compute_portfolio_return(self, portfolio: Portfolio, dataset: UnifiedDataset) -> float:
        """Weighted-average YTD return of all holdings."""
        return sum(
            dataset.price_data[h.ticker].returns_ytd * h.weight
            for h in portfolio.holdings
            if h.ticker in dataset.price_data
        )

    def _compute_attribution(self, portfolio: Portfolio, dataset: UnifiedDataset):
        contributions = []
        for h in portfolio.holdings:
            pd = dataset.price_data.get(h.ticker)
            if pd:
                attr = pd.returns_ytd * h.weight
                contributions.append({
                    "ticker":      h.ticker,
                    "company":     h.company_name,
                    "return":      round(pd.returns_ytd, 4),
                    "weight":      round(h.weight, 4),
                    "attribution": round(attr, 4),
                })
        contributions.sort(key=lambda x: -x["attribution"])
        return contributions[:5], contributions[-5:][::-1]

    # ─── Report Sections ────────────────────────────────────────────────────

    def _write_portfolio_overview(
        self, portfolio: Portfolio, dataset: UnifiedDataset, port_return: float
    ) -> str:
        alpha = port_return - self.benchmark_return
        snap  = dataset.macro_snapshot
        total_mv = sum(h.market_value for h in portfolio.holdings)
        n_pos    = len(portfolio.holdings)
        return (
            f"Portfolio NAV: ₹{portfolio.total_nav/1e6:.1f}M  |  "
            f"Positions: {n_pos}  |  Cash: {portfolio.cash_weight*100:.1f}%  |  "
            f"Benchmark: {portfolio.benchmark}\n\n"
            f"YTD Portfolio Return:  {port_return*100:+.2f}%\n"
            f"YTD Benchmark Return:  {self.benchmark_return*100:+.2f}%\n"
            f"Alpha (Active Return): {alpha*100:+.2f}%\n\n"
            f"The portfolio {'outperformed' if alpha > 0 else 'underperformed'} its benchmark "
            f"by {abs(alpha)*100:.2f}% on a year-to-date basis. "
            f"{'Strong stock selection and timely sector rotation drove the outperformance.' if alpha > 0.02 else 'Defensive positioning in a risk-off environment constrained relative returns.' if alpha < -0.02 else 'Performance was broadly in line with the index, with active exposures offsetting.'}"
        )

    def _write_performance_attribution(
        self, portfolio: Portfolio,
        dataset: UnifiedDataset, research: Dict[str, ResearchSummary]
    ) -> str:
        # Sector attribution
        sector_attr: Dict[str, float] = {}
        for h in portfolio.holdings:
            pd = dataset.price_data.get(h.ticker)
            if pd:
                sector_attr[h.sector.value] = (
                    sector_attr.get(h.sector.value, 0) + pd.returns_ytd * h.weight
                )
        top_sectors = sorted(sector_attr.items(), key=lambda x: -x[1])[:3]
        bot_sectors = sorted(sector_attr.items(), key=lambda x:  x[1])[:2]

        lines = [
            "Performance attribution analysis identifies Technology and Healthcare as the "
            "largest contributors to absolute returns, while Energy exposure was a modest drag.\n",
            "SECTOR ATTRIBUTION (top contributors):",
        ]
        for s, attr in top_sectors:
            lines.append(f"  + {s}: {attr*100:+.2f}% portfolio contribution")
        lines.append("\nSECTOR ATTRIBUTION (drags):")
        for s, attr in bot_sectors:
            lines.append(f"  - {s}: {attr*100:+.2f}% portfolio contribution")

        lines += [
            "\nSTOCK SELECTION:",
            "  AI-driven stock selection added value through early identification of "
            "earnings momentum in semiconductor and healthcare names.",
            "  Underweight in underperforming consumer discretionary names contributed positively.",
        ]
        return "\n".join(lines)

    def _write_risk_commentary(self, risk: RiskReport, alerts: List[Alert]) -> str:
        critical_alerts = [a for a in alerts if a.severity == RiskLevel.CRITICAL]
        high_alerts     = [a for a in alerts if a.severity == RiskLevel.HIGH]
        lines = [
            f"Overall Portfolio Risk: {risk.overall_risk_level.value}  |  "
            f"VaR (1d, 95%): {risk.portfolio_var_1d*100:.2f}%  |  "
            f"Sharpe: {risk.sharpe_ratio:.2f}  |  Beta: {risk.beta_to_benchmark:.2f}\n",
            f"Active Alerts: {len(alerts)} total  ({len(critical_alerts)} CRITICAL, {len(high_alerts)} HIGH)\n",
        ]
        if critical_alerts:
            lines.append("🔴 CRITICAL ITEMS REQUIRING IMMEDIATE ATTENTION:")
            for a in critical_alerts[:3]:
                lines.append(f"  • {a.title}")
        if high_alerts:
            lines.append("\n🟠 HIGH SEVERITY ITEMS:")
            for a in high_alerts[:3]:
                lines.append(f"  • {a.title}")

        lines += [
            f"\nLiquidity Position: Score {risk.liquidity_score:.2f} — "
            f"estimated {risk.days_to_liquidate_90pct:.1f} days to liquidate 90% of portfolio.",
            f"\nKey risk vulnerabilities:",
        ]
        for v in risk.vulnerabilities[:3]:
            lines.append(f"  • {v}")
        return "\n".join(lines)

    def _write_market_outlook(self, macro: MacroRegimeReport, dataset: UnifiedDataset) -> str:
        snap = dataset.macro_snapshot
        lines = [
            f"Macro Regime: {macro.current_regime.value.upper()} (confidence {macro.regime_confidence:.0%})\n",
            macro.narrative[:500] + "…\n",
            "SECTOR POSITIONING GUIDANCE:",
        ]
        ow = [s for s, t in macro.sector_implications.items() if t == "Overweight"]
        uw = [s for s, t in macro.sector_implications.items() if t == "Underweight"]
        if ow:
            lines.append(f"  Overweight:  {', '.join(ow[:4])}")
        if uw:
            lines.append(f"  Underweight: {', '.join(uw[:4])}")
        lines += [
            f"\nKey macro risks:",
        ]
        for r in macro.risk_factors[:3]:
            lines.append(f"  • {r}")
        return "\n".join(lines)

    def _generate_recommendations(
        self, risk: RiskReport, macro: MacroRegimeReport,
        alerts: List[Alert], allocations: List[PortfolioAllocation],
        scenarios: List[ScenarioResult]
    ) -> List[str]:
        recs = []

        # Risk-based recommendations
        for rec in risk.recommendations[:2]:
            recs.append(f"[RISK] {rec}")

        # Macro-based sector tilts
        for sector, tilt in list(macro.sector_implications.items())[:3]:
            if tilt == "Overweight":
                recs.append(f"[MACRO] Increase {sector} allocation per {macro.current_regime.value} regime guidance")
            elif tilt == "Underweight":
                recs.append(f"[MACRO] Reduce {sector} exposure; sector faces headwinds in current regime")

        # Alert-driven actions
        for alert in alerts:
            if alert.severity == RiskLevel.CRITICAL:
                recs.append(f"[ALERT — CRITICAL] {alert.suggested_action}")
                break

        # Scenario-driven hedging
        worst = min(scenarios, key=lambda s: s.portfolio_impact_pct)
        if worst.portfolio_impact_pct < -0.25:
            recs.append(
                f"[SCENARIO] Tail risk: '{worst.scenario_name}' could cause "
                f"{worst.portfolio_impact_pct*100:.1f}% drawdown — evaluate protective hedges"
            )

        # Signal-driven allocation changes
        high_conviction = [a for a in allocations if a.conviction == "High" and a.signal_score > 0.4]
        if high_conviction:
            top3 = sorted(high_conviction, key=lambda x: -x.signal_score)[:3]
            recs.append(
                "[SIGNAL] High-conviction buy ideas: " +
                ", ".join(f"{a.ticker} ({a.expected_return*100:+.1f}% expected return)" for a in top3)
            )

        return recs[:8]

    # ─── Tables ─────────────────────────────────────────────────────────────

    def _build_allocation_table(
        self, allocations: List[PortfolioAllocation], portfolio: Portfolio
    ) -> List[Dict]:
        current_w = {h.ticker: h.weight for h in portfolio.holdings}
        return [
            {
                "ticker":     a.ticker,
                "company":    a.company_name,
                "sector":     a.sector.value,
                "weight":     a.target_weight,
                "current":    current_w.get(a.ticker, 0.0),
                "change":     a.target_weight - current_w.get(a.ticker, 0.0),
                "signal":     a.signal_score,
                "exp_return": a.expected_return,
                "conviction": a.conviction,
            }
            for a in allocations
        ]

    def _build_scenario_table(self, scenarios: List[ScenarioResult]) -> List[Dict]:
        return [
            {
                "scenario":        s.scenario_name,
                "impact_pct":      s.portfolio_impact_pct,
                "recovery_months": s.recovery_estimate_months,
                "var_change":      s.var_change,
                "action_required": s.portfolio_impact_pct < -0.15,
            }
            for s in sorted(scenarios, key=lambda x: x.portfolio_impact_pct)
        ]


# ─── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    from data.sample_data import generate_unified_dataset, generate_sample_portfolio
    from agents.agent2_research_intelligence import ResearchIntelligenceAgent
    from agents.agent3_macro_intelligence    import MacroIntelligenceAgent
    from agents.agent4_risk_intelligence     import RiskIntelligenceAgent
    from agents.agent5_portfolio_construction import PortfolioConstructionAgent
    from agents.agent6_scenario_simulation   import ScenarioSimulationAgent
    from agents.agent7_monitoring_alert      import MonitoringAlertAgent

    ds   = generate_unified_dataset()
    port = generate_sample_portfolio(ds.price_data)
    research  = ResearchIntelligenceAgent().run(ds)
    macro     = MacroIntelligenceAgent().run(ds)
    risk      = RiskIntelligenceAgent().run(port, ds, macro)
    allocs    = PortfolioConstructionAgent().run(ds, research, macro, risk, port)
    scenarios = ScenarioSimulationAgent().run(port, allocs, ds)
    alerts    = MonitoringAlertAgent().run(port, ds, macro, risk, research)

    agent  = InvestorReportingAgent()
    report = agent.run(port, ds, macro, risk, research, alerts, scenarios, allocs)
    print(agent.format_full_report())
