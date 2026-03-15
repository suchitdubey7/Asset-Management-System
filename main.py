"""
AI-Powered Asset Management Intelligence System (AMIS)
=======================================================
Main Orchestrator — coordinates all 8 agents in the correct
workflow sequence and produces the final system output.

Workflow:
  1  DataIngestion        → UnifiedDataset
  2  ResearchIntelligence → ResearchSummaries
  3  MacroIntelligence    → MacroRegimeReport
  4  RiskIntelligence     → RiskReport
  5  PortfolioConstruction→ AllocationTable
  6  ScenarioSimulation   → ScenarioResults
  7  MonitoringAlert      → Alerts
  8  InvestorReporting    → InvestorReport

Human oversight layer: PM reviews all agent outputs before action.
"""

import logging
import json
import time
from datetime import datetime
from typing import Dict, Optional

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from models.data_models import (
    UnifiedDataset, Portfolio, InvestorReport, RiskLevel
)
from data.sample_data import generate_sample_portfolio
from agents import (
    DataIngestionAgent,
    ResearchIntelligenceAgent,
    MacroIntelligenceAgent,
    RiskIntelligenceAgent,
    PortfolioConstructionAgent,
    ScenarioSimulationAgent,
    MonitoringAlertAgent,
    InvestorReportingAgent,
)

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AMIS")


# ─── Orchestrator ─────────────────────────────────────────────────────────────

class AssetManagementIntelligenceSystem:
    """
    Top-level orchestrator for the AMIS multi-agent workflow.

    Each agent is instantiated once and can be called repeatedly (e.g. every
    minute for the ingestion agent, every hour for research, etc.).  In this
    demonstration we run the full pipeline once and print a complete report.

    Human oversight:
      • All agent outputs are available to the PM via the dashboard
      • No trades are executed automatically — the system provides
        decision support, not autonomous execution
    """

    SYSTEM_NAME    = "AI-Powered Asset Management Intelligence System"
    SYSTEM_VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.run_id : Optional[str] = None
        self.results: Dict = {}

        # Instantiate all agents
        self.agent1 = DataIngestionAgent()
        self.agent2 = ResearchIntelligenceAgent()
        self.agent3 = MacroIntelligenceAgent()
        self.agent4 = RiskIntelligenceAgent()
        self.agent5 = PortfolioConstructionAgent()
        self.agent6 = ScenarioSimulationAgent()
        self.agent7 = MonitoringAlertAgent()
        self.agent8 = InvestorReportingAgent()

        logger.info(f"{'='*60}")
        logger.info(f"  {self.SYSTEM_NAME}")
        logger.info(f"  Version {self.SYSTEM_VERSION}")
        logger.info(f"{'='*60}")

    # ─── Full Pipeline ───────────────────────────────────────────────────────

    def run_full_pipeline(
        self,
        portfolio: Optional[Portfolio] = None,
        reporting_period: str = "Monthly",
    ) -> Dict:
        """
        Execute the complete 8-agent workflow.

        Args:
            portfolio:        Current portfolio (generated synthetically if None)
            reporting_period: "Monthly" or "Weekly"

        Returns:
            Dictionary containing all agent outputs.
        """
        self.run_id = datetime.now().strftime("RUN-%Y%m%d-%H%M%S")
        start_time  = time.time()
        logger.info(f"Starting full pipeline — {self.run_id}")

        # ── STAGE 1: Data Ingestion ──────────────────────────────────────────
        logger.info("▶  Stage 1 / 8 — Data Ingestion …")
        t0 = time.time()
        dataset: UnifiedDataset = self.agent1.run()
        logger.info(f"   ✓ Ingested {len(dataset.price_data)} tickers in {time.time()-t0:.1f}s")

        # Build portfolio from ingested price data if not provided
        if portfolio is None:
            portfolio = generate_sample_portfolio(dataset.price_data)
            logger.info(f"   ✓ Portfolio: {portfolio.portfolio_name} | "
                        f"NAV: ${portfolio.total_nav/1e6:.0f}M | "
                        f"{len(portfolio.holdings)} holdings")

        # ── STAGE 2: Research Intelligence ──────────────────────────────────
        logger.info("▶  Stage 2 / 8 — Research Intelligence …")
        t0 = time.time()
        research = self.agent2.run(dataset)
        top_picks = self.agent2.get_top_picks(3)
        logger.info(f"   ✓ {len(research)} research summaries in {time.time()-t0:.1f}s")
        logger.info(f"   ✓ Top picks: {', '.join(s.ticker for s in top_picks)}")

        # ── STAGE 3: Macro Intelligence ──────────────────────────────────────
        logger.info("▶  Stage 3 / 8 — Macro Intelligence …")
        t0 = time.time()
        macro = self.agent3.run(dataset)
        logger.info(f"   ✓ Regime: {macro.current_regime.value} "
                    f"({macro.regime_confidence:.0%} confidence) in {time.time()-t0:.1f}s")

        # ── STAGE 4: Risk Intelligence ───────────────────────────────────────
        logger.info("▶  Stage 4 / 8 — Risk Intelligence …")
        t0 = time.time()
        risk = self.agent4.run(portfolio, dataset, macro)
        logger.info(f"   ✓ Risk level: {risk.overall_risk_level.value} | "
                    f"VaR(1d): {risk.portfolio_var_1d:.2%} | "
                    f"Sharpe: {risk.sharpe_ratio:.2f} in {time.time()-t0:.1f}s")

        # ── STAGE 5: Portfolio Construction ──────────────────────────────────
        logger.info("▶  Stage 5 / 8 — Portfolio Construction …")
        t0 = time.time()
        allocations = self.agent5.run(dataset, research, macro, risk, portfolio)
        changes     = self.agent5.get_changes_vs_current(portfolio, allocations)
        logger.info(f"   ✓ {len(allocations)} positions proposed | "
                    f"{len(changes)} rebalancing trades in {time.time()-t0:.1f}s")

        # ── STAGE 6: Scenario Simulation ──────────────────────────────────────
        logger.info("▶  Stage 6 / 8 — Scenario Simulation …")
        t0 = time.time()
        scenarios = self.agent6.run(portfolio, allocations, dataset)
        worst     = min(scenarios, key=lambda s: s.portfolio_impact_pct)
        logger.info(f"   ✓ {len(scenarios)} scenarios simulated | "
                    f"Worst: {worst.scenario_name[:35]} ({worst.portfolio_impact_pct:.1%}) "
                    f"in {time.time()-t0:.1f}s")

        # ── STAGE 7: Monitoring & Alerts ──────────────────────────────────────
        logger.info("▶  Stage 7 / 8 — Monitoring & Alerts …")
        t0 = time.time()
        alerts = self.agent7.run(portfolio, dataset, macro, risk, research)
        summary = self.agent7.get_alert_summary()
        logger.info(f"   ✓ {summary['total_alerts']} alerts generated "
                    f"({summary['critical']} CRITICAL, {summary['high']} HIGH) "
                    f"in {time.time()-t0:.1f}s")

        # ── STAGE 8: Investor Reporting ───────────────────────────────────────
        logger.info("▶  Stage 8 / 8 — Investor Reporting …")
        t0 = time.time()
        report = self.agent8.run(
            portfolio, dataset, macro, risk, research,
            alerts, scenarios, allocations, reporting_period
        )
        logger.info(f"   ✓ Report {report.report_id} | "
                    f"Return: {report.portfolio_return:.2%} | "
                    f"Alpha: {report.alpha:+.2%} in {time.time()-t0:.1f}s")

        # ── Compile final output ──────────────────────────────────────────────
        elapsed = time.time() - start_time
        self.results = {
            "run_id":         self.run_id,
            "timestamp":      datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "dataset":        dataset,
            "portfolio":      portfolio,
            "research":       research,
            "macro_report":   macro,
            "risk_report":    risk,
            "allocations":    allocations,
            "scenarios":      scenarios,
            "alerts":         alerts,
            "investor_report": report,
            "rebalancing":    changes,
        }

        logger.info(f"{'='*60}")
        logger.info(f"  Pipeline complete in {elapsed:.1f}s — {self.run_id}")
        logger.info(f"{'='*60}")
        return self.results

    # ─── Output Formatters ───────────────────────────────────────────────────

    def print_full_report(self):
        """Print the complete formatted investor report."""
        print(self.agent8.format_full_report())

    def print_alert_digest(self):
        """Print all active alerts."""
        print(self.agent7.format_alert_digest())

    def print_portfolio_dashboard(self):
        """Print a concise portfolio dashboard."""
        if not self.results:
            print("No results yet — run the pipeline first.")
            return
        r    = self.results
        port = r["portfolio"]
        risk = r["risk_report"]
        macro= r["macro_report"]
        rep  = r["investor_report"]

        lines = [
            "",
            "╔══════════════════════════════════════════════════════════════════════╗",
            f"║  AMIS PORTFOLIO DASHBOARD  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}{'':>16}║",
            "╠══════════════════════════════════════════════════════════════════════╣",
            f"║  Fund:   {port.portfolio_name:<32} NAV: ₹{port.total_nav/1e6:.0f}M       ║",
            f"║  PM:     {port.portfolio_manager:<32} Positions: {len(port.holdings):<10}║",
            "╠══════════════════════════════════════════════════════════════════════╣",
            "║  PERFORMANCE                                                         ║",
            f"║    YTD Return:   {rep.portfolio_return*100:>+7.2f}%    Benchmark:  {rep.benchmark_return*100:>+7.2f}%   Alpha: {rep.alpha*100:>+6.2f}%  ║",
            "╠══════════════════════════════════════════════════════════════════════╣",
            "║  RISK METRICS                                                        ║",
            f"║    Risk Level:   {risk.overall_risk_level.value:<12}  VaR(1d):   {risk.portfolio_var_1d*100:>6.2f}%             ║",
            f"║    Sharpe:       {risk.sharpe_ratio:>8.2f}       Beta:      {risk.beta_to_benchmark:>6.2f}             ║",
            f"║    Sortino:      {risk.sortino_ratio:>8.2f}       Liquidity: {risk.liquidity_score:>6.2f}             ║",
            "╠══════════════════════════════════════════════════════════════════════╣",
            "║  MACRO ENVIRONMENT                                                   ║",
            f"║    Regime: {macro.current_regime.value:<20} Confidence: {macro.regime_confidence:.0%}              ║",
            f"║    CPI: {r['dataset'].macro_snapshot.cpi_yoy:.1f}%   Repo: {r['dataset'].macro_snapshot.fed_funds_rate:.2f}%   10Y: {r['dataset'].macro_snapshot.us_10y_yield:.2f}%   VIX: {r['dataset'].macro_snapshot.vix:.1f}  ║",
            "╠══════════════════════════════════════════════════════════════════════╣",
            "║  ALERTS                                                              ║",
        ]
        alert_summary = self.agent7.get_alert_summary()
        lines.append(
            f"║    Total: {alert_summary['total_alerts']:<4}  "
            f"Critical: {alert_summary['critical']:<4}  "
            f"High: {alert_summary['high']:<4}  "
            f"Moderate: {alert_summary['moderate']:<4}  "
            f"Low: {alert_summary['low']:<4}               ║"
        )
        lines += [
            "╠══════════════════════════════════════════════════════════════════════╣",
            "║  TOP SECTOR EXPOSURES                                                ║",
        ]
        for s, w in list(risk.sector_exposures.items())[:5]:
            bar = "█" * int(w * 50)
            lines.append(f"║    {s[:22]:<22} {w*100:>5.1f}%  {bar:<25}              ║")
        lines += [
            "╠══════════════════════════════════════════════════════════════════════╣",
            "║  SCENARIO STRESS — WORST CASES                                       ║",
        ]
        for sc in sorted(r["scenarios"], key=lambda x: x.portfolio_impact_pct)[:4]:
            flag = "⚠️" if sc.portfolio_impact_pct < -0.15 else "  "
            lines.append(f"║  {flag} {sc.scenario_name[:42]:<42} {sc.portfolio_impact_pct*100:>7.1f}%    ║")
        lines.append("╚══════════════════════════════════════════════════════════════════════╝")
        print("\n".join(lines))

    def export_results_json(self, filepath: str = None) -> str:
        """Export key metrics to JSON (suitable for dashboard consumption)."""
        if not self.results:
            return "{}"
        r    = self.results
        port = r["portfolio"]
        risk = r["risk_report"]
        macro= r["macro_report"]
        rep  = r["investor_report"]
        snap = r["dataset"].macro_snapshot

        output = {
            "run_id":        r["run_id"],
            "timestamp":     r["timestamp"],
            "portfolio": {
                "name":      port.portfolio_name,
                "nav":       port.total_nav,
                "n_positions": len(port.holdings),
                "cash_weight": port.cash_weight,
            },
            "performance": {
                "ytd_return":    rep.portfolio_return,
                "benchmark":     rep.benchmark_return,
                "alpha":         rep.alpha,
            },
            "risk": {
                "level":         risk.overall_risk_level.value,
                "var_1d":        risk.portfolio_var_1d,
                "var_5d":        risk.portfolio_var_5d,
                "sharpe":        risk.sharpe_ratio,
                "sortino":       risk.sortino_ratio,
                "beta":          risk.beta_to_benchmark,
                "liquidity":     risk.liquidity_score,
                "max_drawdown":  risk.max_drawdown_ytd,
                "sector_exposures": risk.sector_exposures,
                "factor_exposures": risk.factor_exposures,
                "stress_tests":  risk.stress_test_results,
            },
            "macro": {
                "regime":        macro.current_regime.value,
                "confidence":    macro.regime_confidence,
                "cpi":           snap.cpi_yoy,
                "fed_rate":      snap.fed_funds_rate,
                "yield_10y":     snap.us_10y_yield,
                "yield_curve":   snap.yield_curve_spread,
                "vix":           snap.vix,
                "hy_spread":     snap.hy_spread,
                "oil_wti":       snap.oil_price_wti,
                "sector_tilts":  macro.sector_implications,
                "signals":       macro.investment_signals,
            },
            "alerts": {
                "total":    len(r["alerts"]),
                "critical": len([a for a in r["alerts"] if a.severity == RiskLevel.CRITICAL]),
                "high":     len([a for a in r["alerts"] if a.severity == RiskLevel.HIGH]),
                "list": [
                    {"type": a.alert_type.value, "severity": a.severity.value,
                     "title": a.title, "action": a.suggested_action}
                    for a in r["alerts"][:10]
                ],
            },
            "allocations": [
                {"ticker": a.ticker, "company": a.company_name, "sector": a.sector.value,
                 "weight": a.target_weight, "signal": a.signal_score,
                 "exp_return": a.expected_return, "conviction": a.conviction}
                for a in r["allocations"]
            ],
            "scenarios": [
                {"name": s.scenario_name, "impact": s.portfolio_impact_pct,
                 "recovery_months": s.recovery_estimate_months}
                for s in sorted(r["scenarios"], key=lambda x: x.portfolio_impact_pct)
            ],
            "research": {
                ticker: {
                    "rating": s.analyst_rating, "target": s.price_target,
                    "upside": s.upside_downside, "confidence": s.confidence_score,
                    "revenue_trend": s.revenue_trend[:80],
                }
                for ticker, s in list(r["research"].items())[:20]
            },
            "rebalancing": r["rebalancing"][:10],
            "report_id": rep.report_id,
            "recommended_actions": rep.recommended_actions,
        }

        json_str = json.dumps(output, indent=2, default=str)
        if filepath:
            with open(filepath, "w") as f:
                f.write(json_str)
            logger.info(f"Results exported to {filepath}")
        return json_str


# ─── Main Entry Point ────────────────────────────────────────────────────────

def main():
    print("\n" + "═"*72)
    print("  AI-POWERED ASSET MANAGEMENT INTELLIGENCE SYSTEM (AMIS)")
    print("  Starting full multi-agent workflow …")
    print("═"*72 + "\n")

    system  = AssetManagementIntelligenceSystem()
    results = system.run_full_pipeline(reporting_period="Monthly")

    print("\n")
    system.print_portfolio_dashboard()

    print("\n\n")
    system.print_full_report()

    print("\n\n")
    system.print_alert_digest()

    # Export JSON for dashboard
    json_path = os.path.join(os.path.dirname(__file__), "reports", "amis_results.json")
    system.export_results_json(json_path)
    print(f"\n📊 JSON results exported → {json_path}")

    return results


if __name__ == "__main__":
    results = main()
