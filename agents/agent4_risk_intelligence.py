"""
Agent 4 — Risk Intelligence Agent
=====================================
Continuously monitors portfolio risk — concentration, factor exposure,
liquidity, and tail risk — generating comprehensive risk reports.
"""

import logging
import math
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.data_models import (
    Portfolio, UnifiedDataset, MacroRegimeReport, RiskReport, RiskLevel,
    Sector, Holding
)

logger = logging.getLogger(__name__)


class RiskIntelligenceAgent:
    """
    Agent 4: Risk Intelligence Agent
    ──────────────────────────────────
    Responsibilities:
      • Calculate sector concentration and exposure limits
      • Evaluate factor exposures (momentum, quality, value, beta)
      • Detect liquidity risk and estimate liquidation horizon
      • Run multi-scenario stress tests
      • Compute VaR, drawdown, Sharpe, Sortino metrics
      • Flag vulnerabilities and recommend risk mitigants
    """

    AGENT_ID   = "AGENT-04-RISK"
    AGENT_NAME = "Risk Intelligence Agent"
    VERSION    = "1.0.0"

    # Risk thresholds
    MAX_SECTOR_CONCENTRATION    = 0.35   # flag if single sector > 35%
    MAX_SINGLE_POSITION         = 0.10   # flag if single stock > 10%
    MAX_TOP10_CONCENTRATION     = 0.75   # flag if top-10 > 75%
    MIN_LIQUIDITY_SCORE         = 0.60   # flag if portfolio liquidity < 60%
    MAX_BETA                    = 1.30   # flag if portfolio beta > 1.3
    TARGET_SHARPE               = 1.0
    HIGH_TRACKING_ERROR         = 0.06   # 6%

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.latest_report: Optional[RiskReport] = None
        logger.info(f"[{self.AGENT_NAME}] Initialized (v{self.VERSION})")

    # ─── Public API ─────────────────────────────────────────────────────────

    def run(
        self,
        portfolio: Portfolio,
        dataset: UnifiedDataset,
        macro_report: Optional[MacroRegimeReport] = None,
    ) -> RiskReport:
        """
        Execute the full risk analysis pipeline:
          1. Compute sector concentrations
          2. Evaluate factor exposures
          3. Assess liquidity
          4. Run stress tests
          5. Compute portfolio risk metrics
          6. Identify vulnerabilities
        """
        logger.info(f"[{self.AGENT_NAME}] Running risk analysis for {portfolio.portfolio_name} …")

        sector_exp   = self._compute_sector_exposure(portfolio)
        factor_exp   = self._compute_factor_exposure(portfolio, dataset)
        liq_score, days_to_liq = self._assess_liquidity(portfolio, dataset)
        stress_tests = self._run_stress_tests(portfolio, dataset, macro_report)
        var_1d       = self._compute_var(portfolio, dataset, horizon=1)
        var_5d       = self._compute_var(portfolio, dataset, horizon=5)
        sharpe       = self._compute_sharpe(portfolio, dataset)
        sortino      = self._compute_sortino(portfolio, dataset)
        beta         = self._compute_portfolio_beta(portfolio)
        te           = self._compute_tracking_error(portfolio, dataset)
        max_dd       = self._estimate_max_drawdown(portfolio, dataset)
        top10_wgt    = sum(sorted([h.weight for h in portfolio.holdings], reverse=True)[:10])
        vulnerabilities, recommendations = self._identify_vulnerabilities(
            sector_exp, factor_exp, liq_score, var_1d, beta, top10_wgt, stress_tests
        )
        overall_risk = self._overall_risk_level(var_1d, beta, liq_score, vulnerabilities)

        report = RiskReport(
            portfolio_id=portfolio.portfolio_id,
            generated_at=datetime.now(),
            overall_risk_level=overall_risk,
            portfolio_var_1d=var_1d,
            portfolio_var_5d=var_5d,
            max_drawdown_ytd=max_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            beta_to_benchmark=beta,
            tracking_error=te,
            sector_exposures=sector_exp,
            top_10_holdings_weight=round(top10_wgt, 4),
            factor_exposures=factor_exp,
            liquidity_score=liq_score,
            days_to_liquidate_90pct=days_to_liq,
            stress_test_results=stress_tests,
            vulnerabilities=vulnerabilities,
            recommendations=recommendations,
        )

        self.latest_report = report
        logger.info(f"[{self.AGENT_NAME}] Risk analysis complete — "
                    f"Overall risk: {overall_risk.value}, VaR(1d): {var_1d:.2%}")
        return report

    # ─── Sector Exposure ────────────────────────────────────────────────────

    def _compute_sector_exposure(self, portfolio: Portfolio) -> Dict[str, float]:
        sector_weights: Dict[str, float] = {}
        for h in portfolio.holdings:
            sector_weights[h.sector.value] = sector_weights.get(h.sector.value, 0) + h.weight
        return {k: round(v, 4) for k, v in sorted(sector_weights.items(), key=lambda x: -x[1])}

    # ─── Factor Exposure ────────────────────────────────────────────────────

    def _compute_factor_exposure(self, portfolio: Portfolio, dataset: UnifiedDataset) -> Dict[str, float]:
        """
        Compute exposure to standard risk factors:
          Market Beta | Momentum | Quality | Value | Size | Low Volatility
        """
        exposures: Dict[str, float] = {}

        # Market Beta (weighted average of individual betas)
        betas = [h.beta * h.weight for h in portfolio.holdings]
        exposures["Market Beta"] = round(sum(betas), 4)

        # Momentum factor (weight-average of 3M returns)
        mom_scores = []
        for h in portfolio.holdings:
            pd = dataset.price_data.get(h.ticker)
            if pd:
                mom_scores.append(pd.returns_3m * h.weight)
        exposures["Momentum"] = round(sum(mom_scores) if mom_scores else 0.0, 4)

        # Quality factor (ROIC-based proxy)
        quality_scores = []
        for h in portfolio.holdings:
            fm = dataset.financial_metrics.get(h.ticker)
            if fm:
                quality_scores.append(fm.roic * h.weight)
        exposures["Quality (ROIC)"] = round(sum(quality_scores) if quality_scores else 0.0, 4)

        # Value factor (1/PE weighted)
        value_scores = []
        for h in portfolio.holdings:
            fm = dataset.financial_metrics.get(h.ticker)
            if fm and fm.pe_ratio > 0 and not math.isnan(fm.pe_ratio):
                value_scores.append((1 / fm.pe_ratio) * h.weight)
        exposures["Value (1/PE)"] = round(sum(value_scores) if value_scores else 0.0, 4)

        # Size (log market cap)
        size_scores = []
        for h in portfolio.holdings:
            fm = dataset.financial_metrics.get(h.ticker)
            if fm and fm.market_cap > 0:
                size_scores.append(math.log(fm.market_cap) * h.weight)
        if size_scores:
            raw = sum(size_scores)
            exposures["Size (log MCap)"] = round(raw / 30, 4)  # normalize

        # Low-vol factor (inverse of volatility weighted)
        lvol_scores = []
        for h in portfolio.holdings:
            pd = dataset.price_data.get(h.ticker)
            if pd and pd.volatility_30d > 0:
                lvol_scores.append((1 / pd.volatility_30d) * h.weight)
        if lvol_scores:
            exposures["Low Volatility"] = round(sum(lvol_scores) / 10, 4)

        return exposures

    # ─── Liquidity ──────────────────────────────────────────────────────────

    def _assess_liquidity(self, portfolio: Portfolio, dataset: UnifiedDataset) -> Tuple[float, float]:
        """
        Compute weighted-average liquidity score and estimated days to
        liquidate 90% of the portfolio.
        """
        weighted_liq = sum(h.liquidity_score * h.weight for h in portfolio.holdings)
        weighted_liq += portfolio.cash_weight * 1.0  # cash = perfectly liquid

        # Days to liquidate 90%: assume can sell 20% of avg daily volume/day
        total_days = 0.0
        for h in portfolio.holdings:
            pd = dataset.price_data.get(h.ticker)
            if pd and pd.volume > 0:
                daily_vol_usd = pd.volume * pd.close * 0.20  # 20% participation
                holding_mv    = h.market_value
                days = max(1.0, holding_mv / daily_vol_usd)
                total_days += days * h.weight
            else:
                total_days += 5.0 * h.weight  # assume 5 days if no data

        return round(weighted_liq, 4), round(total_days, 1)

    # ─── Stress Tests ───────────────────────────────────────────────────────

    STRESS_SCENARIOS = {
        "2008 Global Financial Crisis":  {"equity_shock": -0.50, "vix_mult": 4.0, "credit_shock": 0.08, "description": "Severe systemic crisis"},
        "COVID-19 Shock (Mar 2020)":      {"equity_shock": -0.34, "vix_mult": 3.5, "credit_shock": 0.05, "description": "Pandemic-induced market dislocation"},
        "2022 Rate Spike":                {"equity_shock": -0.20, "vix_mult": 1.8, "credit_shock": 0.02, "description": "Rapid Fed tightening cycle"},
        "Global Recession (-3% GDP)":     {"equity_shock": -0.35, "vix_mult": 2.5, "credit_shock": 0.04, "description": "Deep recession scenario"},
        "Oil Price Shock (+60%)":         {"equity_shock": -0.12, "vix_mult": 1.6, "credit_shock": 0.02, "description": "Geopolitical energy supply disruption"},
        "Interest Rate Spike (+200bps)":  {"equity_shock": -0.18, "vix_mult": 1.5, "credit_shock": 0.015, "description": "Sudden hawkish reprice"},
        "USD Currency Crisis (+15% DXY)": {"equity_shock": -0.08, "vix_mult": 1.3, "credit_shock": 0.01, "description": "EM contagion and USD flight-to-safety"},
        "Tech Selloff (-35%)":            {"equity_shock": -0.15, "vix_mult": 2.0, "credit_shock": 0.01, "description": "Valuation compression in growth stocks"},
    }

    def _run_stress_tests(
        self, portfolio: Portfolio, dataset: UnifiedDataset,
        macro_report: Optional[MacroRegimeReport]
    ) -> Dict[str, float]:
        results = {}
        for scenario, params in self.STRESS_SCENARIOS.items():
            impact = self._simulate_scenario_impact(portfolio, dataset, params)
            results[scenario] = round(impact, 4)
        return results

    def _simulate_scenario_impact(self, portfolio: Portfolio, dataset, params: Dict) -> float:
        """
        Estimate portfolio P&L impact under a stress scenario.
        Uses beta-adjusted equity shock + sector-specific sensitivities.
        """
        SECTOR_SENSITIVITIES = {
            "Technology":              1.40,
            "Consumer Discretionary":  1.30,
            "Financials":              1.25,
            "Communication Services":  1.20,
            "Industrials":             1.10,
            "Materials":               1.05,
            "Energy":                  0.90,
            "Healthcare":              0.80,
            "Consumer Staples":        0.65,
            "Utilities":               0.55,
            "Real Estate":             0.95,
        }
        equity_shock = params["equity_shock"]
        total_impact = 0.0

        for h in portfolio.holdings:
            sector_sens = SECTOR_SENSITIVITIES.get(h.sector.value, 1.0)
            beta_adj    = h.beta * sector_sens
            position_impact = h.weight * equity_shock * beta_adj
            total_impact += position_impact

        # Cash is a buffer
        total_impact *= (1 - portfolio.cash_weight)
        return total_impact

    # ─── Portfolio Metrics ──────────────────────────────────────────────────

    def _compute_var(self, portfolio: Portfolio, dataset: UnifiedDataset, horizon: int) -> float:
        """Historical-simulation-proxy VaR at 95% confidence."""
        weighted_vol = 0.0
        for h in portfolio.holdings:
            pd = dataset.price_data.get(h.ticker)
            if pd:
                weighted_vol += (pd.volatility_30d / math.sqrt(252)) * h.weight
        # Annualized daily vol; scale to horizon; 1.645 = 95th percentile Z
        var = weighted_vol * math.sqrt(horizon) * 1.645
        return round(var, 4)

    def _compute_sharpe(self, portfolio: Portfolio, dataset: UnifiedDataset) -> float:
        weighted_ret = sum(
            dataset.price_data[h.ticker].returns_ytd * h.weight
            for h in portfolio.holdings
            if h.ticker in dataset.price_data
        )
        weighted_vol = sum(
            dataset.price_data[h.ticker].volatility_30d * h.weight
            for h in portfolio.holdings
            if h.ticker in dataset.price_data
        )
        risk_free = 0.05  # approximate T-bill rate
        return round((weighted_ret - risk_free) / max(weighted_vol, 0.001), 4)

    def _compute_sortino(self, portfolio: Portfolio, dataset: UnifiedDataset) -> float:
        weighted_ret = sum(
            dataset.price_data[h.ticker].returns_ytd * h.weight
            for h in portfolio.holdings
            if h.ticker in dataset.price_data
        )
        # Downside deviation ≈ vol * 0.707 (assumes ~50% of vol is downside)
        weighted_vol = sum(
            dataset.price_data[h.ticker].volatility_30d * h.weight * 0.707
            for h in portfolio.holdings
            if h.ticker in dataset.price_data
        )
        risk_free = 0.05
        return round((weighted_ret - risk_free) / max(weighted_vol, 0.001), 4)

    def _compute_portfolio_beta(self, portfolio: Portfolio) -> float:
        return round(sum(h.beta * h.weight for h in portfolio.holdings), 4)

    def _compute_tracking_error(self, portfolio: Portfolio, dataset: UnifiedDataset) -> float:
        # Proxy: dispersion of individual stock volatilities vs equal-weight
        vols = [dataset.price_data[h.ticker].volatility_30d
                for h in portfolio.holdings if h.ticker in dataset.price_data]
        if len(vols) < 2:
            return 0.05
        mean_vol = sum(vols) / len(vols)
        variance = sum((v - mean_vol) ** 2 for v in vols) / len(vols)
        return round(math.sqrt(variance) * math.sqrt(12), 4)  # annualize

    def _estimate_max_drawdown(self, portfolio: Portfolio, dataset: UnifiedDataset) -> float:
        weighted_worst = sum(
            min(0, dataset.price_data[h.ticker].returns_ytd) * h.weight
            for h in portfolio.holdings
            if h.ticker in dataset.price_data
        )
        return round(weighted_worst, 4)

    # ─── Risk Flags ─────────────────────────────────────────────────────────

    def _identify_vulnerabilities(
        self, sector_exp, factor_exp, liq_score, var_1d, beta, top10_wgt, stress
    ) -> Tuple[List[str], List[str]]:
        vulnerabilities = []
        recommendations = []

        for sector, weight in sector_exp.items():
            if weight > self.MAX_SECTOR_CONCENTRATION:
                vulnerabilities.append(
                    f"Sector concentration: {sector} at {weight*100:.1f}% — exceeds {self.MAX_SECTOR_CONCENTRATION*100:.0f}% limit")
                recommendations.append(f"Reduce {sector} exposure by ~{(weight - self.MAX_SECTOR_CONCENTRATION)*100:.1f}pp")

        if top10_wgt > self.MAX_TOP10_CONCENTRATION:
            vulnerabilities.append(
                f"Top-10 holdings represent {top10_wgt*100:.1f}% of NAV — idiosyncratic risk elevated")
            recommendations.append("Broaden portfolio diversification across additional names")

        if beta > self.MAX_BETA:
            vulnerabilities.append(
                f"High beta ({beta:.2f}x) — portfolio amplifies market downturns")
            recommendations.append("Add low-beta defensive names to reduce market sensitivity")

        if liq_score < self.MIN_LIQUIDITY_SCORE:
            vulnerabilities.append(
                f"Liquidity score {liq_score:.2f} below minimum {self.MIN_LIQUIDITY_SCORE} threshold")
            recommendations.append("Replace illiquid positions with more liquid equivalents")

        worst_stress = min(stress.values())
        if worst_stress < -0.30:
            scenario = min(stress, key=stress.get)
            vulnerabilities.append(
                f"Severe tail risk: {scenario} could cause {worst_stress*100:.1f}% portfolio loss")
            recommendations.append("Consider tail-risk hedges (put spreads, VIX calls, or defensive allocations)")

        if var_1d > 0.025:
            vulnerabilities.append(f"Daily VaR {var_1d*100:.2f}% is elevated — potential for large daily swings")
            recommendations.append("Review position sizing; ensure largest positions have stop-loss disciplines")

        if not vulnerabilities:
            vulnerabilities.append("No critical risk flags detected — portfolio within acceptable bounds")
            recommendations.append("Continue monitoring; standard monthly risk review schedule appropriate")

        return vulnerabilities, recommendations

    def _overall_risk_level(self, var_1d, beta, liq_score, vulnerabilities) -> RiskLevel:
        critical_count = sum(1 for v in vulnerabilities if "concentration" in v.lower() or "tail risk" in v.lower())
        if var_1d > 0.040 or beta > 1.5 or critical_count >= 3:
            return RiskLevel.CRITICAL
        if var_1d > 0.025 or beta > 1.3 or critical_count >= 2:
            return RiskLevel.HIGH
        if var_1d > 0.015 or beta > 1.1 or critical_count >= 1:
            return RiskLevel.MODERATE
        return RiskLevel.LOW


# ─── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    from data.sample_data import generate_unified_dataset, generate_sample_portfolio
    ds   = generate_unified_dataset()
    port = generate_sample_portfolio(ds.price_data)
    agent = RiskIntelligenceAgent()
    report = agent.run(port, ds)
    print(f"\n✅ Risk analysis complete")
    print(f"   Overall risk: {report.overall_risk_level.value}")
    print(f"   VaR (1d 95%): {report.portfolio_var_1d:.2%}")
    print(f"   Sharpe Ratio: {report.sharpe_ratio:.2f}")
    print(f"   Beta:         {report.beta_to_benchmark:.2f}")
    print(f"   Liq. Score:   {report.liquidity_score:.2f}")
    print(f"\n   Top sector exposures:")
    for s, w in list(report.sector_exposures.items())[:5]:
        print(f"   {s:30s}: {w*100:.1f}%")
    print(f"\n   Stress Tests (worst 3):")
    worst = sorted(report.stress_test_results.items(), key=lambda x: x[1])[:3]
    for s, imp in worst:
        print(f"   {s:45s}: {imp*100:.1f}%")
