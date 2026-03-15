"""
Agent 6 — Scenario Simulation Agent
=======================================
Stress-tests the proposed portfolio across multiple adverse scenarios,
providing granular impact analysis at portfolio, sector, and position level.
"""

import logging
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.data_models import (
    Portfolio, PortfolioAllocation, UnifiedDataset, ScenarioResult, RiskLevel
)

logger = logging.getLogger(__name__)


class ScenarioSimulationAgent:
    """
    Agent 6: Scenario Simulation Agent
    ─────────────────────────────────────
    Responsibilities:
      • Run structured stress tests across historical and hypothetical scenarios
      • Decompose impact by sector and individual holding
      • Estimate recovery timelines
      • Identify scenario-specific key risk drivers
      • Flag scenarios requiring immediate portfolio action
    """

    AGENT_ID   = "AGENT-06-SCENARIO"
    AGENT_NAME = "Scenario Simulation Agent"
    VERSION    = "1.0.0"

    ACTION_THRESHOLD = -0.15    # scenarios with >15% loss trigger action flag

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.results: List[ScenarioResult] = []
        logger.info(f"[{self.AGENT_NAME}] Initialized (v{self.VERSION})")

    # ─── Scenario Definitions ────────────────────────────────────────────────

    SCENARIOS = {
        "Global Recession (-3% GDP)": {
            "description": "Severe global recession: GDP contracts 3%, unemployment rises to 8%, corporate earnings fall 30%.",
            "sector_shocks": {
                "Technology":              -0.38,
                "Consumer Discretionary":  -0.45,
                "Financials":              -0.40,
                "Industrials":             -0.35,
                "Materials":               -0.32,
                "Energy":                  -0.28,
                "Communication Services":  -0.30,
                "Healthcare":              -0.18,
                "Consumer Staples":        -0.12,
                "Utilities":               -0.10,
                "Real Estate":             -0.25,
            },
            "equity_market_shock": -0.35,
            "vix_level": 45,
            "recovery_months": 18,
        },
        "Oil Price Shock (+60% in 3 months)": {
            "description": "Geopolitical supply disruption drives oil to ~$125/bbl, stoking inflation and stagflation fears.",
            "sector_shocks": {
                "Energy":                  +0.25,
                "Materials":               +0.08,
                "Consumer Staples":        -0.08,
                "Consumer Discretionary":  -0.18,
                "Industrials":             -0.14,
                "Technology":              -0.10,
                "Financials":              -0.12,
                "Healthcare":              -0.05,
                "Utilities":               -0.08,
                "Real Estate":             -0.10,
                "Communication Services":  -0.08,
            },
            "equity_market_shock": -0.12,
            "vix_level": 30,
            "recovery_months": 9,
        },
        "Interest Rate Spike (+200bps)": {
            "description": "Fed forced to hike 200bps unexpectedly; bond yields spike, duration assets reprice sharply.",
            "sector_shocks": {
                "Real Estate":             -0.28,
                "Utilities":               -0.22,
                "Consumer Staples":        -0.12,
                "Technology":              -0.18,
                "Communication Services":  -0.15,
                "Consumer Discretionary":  -0.12,
                "Healthcare":              -0.08,
                "Financials":              +0.05,
                "Energy":                  +0.02,
                "Industrials":             -0.10,
                "Materials":               -0.08,
            },
            "equity_market_shock": -0.18,
            "vix_level": 32,
            "recovery_months": 12,
        },
        "2008 Global Financial Crisis Replay": {
            "description": "Systemic banking crisis triggers credit freeze, 50% equity market selloff, and global deleveraging.",
            "sector_shocks": {
                "Financials":              -0.65,
                "Real Estate":             -0.55,
                "Consumer Discretionary":  -0.50,
                "Industrials":             -0.45,
                "Materials":               -0.45,
                "Technology":              -0.42,
                "Communication Services":  -0.38,
                "Energy":                  -0.40,
                "Consumer Staples":        -0.20,
                "Healthcare":              -0.22,
                "Utilities":               -0.18,
            },
            "equity_market_shock": -0.50,
            "vix_level": 80,
            "recovery_months": 48,
        },
        "COVID-19 Style Market Shock": {
            "description": "Pandemic shock: 34% market drawdown in 5 weeks, rapid policy response, V-shaped recovery.",
            "sector_shocks": {
                "Energy":                  -0.60,
                "Financials":              -0.40,
                "Industrials":             -0.38,
                "Consumer Discretionary":  -0.42,
                "Real Estate":             -0.35,
                "Materials":               -0.30,
                "Communication Services":  -0.20,
                "Technology":              -0.18,
                "Healthcare":              -0.12,
                "Consumer Staples":        -0.15,
                "Utilities":               -0.20,
            },
            "equity_market_shock": -0.34,
            "vix_level": 85,
            "recovery_months": 6,
        },
        "Tech Sector Crash (-35%)": {
            "description": "AI/Tech valuation bubble bursts; Nasdaq-style correction with multiple compression.",
            "sector_shocks": {
                "Technology":              -0.38,
                "Communication Services":  -0.30,
                "Consumer Discretionary":  -0.20,
                "Financials":              -0.10,
                "Industrials":             -0.08,
                "Materials":               -0.06,
                "Energy":                  +0.05,
                "Healthcare":              -0.05,
                "Consumer Staples":        +0.02,
                "Utilities":               +0.03,
                "Real Estate":             -0.05,
            },
            "equity_market_shock": -0.18,
            "vix_level": 40,
            "recovery_months": 24,
        },
        "USD Currency Depreciation (-15%)": {
            "description": "Dollar weakens 15% on twin-deficit concerns; benefits international earners, pressures importers.",
            "sector_shocks": {
                "Technology":              +0.05,
                "Materials":               +0.10,
                "Energy":                  +0.08,
                "Industrials":             +0.06,
                "Consumer Staples":        -0.06,
                "Consumer Discretionary":  -0.04,
                "Financials":              -0.05,
                "Healthcare":              +0.02,
                "Real Estate":             +0.03,
                "Utilities":               -0.03,
                "Communication Services":  +0.02,
            },
            "equity_market_shock": -0.05,
            "vix_level": 22,
            "recovery_months": 12,
        },
        "China-Taiwan Geopolitical Escalation": {
            "description": "Military escalation disrupts semiconductor supply chains, trade routes, and global risk appetite.",
            "sector_shocks": {
                "Technology":              -0.30,
                "Consumer Discretionary":  -0.20,
                "Industrials":             -0.15,
                "Materials":               -0.12,
                "Energy":                  +0.12,
                "Financials":              -0.15,
                "Communication Services":  -0.18,
                "Healthcare":              -0.08,
                "Consumer Staples":        -0.05,
                "Utilities":               -0.06,
                "Real Estate":             -0.10,
            },
            "equity_market_shock": -0.22,
            "vix_level": 50,
            "recovery_months": 15,
        },
    }

    # ─── Public API ─────────────────────────────────────────────────────────

    def run(
        self,
        portfolio: Portfolio,
        allocations: List[PortfolioAllocation],
        dataset: UnifiedDataset,
    ) -> List[ScenarioResult]:
        """
        Run all stress scenarios against the portfolio.
        Uses proposed allocations if available, else current holdings.
        """
        logger.info(f"[{self.AGENT_NAME}] Running {len(self.SCENARIOS)} stress scenarios …")

        results = []
        for scenario_name, params in self.SCENARIOS.items():
            result = self._simulate_scenario(
                scenario_name, params, portfolio, allocations, dataset
            )
            results.append(result)
            logger.debug(f"[{self.AGENT_NAME}] {scenario_name}: {result.portfolio_impact_pct*100:.1f}%")

        results.sort(key=lambda x: x.portfolio_impact_pct)
        self.results = results
        worst = results[0]
        logger.info(f"[{self.AGENT_NAME}] Worst scenario: {worst.scenario_name} "
                    f"({worst.portfolio_impact_pct*100:.1f}%)")
        return results

    def get_action_required_scenarios(self) -> List[ScenarioResult]:
        """Return scenarios that breach the action threshold."""
        return [r for r in self.results if r.portfolio_impact_pct < self.ACTION_THRESHOLD]

    def get_summary_table(self) -> List[Dict]:
        return [
            {
                "Scenario":         r.scenario_name,
                "Impact %":         f"{r.portfolio_impact_pct*100:.1f}%",
                "VaR Change":       f"{r.var_change*100:.1f}%",
                "Recovery (months)":r.recovery_estimate_months,
                "Action Required":  "⚠️  YES" if r.portfolio_impact_pct < self.ACTION_THRESHOLD else "No",
            }
            for r in self.results
        ]

    # ─── Simulation Engine ───────────────────────────────────────────────────

    def _simulate_scenario(
        self, name: str, params: Dict,
        portfolio: Portfolio, allocations: List[PortfolioAllocation],
        dataset: UnifiedDataset
    ) -> ScenarioResult:

        sector_shocks  = params["sector_shocks"]
        market_shock   = params["equity_market_shock"]
        vix_level      = params["vix_level"]
        recovery_months= params["recovery_months"]

        # Build effective weight vector from proposed allocations or current holdings
        weight_map: Dict[str, Tuple[float, str]] = {}  # ticker -> (weight, sector)
        if allocations:
            for a in allocations:
                weight_map[a.ticker] = (a.target_weight, a.sector.value)
        else:
            for h in portfolio.holdings:
                weight_map[h.ticker] = (h.weight, h.sector.value)

        # Per-holding impact
        holding_impacts: Dict[str, float] = {}
        sector_impacts:  Dict[str, float] = {}
        total_impact = 0.0

        for ticker, (weight, sector) in weight_map.items():
            # Individual beta adjustment
            pd   = dataset.price_data.get(ticker)
            beta = 1.0
            for h in portfolio.holdings:
                if h.ticker == ticker:
                    beta = h.beta
                    break

            sector_shock  = sector_shocks.get(sector, market_shock)
            idio_factor   = 1.0 + (beta - 1.0) * 0.5   # partial beta adjustment
            stock_impact  = sector_shock * idio_factor

            # Add idiosyncratic noise
            if pd:
                idio_noise = (pd.volatility_30d / math.sqrt(252)) * _pseudo_shock(ticker, name)
                stock_impact += idio_noise * 0.3

            holding_impacts[ticker] = round(stock_impact, 4)
            sector_impacts[sector]  = sector_impacts.get(sector, 0.0) + weight * stock_impact
            total_impact += weight * stock_impact

        # Cash dampens impact
        total_impact *= (1 - portfolio.cash_weight)

        # VaR delta estimate
        current_var = sum(
            (dataset.price_data[t].volatility_30d / math.sqrt(252)) * w * 1.645
            for t, (w, _) in weight_map.items() if t in dataset.price_data
        )
        stressed_var = current_var * (vix_level / 20)  # scale by VIX ratio

        # Key drivers
        driver_pairs = sorted(
            [(t, w * holding_impacts.get(t, 0)) for t, (w, _) in weight_map.items()],
            key=lambda x: x[1]
        )[:5]
        key_drivers = [
            f"{t}: {imp*100:.1f}% attributed impact"
            for t, imp in driver_pairs if abs(imp) > 0.001
        ]

        return ScenarioResult(
            scenario_name=name,
            description=params["description"],
            portfolio_impact_pct=round(total_impact, 4),
            sector_impacts={k: round(v, 4) for k, v in sector_impacts.items()},
            holding_impacts={k: round(v, 4) for k, v in holding_impacts.items()},
            var_change=round(stressed_var - current_var, 4),
            recovery_estimate_months=recovery_months,
            key_drivers=key_drivers,
        )


def _pseudo_shock(ticker: str, scenario: str) -> float:
    """Deterministic pseudo-random shock for idiosyncratic noise."""
    seed = sum(ord(c) for c in ticker + scenario)
    return math.sin(seed) * 0.5   # bounded -0.5 to +0.5


# ─── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    from data.sample_data import generate_unified_dataset, generate_sample_portfolio
    ds   = generate_unified_dataset()
    port = generate_sample_portfolio(ds.price_data)
    agent = ScenarioSimulationAgent()
    results = agent.run(port, [], ds)
    print(f"\n✅ Scenario simulation complete — {len(results)} scenarios")
    print(f"\n{'Scenario':<45} {'Impact %':>10} {'Recovery':>12}")
    print("-" * 70)
    for r in results:
        flag = " ⚠️" if r.portfolio_impact_pct < agent.ACTION_THRESHOLD else ""
        print(f"{r.scenario_name[:43]:<45} {r.portfolio_impact_pct*100:>9.1f}% "
              f"{r.recovery_estimate_months:>10} mo{flag}")
