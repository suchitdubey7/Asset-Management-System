"""
Agent 3 — Macro Intelligence Agent
=====================================
Monitors macroeconomic conditions, detects regime shifts,
identifies sector implications, and generates macro investment signals.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.data_models import (
    UnifiedDataset, MacroRegimeReport, MacroSnapshot,
    MacroRegime, Sector, Sentiment
)

logger = logging.getLogger(__name__)


class MacroIntelligenceAgent:
    """
    Agent 3: Macro Intelligence Agent
    ────────────────────────────────────
    Responsibilities:
      • Detect macroeconomic regime (expansion, contraction, stagflation, etc.)
      • Identify sector-level implications of the current regime
      • Generate asset-class investment signals
      • Flag macro risk factors for portfolio managers
    """

    AGENT_ID   = "AGENT-03-MACRO"
    AGENT_NAME = "Macro Intelligence Agent"
    VERSION    = "1.0.0"

    # Regime detection thresholds
    HIGH_INFLATION   = 3.5    # CPI YoY %
    LOW_INFLATION    = 1.5
    HIGH_RATES       = 4.5    # Fed Funds Rate %
    RECESSION_GDP    = 0.0    # GDP growth threshold
    INVERTED_CURVE   = 0.0    # 10Y-2Y spread
    HIGH_VIX         = 25.0
    STRESS_VIX       = 35.0
    HIGH_HY_SPREAD   = 450    # bps
    STRONG_ISM       = 52.0

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.latest_report: Optional[MacroRegimeReport] = None
        logger.info(f"[{self.AGENT_NAME}] Initialized (v{self.VERSION})")

    # ─── Public API ─────────────────────────────────────────────────────────

    def run(self, dataset: UnifiedDataset) -> MacroRegimeReport:
        """
        Execute macro analysis pipeline:
          1. Detect current macroeconomic regime
          2. Score regime confidence
          3. Identify sector implications
          4. Generate cross-asset investment signals
          5. Compile macro report
        """
        logger.info(f"[{self.AGENT_NAME}] Running macro analysis …")
        snap = dataset.macro_snapshot

        regime, confidence = self._detect_regime(snap)
        inflation_trend   = self._assess_inflation(snap)
        rate_outlook      = self._assess_rates(snap)
        growth_outlook    = self._assess_growth(snap)
        sector_implications = self._map_sector_implications(regime, snap)
        risk_factors      = self._identify_macro_risks(snap)
        signals           = self._generate_investment_signals(regime, snap)
        narrative         = self._write_narrative(regime, snap, inflation_trend,
                                                   rate_outlook, growth_outlook)

        report = MacroRegimeReport(
            generated_at=datetime.now(),
            current_regime=regime,
            regime_confidence=confidence,
            inflation_trend=inflation_trend,
            interest_rate_outlook=rate_outlook,
            growth_outlook=growth_outlook,
            sector_implications=sector_implications,
            risk_factors=risk_factors,
            investment_signals=signals,
            narrative=narrative,
        )

        self.latest_report = report
        logger.info(f"[{self.AGENT_NAME}] Macro regime: {regime.value} "
                    f"(confidence {confidence:.0%})")
        return report

    # ─── Regime Detection ───────────────────────────────────────────────────

    def _detect_regime(self, snap: MacroSnapshot):
        """
        Classify the current macro regime using a rules-based scoring engine.
        Rules incorporate: GDP growth, inflation, yield curve, credit spreads, ISM.
        """
        scores: Dict[MacroRegime, float] = {r: 0.0 for r in MacroRegime}

        # ── GDP Growth signal ────────────────────────────────────────────────
        gdp_ann = snap.gdp_growth_qoq * 4  # annualize
        if gdp_ann >= 0.03:
            scores[MacroRegime.EXPANSION] += 2.0
            scores[MacroRegime.RECOVERY]  += 1.0
        elif gdp_ann >= 0.01:
            scores[MacroRegime.EXPANSION] += 1.0
            scores[MacroRegime.PEAK]      += 1.0
        elif gdp_ann >= 0:
            scores[MacroRegime.PEAK]       += 1.5
        else:
            scores[MacroRegime.CONTRACTION]+= 2.5
            scores[MacroRegime.TROUGH]     += 0.5

        # ── Inflation signal ─────────────────────────────────────────────────
        if snap.cpi_yoy >= self.HIGH_INFLATION:
            if gdp_ann < 0.01:
                scores[MacroRegime.STAGFLATION] += 2.5
            else:
                scores[MacroRegime.PEAK]        += 1.5
        elif snap.cpi_yoy >= 2.0:
            scores[MacroRegime.EXPANSION]       += 0.5
        else:
            scores[MacroRegime.RECOVERY]        += 1.0
            scores[MacroRegime.TROUGH]          += 0.5

        # ── Yield Curve signal ───────────────────────────────────────────────
        if snap.yield_curve_spread < self.INVERTED_CURVE:
            scores[MacroRegime.PEAK]            += 1.5
            scores[MacroRegime.CONTRACTION]     += 1.0
        elif snap.yield_curve_spread > 0.5:
            scores[MacroRegime.EXPANSION]       += 1.0
            scores[MacroRegime.RECOVERY]        += 1.0

        # ── Credit spread signal ─────────────────────────────────────────────
        if snap.hy_spread > self.HIGH_HY_SPREAD:
            scores[MacroRegime.CONTRACTION]     += 1.5
            scores[MacroRegime.TROUGH]          += 0.5
        else:
            scores[MacroRegime.EXPANSION]       += 0.5

        # ── ISM / Activity signal ────────────────────────────────────────────
        if snap.ism_manufacturing >= self.STRONG_ISM:
            scores[MacroRegime.EXPANSION]       += 1.0
        elif snap.ism_manufacturing < 48:
            scores[MacroRegime.CONTRACTION]     += 0.8

        # ── VIX / Risk signal ────────────────────────────────────────────────
        if snap.vix >= self.STRESS_VIX:
            scores[MacroRegime.CONTRACTION]     += 1.0
            scores[MacroRegime.TROUGH]          += 1.0
        elif snap.vix >= self.HIGH_VIX:
            scores[MacroRegime.PEAK]            += 0.5

        # Select dominant regime
        best = max(scores, key=lambda r: scores[r])
        total = sum(scores.values())
        confidence = scores[best] / max(total, 0.01)
        return best, round(min(confidence, 0.95), 4)

    # ─── Assessment Modules ─────────────────────────────────────────────────

    def _assess_inflation(self, snap: MacroSnapshot) -> str:
        cpi = snap.cpi_yoy
        pce = snap.pce_yoy
        if cpi >= self.HIGH_INFLATION:
            return (f"Persistently elevated — CPI at {cpi:.1f}% YoY ({pce:.1f}% PCE). "
                    f"Above Fed target; risk of re-acceleration warrants caution.")
        if cpi >= 2.5:
            return (f"Moderating but sticky — CPI at {cpi:.1f}% ({pce:.1f}% PCE). "
                    f"Progress toward 2% target ongoing; rate cut path uncertain.")
        if cpi >= self.LOW_INFLATION:
            return (f"Benign — CPI at {cpi:.1f}% ({pce:.1f}% PCE). "
                    f"Within Fed tolerance; policy flexibility returning.")
        return (f"Deflationary risk — CPI at {cpi:.1f}% ({pce:.1f}% PCE). "
                f"Demand weakness may require easing.")

    def _assess_rates(self, snap: MacroSnapshot) -> str:
        ffr  = snap.fed_funds_rate
        y10  = snap.us_10y_yield
        y2   = snap.us_2y_yield
        spread = y10 - y2
        curve_desc = "inverted" if spread < 0 else "normal"
        if ffr >= self.HIGH_RATES:
            return (f"Restrictive policy — Fed Funds at {ffr:.2f}%; 10Y at {y10:.2f}%. "
                    f"Yield curve {curve_desc} ({spread:+.2f}%). Rate cuts conditional on inflation normalization.")
        if ffr >= 3.0:
            return (f"Modestly tight — Fed Funds at {ffr:.2f}%; 10Y at {y10:.2f}%. "
                    f"Yield curve {curve_desc} ({spread:+.2f}%). Easing cycle possible if data cooperates.")
        return (f"Accommodative — Fed Funds at {ffr:.2f}%; 10Y at {y10:.2f}%. "
                f"Yield curve {curve_desc} ({spread:+.2f}%). Stimulus effect still permeating economy.")

    def _assess_growth(self, snap: MacroSnapshot) -> str:
        gdp_ann = snap.gdp_growth_qoq * 4
        unemp   = snap.unemployment_rate * 100
        ism_m   = snap.ism_manufacturing
        ism_s   = snap.ism_services
        if gdp_ann >= 0.03:
            return (f"Solid expansion — GDP tracking ~{gdp_ann*100:.1f}% annualized. "
                    f"Unemployment {unemp:.1f}%. ISM Mfg {ism_m:.1f} / Svcs {ism_s:.1f}.")
        if gdp_ann >= 0.01:
            return (f"Modest growth — GDP ~{gdp_ann*100:.1f}% annualized; resilient but losing momentum. "
                    f"Unemployment {unemp:.1f}%.")
        if gdp_ann >= 0:
            return (f"Stalling — GDP barely positive at {gdp_ann*100:.1f}%. "
                    f"Unemployment edging up to {unemp:.1f}%.")
        return (f"Contraction — GDP {gdp_ann*100:.1f}% annualized. "
                f"Unemployment rising ({unemp:.1f}%). Recession risk elevated.")

    # ─── Sector Implications ────────────────────────────────────────────────

    # Regime → sector tilt lookup
    REGIME_SECTOR_MAP: Dict[MacroRegime, Dict[str, str]] = {
        MacroRegime.EXPANSION: {
            "Technology":              "Overweight",
            "Consumer Discretionary":  "Overweight",
            "Industrials":             "Overweight",
            "Financials":              "Overweight",
            "Communication Services":  "Neutral",
            "Healthcare":              "Neutral",
            "Materials":               "Overweight",
            "Energy":                  "Neutral",
            "Consumer Staples":        "Underweight",
            "Utilities":               "Underweight",
            "Real Estate":             "Neutral",
        },
        MacroRegime.PEAK: {
            "Energy":                  "Overweight",
            "Materials":               "Overweight",
            "Financials":              "Neutral",
            "Technology":              "Neutral",
            "Consumer Discretionary":  "Underweight",
            "Industrials":             "Neutral",
            "Communication Services":  "Underweight",
            "Healthcare":              "Neutral",
            "Consumer Staples":        "Overweight",
            "Utilities":               "Overweight",
            "Real Estate":             "Underweight",
        },
        MacroRegime.CONTRACTION: {
            "Healthcare":              "Overweight",
            "Consumer Staples":        "Overweight",
            "Utilities":               "Overweight",
            "Technology":              "Underweight",
            "Consumer Discretionary":  "Underweight",
            "Financials":              "Underweight",
            "Industrials":             "Underweight",
            "Energy":                  "Neutral",
            "Materials":               "Underweight",
            "Communication Services":  "Neutral",
            "Real Estate":             "Neutral",
        },
        MacroRegime.TROUGH: {
            "Consumer Discretionary":  "Overweight",
            "Technology":              "Overweight",
            "Financials":              "Overweight",
            "Industrials":             "Overweight",
            "Healthcare":              "Neutral",
            "Consumer Staples":        "Neutral",
            "Energy":                  "Neutral",
            "Materials":               "Overweight",
            "Communication Services":  "Neutral",
            "Utilities":               "Underweight",
            "Real Estate":             "Overweight",
        },
        MacroRegime.STAGFLATION: {
            "Energy":                  "Overweight",
            "Materials":               "Overweight",
            "Consumer Staples":        "Overweight",
            "Healthcare":              "Overweight",
            "Utilities":               "Neutral",
            "Technology":              "Underweight",
            "Consumer Discretionary":  "Underweight",
            "Financials":              "Underweight",
            "Industrials":             "Underweight",
            "Communication Services":  "Underweight",
            "Real Estate":             "Underweight",
        },
        MacroRegime.RECOVERY: {
            "Technology":              "Overweight",
            "Consumer Discretionary":  "Overweight",
            "Financials":              "Overweight",
            "Industrials":             "Overweight",
            "Materials":               "Overweight",
            "Real Estate":             "Overweight",
            "Healthcare":              "Neutral",
            "Communication Services":  "Neutral",
            "Consumer Staples":        "Underweight",
            "Utilities":               "Underweight",
            "Energy":                  "Neutral",
        },
    }

    def _map_sector_implications(self, regime: MacroRegime, snap: MacroSnapshot) -> Dict[str, str]:
        base = self.REGIME_SECTOR_MAP.get(regime, {})
        # Rate-sensitivity override
        if snap.us_10y_yield > 4.5:
            base["Utilities"]     = "Underweight"
            base["Real Estate"]   = "Underweight"
        if snap.oil_price_wti > 90:
            base["Energy"]        = "Overweight"
        return base

    # ─── Investment Signals ─────────────────────────────────────────────────

    def _generate_investment_signals(self, regime: MacroRegime, snap: MacroSnapshot) -> Dict[str, float]:
        """Generate cross-asset signal scores (-1 to +1) for asset classes."""
        signals: Dict[str, float] = {}

        # Equities
        if regime in (MacroRegime.EXPANSION, MacroRegime.RECOVERY):
            signals["US Equities"]        = +0.70
            signals["International Equities"] = +0.50
            signals["EM Equities"]        = +0.45
        elif regime in (MacroRegime.PEAK, MacroRegime.STAGFLATION):
            signals["US Equities"]        = -0.20
            signals["International Equities"] = -0.10
            signals["EM Equities"]        = -0.35
        else:  # Contraction / Trough
            signals["US Equities"]        = -0.50
            signals["EM Equities"]        = -0.60

        # Fixed Income
        if snap.us_10y_yield > 4.5:
            signals["US Treasuries"]      = +0.40
            signals["Investment Grade"]   = +0.25
        else:
            signals["US Treasuries"]      = +0.10
        signals["High Yield"]             = -0.30 if snap.hy_spread > self.HIGH_HY_SPREAD else +0.10

        # Real Assets
        signals["Gold"]      = +0.30 if snap.cpi_yoy > self.HIGH_INFLATION else 0.0
        signals["Oil/Energy"]= +0.40 if snap.oil_price_wti > 85 else -0.10
        signals["TIPS"]      = +0.35 if snap.cpi_yoy > 3.0 else -0.10

        # Cash
        signals["Cash"]      = +0.60 if regime == MacroRegime.CONTRACTION else +0.10

        return {k: round(v, 4) for k, v in signals.items()}

    # ─── Risk Factors ───────────────────────────────────────────────────────

    def _identify_macro_risks(self, snap: MacroSnapshot) -> List[str]:
        risks = []
        if snap.cpi_yoy > self.HIGH_INFLATION:
            risks.append(f"Sticky inflation ({snap.cpi_yoy:.1f}% CPI) limits Fed flexibility")
        if snap.yield_curve_spread < 0:
            risks.append(f"Inverted yield curve ({snap.yield_curve_spread:+.2f}%) — historical recession predictor")
        if snap.hy_spread > self.HIGH_HY_SPREAD:
            risks.append(f"Elevated high-yield spreads ({snap.hy_spread:.0f}bps) signal credit stress")
        if snap.vix > self.HIGH_VIX:
            risks.append(f"Elevated equity volatility (VIX {snap.vix:.1f}) — risk-off sentiment")
        if snap.oil_price_wti > 95:
            risks.append(f"High oil prices (${snap.oil_price_wti:.0f}/bbl) risk stagflation feedback")
        if snap.dxy_index > 108:
            risks.append(f"Strong USD (DXY {snap.dxy_index:.1f}) pressures EM and US multinationals")
        if snap.ism_manufacturing < 47:
            risks.append(f"Manufacturing PMI in contraction territory ({snap.ism_manufacturing:.1f})")
        if not risks:
            risks.append("No acute macro risks identified; standard cyclical risks apply")
        return risks

    # ─── Narrative ──────────────────────────────────────────────────────────

    def _write_narrative(
        self, regime: MacroRegime, snap: MacroSnapshot,
        inflation: str, rates: str, growth: str
    ) -> str:
        regime_intros = {
            MacroRegime.EXPANSION:   "The economy remains in solid expansion mode, supporting risk assets broadly.",
            MacroRegime.PEAK:        "We appear to be near the peak of the cycle; selectivity is increasingly important.",
            MacroRegime.CONTRACTION: "Economic contraction is underway; defensive positioning and quality tilt advised.",
            MacroRegime.TROUGH:      "Indicators suggest we may be approaching a trough; early-cycle positioning attractive.",
            MacroRegime.STAGFLATION: "Stagflationary conditions — high inflation with stagnant growth — present the most challenging backdrop.",
            MacroRegime.RECOVERY:    "Recovery dynamics are building; cyclical sectors and risk assets offer opportunity.",
        }
        intro = regime_intros.get(regime, "Macro environment is mixed.")
        return (
            f"MACRO REGIME: {regime.value.upper()}\n\n"
            f"{intro}\n\n"
            f"INFLATION: {inflation}\n\n"
            f"MONETARY POLICY / RATES: {rates}\n\n"
            f"GROWTH: {growth}\n\n"
            f"KEY MACRO INDICATORS:\n"
            f"  CPI (YoY): {snap.cpi_yoy:.2f}%\n"
            f"  Fed Funds:  {snap.fed_funds_rate:.2f}%\n"
            f"  10Y Yield:  {snap.us_10y_yield:.2f}%\n"
            f"  Yield Curve:{snap.yield_curve_spread:+.2f}%\n"
            f"  GDP (QoQ):  {snap.gdp_growth_qoq*100:.2f}% ({snap.gdp_growth_qoq*400:.1f}% ann.)\n"
            f"  VIX:        {snap.vix:.1f}\n"
            f"  HY Spread:  {snap.hy_spread:.0f}bps\n"
            f"  Oil (WTI):  ${snap.oil_price_wti:.1f}/bbl"
        )


# ─── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    from data.sample_data import generate_unified_dataset
    ds = generate_unified_dataset()
    agent = MacroIntelligenceAgent()
    report = agent.run(ds)
    print(f"\n✅ Macro analysis complete")
    print(f"   Regime:     {report.current_regime.value} ({report.regime_confidence:.0%})")
    print(f"   Inflation:  {report.inflation_trend[:60]}…")
    print(f"   Signals:    {list(report.investment_signals.items())[:4]}")
    print(f"\n   Sector tilts (first 5):")
    for s, tilt in list(report.sector_implications.items())[:5]:
        print(f"   {s:30s}: {tilt}")
