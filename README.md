# AMIS — AI-Powered Asset Management Intelligence System

A production-grade **multi-agent workflow** that automates investment research, portfolio analysis, risk monitoring, and investor reporting for an asset management firm.

---

## Architecture

```
Data Ingestion → Research Intelligence → Macro Intelligence → Risk Intelligence
                                                                      ↓
                              Investor Reporting ← Monitoring & Alerts ← Portfolio Construction
                                                                                ↓
                                                               Scenario Simulation
```

## 8-Agent Workflow

| # | Agent | Purpose | Key Outputs |
|---|-------|---------|-------------|
| 1 | **Data Ingestion** | Collects & normalizes market, financial, macro, news data | Unified dataset, quality score |
| 2 | **Research Intelligence** | Fundamental analysis, multi-factor stock scoring | Analyst ratings, price targets, EPS/margin analysis |
| 3 | **Macro Intelligence** | Regime detection (6 regimes), sector implications | Macro report, sector tilts, cross-asset signals |
| 4 | **Risk Intelligence** | VaR, Sharpe, beta, liquidity, factor exposure | Risk report, stress test results |
| 5 | **Portfolio Construction** | Signal-weighted, constraint-optimized allocation | Target allocation table, rebalancing trades |
| 6 | **Scenario Simulation** | 8 stress scenarios (GFC, COVID, recession, rate spike…) | Scenario impact table, recovery timelines |
| 7 | **Monitoring & Alerts** | Real-time 7-category surveillance | Ranked alerts with suggested actions |
| 8 | **Investor Reporting** | Full monthly/weekly investor reports | Attribution, outlook, recommendations |

---

## Project Structure

```
asset_management_system/
├── main.py                          # Main orchestrator — runs full 8-agent pipeline
├── requirements.txt
├── models/
│   └── data_models.py               # Shared data structures (40+ dataclasses)
├── data/
│   └── sample_data.py               # Realistic synthetic data generator (20 stocks)
├── agents/
│   ├── agent1_data_ingestion.py
│   ├── agent2_research_intelligence.py
│   ├── agent3_macro_intelligence.py
│   ├── agent4_risk_intelligence.py
│   ├── agent5_portfolio_construction.py
│   ├── agent6_scenario_simulation.py
│   ├── agent7_monitoring_alert.py
│   └── agent8_investor_reporting.py
├── dashboard/
│   └── index.html                   # Interactive 9-tab portfolio dashboard
└── reports/
    └── amis_results.json            # JSON output for API/dashboard integration
```

---

## Quick Start

```bash
# No external dependencies required (pure stdlib)
python main.py
```

This runs the full pipeline and outputs:
- Terminal portfolio dashboard
- Complete investor report
- Alert digest
- JSON results file (`reports/amis_results.json`)

### View the Dashboard
Open `dashboard/index.html` in any browser — no server required.

---

## Sample Output

```
╔══════════════════════════════════════════════════════════════════════╗
║  AMIS PORTFOLIO DASHBOARD  —  2026-03-15 12:51                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  Fund:   AMIS Alpha Equity Fund           NAV: $500M                ║
║  PM:     Suchit Dubey                     Positions: 20              ║
╠══════════════════════════════════════════════════════════════════════╣
║  PERFORMANCE                                                         ║
║    YTD Return:     +7.82%    Benchmark:    +8.20%   Alpha:  -0.38%  ║
╠══════════════════════════════════════════════════════════════════════╣
║  RISK METRICS                                                        ║
║    Risk Level:   High          VaR(1d):     2.94%                   ║
║    Sharpe:           0.10       Beta:        0.93                   ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## Key Features

- **Zero external dependencies** — pure Python 3.8+ stdlib for the core system
- **Human-in-the-loop design** — all AI signals are advisory; no trades auto-execute
- **Modular agents** — each agent runs independently and can be individually scheduled
- **Extensible** — swap synthetic data for real APIs (Bloomberg, Refinitiv, FRED) with minimal changes
- **Interactive dashboard** — dark-mode HTML dashboard with Chart.js visualizations

---

## Production Integration (Optional Dependencies)

```
numpy, pandas, scipy, cvxpy     # Numerical optimization
yfinance, pandas-datareader      # Live market & macro data
anthropic, openai                # LLM-powered agent reasoning
fastapi, uvicorn, pydantic       # REST API server
```

---

## Human Oversight

> All agent outputs are **advisory only**. Portfolio managers retain full decision authority. No trades are executed automatically. Every recommendation requires explicit PM approval.

---

*Built with the Claude Agent SDK · AMIS v1.0.0*
