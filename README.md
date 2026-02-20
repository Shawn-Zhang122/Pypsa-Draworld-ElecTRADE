# Pypsa-China-ElecTRADE-Draworld

Daily zonal electricity market simulation for China  
Rolling welfare-maximizing dispatch with automated publication

---

## Overview

Pypsa-China-ElecTRADE-Draworld is a reproducible day-ahead (D-1) market clearing engine built on PyPSA.  
It performs rolling optimization, exports results, and publishes them via a static web interface.

The system separates:

- **Model year (data structure):** 2025 representative system year  
- **Publication year (timestamp):** Real calendar year (e.g. 2026 and later)

This preserves physical consistency while maintaining operational relevance.

---

## Model Design

- 33 provincial zones
- Linear welfare maximization
- 48h rolling horizon, 24h publication
- Interzonal transmission constraints
- Storage intertemporal optimization
- Locational marginal pricing (zonal aggregation)

Outputs include:

- Hourly zonal prices
- Interzonal flows
- Storage SOC
- Full solved PyPSA network (`.nc`)
- Others

---

## Web Interface

- Interactive zonal map (MapLibre)
- Hourly price curves (Plotly)
- Storage visualization (Plotly)
- Attribution labeling (Draworld / PyPSA / MapLibre)

---

## Applications

The platform can support:

- Congestion price analysis  
- Coal flexibility diagnostics  
- Renewable integration stress tests  
- Storage arbitrage demonstration  
- Interprovincial trade evaluation  
- Market design comparison 

Potential extension in 2020/2030s:

- Electricity trading sandbox  
- Training and simulation environment  
- Policy workshop demonstration tool  
- Strategic bidding experiments  


## License
AGPL-3.0, Aligned with the open modeling ecosystem of PyPSA.

---
## Statement

Pypsa-China-ElecTRADE-Draworld is a research, toy-electricity-trade-game and training dispatch engine. It is not an official trading system or regulatory platform.
