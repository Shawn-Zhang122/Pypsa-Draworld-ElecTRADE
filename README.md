# China-Daily-ElecTRADE

China Day-Ahead Zonal Electricity Market Simulation  
(PCR-style continuous welfare maximization)

## What this is
- A **daily (D-1)** national electricity market clearing simulation
- **Zonal pricing** (province as zone)
- **Batch computation + GitHub-driven publication**
- Inspired by PyPSA/nowcast and European PCR practice

## What this is NOT
- Not a real trading platform
- Not a regulatory system
- Not real-time dispatch

## Core idea
> Git is the database, the scheduler, and the publication record.

## Main components
- `model/` — clearing engine (PyPSA initially)
- `data/snapshots/` — versioned published results
- `web/` — static map frontend (MapLibre)
- `.github/workflows/` — daily automation

## License
AGPL-3.0 (same spirit as PyPSA/nowcast)
