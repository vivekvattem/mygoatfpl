# FPL AI Predictor

FPL AI Predictor is a data-driven Fantasy Premier League decision engine. Its long-term purpose is to optimize expected points, transfers, starting XI selection, captaincy, bench order, Wildcards, and multi-Gameweek strategy for a real FPL squad.

> The system aims to maximize expected Fantasy Premier League points. It cannot guarantee maximum realized points because football outcomes contain significant uncertainty.

## Current Status

**Phase 1 — Data Engine**

The current release downloads the official FPL bootstrap and fixture datasets, archives every refresh, creates clean player and team-perspective fixture tables, engineers transparent per-90 and value features, and summarizes official fixture difficulty over the next three and five Gameweeks.

The `attacking_score` is deliberately a simple, explainable baseline:

```text
4.0 × xG/90 + 3.0 × xA/90 + 0.01 × threat/90 + 0.005 × creativity/90
```

It is not an expected-points prediction model. No ML model, optimizer, squad import, UI, scheduling, or AI explanation layer is implemented in Phase 1.

## Architecture

```text
API ingestion → clean data → feature engineering → prediction
              → optimization → recommendations → UI / AI explanation
```

Each layer is kept replaceable. Current transformations fetch once and operate locally. The future historical dataset will use one row per player per Gameweek, with every predictive feature restricted to information available before the target Gameweek to prevent leakage.

## Setup

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/refresh_data.py
pytest
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

For an editable development installation, use `pip install -e .`.

## Generated Data

One refresh creates current snapshots and timestamped archives under `data/raw/`, plus:

```text
data/processed/players.csv
data/processed/fixtures.csv
data/processed/fixture_summary_3gw.csv
data/processed/fixture_summary_5gw.csv
```

The fixture table has one row for each team's perspective. This naturally preserves Blank and Double Gameweeks instead of assuming one fixture per team per week.

## Tests

The test suite is fully offline. It covers mocked API success and failure, optional player fields, dynamic team and position mappings, numeric coercion, zero-minute per-90 behavior, value and attacking formulas, fixture normalization, forecast horizons, and Double Gameweeks.

## Roadmap

- Phase 1 — Current FPL data ingestion and baseline analytics
- Phase 2 — Historical Gameweek dataset
- Phase 3 — Leakage-safe rolling features and team-strength model
- Phase 4 — Expected-points ML model
- Phase 5 — Personalized squad import
- Phase 6 — Transfer / XI / captain optimizer
- Phase 7 — Streamlit dashboard
- Phase 8 — AI analyst
- Phase 9 — Automated weekly evaluation and model monitoring

Future transfer recommendations will compare multi-Gameweek expected gains after hit costs and legal FPL constraints. If no legal move clears the improvement threshold, the recommendation will explicitly be `ROLL TRANSFER`.
