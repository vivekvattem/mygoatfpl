# FPL AI Predictor

FPL AI Predictor is a data-driven Fantasy Premier League decision engine. Its long-term purpose is to optimize expected points, transfers, starting XI selection, captaincy, bench order, Wildcards, and multi-Gameweek strategy for a real FPL squad.

> The system aims to maximize expected Fantasy Premier League points. It cannot guarantee maximum realized points because football outcomes contain significant uncertainty.

## Current Status

**Phase 2 — Historical Dataset**

Phase 1's current-data engine remains intact. Phase 2 adds a season-aware supervised-learning dataset with one row per player × Gameweek, leakage-safe player/team/opponent rolling form, fixture context, explicit Blank and Double Gameweek handling, targets, validation, and chronological split utilities. It does not train an ML model.

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

Each layer is kept replaceable. Current and historical ingestion are separate, and historical IDs are never joined directly to current API IDs. The historical dataset uses one row per player per Gameweek, with every outcome-derived predictor restricted to information available before the target Gameweek.

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

Build the historical dataset with:

```bash
.venv/bin/python scripts/build_historical_dataset.py
```

## Generated Data

One refresh creates current snapshots and timestamped archives under `data/raw/`, plus:

```text
data/processed/players.csv
data/processed/fixtures.csv
data/processed/fixture_summary_3gw.csv
data/processed/fixture_summary_5gw.csv
```

The fixture table has one row for each team's perspective. This naturally preserves Blank and Double Gameweeks instead of assuming one fixture per team per week.

## Historical Data Source

Historical data comes from [Vaastav Anand's Fantasy Premier League historical dataset](https://github.com/vaastav/Fantasy-Premier-League), specifically each season's `gws/merged_gw.csv`, `fixtures.csv`, `teams.csv`, and `players_raw.csv`. Phase 2 successfully ingests these completed seasons:

- 2022/23
- 2023/24
- 2024/25
- 2025/26

The configured list lives in `config.py`. The downloader caches exact source CSVs under `data/historical/raw/<season>/`; the `HistoricalSource` boundary makes the provider replaceable. Source differences are normalized without fabricating unavailable fields. Historical FPL element IDs are season-scoped as values such as `2024-25_123`; no fuzzy cross-season identity matching is attempted.

The source documents that its scraped `xP` field may reflect post-match updates. This project excludes `xP` entirely rather than allowing it into the training feature set.

## Historical Dataset Semantics

Source fixture records are first preserved and then aggregated to one player × Gameweek row. In a Double Gameweek, performance counts and expected statistics are summed while fixture count, home/away counts, and mean/min/max official FDR are retained. Mixed home/away doubles have an undefined single `is_home` value and retain the explicit counts instead.

Missing player-GW rows inside a player's active season range are reconciled with the official team fixture schedule. Genuine team blanks receive `fixture_count = 0` and `did_not_play_because_team_blank = true`. Their zero target remains distinguishable from a player who scored zero while their team had a fixture.

All player rolling features use `groupby(player-season).shift(1).rolling(...)`. Team and opponent form use the same shift-before-roll rule within season and team. Aggregated per-90 figures divide prior-window sums by prior-window minutes; zero-minute denominators become missing values, never infinity. Form is not carried across seasons.

The ML output explicitly partitions identifiers, features, and `target_points` / `target_minutes`. Same-GW outcomes such as minutes, points, goals, assists, xG, bonus, and BPS are targets/source data only—not predictors. The build fails on suspicious unlagged outcomes, duplicates, infinities, invalid positions/GWs/prices, implausible targets, or populated GW1 rolling features.

Generated historical artifacts are:

```text
data/historical/processed/player_gameweeks_<season>.csv
data/historical/ml/player_gameweek_dataset.csv
data/historical/ml/dataset_summary.json
```

These generated files are intentionally ignored by Git because the ML CSV is large. `chronological_split` assigns whole seasons to train/validation/test sets; it never randomizes rows. An expanding-window evaluator can be added in Phase 3.

## Tests

The test suite is fully offline. In addition to Phase 1, it covers source replacement, season-scoped identity, player-GW normalization, Blank/Double Gameweeks, rolling 1/3/5 windows, explicit target-change leakage regression, zero-minute per-90 behavior, team/opponent lags, chronological splitting, target separation, and dataset validation.

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
