# FPL AI Predictor

FPL AI Predictor is a data-driven Fantasy Premier League decision engine. Its long-term purpose is to optimize expected points, transfers, starting XI selection, captaincy, bench order, Wildcards, and multi-Gameweek strategy for a real FPL squad.

> The system aims to maximize expected Fantasy Premier League points. It cannot guarantee maximum realized points because football outcomes contain significant uncertainty.

## Current Status

**Phase 5 — Personalized Live Inference**

Phases 1–4 remain intact. Phase 5 connects the frozen position-specific Ridge model to the current official FPL season, validates live/training feature parity, predicts every current player, and optionally imports a public FPL entry and its current 15-player squad. It does not optimize transfers, lineups, or captaincy.

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

Run the Phase 3 experiment with:

```bash
.venv/bin/python scripts/run_baselines.py
```

Train and evaluate Phase 4 separately so the final test season remains isolated:

```bash
.venv/bin/python scripts/train_models.py
.venv/bin/python scripts/evaluate_models.py
```

Generate current player predictions without an entry ID:

```bash
.venv/bin/python scripts/live_predictions.py
```

Generate a personalized report using the numeric ID visible in an FPL team URL:

```bash
.venv/bin/python scripts/live_squad_report.py --entry-id 123456
```

Alternatively set `FPL_ENTRY_ID` in `.env`; the CLI argument takes precedence.

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

These generated files are intentionally ignored by Git because the ML CSV is large. Temporal split utilities assign whole seasons to train/validation/test sets and provide expanding-window folds; they never randomize rows.

## Phase 3 Evaluation

Random splitting is prohibited because it can train on future football outcomes and evaluate on the past. The default holdout is:

```text
Train:      2022/23 + 2023/24
Validation: 2024/25
Test:       2025/26
```

For final test reporting, fitted benchmarks are retrained on train plus validation, while 2025/26 targets remain untouched. Expanding-window utilities additionally generate the three season-by-season folds needed for honest backtesting.

The machine-readable [feature registry](config/features.yaml) expands into one concrete provenance record per feature. Every record includes its category, dtype, source, timing, window, positional applicability, and description. Predictors must be registered as `pre_gw`; identifiers, targets, excluded diagnostic columns, and unknown fields are rejected before fitting.

The evaluated models are Previous GW points, prior-three and prior-five Gameweek means, minutes-adjusted recent form, training-only position mean, training-only price-plus-position regression, and a fixed-alpha Ridge benchmark. Ridge uses named basic, form, minutes, fixture, value, and team-strength groups rather than blindly consuming every available column.

Numeric missing values are median-imputed inside the fitted pipeline, so validation/test values cannot influence imputation. Categorical values use training-set most-frequent imputation and unknown-safe one-hot encoding. Important history gaps also have explicit `xGI_per90_last_3_missing` and `minutes_last_3_missing` flags. Transparent baselines leave unavailable histories missing and are scored only where predictions exist.

Team attack strength is prior-fixture goals scored per match. Defense strength is prior-fixture goals conceded per match; therefore a league-relative defensive value above `1.0` means a weaker defense that concedes more than average. Windows of 3, 5, and 10 fixtures, league-relative variants, and simple home/away splits are calculated strictly from Gameweeks before the target. Opponent ratings retain mean/min/max aggregation across all Double Gameweek opponents.

Evaluation reports aggregate and position-specific MAE, RMSE, Spearman correlation, Gameweek-level Top-10/25/50 precision and recall, plus top-pick points and regret. “Top pick” is not called captain accuracy because it ignores squad membership and ownership.

Generated Phase 3 reports are:

```text
data/historical/ml/baseline_results.csv
data/historical/ml/baseline_results_by_position.csv
data/historical/ml/baseline_results_by_gw.csv
data/historical/ml/phase3_summary.json
```

## Tests

The test suite is fully offline. In addition to Phases 1–2, it covers holdout/expanding temporal order, non-destructive eligibility masks, baseline formulas, training-only fitted baselines, missing-value pipelines, feature timing, Ridge execution, target leakage, shifted team ratings, Double Gameweek opponent aggregation, regression/ranking metrics, and top-pick regret.

Current limitations include source-snapshot timing for historic prices/FDR, no expected-minutes model, no external xG source, modest home/away sample sizes early in a season, no cross-season player matching, and no squad-aware decision optimization.

## Phase 4 Expected-Points Models

Phase 4 compares fixed, conservative Ridge, Random Forest, and `HistGradientBoostingRegressor` pipelines. XGBoost is intentionally omitted: the scikit-learn benchmarks cover the nonlinear comparison without adding a third-party dependency. Numeric features use training-only median imputation, categorical features use training-only most-frequent imputation and unknown-safe one-hot encoding, and every input must pass the `pre_gw` registry gate.

Model/feature selection uses 2024/25 only. The weighted review score is documented as 30% MAE, 25% Spearman, 15% Top-25 precision, 20% NDCG@25, and 10% top-pick regret, after direction-aware min-max normalization. Raw metrics remain available for manual review. The frozen winner is then refitted on 2022/23–2024/25 and evaluated once on 2025/26.

Expanding-window OOF predictions cover 2023/24, 2024/25, and 2025/26. Every row records its fold and latest training season, making future-row contamination auditable. Feature ablations remove fixtures, team strength, value, minutes, three-GW form, or five-GW form. A transparent expected-minutes proxy combines only lagged three/five-GW minutes, last-GW minutes, and lagged start rates.

The selected production configuration is position-specific Ridge with `feature_set_form`. It won the validation multi-metric selection despite HistGradientBoosting having slightly lower validation MAE. Model binaries and metadata are stored under `models/` and ignored by Git.

Calibration is reported in fixed `<2`, `2–4`, `4–6`, `6–8`, and `8+` xPts bands. Uncertainty uses position-specific 10th/90th percentile residuals learned from pre-test OOF predictions; ranges are empirical diagnostics, not probabilistic guarantees. Residual reports segment errors by position, price, minutes history, venue, FDR, prediction band, Gameweek, and season stage.

Phase 4 outputs include `model_results*.csv`, `oof_predictions.csv`, `calibration_results.csv`, `residual_analysis.csv`, `test_predictions_with_uncertainty.csv`, and `phase4_summary.json` under `data/historical/ml/`.

Current limitations remain substantial: predictions are compressed toward the mean, high-ceiling recall is weak, uncertainty is empirical rather than conditional/probabilistic, historic price/FDR timing is snapshot-dependent, and the expected-minutes proxy is not a dedicated availability model.

## Phase 5 Live Inference

Live inference uses only official JSON endpoints: `bootstrap-static/`, `fixtures/`, `event/{gw}/live/`, `entry/{id}/`, `entry/{id}/event/{gw}/picks/`, and `entry/{id}/history/`. Completed Gameweek live payloads are normalized to player × Gameweek rows; target rows are then appended and passed through the same Phase 2 shift-before-roll implementation. Target-GW outcomes never enter rolling features.

The persisted metadata is authoritative. All required feature names must exist with compatible dtypes before a position model is loaded. Missing or mismatched inputs fail loudly. Sparse early-season rolling fields remain missing and are handled by the model's training-fitted imputers; they are not silently filled with zero.

The lightweight minutes proxy uses only lagged one/three/five-GW minutes and start rates. `minutes_confidence` describes history depth and published availability uncertainty. Official `status`, `chance_of_playing_next_round`, and `news` are preserved. When FPL publishes a percentage, displayed xPts are multiplied by that percentage; when it does not, no probability is invented and raw display xPts remain unchanged.

Raw Ridge predictions are preserved. User-facing xPts and lower uncertainty bounds are floored at zero. Position-specific empirical residual ranges come from Phase 4 pre-test OOF residuals and are descriptive rather than guaranteed probabilities.

Public picks preserve purchase price, selling price, current price, multiplier, captain/vice flags, and bench order. The public API does not reliably expose the current free-transfer balance, so it is reported as unknown. No credentials are stored.

Current xPts estimates are optimized for average expected-points ranking and are not yet reliable estimates of explosive 10+ point outcomes. This is especially important for future captaincy decisions.

Live outputs are `data/live/player_predictions.csv`, optional `my_squad.csv`, and `live_summary.json`. The refreshed Phase 1 fixture summaries continue to provide next-three and next-five-Gameweek context; Phase 5 does not fabricate future availability or apply the one-GW model repeatedly to future dates.

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
