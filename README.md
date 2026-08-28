# FPL AI Predictor

FPL AI Predictor is a data-driven Fantasy Premier League decision engine. Its long-term purpose is to optimize expected points, transfers, starting XI selection, captaincy, bench order, Wildcards, and multi-Gameweek strategy for a real FPL squad.

> The system aims to maximize expected Fantasy Premier League points. It cannot guarantee maximum realized points because football outcomes contain significant uncertainty.

## Current Status

**Phase 9 — AI Analyst**

Phases 1–8 remain intact. Phase 9 adds a conversational, evidence-grounded explanation layer over the existing model, optimizers, fixture engine, signals, availability, and chip planner. The analyst is not a separate prediction engine. All major results render directly in Streamlit, and the app never logs in to FPL, executes transfers, or activates chips.

The original Phase 1 `attacking_score` remains a simple, explainable baseline:

```text
4.0 × xG/90 + 3.0 × xA/90 + 0.01 × threat/90 + 0.005 × creativity/90
```

It is not the expected-points model used by the dashboard.

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

Optional analyst-provider configuration belongs in `.env` locally or uncommitted Streamlit secrets:

```text
FPL_ANALYST_PROVIDER=openai
FPL_ANALYST_API_KEY=
FPL_ANALYST_MODEL=
```

Leaving these empty disables provider calls and keeps the complete deterministic analyst available.

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

Run lineup and captaincy while leaving unknown financial state explicit:

```bash
.venv/bin/python scripts/optimize_squad.py --entry-id 8974446 --squad-file data/live/manual_squad.json
```

Run an explicitly labelled transfer scenario when bank/free transfers are known but selling prices are not:

```bash
.venv/bin/python scripts/optimize_squad.py --entry-id 8974446 --squad-file data/live/manual_squad.json \
  --bank 0.0 --free-transfers 1 --horizon 5 --risk-profile balanced \
  --assume-selling-price-current
```

## Local App

```bash
source .venv/bin/activate
streamlit run app.py
```

The app opens with entry `8974446` explicitly labelled as a demo default. Select a squad source in the sidebar, preserve bank/free transfers as **Unknown** unless authoritative values are supplied, and click **Refresh FPL Data** when a new pipeline run is wanted. Widget changes and page navigation load cached outputs; they do not call the FPL API or rerun five-Gameweek optimization.

Refresh performs one of three explicit workflows:

- a valid manual squad runs the existing live-inference and Phase 6 optimizer CLI;
- public-picks mode refreshes predictions and imports the latest public post-deadline snapshot for inspection, without treating it as a current pre-deadline squad;
- a missing manual squad still refreshes the all-player rankings, while clearly reporting that personalization was skipped.

If refresh fails, the last successful files remain visible and are marked **STALE DATA** when older than the configured TTL. Expected API, entry, squad, financial-state, artifact, and schema errors are shown as user-facing messages.

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

The selected production configuration is position-specific Ridge with `feature_set_form`. It won the validation multi-metric selection despite HistGradientBoosting having slightly lower validation MAE. The small frozen global and position-specific Ridge binaries and metadata under `models/` are intentionally committed for deterministic dashboard deployment; experimental/non-production artifacts remain ignored.

Calibration is reported in fixed `<2`, `2–4`, `4–6`, `6–8`, and `8+` xPts bands. Uncertainty uses position-specific 10th/90th percentile residuals learned from pre-test OOF predictions; ranges are empirical diagnostics, not probabilistic guarantees. Residual reports segment errors by position, price, minutes history, venue, FDR, prediction band, Gameweek, and season stage.

Phase 4 outputs include `model_results*.csv`, `oof_predictions.csv`, `calibration_results.csv`, `residual_analysis.csv`, `test_predictions_with_uncertainty.csv`, and `phase4_summary.json` under `data/historical/ml/`.

Current limitations remain substantial: predictions are compressed toward the mean, high-ceiling recall is weak, uncertainty is empirical rather than conditional/probabilistic, historic price/FDR timing is snapshot-dependent, and the expected-minutes proxy is not a dedicated availability model.

## Phase 5 Live Inference

Live inference uses only official JSON endpoints: `bootstrap-static/`, `fixtures/`, `event/{gw}/live/`, `entry/{id}/`, `entry/{id}/event/{gw}/picks/`, and `entry/{id}/history/`. Completed Gameweek live payloads are normalized to player × Gameweek rows; target rows are then appended and passed through the same Phase 2 shift-before-roll implementation. Target-GW outcomes never enter rolling features.

The persisted metadata is authoritative. All required feature names must exist with compatible dtypes before a position model is loaded. Missing or mismatched inputs fail loudly. Sparse early-season rolling fields remain missing and are handled by the model's training-fitted imputers; they are not silently filled with zero.

The lightweight minutes proxy uses only lagged one/three/five-GW minutes and start rates. `minutes_confidence` describes history depth and published availability uncertainty. Official `status`, `chance_of_playing_next_round`, and `news` are preserved. When FPL publishes a percentage, displayed xPts are multiplied by that percentage; when it does not, no probability is invented and raw display xPts remain unchanged.

Raw Ridge predictions are preserved. User-facing xPts and lower uncertainty bounds are floored at zero. Position-specific empirical residual ranges come from Phase 4 pre-test OOF residuals and are descriptive rather than guaranteed probabilities.

Public picks preserve purchase price, selling price, current price, multiplier, captain/vice flags, and bench order. They are resolved only from Gameweeks evidenced by entry history and are explicitly labelled **latest public squad**: a public historical snapshot is never assumed to be the current pre-deadline squad. Late-created entries can validly have no historical picks; this produces a clean unavailable report rather than an error.

For a current pre-deadline view, provide a local JSON file:

```bash
.venv/bin/python scripts/live_squad_report.py --entry-id 123456 --squad-file data/live/manual_squad.json
```

It must contain a `players` list of exactly 15 current players (by `player_id`, or exact normalized full name then unique web name), with the legal 2 GK / 5 DEF / 5 MID / 3 FWD composition and at most three per club. Optional `purchase_price`, `selling_price`, captain/vice flags, `bank`, and `free_transfers` are preserved; omitted historical prices remain unknown. Ambiguous names and invalid squads fail clearly. The public API does not reliably expose the current free-transfer balance, so it is otherwise reported as unknown. No credentials are stored.

Current xPts estimates are optimized for average expected-points ranking and are not yet reliable estimates of explosive 10+ point outcomes. This is especially important for future captaincy decisions.

Live outputs are `data/live/player_predictions.csv`, optional `my_squad.csv`, and `live_summary.json`; the latter records `squad_source` (`public_api`, `manual_file`, or `unavailable`) and `squad_gameweek`. The refreshed Phase 1 fixture summaries continue to provide next-three and next-five-Gameweek context; Phase 5 does not fabricate future availability or apply the one-GW model repeatedly to future dates. Any future transfer workflow must reject unavailable, stale public, or otherwise unverified current-squad state rather than optimizing it.

## Phase 6 Decision Optimizer

The optimizer uses SciPy's bundled HiGHS mixed-integer solver. Starting-XI selection enforces 11 players, exactly one goalkeeper, 3–5 defenders, 2–5 midfielders, and 1–3 forwards. The remaining goalkeeper occupies the dedicated goalkeeper bench slot; the three outfield substitutes are ordered by availability-adjusted xPts. Full-squad selection enforces 2 GK / 5 DEF / 5 MID / 3 FWD, a maximum of three players per club, and the supplied budget.

Captaincy profiles are transparent. `safe` emphasizes expected points and minutes while penalizing uncertainty; `balanced` adds moderate ceiling weight; `aggressive` gives the heuristic ceiling score more influence. The ceiling score combines pre-GW mean xPts, xGI/90, ICT form, expected minutes, and fixture quality. It is a secondary ranking aid—not a calibrated probability of a haul.

For GW+1 through GW+5, known rolling form, availability, and expected-minutes inputs remain fixed. The production model is reapplied with each future Gameweek's known official fixture context. Blanks receive zero and Doubles retain aggregate fixture context. Weighted horizons are `1.0`; `1.0/0.9/0.8`; and `1.0/0.9/0.8/0.7/0.6`.

Transfer analysis evaluates zero, one, and pruned two-transfer paths, preserving position, budget, uniqueness, and club legality. Hits cost `4 × max(0, transfers - free_transfers)`. A free move must clear the configurable 1.5-point horizon threshold or the result is `ROLL TRANSFER`. One-transfer pools retain the strongest projection and value candidates per position. Two-transfer analysis additionally limits outgoing consideration to the weakest projected squad players and uses a smaller top-candidate pool per position; these bounds are configurable function parameters.

Bank and free transfers are never inferred. Supply them with `--bank` and `--free-transfers` or `data/live/manual_state.json`; CLI values win. Unknown selling prices block transfer analysis unless `--assume-selling-price-current` is explicitly supplied, in which case every report identifies scenario mode. Stale public squads require `--allow-stale-squad`; manual pre-deadline squads are accepted normally.

Synchronize the manual snapshot after making a transfer on the FPL website:

```bash
.venv/bin/python scripts/update_my_squad.py --out "Dominik Szoboszlai" --in "Bukayo Saka"
```

The updater uses exact current-bootstrap identity resolution, preserves lineup/captaincy metadata where possible, requires explicit captain/vice reassignment when removed, validates the final squad, and creates a timestamped backup before overwriting. Unknown prices remain null.

Decision outputs are `optimized_xi.csv`, `transfer_candidates.csv`, `two_transfer_candidates.csv`, `replacement_shortlists.csv`, and `decision_summary.json` under `data/live/`. Experimental full-squad selection is available with `--full-squad-budget`; it is not labelled a Wildcard recommendation.

## Phase 7 Dashboard

The Streamlit layer lives in `app.py`, `pages/`, and `src/fpl_predictor/ui/`. UI modules load and format output artifacts, manage runtime-only settings, render shared status/pitch/download components, and provide Plotly charts. The live model, multi-Gameweek features, lineup/captaincy, transfer legality, decision thresholds, and squad updates continue to run through the established modules and CLI boundaries.

The application pages are:

- **Dashboard** — target-GW context, five KPIs, optimized pitch, bench, limitations, and downloads.
- **My Squad** — sortable 15-player details plus explicit current/optimized role differences.
- **Player Rankings** — position, team, price, availability, ownership, and minutes filters with next-GW/3-GW/5-GW/value/ceiling/risk sorting.
- **Transfers** — roll/make decision, threshold reasoning, one- and two-transfer paths, hit costs, and replacement shortlists. These are inspection-only.
- **Captaincy** — safe, balanced, and aggressive profiles plus xPts-versus-ceiling diagnostics.
- **Fixtures** — numeric official FDR, model opponent strength, and best/worst three- and five-GW runs.
- **Player Comparison** — a focused two-player form, minutes, fixture, utility, and ownership comparison.
- **Model Performance** — stored Phase 4 frozen-test metrics, model comparisons, calibration, and segmented residual diagnostics.
- **Settings** — runtime assumptions, TTL, JSON squad upload, and confirmed local/session manual-squad updates.

Live file loading uses `st.cache_data` with a default 600-second TTL. Mutable settings and uploaded squads remain in `st.session_state`; the explicit Refresh button invalidates cached data only after the underlying CLI succeeds. Model files are loaded by the existing persisted-artifact code during inference rather than exposed for download. Available CSV/JSON decision reports can be downloaded from the dashboard.

### Personal Squad Semantics

Public FPL picks are a post-deadline/historical snapshot and may not represent current edits. A manual JSON squad is the authoritative local pre-deadline workflow. The Settings page validates uploaded files against the current bootstrap universe and requires explicit confirmation before calling the existing timestamped-backup squad updater. The app never replaces a manual squad with public picks automatically.

Streamlit Community Cloud has an ephemeral filesystem. Uploaded squads and UI changes may disappear after an app restart; use the local CLI and version-controlled workflow for persistence. No database or private-account authentication is included.

### Streamlit Community Cloud Deployment

Configure Community Cloud with:

```text
Repository: vivekvattem/mygoatfpl
Branch:     main
Main file:  app.py
```

Push the repository including `app.py`, `pages/`, `.streamlit/config.toml`, package source, the small production `models/ridge_*` artifacts, and compact Phase 4 report files. No secrets are required for the official public FPL endpoints. Future secrets belong in the Cloud console or an uncommitted `.streamlit/secrets.toml`.

All runtime paths derive from the repository root. Large historical datasets, current API snapshots, and generated live CSV/JSON files remain excluded from Git; the dashboard can recreate live outputs through Refresh.

### Current Dashboard Limitations

- Public FPL picks do not reveal reliable current pre-deadline edits.
- Current-season early-Gameweek form is sparse and falls back to training-time model imputation.
- Ridge xPts compresses rare high-ceiling outcomes; captaincy therefore adds a transparent ceiling/minutes heuristic.
- Expected minutes are a historical proxy, not a dedicated minutes model.
- Future-GW projections hold current form, availability, and minutes assumptions fixed while changing known fixtures.
- Unknown selling prices block strict transfer analysis; explicit current-price substitution is labelled **SCENARIO MODE**.
- Cloud-local squad uploads and edits are session/runtime conveniences, not durable storage.
- The app is decision support, not an automated transfer, chip, or account-management system.

## Phase 8 Strategic Planning

The fixture calendar expands the official FPL `fixtures/` response into a complete team × Gameweek grid for the next 10 Gameweeks. Zero scheduled fixtures are explicit BGWs, two are DGWs, and three or more are TGW/congestion rows. Opponents, venue, official FDR, kickoff time, and fixture IDs are retained. Only fixtures present in the official API are labelled **CONFIRMED**; rumours and tentative rearrangements are excluded.

Player signals are deterministic. Availability uses official status/chance, minutes uses the expected-minutes proxy (`GREEN ≥75`, `YELLOW 45–74 or low confidence`, `RED <45`), form and value are position-relative tertiles, and fixtures use numeric five-Gameweek FDR/counts. Overall signal weights are availability 25%, minutes 20%, fixtures 20%, form 15%, value 10%, and five-Gameweek outlook 10%; an official red availability state is a hard red override. Every signal includes structured reasons.

Actions are conservative: unowned green players are `BUY`; owned green players are `HOLD`; yellow players are `WATCH`; an owned red player is `SELL` only when the legal transfer analysis also clears the configured gain threshold, otherwise it is `WATCH / HOLD`.

The chip planner is advisory and shows at least eight upcoming Gameweeks. Chip state is `available`, `used`, or `unknown`; official history can prove usage, while availability remains unknown unless manually supplied in Settings. Wildcard compares a budget-legal rebuilt squad when bank is known. Free Hit combines confirmed active-player count with a budget-legal one-GW XI gain. Bench Boost uses bench xPts, minutes, availability, and all-15 fixture exposure. Triple Captain combines captain xPts with the existing ceiling/minutes heuristic and confirmed fixture count; Ridge xPts alone never determines it.

Chip signals are thresholded and visible as text: `GREEN` is a strong modeled opportunity, `YELLOW` is plausible but not exceptional, `RED` is poor timing, and `GREY` means used/unknown/insufficient data. Future fixtures may be rescheduled after refresh, expected minutes remain heuristic, high-ceiling compression remains, and projections beyond GW+5 lack player-level model output. There are no bookmaker odds, no private login, no automatic chip execution, and no private pre-deadline chip state unless supplied.

## Phase 9 AI Analyst

`Ask FPL AI` follows a guarded pipeline: deterministic intent routing, conservative player-name resolution, compact evidence selection, deterministic recommendation, optional provider explanation, and post-generation grounding validation. Relevant context is selected by intent rather than sending all 616 players. Transfer questions receive the selected player and legal alternatives; captaincy receives the existing profile candidates; comparisons receive two resolved players; budget searches are filtered programmatically before explanation.

Evidence labels identify the internal source of every answer: **ML Projection**, **Transfer Engine**, **Fixture Calendar**, **Signal Engine**, **Captaincy Engine**, **Chip Planner**, **Availability**, and **Squad State**. The expandable evidence panel is built from deterministic application context, never from provider prose. Confidence is also deterministic: missing financial legality or chip state, stale data, sparse minutes confidence, and missing required context reduce it from `HIGH` to `MODERATE` or `LOW`.

Without a provider key, the page displays `AI provider: Disabled · Using deterministic analyst`. Primary transfer, captaincy, comparison, ranking, risk, fixture, DGW/BGW, chip, player, and weekly-brief questions still work. The optional provider uses a stateless OpenAI Responses API request with bounded output and no response storage. Provider timeout, rate-limit/error, malformed output, or failed grounding automatically discards provider prose and shows the deterministic answer.

Grounding checks reject current players absent from supplied context, unsupported material xPts/price/gain/minutes numbers, false confirmed DGW/BGW claims, and false chip-state claims. The system prompt forbids pretrained live-football knowledge, fabricated data, optimizer overrides, guaranteed-points language, and unsupported certainty. Conversation history exists only in the current Streamlit session and is not stored externally.

The analyst explains structured estimates; it does not browse news, speculate about fixtures, or mutate an FPL account. Its answers depend on data freshness. Ridge ceiling compression and heuristic expected minutes remain, and unknown bank, free transfers, selling prices, or private chip state restrict authoritative advice. Disabling the AI provider does not reduce the model, optimizer, or deterministic analyst functionality.

## Roadmap

- Phase 1 — Current FPL data ingestion and baseline analytics
- Phase 2 — Historical Gameweek dataset
- Phase 3 — Leakage-safe rolling features and team-strength model
- Phase 4 — Expected-points ML model
- Phase 5 — Personalized squad import
- Phase 6 — Transfer / XI / captain optimizer
- Phase 7 — Streamlit dashboard
- Phase 8 — DGW/BGW, signals, and chip planner
- Phase 9 — grounded AI analyst and explainable decision layer (current)
- Phase 10 — monitoring and automated evaluation (not started)

Future transfer recommendations will compare multi-Gameweek expected gains after hit costs and legal FPL constraints. If no legal move clears the improvement threshold, the recommendation will explicitly be `ROLL TRANSFER`.
