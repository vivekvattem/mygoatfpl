#!/usr/bin/env python3
"""Build a personalized legal lineup, captaincy, and transfer decision report."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src")); sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from fpl_predictor.captaincy import rank_captains  # noqa: E402
from fpl_predictor.config import HISTORICAL_ML_DIR, LIVE_DATA_DIR  # noqa: E402
from fpl_predictor.decision_report import render_decision_report  # noqa: E402
from fpl_predictor.lineup_optimizer import optimize_starting_xi  # noqa: E402
from fpl_predictor.loaders import load_players  # noqa: E402
from fpl_predictor.multigw import build_multi_gw_projections  # noqa: E402
from fpl_predictor.optimizer import best_15_player_squad  # noqa: E402
from fpl_predictor.squad_state import load_manual_squad_state, validate_squad_freshness  # noqa: E402
from fpl_predictor.transfer_optimizer import (  # noqa: E402
    attach_squad_projections, optimize_one_transfer, optimize_two_transfers,
    replacement_shortlists, transfer_decision,
)
from fpl_predictor.utility import add_player_utilities  # noqa: E402
from live_predictions import run_live_predictions  # noqa: E402


def run(args: argparse.Namespace) -> dict:
    live = run_live_predictions(); target_gw = int(live["summary"]["target_gw"])
    current_players = load_players(live["bootstrap"])
    state = load_manual_squad_state(args.entry_id, target_gw, args.squad_file,
                                    current_players, args.state_file, args.bank, args.free_transfers)
    validate_squad_freshness(state, args.allow_stale_squad)
    multi = build_multi_gw_projections(live["features"], live["bootstrap"], live["fixtures"],
                                       live["artifacts"], HISTORICAL_ML_DIR / "phase4_summary.json",
                                       target_gw, horizon=5)
    identity = {"player", "team", "position", "price"}
    universe = live["predictions"].merge(
        multi.drop(columns=[column for column in identity if column in multi]), on="player_id", validate="one_to_one"
    )
    universe = add_player_utilities(universe, args.horizon, args.risk_lambda)
    squad = attach_squad_projections(state, universe)
    lineup = optimize_starting_xi(squad, f"weighted_xpts_{args.horizon}")
    captaincy = rank_captains(lineup.starting_11, args.risk_profile)
    one = two = replacements = None
    decision = "TRANSFER STATE REQUIRED"
    transfer_note = None
    try:
        one = optimize_one_transfer(state, squad, universe, args.assume_selling_price_current, args.horizon)
        two = optimize_two_transfers(state, squad, universe, args.assume_selling_price_current, args.horizon)
        replacements = replacement_shortlists(one, horizon=args.horizon)
        decision = transfer_decision(one, two, args.horizon, args.minimum_gain)
        if args.assume_selling_price_current:
            transfer_note = "SCENARIO MODE: unknown selling prices were explicitly assumed equal to current prices."
    except ValueError as exc:
        transfer_note = str(exc)
        one = one if one is not None else None
        two = two if two is not None else None
    optimized = lineup.starting_11.copy()
    optimized["captain"] = optimized.player_id.eq(captaincy.captain.player_id)
    optimized["vice_captain"] = optimized.player_id.eq(captaincy.vice_captain.player_id)
    optimized.to_csv(LIVE_DATA_DIR / "optimized_xi.csv", index=False)
    (one if one is not None else pd.DataFrame()).to_csv(LIVE_DATA_DIR / "transfer_candidates.csv", index=False)
    (two if two is not None else pd.DataFrame()).to_csv(LIVE_DATA_DIR / "two_transfer_candidates.csv", index=False)
    (replacements if replacements is not None else pd.DataFrame()).to_csv(LIVE_DATA_DIR / "replacement_shortlists.csv", index=False)
    current_starters = squad[pd_numeric(squad.get("multiplier", 0)).gt(0)]
    current_xi = float(current_starters.availability_adjusted_xpts.sum())
    optimized_next = float(lineup.starting_11.availability_adjusted_xpts.sum())
    best_one = None if one is None or one.empty else one.iloc[0].to_dict()
    best_two = None if two is None or two.empty else two.iloc[0].to_dict()
    summary = {"entry_id": args.entry_id, "target_gw": target_gw, "horizon": args.horizon,
               "risk_profile": args.risk_profile, "squad_source": state.squad_source,
               "bank": state.bank, "free_transfers": state.free_transfers,
               "financial_state_scenario": bool(args.assume_selling_price_current),
               "current_squad_xpts": float(squad.availability_adjusted_xpts.sum()),
               "current_xi_xpts": current_xi, "optimized_xi_xpts": optimized_next,
               "optimized_planning_xpts": lineup.starting_xpts,
               "captain": captaincy.captain.player, "vice_captain": captaincy.vice_captain.player,
               "formation": lineup.formation, "transfer_decision": decision,
               "best_transfer": best_one, "best_two_transfer_path": best_two,
               "transfer_note": transfer_note, "created_at": datetime.now(timezone.utc).isoformat()}
    (LIVE_DATA_DIR / "decision_summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    if args.full_squad_budget is not None:
        full = best_15_player_squad(universe, args.full_squad_budget, f"weighted_xpts_{args.horizon}")
        full.optimal_15.to_csv(LIVE_DATA_DIR / "experimental_optimal_15.csv", index=False)
    report = render_decision_report(args.entry_id, target_gw, lineup, captaincy, decision,
                                    args.horizon, args.risk_profile, one, two, replacements, transfer_note)
    print("\n" + report)
    return summary


def pd_numeric(values):
    return pd.to_numeric(values, errors="coerce").fillna(0)


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if value != value:
        return None
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--entry-id", type=int, required=True)
    result.add_argument("--squad-file", type=Path, required=True)
    result.add_argument("--state-file", type=Path, default=LIVE_DATA_DIR / "manual_state.json")
    result.add_argument("--bank", type=float); result.add_argument("--free-transfers", type=int)
    result.add_argument("--horizon", type=int, choices=(1, 3, 5), default=5)
    result.add_argument("--risk-profile", choices=("safe", "balanced", "aggressive"), default="balanced")
    result.add_argument("--risk-lambda", type=float, default=0.10)
    result.add_argument("--minimum-gain", type=float, default=1.5)
    result.add_argument("--assume-selling-price-current", action="store_true")
    result.add_argument("--allow-stale-squad", action="store_true")
    result.add_argument("--full-squad-budget", type=float)
    return result


if __name__ == "__main__":
    try:
        run(parser().parse_args())
    except (ValueError, RuntimeError) as exc:
        print(f"Optimization failed: {exc}", file=sys.stderr); raise SystemExit(1)
