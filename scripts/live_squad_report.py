#!/usr/bin/env python3
"""Produce a personalized report without overstating public squad freshness."""

import argparse
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src")); sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from fpl_predictor.api import FPLAPIError  # noqa: E402
from fpl_predictor.config import LIVE_DATA_DIR  # noqa: E402
from fpl_predictor.entry import load_manual_squad, parse_entry, parse_picks, resolve_entry_squad  # noqa: E402
from fpl_predictor.loaders import load_events, load_players  # noqa: E402
from live_predictions import run_live_predictions  # noqa: E402


def _update_summary(summary: dict, **values: object) -> None:
    summary.update(values)
    (LIVE_DATA_DIR / "live_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _print_unavailable(entry, target_gw: int, reason: str) -> None:
    print(f"\nFPL LIVE SQUAD REPORT\n\nEntry: {entry.manager_name}\nTeam: {entry.team_name}\nTarget Gameweek: {target_gw}")
    print("\nLive predictions: READY\nPublic squad import: UNAVAILABLE")
    print(f"Reason: {reason}")
    print("\nTo supply your current pre-deadline squad, run:")
    print("  .venv/bin/python scripts/live_squad_report.py --entry-id " + str(entry.entry_id) + " --squad-file data/live/manual_squad.json")
    print("Manual files must declare exactly 15 valid current players (2 GK, 5 DEF, 5 MID, 3 FWD; max 3 per club).")


def _print_report(entry, squad, predictions, summary: dict, source: str, squad_gameweek: int | None) -> None:
    if source == "public_api":
        heading = f"LATEST PUBLIC SQUAD — GW {squad_gameweek}"
        freshness = "This is a historical public snapshot, not a claimed current pre-deadline squad."
    else:
        heading = "CURRENT PRE-DEADLINE SQUAD — MANUAL FILE"
        freshness = "Current-squad status is supplied by the local manual file."
    print(f"\nFPL LIVE SQUAD REPORT\n\nEntry: {entry.manager_name}\nTeam: {entry.team_name}\nTarget Gameweek: {summary['target_gw']}")
    print("Model: Position-specific Ridge\nFeature set: form\nSchema parity: PASSED")
    print(f"\n{heading}\n{freshness}")
    for position in ("GK", "DEF", "MID", "FWD"):
        rows = squad[squad.position.eq(position)][["player", "current_price", "display_xpts", "availability_adjusted_xpts", "expected_minutes_proxy"]]
        print(f"\n{position}\n" + rows.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    captain = squad[squad.is_captain.eq(True)]; vice = squad[squad.is_vice_captain.eq(True)]
    print(f"\nCAPTAIN\n{captain.player.iloc[0] if not captain.empty else 'Unknown'}")
    print(f"VICE-CAPTAIN\n{vice.player.iloc[0] if not vice.empty else 'Unknown'}")
    print(f"\nSQUAD RAW xPTS TOTAL: {squad.display_xpts.sum():.2f}\nAVAILABILITY-ADJUSTED xPTS TOTAL: {squad.availability_adjusted_xpts.sum():.2f}")
    print("\nTOP NON-OWNED PROJECTIONS\n" + predictions[~predictions.owned].nlargest(10, "display_xpts")[["player", "team", "display_xpts"]].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nFree transfers: unknown unless supplied in the manual file; public FPL data does not expose them reliably.")
    print("Current xPts estimates are optimized for average expected-points ranking\nand are not yet reliable estimates of explosive 10+ point outcomes.")


def run(entry_id: int, squad_file: str | Path | None = None) -> int:
    """Run predictions then resolve a manual current squad or public historical one."""
    live = run_live_predictions(); api = live["api"]
    try:
        entry = parse_entry(api.get_entry(entry_id))
        history = api.get_entry_history(entry_id)
    except FPLAPIError as exc:
        print(f"Entry import unavailable for {entry_id}: {exc}", file=sys.stderr)
        return 1
    summary = dict(live["summary"])
    players = load_players(live["bootstrap"])
    source, gameweek, kind = "unavailable", None, None
    bank, free_transfers = None, None
    if squad_file:
        try:
            manual = load_manual_squad(squad_file, players)
        except ValueError as exc:
            print(f"Manual squad import failed: {exc}", file=sys.stderr)
            return 1
        squad, source, kind = manual.picks, "manual_file", "current_pre_deadline_squad"
        bank, free_transfers = manual.bank, manual.free_transfers
    else:
        events = load_events(live["bootstrap"])
        completed = events.loc[events.finished.fillna(False), "id"].astype(int).tolist()
        resolved = resolve_entry_squad(api, entry_id, history, completed)
        source, gameweek, kind = resolved.source, resolved.squad_gameweek, resolved.squad_kind
        if source == "unavailable":
            _update_summary(summary, entry_id=entry_id, squad_source=source, squad_gameweek=None,
                            squad_player_count=None, squad_raw_xpts=None, squad_adjusted_xpts=None,
                            free_transfers=None)
            _print_unavailable(entry, int(summary["target_gw"]), str(resolved.reason))
            return 0
        squad = parse_picks(resolved.payload or {}, players)
    predictions = live["predictions"].copy()
    squad = squad.merge(predictions[["player_id", "raw_xpts", "display_xpts", "availability_adjusted_xpts", "expected_minutes_proxy", "minutes_confidence", "status", "news"]], on="player_id", how="left")
    squad["squad_source"] = source; squad["squad_gameweek"] = gameweek; squad["squad_kind"] = kind
    predictions.loc[predictions.player_id.isin(squad.player_id), "owned"] = True
    squad.to_csv(LIVE_DATA_DIR / "my_squad.csv", index=False)
    columns = ["player_id", "player", "team", "position", "price", "raw_xpts", "display_xpts", "availability_adjusted_xpts", "xpts_lower", "xpts_upper", "expected_minutes_proxy", "minutes_confidence", "status", "chance_of_playing_next_round", "news", "owned"]
    predictions[columns].to_csv(LIVE_DATA_DIR / "player_predictions.csv", index=False)
    _update_summary(summary, entry_id=entry_id, squad_source=source, squad_gameweek=gameweek,
                    squad_kind=kind, squad_player_count=len(squad),
                    squad_raw_xpts=float(squad.display_xpts.sum()),
                    squad_adjusted_xpts=float(squad.availability_adjusted_xpts.sum()),
                    bank=bank if source == "manual_file" else entry.bank,
                    free_transfers=free_transfers)
    _print_report(entry, squad, predictions, summary, source, gameweek)
    return 0


if __name__ == "__main__":
    load_dotenv(); parser = argparse.ArgumentParser()
    parser.add_argument("--entry-id", type=int); parser.add_argument("--squad-file", type=Path)
    args = parser.parse_args(); value = args.entry_id or os.getenv("FPL_ENTRY_ID")
    if not value: parser.error("Provide --entry-id or set FPL_ENTRY_ID")
    raise SystemExit(run(int(value), args.squad_file))
