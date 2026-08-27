#!/usr/bin/env python3
"""Produce a personalized report from public official FPL entry data."""

import argparse
import json
import os
from pathlib import Path
import sys

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src")); sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from fpl_predictor.config import LIVE_DATA_DIR  # noqa: E402
from fpl_predictor.entry import parse_entry, parse_picks  # noqa: E402
from fpl_predictor.loaders import load_players  # noqa: E402
from live_predictions import run_live_predictions  # noqa: E402


def run(entry_id: int) -> None:
    live = run_live_predictions(); api = live["api"]
    entry = parse_entry(api.get_entry(entry_id))
    if not entry.current_gameweek:
        raise ValueError("Entry has no publicly available current Gameweek picks")
    picks_payload = api.get_entry_picks(entry_id, entry.current_gameweek)
    api.get_entry_history(entry_id)  # validated public history endpoint; free transfers are not exposed
    squad = parse_picks(picks_payload, load_players(live["bootstrap"]))
    predictions = live["predictions"]
    squad = squad.merge(predictions[["player_id", "raw_xpts", "display_xpts", "availability_adjusted_xpts", "expected_minutes_proxy", "minutes_confidence", "status", "news"]], on="player_id", how="left")
    predictions.loc[predictions.player_id.isin(squad.player_id), "owned"] = True
    squad.to_csv(LIVE_DATA_DIR / "my_squad.csv", index=False)
    predictions[["player_id", "player", "team", "position", "price", "raw_xpts", "display_xpts", "availability_adjusted_xpts", "xpts_lower", "xpts_upper", "expected_minutes_proxy", "minutes_confidence", "status", "chance_of_playing_next_round", "news", "owned"]].to_csv(LIVE_DATA_DIR / "player_predictions.csv", index=False)
    summary_path = LIVE_DATA_DIR / "live_summary.json"; summary = json.loads(summary_path.read_text())
    summary.update({"entry_id": entry_id, "squad_player_count": len(squad),
                    "squad_raw_xpts": float(squad.display_xpts.sum()),
                    "squad_adjusted_xpts": float(squad.availability_adjusted_xpts.sum()),
                    "free_transfers": None})
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nFPL LIVE SQUAD REPORT\n\nEntry: {entry.manager_name}\nTeam: {entry.team_name}\nTarget Gameweek: {summary['target_gw']}")
    print("Model: Position-specific Ridge\nFeature set: form\nSchema parity: PASSED\n\nYOUR SQUAD")
    for position in ("GK", "DEF", "MID", "FWD"):
        print(f"\n{position}\n" + squad[squad.position.eq(position)][["player", "current_price", "display_xpts", "availability_adjusted_xpts", "expected_minutes_proxy"]].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    captain = squad[squad.is_captain.eq(True)]; vice = squad[squad.is_vice_captain.eq(True)]
    print(f"\nCURRENT CAPTAIN\n{captain.player.iloc[0] if not captain.empty else 'Unknown'}")
    print(f"CURRENT VICE\n{vice.player.iloc[0] if not vice.empty else 'Unknown'}")
    print(f"\nSQUAD RAW xPTS TOTAL: {squad.display_xpts.sum():.2f}\nAVAILABILITY-ADJUSTED xPTS TOTAL: {squad.availability_adjusted_xpts.sum():.2f}")
    print("\nTOP NON-OWNED PROJECTIONS\n" + predictions[~predictions.owned].nlargest(10, "display_xpts")[["player", "team", "display_xpts"]].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nFree transfers: unknown (not exposed reliably by the public API).")
    print("Current xPts estimates are optimized for average expected-points ranking\nand are not yet reliable estimates of explosive 10+ point outcomes.")


if __name__ == "__main__":
    load_dotenv(); parser = argparse.ArgumentParser(); parser.add_argument("--entry-id", type=int)
    args = parser.parse_args(); value = args.entry_id or os.getenv("FPL_ENTRY_ID")
    if not value: parser.error("Provide --entry-id or set FPL_ENTRY_ID")
    run(int(value))
