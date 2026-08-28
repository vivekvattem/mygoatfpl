#!/usr/bin/env python3
"""Safely synchronize data/live/manual_squad.json after a manual FPL transfer."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from fpl_predictor.config import LIVE_DATA_DIR, RAW_DATA_DIR  # noqa: E402
from fpl_predictor.loaders import load_players  # noqa: E402
from fpl_predictor.squad_update import update_manual_squad  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--squad-file", type=Path, default=LIVE_DATA_DIR / "manual_squad.json")
    parser.add_argument("--out", dest="player_out"); parser.add_argument("--in", dest="player_in")
    parser.add_argument("--captain"); parser.add_argument("--vice-captain")
    parser.add_argument("--purchase-price", type=float); parser.add_argument("--selling-price", type=float)
    parser.add_argument("--bank", type=float)
    args = parser.parse_args()
    if not args.player_out:
        args.player_out = input("Player out: ").strip()
    if not args.player_in:
        args.player_in = input("Player in: ").strip()
    bootstrap_path = RAW_DATA_DIR / "bootstrap_static.json"
    if not bootstrap_path.exists():
        print("Current bootstrap data is missing; run scripts/refresh_data.py first.", file=sys.stderr); return 1
    players = load_players(json.loads(bootstrap_path.read_text(encoding="utf-8")))
    try:
        backup, squad = update_manual_squad(args.squad_file, players, args.player_out, args.player_in,
                                            args.captain, args.vice_captain,
                                            args.purchase_price, args.selling_price, args.bank)
    except ValueError as exc:
        print(f"Squad update failed: {exc}", file=sys.stderr); return 1
    print("MANUAL SQUAD UPDATED\n")
    print(squad[["player", "position", "team", "current_price", "bench_position", "is_captain", "is_vice_captain"]]
          .sort_values(["position", "bench_position"]).to_string(index=False))
    print(f"\nBackup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
