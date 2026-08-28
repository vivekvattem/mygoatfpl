"""Advisory-only FPL chip state and transparent planning heuristics."""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .lineup_optimizer import optimize_starting_xi
from .optimizer import best_15_player_squad


CHIP_RULES = {
    "wildcard": {"api_names": {"wildcard", "wildcard1", "wildcard2"}, "label": "Wildcard"},
    "free_hit": {"api_names": {"freehit", "free_hit"}, "label": "Free Hit"},
    "bench_boost": {"api_names": {"bboost", "benchboost", "bench_boost"}, "label": "Bench Boost"},
    "triple_captain": {"api_names": {"3xc", "triplecaptain", "triple_captain"}, "label": "Triple Captain"},
}


@dataclass(frozen=True)
class ChipState:
    chip: str
    state: str
    source: str


def resolve_chip_states(history: dict[str, Any] | None = None,
                        overrides: dict[str, str] | None = None) -> dict[str, ChipState]:
    """Official history proves usage; availability remains unknown unless overridden."""
    used_names = {str(item.get("name", "")).casefold() for item in (history or {}).get("chips", [])
                  if isinstance(item, dict)}
    overrides = overrides or {}
    result = {}
    for chip, rule in CHIP_RULES.items():
        override = overrides.get(chip)
        if override in {"available", "used"}:
            result[chip] = ChipState(chip, override, "manual_override")
        elif used_names & rule["api_names"]:
            result[chip] = ChipState(chip, "used", "official_entry_history")
        else:
            result[chip] = ChipState(chip, "unknown", "not publicly verifiable pre-deadline")
    return result


def chip_signal(score: object, state: ChipState, green: float, yellow: float) -> str:
    value = pd.to_numeric(score, errors="coerce")
    if state.state != "available" or pd.isna(value):
        return "GREY"
    return "GREEN" if value >= green else "YELLOW" if value >= yellow else "RED"


def build_chip_plan(calendar: pd.DataFrame, squad: pd.DataFrame, players: pd.DataFrame,
                    states: dict[str, ChipState], start_gw: int, horizon: int = 8,
                    wildcard_gain: float | None = None,
                    free_hit_gains: dict[int, float] | None = None) -> pd.DataFrame:
    """Build an eight-GW comparison using only current structured projections."""
    free_hit_gains = free_hit_gains or {}
    rows = []
    owned = squad.copy()
    for gw in range(start_gw, start_gw + horizon):
        offset = gw - start_gw + 1
        team_counts = calendar[calendar.gw.eq(gw)].set_index("team").fixture_count.to_dict()
        active = int(owned.team.map(team_counts).fillna(0).gt(0).sum()) if not owned.empty else 0
        blank_count = max(0, len(owned) - active)
        double_teams = int(calendar[calendar.gw.eq(gw)].fixture_count.ge(2).sum())
        red = int(owned.get("overall_signal", pd.Series(index=owned.index)).eq("RED").sum())
        yellow = int(owned.get("overall_signal", pd.Series(index=owned.index)).eq("YELLOW").sum())
        wc_score = (None if wildcard_gain is None else float(wildcard_gain) + 2 * red + 0.5 * yellow +
                    1.5 * blank_count + 0.5 * double_teams)
        wc_signal = chip_signal(wc_score, states["wildcard"], 15, 8)
        wc_reason = (f"Budget-legal rebuild gain, {red} red/{yellow} yellow players, "
                     f"{blank_count} blanks and {double_teams} DGW/TGW teams" if wc_score is not None
                     else "Requires known bank and budget-legal rebuilt-squad comparison")

        fh_gain = free_hit_gains.get(gw)
        fh_score = None if fh_gain is None else float(fh_gain) + max(0, 11 - active) * 2
        fh_signal = chip_signal(fh_score, states["free_hit"], 12, 6)
        fh_reason = f"{active}/15 squad players have a confirmed fixture; one-GW gain " + (
            f"{fh_gain:.2f}" if fh_gain is not None else "unavailable")

        xpts_column = f"xpts_gw{offset}"
        fixture_column = f"fixture_count_gw{offset}"
        bench = owned[pd.to_numeric(owned.get("multiplier", 1), errors="coerce").fillna(1).eq(0)] if not owned.empty else owned
        if offset <= 5 and xpts_column in bench:
            bench_points = float(pd.to_numeric(bench[xpts_column], errors="coerce").fillna(0).sum())
            bench_minutes = pd.to_numeric(bench.get("expected_minutes_proxy"), errors="coerce")
            all_active = bool(pd.to_numeric(owned.get(fixture_column), errors="coerce").fillna(0).gt(0).all())
            bb_score = bench_points + (2 if all_active else 0)
            bb_signal = chip_signal(bb_score, states["bench_boost"], 12, 7)
            if not bench_minutes.empty and bench_minutes.min() < 45 and bb_signal == "GREEN": bb_signal = "YELLOW"
            bb_reason = f"Bench projects {bench_points:.2f} xPts; all 15 active: {all_active}"
        else:
            bench_points, bb_score, bb_signal = None, None, "GREY"
            bb_reason = "No player-level projection is available beyond GW+5"

        if offset <= 5 and xpts_column in players:
            candidates = players[pd.to_numeric(players.get(fixture_column), errors="coerce").fillna(0).gt(0)]
            if candidates.empty:
                captain, captain_xpts, fixtures, tc_score, tc_signal = None, None, 0, None, "GREY"
                tc_reason = "No projected captain has a confirmed fixture"
            else:
                ranked = candidates.assign(_tc=(pd.to_numeric(candidates[xpts_column], errors="coerce").fillna(0) +
                                                 pd.to_numeric(candidates.get("ceiling_score"), errors="coerce").fillna(0) / 100))
                top = ranked.nlargest(1, "_tc").iloc[0]
                captain, captain_xpts = top.player, float(top[xpts_column])
                fixtures = int(top.get(fixture_column, 1)); minutes = float(top.get("expected_minutes_proxy", 0) or 0)
                ceiling = float(top.get("ceiling_score", 0) or 0)
                tc_score = captain_xpts + (3 if fixtures >= 2 else 0) + ceiling / 25
                tc_signal = chip_signal(tc_score, states["triple_captain"], 12, 8)
                if minutes < 60: tc_signal = "RED"
                elif minutes < 75 and tc_signal == "GREEN": tc_signal = "YELLOW"
                tc_reason = f"{captain}: {fixtures} fixture(s), {captain_xpts:.2f} base xPts, ceiling {ceiling:.0f}"
        else:
            captain, captain_xpts, fixtures, tc_score, tc_signal = None, None, None, None, "GREY"
            tc_reason = "No player-level projection is available beyond GW+5"
        rows.append({
            "gw": gw, "active_squad_players": active,
            "wildcard_score": wc_score, "wildcard_signal": wc_signal, "wildcard_reason": wc_reason,
            "free_hit_score": fh_score, "free_hit_signal": fh_signal, "free_hit_gain": fh_gain,
            "free_hit_reason": fh_reason, "bench_boost_score": bb_score, "bench_boost_signal": bb_signal,
            "bench_points": bench_points, "bench_boost_reason": bb_reason,
            "triple_captain_score": tc_score, "triple_captain_signal": tc_signal,
            "triple_captain": captain, "captain_xpts": captain_xpts, "captain_fixtures": fixtures,
            "triple_captain_reason": tc_reason,
        })
    return pd.DataFrame(rows)


def budget_legal_chip_gains(players: pd.DataFrame, squad: pd.DataFrame, bank: float | None,
                            start_gw: int) -> tuple[float | None, dict[int, float]]:
    """Compare current and optimized legal squads; unknown bank stays unknown."""
    if bank is None or players.empty or squad.empty:
        return None, {}
    budget = float(pd.to_numeric(squad.price, errors="coerce").sum()) + float(bank)
    wildcard_gain = None
    try:
        optimal = best_15_player_squad(players, budget, "weighted_xpts_5").optimal_15
        wildcard_gain = float(optimal.weighted_xpts_5.sum() - squad.weighted_xpts_5.sum())
    except (ValueError, RuntimeError):
        pass
    gains = {}
    for offset in range(1, 6):
        metric = f"xpts_gw{offset}"
        if metric not in players or metric not in squad:
            continue
        try:
            optimal = best_15_player_squad(players, budget, metric).optimal_15
            optimal_xi = optimize_starting_xi(optimal, metric).starting_xpts
            current_xi = optimize_starting_xi(squad, metric).starting_xpts
            gains[start_gw + offset - 1] = float(optimal_xi - current_xi)
        except (ValueError, RuntimeError):
            continue
    return wildcard_gain, gains
