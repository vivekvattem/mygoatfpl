"""Confirmed official-FPL fixture calendar with explicit Blank/Double/Triple rows."""

from itertools import product
from typing import Any

import pandas as pd


CALENDAR_COLUMNS = [
    "team_id", "team", "gw", "fixture_count", "is_blank", "is_double", "is_triple",
    "is_congested", "opponents", "home_away", "official_fdr", "average_fdr",
    "kickoff_times", "fixture_ids", "schedule_status", "schedule_label",
]


def build_fixture_calendar(fixtures: list[dict[str, Any]], teams: pd.DataFrame,
                           start_gw: int, horizon: int = 10) -> pd.DataFrame:
    """Return one row per team×GW, including explicit zero-fixture rows."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    required = {"id", "name"}
    if not required.issubset(teams.columns):
        raise ValueError("teams must contain id and name")
    names = teams.set_index("id")["name"].to_dict()
    rows: list[dict[str, Any]] = []
    end_gw = start_gw + horizon - 1
    for fixture in fixtures:
        gw, home, away = fixture.get("event"), fixture.get("team_h"), fixture.get("team_a")
        if gw is None or home not in names or away not in names or not start_gw <= int(gw) <= end_gw:
            continue
        common = {"gw": int(gw), "fixture_id": fixture.get("id"), "kickoff_time": fixture.get("kickoff_time")}
        rows.extend([
            {**common, "team_id": home, "opponent": names[away], "venue": "H",
             "fdr": fixture.get("team_h_difficulty")},
            {**common, "team_id": away, "opponent": names[home], "venue": "A",
             "fdr": fixture.get("team_a_difficulty")},
        ])
    fixture_rows = pd.DataFrame(rows)
    grouped: dict[tuple[int, int], dict[str, Any]] = {}
    if not fixture_rows.empty:
        fixture_rows["fdr"] = pd.to_numeric(fixture_rows.fdr, errors="coerce")
        fixture_rows["kickoff_time"] = pd.to_datetime(fixture_rows.kickoff_time, errors="coerce", utc=True)
        for (team_id, gw), group in fixture_rows.sort_values("kickoff_time").groupby(["team_id", "gw"]):
            grouped[(int(team_id), int(gw))] = {
                "opponents": tuple(group.opponent.astype(str)), "home_away": tuple(group.venue.astype(str)),
                "official_fdr": tuple(group.fdr.tolist()), "average_fdr": float(group.fdr.mean()),
                "kickoff_times": tuple(value.isoformat() if pd.notna(value) else "unknown" for value in group.kickoff_time),
                "fixture_ids": tuple(group.fixture_id.tolist()),
            }
    output = []
    team_rows = teams[["id", "name"]].drop_duplicates("id")
    for team, gw in product(team_rows.itertuples(index=False), range(start_gw, end_gw + 1)):
        context = grouped.get((int(team.id), gw), {})
        count = len(context.get("opponents", ()))
        marker = "BGW" if count == 0 else "DGW" if count == 2 else "TGW" if count >= 3 else "NORMAL"
        output.append({
            "team_id": int(team.id), "team": team.name, "gw": gw, "fixture_count": count,
            "is_blank": count == 0, "is_double": count == 2, "is_triple": count >= 3,
            "is_congested": count >= 2, "opponents": context.get("opponents", ()),
            "home_away": context.get("home_away", ()), "official_fdr": context.get("official_fdr", ()),
            "average_fdr": context.get("average_fdr"), "kickoff_times": context.get("kickoff_times", ()),
            "fixture_ids": context.get("fixture_ids", ()), "schedule_status": "CONFIRMED",
            "schedule_label": marker,
        })
    return pd.DataFrame(output, columns=CALENDAR_COLUMNS)


def team_fixture_signals(calendar: pd.DataFrame, start_gw: int, horizon: int = 5) -> pd.DataFrame:
    """Transparent team fixture signal over a fixed Gameweek window."""
    selected = calendar[calendar.gw.between(start_gw, start_gw + horizon - 1)].copy()
    rows = []
    for team, group in selected.groupby("team", sort=True):
        next_row = group[group.gw.eq(start_gw)]
        fixtures = int(group.fixture_count.sum())
        fdr_values = [float(value) for values in group.official_fdr for value in values if pd.notna(value)]
        average = sum(fdr_values) / len(fdr_values) if fdr_values else float("nan")
        if not next_row.empty and bool(next_row.iloc[0].is_blank):
            signal, reason = "RED", "No confirmed fixture in the next Gameweek"
        elif group.is_triple.any():
            signal, reason = "YELLOW", "Three or more fixtures create congestion and rotation risk"
        elif group.is_double.any() and (pd.isna(average) or average <= 3.25):
            signal, reason = "GREEN", "Confirmed Double Gameweek in the planning window"
        elif fixtures >= horizon and pd.notna(average) and average <= 2.75:
            signal, reason = "GREEN", "Favorable official FDR over the planning window"
        elif pd.notna(average) and average >= 3.75:
            signal, reason = "RED", "Difficult official FDR over the planning window"
        else:
            signal, reason = "YELLOW", "Mixed confirmed fixture run"
        rows.append({"team": team, "fixture_signal": signal, "fixture_reason": reason,
                     "fixtures_next_5": fixtures, "average_fdr_5": average})
    return pd.DataFrame(rows)


def fixture_matrix(calendar: pd.DataFrame, start_gw: int, horizon: int = 5) -> pd.DataFrame:
    """Human-readable matrix; text markers make status accessible without colour."""
    selected = calendar[calendar.gw.between(start_gw, start_gw + horizon - 1)].copy()
    selected["cell"] = selected.apply(_fixture_cell, axis=1)
    matrix = selected.pivot(index="team", columns="gw", values="cell").reset_index()
    return matrix.rename(columns={gw: f"GW+{int(gw) - start_gw + 1}" for gw in selected.gw.unique()})


def _fixture_cell(row: pd.Series) -> str:
    if row.fixture_count == 0:
        return "BGW — no fixture"
    fixtures = ", ".join(
        f"{opponent}({venue}) FDR {float(fdr):.0f}"
        for opponent, venue, fdr in zip(row.opponents, row.home_away, row.official_fdr)
    )
    marker = f" — {row.schedule_label}" if row.schedule_label != "NORMAL" else ""
    return fixtures + marker
