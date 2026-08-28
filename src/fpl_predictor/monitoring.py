"""Deterministic, side-effect-free monitoring for current public FPL state."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Iterable

import numpy as np
import pandas as pd


CHANGE_CATEGORIES = {
    "NO_CHANGE", "PLAYER_DATA_CHANGED", "FIXTURES_CHANGED", "GAMEWEEK_CHANGED",
    "SCHEMA_CHANGED", "MULTIPLE_CHANGES",
}


def _hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _select(rows: Iterable[dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    return [{field: row.get(field) for field in fields} for row in sorted(rows, key=lambda item: item.get("id", 0))]


def live_fingerprint(bootstrap: dict[str, Any], fixtures: list[dict[str, Any]]) -> dict[str, str]:
    """Hash compact, meaningful serving state instead of entire raw payloads."""
    players = _select(bootstrap.get("elements", []), (
        "id", "now_cost", "status", "chance_of_playing_next_round", "selected_by_percent",
        "transfers_in_event", "transfers_out_event", "news_added",
    ))
    fixture_rows = _select(fixtures, (
        "id", "event", "team_h", "team_a", "kickoff_time", "team_h_difficulty", "team_a_difficulty",
    ))
    events = _select(bootstrap.get("events", []), (
        "id", "is_current", "is_next", "finished", "data_checked", "deadline_time",
    ))
    element_keys = sorted({key for row in bootstrap.get("elements", []) for key in row})
    fixture_keys = sorted({key for row in fixtures for key in row})
    components = {
        "players": _hash(players), "fixtures": _hash(fixture_rows), "gameweek": _hash(events),
        "schema": _hash({"elements": element_keys, "fixtures": fixture_keys}),
    }
    components["combined"] = _hash(components)
    return components


def classify_change(previous: dict[str, str] | None, current: dict[str, str]) -> str:
    if not previous:
        return "MULTIPLE_CHANGES"
    changed = [name for name in ("players", "fixtures", "gameweek", "schema")
               if previous.get(name) != current.get(name)]
    if not changed:
        return "NO_CHANGE"
    if len(changed) > 1:
        return "MULTIPLE_CHANGES"
    return {"players": "PLAYER_DATA_CHANGED", "fixtures": "FIXTURES_CHANGED",
            "gameweek": "GAMEWEEK_CHANGED", "schema": "SCHEMA_CHANGED"}[changed[0]]


@dataclass(frozen=True)
class PlayerChange:
    player_id: int
    player: str
    field: str
    previous: Any
    current: Any


@dataclass(frozen=True)
class FixtureChange:
    fixture_id: int
    kind: str
    detail: str


@dataclass(frozen=True)
class ChangeReport:
    category: str
    fingerprint: dict[str, str]
    detected_at: datetime
    player_changes: tuple[PlayerChange, ...] = ()
    fixture_changes: tuple[FixtureChange, ...] = ()
    schedule_alerts: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return self.category != "NO_CHANGE"


def _player_changes(previous: dict[str, Any], current: dict[str, Any]) -> tuple[PlayerChange, ...]:
    old = {int(row["id"]): row for row in previous.get("elements", []) if row.get("id") is not None}
    new = {int(row["id"]): row for row in current.get("elements", []) if row.get("id") is not None}
    names = {player_id: str(row.get("web_name") or row.get("second_name") or player_id)
             for player_id, row in new.items()}
    fields = ("now_cost", "status", "chance_of_playing_next_round", "selected_by_percent")
    changes = []
    for player_id in old.keys() & new.keys():
        for field_name in fields:
            before, after = old[player_id].get(field_name), new[player_id].get(field_name)
            if before != after:
                if field_name == "selected_by_percent":
                    try:
                        if abs(float(after) - float(before)) < 0.5:
                            continue
                    except (TypeError, ValueError):
                        pass
                changes.append(PlayerChange(player_id, names[player_id], field_name, before, after))
    return tuple(changes)


def _fixture_counts(fixtures: list[dict[str, Any]]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for row in fixtures:
        event = row.get("event")
        if event is None:
            continue
        for team_field in ("team_h", "team_a"):
            team = row.get(team_field)
            if team is not None:
                counts[(int(team), int(event))] = counts.get((int(team), int(event)), 0) + 1
    return counts


def _fixture_changes(previous: list[dict[str, Any]], current: list[dict[str, Any]],
                     bootstrap: dict[str, Any]) -> tuple[tuple[FixtureChange, ...], tuple[str, ...]]:
    old = {int(row["id"]): row for row in previous if row.get("id") is not None}
    new = {int(row["id"]): row for row in current if row.get("id") is not None}
    changes: list[FixtureChange] = []
    for fixture_id in sorted(new.keys() - old.keys()):
        changes.append(FixtureChange(fixture_id, "added", "Fixture added"))
    for fixture_id in sorted(old.keys() - new.keys()):
        changes.append(FixtureChange(fixture_id, "removed", "Fixture removed"))
    fields = {"event": "Gameweek", "kickoff_time": "Kickoff", "team_h_difficulty": "Home FDR",
              "team_a_difficulty": "Away FDR", "team_h": "Home team", "team_a": "Away team"}
    for fixture_id in sorted(old.keys() & new.keys()):
        for field_name, label in fields.items():
            if old[fixture_id].get(field_name) != new[fixture_id].get(field_name):
                changes.append(FixtureChange(
                    fixture_id, "moved_gw" if field_name == "event" else "changed",
                    f"{label}: {old[fixture_id].get(field_name)} → {new[fixture_id].get(field_name)}",
                ))
    old_counts, new_counts = _fixture_counts(previous), _fixture_counts(current)
    team_names = {int(row["id"]): row.get("name", f"Team {row['id']}")
                  for row in bootstrap.get("teams", []) if row.get("id") is not None}
    alerts = []
    for key in sorted(old_counts.keys() | new_counts.keys()):
        before, after = old_counts.get(key, 0), new_counts.get(key, 0)
        if before == after:
            continue
        team, gw = key
        before_label = "BGW" if before == 0 else "DGW" if before == 2 else "TGW" if before >= 3 else "NORMAL"
        after_label = "BGW" if after == 0 else "DGW" if after == 2 else "TGW" if after >= 3 else "NORMAL"
        if before_label != after_label:
            alerts.append(f"GW{gw}: {team_names.get(team, f'Team {team}')} changed {before_label} → {after_label}")
    return tuple(changes), tuple(alerts)


def compare_live_state(previous_bootstrap: dict[str, Any] | None,
                       previous_fixtures: list[dict[str, Any]] | None,
                       bootstrap: dict[str, Any], fixtures: list[dict[str, Any]]) -> ChangeReport:
    previous_fingerprint = (live_fingerprint(previous_bootstrap, previous_fixtures)
                            if previous_bootstrap is not None and previous_fixtures is not None else None)
    fingerprint = live_fingerprint(bootstrap, fixtures)
    category = classify_change(previous_fingerprint, fingerprint)
    players = _player_changes(previous_bootstrap or {}, bootstrap) if previous_bootstrap is not None else ()
    fixture_changes, alerts = (_fixture_changes(previous_fixtures or [], fixtures, bootstrap)
                               if previous_fixtures is not None else ((), ()))
    return ChangeReport(category, fingerprint, datetime.now(timezone.utc), players, fixture_changes, alerts)


def change_summary(report: ChangeReport) -> str:
    """Return a compact human summary without dumping the full player universe."""
    field_counts: dict[str, int] = {}
    for item in report.player_changes:
        field_counts[item.field] = field_counts.get(item.field, 0) + 1
    labels = {"now_cost": "price", "status": "availability",
              "chance_of_playing_next_round": "chance-of-playing", "selected_by_percent": "ownership"}
    parts = [f"{count} {labels.get(field, field)} change{'s' if count != 1 else ''}"
             for field, count in field_counts.items()]
    if report.fixture_changes:
        parts.append(f"{len(report.fixture_changes)} fixture change{'s' if len(report.fixture_changes) != 1 else ''}")
    parts.append(f"{len(report.schedule_alerts)} new schedule alert{'s' if len(report.schedule_alerts) != 1 else ''}")
    return " · ".join(parts)


def prediction_distribution(frame: pd.DataFrame, column: str = "availability_adjusted_xpts") -> dict[str, Any]:
    if frame.empty or column not in frame:
        return {"status": "UNAVAILABLE", "count": 0}
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return {"status": "UNAVAILABLE", "count": 0}
    position_means = {}
    if "position" in frame:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        position_means = numeric.groupby(frame["position"], dropna=False).mean().dropna().round(3).to_dict()
    maximum, p95 = float(values.max()), float(values.quantile(0.95))
    drift = "ALERT" if maximum > 15 or p95 > 10 else "WATCH" if maximum > 10 or p95 > 7 else "NORMAL"
    return {"status": drift, "count": int(len(values)), "mean": float(values.mean()),
            "median": float(values.median()), "p95": p95, "maximum": maximum,
            "position_means": position_means}


def completed_gw_monitoring(scored: pd.DataFrame, minimum_gameweeks: int = 3) -> dict[str, Any]:
    required = {"gw", "actual_points", "predicted_points"}
    if scored.empty or not required.issubset(scored):
        return {"status": "INSUFFICIENT SAMPLE", "completed_gameweeks": 0}
    clean = scored.dropna(subset=list(required)).copy()
    gameweeks = clean.gw.nunique()
    if gameweeks < minimum_gameweeks:
        return {"status": "INSUFFICIENT SAMPLE", "completed_gameweeks": int(gameweeks)}
    clean["actual_points"] = pd.to_numeric(clean.actual_points, errors="coerce")
    clean["predicted_points"] = pd.to_numeric(clean.predicted_points, errors="coerce")
    clean = clean.dropna(subset=["actual_points", "predicted_points"])
    residual = clean.actual_points - clean.predicted_points
    top_k_scores = []
    for _, group in clean.groupby("gw"):
        size = min(10, len(group))
        predicted_top = set(group.nlargest(size, "predicted_points").index)
        actual_top = set(group.nlargest(size, "actual_points").index)
        top_k_scores.append(len(predicted_top & actual_top) / size if size else np.nan)
    try:
        bins = pd.qcut(clean.predicted_points, q=min(5, clean.predicted_points.nunique()),
                       duplicates="drop")
        calibration = (clean.assign(_bin=bins).groupby("_bin", observed=True)
                       .agg(count=("gw", "size"), predicted_mean=("predicted_points", "mean"),
                            actual_mean=("actual_points", "mean")).reset_index(drop=True)
                       .round(3).to_dict("records"))
    except ValueError:
        calibration = []
    return {"status": "NORMAL", "completed_gameweeks": int(gameweeks),
            "mae": float(residual.abs().mean()), "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "spearman": float(clean.actual_points.corr(clean.predicted_points, method="spearman")),
            "top_10_overlap": float(np.nanmean(top_k_scores)), "calibration_bins": calibration}
