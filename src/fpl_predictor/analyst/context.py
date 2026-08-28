"""Compact, intent-specific context built from existing structured engines."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fpl_predictor.captaincy import rank_captains
from fpl_predictor.chips import budget_legal_chip_gains, build_chip_plan, resolve_chip_states
from fpl_predictor.ui.data import dashboard_summary

from .intents import detect_intent, extract_budget, extract_position, normalize_text, resolve_question_players


@dataclass(frozen=True)
class AnalystContext:
    intent: str
    payload: dict[str, Any]
    evidence: tuple[str, ...]
    evidence_details: dict[str, Any]
    confidence: str
    clarification: str | None = None


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return None if pd.isna(value) or not np.isfinite(value) else round(float(value), 4)
    if pd.isna(value) if not isinstance(value, (str, bool)) else False:
        return None
    return value


def _player(row: pd.Series) -> dict[str, Any]:
    fields = {
        "player_id": "player_id", "name": "player", "team": "team", "position": "position",
        "owned": "owned", "price": "price", "xpts_next": "availability_adjusted_xpts",
        "xpts_3gw": "total_xpts_3", "xpts_5gw": "total_xpts_5",
        "expected_minutes": "expected_minutes_proxy", "minutes_confidence": "minutes_confidence",
        "fixture_signal": "fixture_signal", "overall_signal": "overall_signal", "action": "action",
        "availability": "availability", "availability_signal": "availability_signal",
        "ceiling": "ceiling_score", "uncertainty_width": "uncertainty_width",
        "average_fdr_5": "average_fdr_5", "fixtures_next_5": "fixtures_next_5",
        "risk_reason": "risk_reason", "signal_reason": "signal_reason",
    }
    return _clean({target: row.get(source) for target, source in fields.items()})


def _players_by_id(frame: pd.DataFrame, ids: tuple[int, ...]) -> list[dict[str, Any]]:
    indexed = frame.drop_duplicates("player_id").set_index("player_id")
    return [_player(indexed.loc[player_id]) for player_id in ids if player_id in indexed.index]


def _captain_context(bundle: Any, risk_profile: str) -> dict[str, Any]:
    xi = bundle.optimized_xi.copy()
    if xi.empty:
        return {"profile": risk_profile, "candidates": []}
    signal_columns = [column for column in ("player_id", "overall_signal", "availability_signal", "minutes_signal")
                      if column in bundle.predictions]
    if len(signal_columns) > 1 and "overall_signal" not in xi:
        xi = xi.merge(bundle.predictions[signal_columns].drop_duplicates("player_id"), on="player_id", how="left")
    profiles = {}
    for profile in ("safe", "balanced", "aggressive"):
        result = rank_captains(xi, profile)
        profiles[profile] = {
            "captain": _player(result.captain), "vice": _player(result.vice_captain),
            "captaincy_score": _clean(result.captain.get("captaincy_score")),
        }
    selected = profiles.get(risk_profile, profiles["balanced"])
    return {"profile": risk_profile, "selected": selected, "profiles": profiles}


def _schedule_context(calendar: pd.DataFrame) -> dict[str, Any]:
    if calendar.empty:
        return {"status": "unavailable", "double_gameweeks": [], "blank_gameweeks": [], "triple_gameweeks": []}
    def events(column: str) -> list[dict[str, Any]]:
        selected = calendar[calendar[column]]
        return [{"gw": int(gw), "teams": sorted(group.team.astype(str).tolist()), "status": "CONFIRMED"}
                for gw, group in selected.groupby("gw")]
    return {"status": "CONFIRMED", "double_gameweeks": events("is_double"),
            "blank_gameweeks": events("is_blank"), "triple_gameweeks": events("is_triple")}


def _risk_players(squad: pd.DataFrame, limit: int = 3) -> list[dict[str, Any]]:
    if squad.empty:
        return []
    order = {"RED": 0, "YELLOW": 1, "GREY": 2, "GREEN": 3}
    signals = squad["overall_signal"] if "overall_signal" in squad else pd.Series("GREY", index=squad.index)
    ranked = squad.assign(_risk=signals.map(order).fillna(2),
                          _uncertainty=pd.to_numeric(squad.get("uncertainty_width"), errors="coerce").fillna(0))
    return [_player(row) for _, row in ranked.sort_values(["_risk", "_uncertainty"], ascending=[True, False]).head(limit).iterrows()]


def _weekly_payload(bundle: Any, summary: dict[str, Any], settings: Any,
                    captain: dict[str, Any], schedule: dict[str, Any], chip: dict[str, Any]) -> dict[str, Any]:
    xi = bundle.optimized_xi
    projected = summary.get("projected_xi")
    if projected is None and not xi.empty and "availability_adjusted_xpts" in xi:
        projected = float(pd.to_numeric(xi.availability_adjusted_xpts, errors="coerce").sum())
    owned = (bundle.predictions["owned"].fillna(False) if "owned" in bundle.predictions
             else pd.Series(False, index=bundle.predictions.index))
    opportunities = bundle.predictions[~owned] if not bundle.predictions.empty else bundle.predictions
    if not opportunities.empty and "weighted_xpts_5" in opportunities:
        opportunities = opportunities.sort_values("weighted_xpts_5", ascending=False)
    selected_captain = captain.get("selected", {})
    transfer = summary.get("transfer_decision", "UNAVAILABLE")
    return {
        "projected_xi_xpts": _clean(projected),
        "starting_xi": xi.player.astype(str).tolist() if not xi.empty and "player" in xi else [],
        "captain": selected_captain.get("captain"), "vice": selected_captain.get("vice"),
        "transfer_decision": transfer, "transfer_threshold": settings.minimum_gain,
        "top_risk": (_risk_players(bundle.squad, 1) or [None])[0],
        "top_opportunity": _player(opportunities.iloc[0]) if not opportunities.empty else None,
        "chip": chip, "schedule": schedule,
    }


def _chip_context(bundle: Any, settings: Any, start_gw: int,
                  chip_overrides: dict[str, str] | None) -> dict[str, Any]:
    history = {"chips": bundle.live_summary.get("entry_history_chips", [])}
    states = resolve_chip_states(history, chip_overrides)
    wildcard_gain, free_hit_gains = budget_legal_chip_gains(
        bundle.predictions, bundle.squad, settings.bank, start_gw)
    plan = build_chip_plan(bundle.fixture_calendar, bundle.squad, bundle.predictions, states,
                           start_gw, 8, wildcard_gain, free_hit_gains)
    state_values = {chip: state.state for chip, state in states.items()}
    rows = _clean(plan.to_dict(orient="records")) if not plan.empty else []
    return {"chip_states": state_values, "plan": rows}


def _transfer_context(bundle: Any, settings: Any, players: list[dict[str, Any]]) -> dict[str, Any]:
    summary = bundle.decision_summary
    best = summary.get("best_transfer") or {}
    horizon_column = f"net_gain_{settings.horizon}gw"
    selected_id = players[0].get("player_id") if players else None
    selected_owned = bool(players and players[0].get("owned"))
    alternatives = bundle.replacements if selected_owned else bundle.one_transfers
    if selected_id is not None and not alternatives.empty:
        identity_column = "out_id" if selected_owned else "in_id"
        if identity_column in alternatives:
            alternatives = alternatives[alternatives[identity_column].eq(selected_id)]
    candidates = []
    for _, row in alternatives.head(3).iterrows():
        candidates.append(_clean({"out": row.get("out"), "out_id": row.get("out_id"), "in": row.get("in"),
                                  "in_id": row.get("in_id"), "gain": row.get(horizon_column),
                                  "new_bank": row.get("new_bank"), "hit_cost": row.get("hit_cost")}))
    return {
        "players": players, "decision": summary.get("transfer_decision", "UNAVAILABLE"),
        "best_transfer": _clean(best), "alternatives": candidates,
        "horizon": settings.horizon, "threshold": settings.minimum_gain,
        "bank": settings.bank, "free_transfers": settings.free_transfers,
        "selling_price_scenario": settings.assume_selling_price_current,
        "legality_verified": bool(summary),
    }


def _ranking_context(bundle: Any, question: str) -> dict[str, Any]:
    budget, position = extract_budget(question), extract_position(question)
    candidates = bundle.predictions.copy()
    if budget is not None:
        candidates = candidates[pd.to_numeric(candidates.price, errors="coerce").le(budget)]
    if position:
        candidates = candidates[candidates.position.eq(position)]
    if "weighted_xpts_5" in candidates:
        candidates = candidates.sort_values("weighted_xpts_5", ascending=False)
    return {"budget": budget, "position": position,
            "candidates": [_player(row) for _, row in candidates.head(5).iterrows()]}


def _fixture_context(bundle: Any, question: str, start_gw: int | None) -> dict[str, Any]:
    signals = bundle.team_fixture_signals.copy()
    if not signals.empty:
        order = {"GREEN": 0, "YELLOW": 1, "GREY": 2, "RED": 3}
        signals = signals.assign(_signal=signals.fixture_signal.map(order).fillna(2)).sort_values(
            ["_signal", "average_fdr_5"], ascending=[True, True])
    text = normalize_text(question)
    teams = sorted(bundle.fixture_calendar.team.astype(str).unique()) if not bundle.fixture_calendar.empty else []
    mentioned = [team for team in teams if normalize_text(team) in text]
    fixtures = bundle.fixture_calendar
    if mentioned:
        fixtures = fixtures[fixtures.team.isin(mentioned)]
    elif start_gw is not None:
        fixtures = fixtures[fixtures.gw.eq(start_gw)]
    keep = [column for column in ("team", "gw", "fixture_count", "opponents", "home_away", "average_fdr",
                                   "schedule_label", "schedule_status") if column in fixtures]
    return {"matched_teams": mentioned, "fixtures": _clean(fixtures[keep].head(20).to_dict(orient="records")),
            "best_runs": _clean(signals.drop(columns=["_signal"], errors="ignore").head(5).to_dict(orient="records"))}


def _confidence(intent: str, payload: dict[str, Any], stale: bool) -> str:
    concerns = int(stale)
    if intent == "transfer" and not payload.get("transfer", {}).get("legality_verified"):
        concerns += 2
    if intent in {"captaincy", "player_comparison", "player_lookup"}:
        players = payload.get("players", [])
        if not players and intent == "captaincy":
            players = [payload.get("captaincy", {}).get("selected", {}).get("captain", {})]
        if not players:
            concerns += 2
        if any(player and player.get("minutes_confidence") == "low" for player in players):
            concerns += 1
    if intent == "chip" and all(state != "available" for state in payload.get("chip_states", {}).values()):
        concerns += 2
    return "LOW" if concerns >= 2 else "MODERATE" if concerns == 1 else "HIGH"


def build_analyst_context(bundle: Any, settings: Any, question: str,
                          chip_overrides: dict[str, str] | None = None) -> AnalystContext:
    """Select only the structured evidence required by the detected intent."""
    intent = detect_intent(question)
    summary = dashboard_summary(bundle)
    target_gw = int(summary["target_gw"]) if summary.get("target_gw") is not None else None
    base = {"target_gw": target_gw, "data_stale": bundle.status.stale,
            "squad_source": summary.get("squad_source"), "risk_profile": settings.risk_profile}
    resolution = resolve_question_players(question, bundle.predictions)
    if resolution.status == "ambiguous":
        choices = ", ".join(resolution.candidates)
        return AnalystContext(intent, base, ("squad_state",), {"Ambiguous name": resolution.query}, "LOW",
                              f"Which {resolution.query.title()} did you mean? Current matches: {choices}.")
    players = _players_by_id(bundle.predictions, resolution.players)
    evidence: list[str] = []
    details: dict[str, Any] = {"Target GW": target_gw, "Data freshness": "STALE" if bundle.status.stale else "LIVE"}
    payload = dict(base)
    schedule = _schedule_context(bundle.fixture_calendar)

    if intent == "transfer":
        payload["transfer"] = _transfer_context(bundle, settings, players)
        evidence += ["transfer_engine", "ml_projection", "signal_engine", "availability", "squad_state"]
        details.update({"Decision": payload["transfer"]["decision"], "Threshold": settings.minimum_gain,
                        "Legality verified": payload["transfer"]["legality_verified"]})
    elif intent == "captaincy":
        payload["captaincy"] = _captain_context(bundle, settings.risk_profile)
        evidence += ["captaincy_engine", "ml_projection", "availability", "fixture_calendar"]
        selected = payload["captaincy"].get("selected", {})
        details.update({"Profile": settings.risk_profile, "Captain": (selected.get("captain") or {}).get("name")})
    elif intent in {"player_comparison", "player_lookup"}:
        payload["players"] = players
        evidence += ["ml_projection", "signal_engine", "availability", "fixture_calendar"]
        details["Players"] = ", ".join(player["name"] for player in players) if players else "Not resolved"
    elif intent in {"ranking", "budget"}:
        payload["ranking"] = _ranking_context(bundle, question)
        evidence += ["ml_projection", "signal_engine", "availability"]
        details.update({"Budget": payload["ranking"]["budget"], "Position": payload["ranking"]["position"]})
    elif intent in {"fixture", "dgw_bgw"}:
        payload["schedule"] = schedule
        payload["fixture_outlook"] = _fixture_context(bundle, question, target_gw)
        evidence += ["fixture_calendar"]
        details.update({"Confirmed DGWs": len(schedule["double_gameweeks"]),
                        "Confirmed BGWs": len(schedule["blank_gameweeks"])})
    elif intent == "risk":
        payload["risks"] = _risk_players(bundle.squad)
        evidence += ["signal_engine", "availability", "fixture_calendar", "ml_projection", "squad_state"]
        details["Risk players"] = ", ".join(player["name"] for player in payload["risks"])
    elif intent == "chip":
        chip = _chip_context(bundle, settings, target_gw or 1, chip_overrides)
        text = normalize_text(question)
        chip["requested_chip"] = next((name for phrase, name in (("wildcard", "wildcard"),
            ("free hit", "free_hit"), ("bench boost", "bench_boost"),
            ("triple captain", "triple_captain")) if phrase in text), None)
        payload.update(chip); payload["schedule"] = schedule
        evidence += ["chip_planner", "fixture_calendar", "availability", "squad_state"]
        details["Chip states"] = ", ".join(f"{key}: {value}" for key, value in chip["chip_states"].items())
    elif intent == "squad_summary":
        captain = _captain_context(bundle, settings.risk_profile)
        chip = _chip_context(bundle, settings, target_gw or 1, chip_overrides)
        payload["weekly_brief"] = _weekly_payload(bundle, summary, settings, captain, schedule, chip)
        payload["schedule"] = schedule; payload["chip_states"] = chip["chip_states"]
        evidence += ["squad_state", "captaincy_engine", "transfer_engine", "signal_engine",
                     "chip_planner", "fixture_calendar", "ml_projection"]
        details.update({"Projected XI": payload["weekly_brief"]["projected_xi_xpts"],
                        "Transfer": payload["weekly_brief"]["transfer_decision"]})
    else:
        payload["players"] = players
        payload["supported_topics"] = ["squad", "transfers", "captaincy", "players", "fixtures", "chips"]
        evidence += ["squad_state"]
    confidence = _confidence(intent, payload, bundle.status.stale)
    payload["confidence"] = confidence
    return AnalystContext(intent, _clean(payload), tuple(dict.fromkeys(evidence)), _clean(details), confidence)
