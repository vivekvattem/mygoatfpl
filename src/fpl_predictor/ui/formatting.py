"""Pure dashboard labels and compact table formatting."""

from typing import Any

import pandas as pd


def decision_status_label(decision: str | None) -> str:
    value = (decision or "UNAVAILABLE").upper()
    return {"ROLL TRANSFER": "ROLL TRANSFER", "MAKE TRANSFER": "MAKE TRANSFER",
            "TRANSFER STATE REQUIRED": "TRANSFER STATE REQUIRED"}.get(value, value)


def scenario_mode_label(enabled: bool) -> str:
    return "SCENARIO MODE — selling price assumed current" if enabled else "STRICT MODE"


def money(value: Any) -> str:
    return "Unknown" if value is None or pd.isna(value) else f"£{float(value):.1f}m"


def points(value: Any) -> str:
    return "—" if value is None or pd.isna(value) else f"{float(value):.2f}"


def format_player_table(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    rename = {"player": "Player", "team": "Team", "position": "Pos", "price": "Price",
              "availability_adjusted_xpts": "xPts", "weighted_xpts_3": "3GW",
              "weighted_xpts_5": "5GW", "expected_minutes_proxy": "Minutes",
              "ceiling_score": "Ceiling", "uncertainty_width": "Uncertainty",
              "selected_by_percent": "Ownership", "availability": "Availability",
              "overall_signal": "Overall Signal", "action": "Action", "risk_reason": "Risk Reason",
              "signal_reason": "Why"}
    result = result.rename(columns=rename)
    for column in ("Price", "xPts", "3GW", "5GW", "Minutes", "Ceiling", "Uncertainty", "Ownership"):
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce").round(2)
    return result


def safe_public_summary(payload: dict[str, Any]) -> dict[str, Any]:
    blocked = {"password", "token", "secret", "cookie", "authorization"}
    return {key: value for key, value in payload.items() if key.casefold() not in blocked}


def transfer_signal(net_gain: Any, threshold: float, blocked: bool = False) -> str:
    """A text signal accompanies colour so transfer advice is never colour-only."""
    if blocked or net_gain is None or pd.isna(net_gain):
        return "GREY — BLOCKED"
    value = float(net_gain)
    if value >= threshold:
        return "GREEN — BENEFICIAL"
    if value > 0:
        return "YELLOW — MARGINAL / ROLL"
    return "RED — NEGATIVE"


def transfer_status_badge(decision: str | None, net_gain: Any, threshold: float,
                          blocked: bool = False) -> str:
    if blocked:
        return "GREY — INSUFFICIENT FINANCIAL STATE"
    if (decision or "").upper() == "MAKE TRANSFER":
        return "GREEN — BENEFICIAL MOVE"
    if (decision or "").upper() == "ROLL TRANSFER":
        return "YELLOW — MARGINAL / ROLL"
    if net_gain is not None and not pd.isna(net_gain) and float(net_gain) <= 0:
        return "RED — NEGATIVE"
    return "GREY — READY TO ANALYSE"


def prepare_one_transfer_table(frame: pd.DataFrame, horizon: int, threshold: float) -> pd.DataFrame:
    columns = ["out", "in", "selling_price", "buy_price", "new_bank", "gain_1gw", "gain_3gw", "gain_5gw",
               "hit_cost", f"net_gain_{horizon}gw"]
    result = frame[[column for column in columns if column in frame]].copy()
    net_column = f"net_gain_{horizon}gw"
    result["Signal"] = result.get(net_column, pd.Series(index=result.index, dtype=float)).map(
        lambda value: transfer_signal(value, threshold)
    )
    return result.rename(columns={
        "out": "OUT", "in": "IN", "selling_price": "Price Out", "buy_price": "Price In",
        "new_bank": "Bank After", "gain_1gw": "GW1 Gain", "gain_3gw": "3-GW Gain", "gain_5gw": "5-GW Gain",
        "hit_cost": "Hit", net_column: "Net Gain",
    }).round(2)


def prepare_two_transfer_table(frame: pd.DataFrame, horizon: int, threshold: float) -> pd.DataFrame:
    columns = ["out_1", "out_2", "in_1", "in_2", "hit_cost", "gain_3gw", "gain_5gw", f"net_gain_{horizon}gw"]
    result = frame[[column for column in columns if column in frame]].copy()
    net_column = f"net_gain_{horizon}gw"
    result["Signal"] = result.get(net_column, pd.Series(index=result.index, dtype=float)).map(
        lambda value: transfer_signal(value, threshold)
    )
    return result.rename(columns={
        "out_1": "OUT 1", "out_2": "OUT 2", "in_1": "IN 1", "in_2": "IN 2", "hit_cost": "Hit",
        "gain_3gw": "3-GW Gain", "gain_5gw": "5-GW Gain", net_column: "Net Gain",
    }).round(2)


def prepare_replacement_table(frame: pd.DataFrame, predictions: pd.DataFrame,
                              horizon: int, threshold: float) -> pd.DataFrame:
    """Attach existing live-projection context to Phase 6 replacement paths."""
    incoming_columns = [column for column in ["player_id", "team", "price", "availability_adjusted_xpts",
                        "weighted_xpts_3", "weighted_xpts_5", "availability", "overall_signal", "action"]
                        if column in predictions]
    incoming = predictions[incoming_columns].rename(columns={"player_id": "in_id"})
    result = frame.merge(incoming, on="in_id", how="left") if "in_id" in frame else frame.copy()
    net_column = f"net_gain_{horizon}gw"
    columns = ["in", "team", "price", "availability_adjusted_xpts", "weighted_xpts_3", "weighted_xpts_5",
               net_column, "availability", "overall_signal", "action"]
    result = result[[column for column in columns if column in result]].copy()
    result["Signal"] = result.get(net_column, pd.Series(index=result.index, dtype=float)).map(
        lambda value: transfer_signal(value, threshold)
    )
    return result.rename(columns={
        "in": "Player", "team": "Team", "price": "Price", "availability_adjusted_xpts": "Next GW xPts",
        "weighted_xpts_3": "3-GW xPts", "weighted_xpts_5": "5-GW xPts", net_column: "Expected Gain",
        "availability": "Availability", "overall_signal": "Overall Signal", "action": "Action",
    }).round(2)
