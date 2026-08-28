"""Pure dashboard labels and compact table formatting."""

from typing import Any

import numpy as np
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
              "selected_by_percent": "Ownership", "availability": "Availability"}
    result = result.rename(columns=rename)
    for column in ("Price", "xPts", "3GW", "5GW", "Minutes", "Ceiling", "Uncertainty", "Ownership"):
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce").round(2)
    return result


def safe_public_summary(payload: dict[str, Any]) -> dict[str, Any]:
    blocked = {"password", "token", "secret", "cookie", "authorization"}
    return {key: value for key, value in payload.items() if key.casefold() not in blocked}
