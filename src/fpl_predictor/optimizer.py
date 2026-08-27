"""Core legal FPL optimization primitives."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from .squad_state import SQUAD_POSITION_COUNTS, validate_squad


@dataclass(frozen=True)
class OptimalSquadResult:
    optimal_15: pd.DataFrame
    budget_used: float
    projected_1gw: float
    projected_3gw: float | None
    projected_5gw: float | None


def best_15_player_squad(players: pd.DataFrame, budget: float,
                         score_column: str = "weighted_xpts_5") -> OptimalSquadResult:
    """Experimental full-squad MILP; this is not labelled a Wildcard plan."""
    if budget <= 0:
        raise ValueError("Budget must be positive")
    required = {"player_id", "position", "team", "price", score_column}
    missing = required - set(players.columns)
    if missing:
        raise ValueError(f"Player pool missing columns: {sorted(missing)}")
    pool = players.drop_duplicates("player_id").reset_index(drop=True)
    n = len(pool); rows = []; lower = []; upper = []
    rows.append(np.ones(n)); lower.append(15); upper.append(15)
    for position, count in SQUAD_POSITION_COUNTS.items():
        rows.append(pool.position.eq(position).astype(float).to_numpy()); lower.append(count); upper.append(count)
    for team in sorted(pool.team.dropna().unique()):
        rows.append(pool.team.eq(team).astype(float).to_numpy()); lower.append(0); upper.append(3)
    rows.append(pd.to_numeric(pool.price, errors="coerce").fillna(1e6).to_numpy()); lower.append(0); upper.append(budget)
    result = milp(c=-pd.to_numeric(pool[score_column], errors="coerce").fillna(-1e6).to_numpy(),
                  integrality=np.ones(n), bounds=Bounds(np.zeros(n), np.ones(n)),
                  constraints=LinearConstraint(np.vstack(rows), np.asarray(lower), np.asarray(upper)),
                  options={"disp": False})
    if not result.success:
        raise ValueError("No legal 15-player squad exists for the supplied budget and pool")
    selected = pool.loc[np.rint(result.x).astype(bool)].copy()
    validate_squad(selected)
    def total(column: str) -> float | None:
        return float(selected[column].sum()) if column in selected else None
    return OptimalSquadResult(selected, float(selected.price.sum()), total("weighted_xpts_1") or 0.0,
                              total("weighted_xpts_3"), total("weighted_xpts_5"))
