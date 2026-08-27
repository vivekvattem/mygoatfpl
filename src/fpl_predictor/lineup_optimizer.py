"""MILP-backed legal starting-XI and bench selection."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class LineupResult:
    starting_11: pd.DataFrame
    bench: pd.DataFrame
    bench_gk: pd.Series
    formation: str
    starting_xpts: float
    forced_unavailable: bool = False


def _solve_lineup(squad: pd.DataFrame, score_column: str, exclude_unavailable: bool) -> np.ndarray | None:
    n = len(squad)
    rows, lower, upper = [], [], []
    rows.append(np.ones(n)); lower.append(11); upper.append(11)
    limits = {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
    for position, (minimum, maximum) in limits.items():
        rows.append(squad.position.eq(position).astype(float).to_numpy())
        lower.append(minimum); upper.append(maximum)
    ub = np.ones(n)
    if exclude_unavailable:
        unavailable = squad.get("availability", pd.Series("available", index=squad.index)).isin(
            ["injured", "suspended/unavailable"]
        )
        ub[unavailable.to_numpy()] = 0
    result = milp(c=-pd.to_numeric(squad[score_column], errors="coerce").fillna(-1e6).to_numpy(),
                  integrality=np.ones(n), bounds=Bounds(np.zeros(n), ub),
                  constraints=LinearConstraint(np.vstack(rows), np.asarray(lower), np.asarray(upper)),
                  options={"disp": False})
    return None if not result.success else np.rint(result.x).astype(int)


def optimize_starting_xi(squad: pd.DataFrame,
                         score_column: str = "availability_adjusted_xpts") -> LineupResult:
    if len(squad) != 15:
        raise ValueError("Starting-XI optimization requires a complete 15-player squad")
    if score_column not in squad:
        raise ValueError(f"Missing lineup score column: {score_column}")
    selected = _solve_lineup(squad, score_column, exclude_unavailable=True)
    forced = False
    if selected is None:
        selected = _solve_lineup(squad, score_column, exclude_unavailable=False)
        forced = True
    if selected is None:
        raise ValueError("No legal starting formation can be built from this squad")
    starters = squad.loc[selected.astype(bool)].copy().sort_values(["position", score_column], ascending=[True, False])
    bench = squad.loc[~selected.astype(bool)].copy()
    bench_gks = bench[bench.position.eq("GK")]
    if len(bench_gks) != 1:
        raise ValueError("A legal optimized bench must contain exactly one goalkeeper")
    outfield = bench[~bench.position.eq("GK")].sort_values(score_column, ascending=False).copy()
    outfield["bench_order"] = range(1, len(outfield) + 1)
    counts = starters.position.value_counts()
    formation = f"{counts.get('DEF', 0)}-{counts.get('MID', 0)}-{counts.get('FWD', 0)}"
    return LineupResult(starters, outfield, bench_gks.iloc[0], formation,
                        float(pd.to_numeric(starters[score_column], errors="coerce").sum()), forced)
