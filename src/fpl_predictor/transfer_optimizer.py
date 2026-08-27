"""Legal, financial-state-aware one- and two-transfer analysis."""

from itertools import combinations
import warnings

import numpy as np
import pandas as pd

from .squad_state import SquadState, require_transfer_state, validate_squad

LEGAL_FORMATIONS = ((3, 4, 3), (3, 5, 2), (4, 4, 2), (4, 3, 3),
                    (4, 5, 1), (5, 4, 1), (5, 3, 2), (5, 2, 3))


def transfer_hit_cost(transfer_count: int, free_transfers: int) -> int:
    if transfer_count < 0 or free_transfers < 0:
        raise ValueError("Transfer counts cannot be negative")
    return 4 * max(0, transfer_count - free_transfers)


def attach_squad_projections(state: SquadState, projections: pd.DataFrame) -> pd.DataFrame:
    projection_columns = [column for column in projections if column not in {"player_id", "player", "team", "position", "price"}]
    merged = state.players.merge(projections[["player_id", *projection_columns]], on="player_id", how="left", validate="one_to_one")
    if merged["weighted_xpts_1"].isna().any():
        raise ValueError("Some squad players are missing projections")
    validate_squad(merged)
    return merged


def _sale_value(row: pd.Series, assume_current: bool) -> float:
    value = pd.to_numeric(row.get("selling_price"), errors="coerce")
    if pd.isna(value):
        if not assume_current:
            raise ValueError(f"Selling price is unknown for {row.get('player', row.player_id)}")
        value = pd.to_numeric(row.get("current_price", row.get("price")), errors="coerce")
    return float(value)


def _legal_replacement(squad: pd.DataFrame, out_id: int, candidate: pd.Series,
                       available_funds: float) -> bool:
    outgoing = squad.loc[squad.player_id.eq(out_id)].iloc[0]
    if candidate.player_id in set(squad.player_id) or candidate.position != outgoing.position:
        return False
    if float(candidate.price) > available_funds + 1e-9:
        return False
    clubs = squad.loc[~squad.player_id.eq(out_id), "team"].value_counts().to_dict()
    return clubs.get(candidate.team, 0) < 3


def _replace(squad: pd.DataFrame, out_id: int, candidate: pd.Series) -> pd.DataFrame:
    result = squad.loc[~squad.player_id.eq(out_id)].copy().reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*DataFrame concatenation.*")
        result.loc[len(result)] = candidate.reindex(result.columns)
    result["player_id"] = pd.to_numeric(result.player_id).astype(int)
    validate_squad(result)
    return result


def _xi_scores(squad: pd.DataFrame) -> dict[int, float]:
    """Exact fixed-squad scorer over every legal formation.

    Full-universe and primary lineup selection remain MILP-backed. Repeated
    transfer-path evaluation needs only eight deterministic formation checks.
    """
    scores = {}
    unavailable = squad.get("availability", pd.Series("available", index=squad.index)).isin(
        ["injured", "suspended/unavailable"]
    )
    eligible = squad.loc[~unavailable]
    needed = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
    if any(eligible.position.value_counts().get(position, 0) < count for position, count in needed.items()):
        eligible = squad
    for horizon in (1, 3, 5):
        column = f"weighted_xpts_{horizon}"
        if column in squad:
            position_scores = {
                position: np.sort(pd.to_numeric(eligible.loc[eligible.position.eq(position), column],
                                                errors="coerce").fillna(-1e6).to_numpy())[::-1]
                for position in ("GK", "DEF", "MID", "FWD")
            }
            formation_scores = []
            for defenders, midfielders, forwards in LEGAL_FORMATIONS:
                counts = {"GK": 1, "DEF": defenders, "MID": midfielders, "FWD": forwards}
                if all(len(position_scores[position]) >= count for position, count in counts.items()):
                    formation_scores.append(sum(float(position_scores[position][:count].sum())
                                                for position, count in counts.items()))
            if not formation_scores:
                raise ValueError("No legal starting XI exists for transfer-path scoring")
            scores[horizon] = max(formation_scores)
    return scores


def optimize_one_transfer(state: SquadState, squad: pd.DataFrame, universe: pd.DataFrame,
                          assume_selling_price_current: bool = False,
                          selected_horizon: int = 5,
                          candidate_limit_per_position: int = 15) -> pd.DataFrame:
    require_transfer_state(state, assume_selling_price_current)
    baseline = _xi_scores(squad); rows = []
    owned = set(squad.player_id)
    unavailable = universe.get("availability", pd.Series("available", index=universe.index)).isin(
        ["injured", "suspended/unavailable"]
    )
    unowned = universe.loc[~universe.player_id.isin(owned) & ~unavailable].copy()
    score_column = f"weighted_xpts_{selected_horizon}"
    candidate_groups = []
    for _, group in unowned.groupby("position"):
        ranked = group.assign(_transfer_value=pd.to_numeric(group[score_column], errors="coerce") /
                              pd.to_numeric(group.price, errors="coerce").replace(0, np.nan))
        candidate_groups.extend([ranked.nlargest(candidate_limit_per_position, score_column),
                                 ranked.nlargest(max(5, candidate_limit_per_position // 2), "_transfer_value")])
    candidates = pd.concat(candidate_groups).drop_duplicates("player_id").drop(columns="_transfer_value")
    for outgoing in squad.itertuples(index=False):
        out = pd.Series(outgoing._asdict())
        sale = _sale_value(out, assume_selling_price_current)
        funds = float(state.bank) + sale
        for candidate in candidates[candidates.position.eq(out.position)].itertuples(index=False):
            incoming = pd.Series(candidate._asdict())
            if not _legal_replacement(squad, int(out.player_id), incoming, funds):
                continue
            new_squad = _replace(squad, int(out.player_id), incoming)
            scores = _xi_scores(new_squad); hit = transfer_hit_cost(1, int(state.free_transfers))
            row = {"out_id": int(out.player_id), "out": out.player, "in_id": int(incoming.player_id),
                   "in": incoming.player, "selling_price": sale, "buy_price": float(incoming.price),
                   "new_bank": funds - float(incoming.price), "hit_cost": hit}
            for horizon, score in scores.items():
                row[f"gain_{horizon}gw"] = score - baseline[horizon]
                row[f"net_gain_{horizon}gw"] = row[f"gain_{horizon}gw"] - hit
            rows.append(row)
    result = pd.DataFrame(rows)
    sort = f"net_gain_{selected_horizon}gw"
    return result.sort_values(sort, ascending=False).reset_index(drop=True) if not result.empty else result


def optimize_two_transfers(state: SquadState, squad: pd.DataFrame, universe: pd.DataFrame,
                           assume_selling_price_current: bool = False,
                           selected_horizon: int = 5,
                           candidate_limit_per_position: int = 3,
                           outgoing_candidate_limit: int = 6) -> pd.DataFrame:
    require_transfer_state(state, assume_selling_price_current)
    baseline = _xi_scores(squad); rows = []; seen = set()
    owned = set(squad.player_id)
    score_column = f"weighted_xpts_{selected_horizon}"
    pools = {}
    for position in ("GK", "DEF", "MID", "FWD"):
        unavailable = universe.get("availability", pd.Series("available", index=universe.index)).isin(
            ["injured", "suspended/unavailable"]
        )
        group = universe[(universe.position.eq(position)) & ~universe.player_id.isin(owned) & ~unavailable].copy()
        group["_transfer_value"] = pd.to_numeric(group[score_column], errors="coerce") / pd.to_numeric(group.price, errors="coerce").replace(0, np.nan)
        pools[position] = pd.concat([
            group.nlargest(candidate_limit_per_position, score_column),
            group.nlargest(candidate_limit_per_position, "_transfer_value"),
            group.nsmallest(candidate_limit_per_position, "price"),
        ]).drop_duplicates("player_id").drop(columns="_transfer_value")
    outgoing_pool = squad.nsmallest(outgoing_candidate_limit, score_column)
    for out_a, out_b in combinations([pd.Series(row._asdict()) for row in outgoing_pool.itertuples(index=False)], 2):
        sale_a = _sale_value(out_a, assume_selling_price_current)
        sale_b = _sale_value(out_b, assume_selling_price_current)
        funds = float(state.bank) + sale_a + sale_b
        for in_a in pools[out_a.position].itertuples(index=False):
            incoming_a = pd.Series(in_a._asdict())
            for in_b in pools[out_b.position].itertuples(index=False):
                incoming_b = pd.Series(in_b._asdict())
                if incoming_a.player_id == incoming_b.player_id:
                    continue
                key = (tuple(sorted((int(out_a.player_id), int(out_b.player_id)))),
                       tuple(sorted((int(incoming_a.player_id), int(incoming_b.player_id)))))
                if key in seen or float(incoming_a.price) + float(incoming_b.price) > funds + 1e-9:
                    continue
                new_squad = squad.loc[~squad.player_id.isin([out_a.player_id, out_b.player_id])].copy().reset_index(drop=True)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning, message=".*DataFrame concatenation.*")
                    new_squad.loc[len(new_squad)] = incoming_a.reindex(new_squad.columns)
                    new_squad.loc[len(new_squad)] = incoming_b.reindex(new_squad.columns)
                new_squad["player_id"] = pd.to_numeric(new_squad.player_id).astype(int)
                try:
                    validate_squad(new_squad)
                except ValueError:
                    continue
                seen.add(key); scores = _xi_scores(new_squad)
                hit = transfer_hit_cost(2, int(state.free_transfers))
                row = {"out_1_id": int(out_a.player_id), "out_1": out_a.player,
                       "out_2_id": int(out_b.player_id), "out_2": out_b.player,
                       "in_1_id": int(incoming_a.player_id), "in_1": incoming_a.player,
                       "in_2_id": int(incoming_b.player_id), "in_2": incoming_b.player,
                       "new_bank": funds - float(incoming_a.price) - float(incoming_b.price),
                       "hit_cost": hit}
                for horizon, score in scores.items():
                    row[f"gain_{horizon}gw"] = score - baseline[horizon]
                    row[f"net_gain_{horizon}gw"] = row[f"gain_{horizon}gw"] - hit
                rows.append(row)
    result = pd.DataFrame(rows); sort = f"net_gain_{selected_horizon}gw"
    return result.sort_values(sort, ascending=False).head(10).reset_index(drop=True) if not result.empty else result


def replacement_shortlists(one_transfers: pd.DataFrame, limit: int = 5,
                           horizon: int = 5) -> pd.DataFrame:
    if one_transfers.empty:
        return one_transfers.copy()
    return (one_transfers.sort_values(f"net_gain_{horizon}gw", ascending=False)
            .groupby("out_id", as_index=False, group_keys=False).head(limit).reset_index(drop=True))


def transfer_decision(one_transfers: pd.DataFrame, two_transfers: pd.DataFrame,
                      horizon: int = 5, minimum_gain: float = 1.5) -> str:
    candidates = []
    for frame in (one_transfers, two_transfers):
        column = f"net_gain_{horizon}gw"
        if not frame.empty and column in frame:
            candidates.append(float(frame[column].max()))
    return "MAKE TRANSFER" if candidates and max(candidates) >= minimum_gain else "ROLL TRANSFER"
