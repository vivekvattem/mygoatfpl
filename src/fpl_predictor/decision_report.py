"""Text rendering for personalized Phase 6 decisions."""

import pandas as pd


LIMITATIONS = """Current xPts estimates are optimized for average expected-points ranking.

The production model compresses high-ceiling outcomes, so captaincy and
aggressive differential recommendations also use a separate transparent
ceiling heuristic.

Future-GW projections assume current form/minutes information remains
approximately stable and should be treated as planning estimates."""


def render_decision_report(entry_id: int, target_gw: int, lineup, captaincy,
                           decision: str, horizon: int, risk_profile: str,
                           one_transfers: pd.DataFrame | None = None,
                           two_transfers: pd.DataFrame | None = None,
                           replacements: pd.DataFrame | None = None,
                           transfer_note: str | None = None) -> str:
    starters = lineup.starting_11[["player", "position", "availability_adjusted_xpts"]]
    bench_rows = [f"GK: {lineup.bench_gk.player}"] + [f"{int(row.bench_order)}. {row.player}"
                                                       for row in lineup.bench.itertuples(index=False)]
    def projected(column: str) -> float | None:
        return float(lineup.starting_11[column].sum()) if column in lineup.starting_11 else None
    one_gw = projected("xpts_gw1") or projected("availability_adjusted_xpts")
    three_total, three_weighted = projected("total_xpts_3"), projected("weighted_xpts_3")
    five_total, five_weighted = projected("total_xpts_5"), projected("weighted_xpts_5")
    outlook = [f"CURRENT XI xPTS — GW+1\n{one_gw:.2f}" if one_gw is not None else "CURRENT XI xPTS\nUnavailable"]
    if three_total is not None and three_weighted is not None:
        outlook.append(f"3-GW OUTLOOK\n{three_total:.2f} unweighted | {three_weighted:.2f} weighted")
    if five_total is not None and five_weighted is not None:
        outlook.append(f"5-GW OUTLOOK\n{five_total:.2f} unweighted | {five_weighted:.2f} weighted")
    parts = [f"FPL DECISION REPORT — GW{target_gw}",
             f"ENTRY\n{entry_id}",
             f"PLANNING\n{horizon}-GW horizon | {risk_profile} risk profile",
             "PROJECTED XI\n" + starters.to_string(index=False, float_format=lambda x: f"{x:.2f}"),
             f"FORMATION\n{lineup.formation}",
             f"CAPTAIN\n{captaincy.captain.player} — score {captaincy.captain.captaincy_score:.1f}",
             f"VICE CAPTAIN\n{captaincy.vice_captain.player} — score {captaincy.vice_captain.captaincy_score:.1f}",
             "BENCH\n" + "\n".join(bench_rows),
             f"OPTIMIZED XI PLANNING xPTS\n{lineup.starting_xpts:.2f}",
             *outlook,
             f"TRANSFER DECISION\n{decision}"]
    if transfer_note:
        parts.append("TRANSFER STATE\n" + transfer_note)
    if one_transfers is not None and not one_transfers.empty:
        columns = ["out", "in", f"net_gain_{horizon}gw", "new_bank", "hit_cost"]
        parts.append("BEST 1-TRANSFER MOVES\n" + one_transfers.head(5)[columns].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    if two_transfers is not None and not two_transfers.empty:
        columns = ["out_1", "out_2", "in_1", "in_2", f"net_gain_{horizon}gw", "hit_cost"]
        parts.append("BEST 2-TRANSFER MOVES\n" + two_transfers.head(5)[columns].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    if replacements is not None and not replacements.empty:
        columns = ["out", "in", f"net_gain_{horizon}gw", "new_bank"]
        parts.append("TOP LEGAL REPLACEMENT OPTIONS\n" + replacements.head(10)[columns].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    if lineup.forced_unavailable:
        parts.append("RISK FLAGS\nAn unavailable player was required to satisfy a legal formation.")
    parts.append("MODEL LIMITATIONS\n" + LIMITATIONS)
    return "\n\n".join(parts)
