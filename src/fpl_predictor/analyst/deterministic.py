"""Useful analyst answers that never require an LLM provider."""

from typing import Any


def _number(value: Any, digits: int = 2) -> str:
    return "unavailable" if value is None else f"{float(value):.{digits}f}"


def _player_line(player: dict[str, Any]) -> str:
    return (f"**{player.get('name', 'Unknown')}** — {_number(player.get('xpts_next'))} next-GW xPts, "
            f"{_number(player.get('xpts_5gw'))} over 5 GWs, {player.get('overall_signal', 'GREY')} / "
            f"{player.get('action', 'WATCH')}, expected minutes {_number(player.get('expected_minutes'), 0)}.")


def build_weekly_brief(payload: dict[str, Any], target_gw: int | None, confidence: str) -> str:
    brief = payload.get("weekly_brief", {})
    captain = brief.get("captain") or {}; vice = brief.get("vice") or {}
    risk = brief.get("top_risk") or {}; opportunity = brief.get("top_opportunity") or {}
    schedule = brief.get("schedule", {})
    schedule_text = ("No confirmed DGW/BGW in the current window" if not schedule.get("double_gameweeks")
                     and not schedule.get("blank_gameweeks") else "Confirmed schedule alerts are present")
    chip_states = (brief.get("chip") or {}).get("chip_states", {})
    chip_text = ("Chip availability is unverified" if not any(value == "available" for value in chip_states.values())
                 else "Review the strongest available chip signal")
    return f"""### GW{target_gw or '—'} Brief

**Projected XI:** {_number(brief.get('projected_xi_xpts'))} xPts  
**Captain:** {captain.get('name', 'Unavailable')}  
**Vice:** {vice.get('name', 'Unavailable')}  
**Transfer:** {brief.get('transfer_decision', 'UNAVAILABLE')}  
**Top opportunity:** {opportunity.get('name', 'Unavailable')}  
**Main risk:** {risk.get('name', 'Unavailable')} — {risk.get('risk_reason', 'no verified risk detail')}  
**Chip:** {chip_text}  
**Schedule:** {schedule_text}  
**Confidence:** {confidence}"""


def deterministic_answer(intent: str, payload: dict[str, Any], confidence: str) -> str:
    if intent == "squad_summary":
        return build_weekly_brief(payload, payload.get("target_gw"), confidence)
    if intent == "transfer":
        transfer = payload.get("transfer", {})
        players = transfer.get("players", [])
        if not transfer.get("legality_verified"):
            player_text = f"\n\n{_player_line(players[0])}" if players else ""
            return ("### Recommendation: FINANCIAL STATE REQUIRED\n\n"
                    "I cannot verify a legal transfer or roll decision because the current optimizer result, bank, "
                    "free transfers, or authoritative selling prices are unavailable. I will not guess them."
                    f"{player_text}\n\n**Confidence:** LOW")
        decision = transfer.get("decision", "UNAVAILABLE")
        best = transfer.get("best_transfer", {})
        gain = best.get(f"net_gain_{transfer.get('horizon')}gw")
        move = f"{best.get('out', '—')} → {best.get('in', '—')}"
        return (f"### Recommendation: {decision}\n\n**Best legal move:** {move}  \n"
                f"**Projected gain:** {_number(gain)} points over {transfer.get('horizon')} GWs  \n"
                f"**Threshold:** {_number(transfer.get('threshold'))} points  \n"
                f"**Scenario selling prices:** {'Yes' if transfer.get('selling_price_scenario') else 'No'}  \n"
                f"**Confidence:** {confidence}")
    if intent == "captaincy":
        selected = payload.get("captaincy", {}).get("selected", {})
        captain, vice = selected.get("captain") or {}, selected.get("vice") or {}
        if not captain:
            return "### Captain recommendation unavailable\n\nRun or refresh the optimized XI first.\n\n**Confidence:** LOW"
        return (f"### Captain: {captain.get('name')}\n\n{_player_line(captain)}\n\n"
                f"**Vice:** {vice.get('name', 'Unavailable')}  \n"
                f"**Ceiling score:** {_number(captain.get('ceiling'))}  \n"
                f"**Uncertainty width:** {_number(captain.get('uncertainty_width'))}  \n"
                f"**Confidence:** {confidence}")
    if intent == "player_comparison":
        players = payload.get("players", [])
        if len(players) != 2:
            return "### Comparison needs two unambiguous players\n\nPlease provide two full player names.\n\n**Confidence:** LOW"
        winner = max(players, key=lambda player: player.get("xpts_5gw") or float("-inf"))
        rows = ["| Metric | " + " | ".join(player["name"] for player in players) + " |", "|---|---:|---:|"]
        for label, key in (("Next GW xPts", "xpts_next"), ("3-GW xPts", "xpts_3gw"),
                           ("5-GW xPts", "xpts_5gw"), ("Expected minutes", "expected_minutes"),
                           ("Fixture signal", "fixture_signal"), ("Overall signal", "overall_signal"),
                           ("Ceiling", "ceiling"), ("Availability", "availability")):
            rows.append(f"| {label} | " + " | ".join(str(player.get(key, "—")) for player in players) + " |")
        return (f"### Preference: {winner['name']}\n\n" + "\n".join(rows) +
                f"\n\nThe preference follows the supplied five-GW projection and structured signals.\n\n**Confidence:** {confidence}")
    if intent == "player_lookup":
        players = payload.get("players", [])
        return (_player_line(players[0]) + f"\n\n**Confidence:** {confidence}" if players else
                "I don't currently have verified data for that player. Please use a full current FPL name.\n\n**Confidence:** LOW")
    if intent in {"ranking", "budget"}:
        ranking = payload.get("ranking", {}); candidates = ranking.get("candidates", [])
        if not candidates:
            return "No eligible players match the supplied structured filters.\n\n**Confidence:** LOW"
        constraint = []
        if ranking.get("position"): constraint.append(ranking["position"])
        if ranking.get("budget") is not None: constraint.append(f"under £{ranking['budget']:.1f}m")
        lines = [f"{index}. {_player_line(player)}" for index, player in enumerate(candidates, 1)]
        return f"### Top projections {' '.join(constraint)}\n\n" + "\n".join(lines) + f"\n\n**Confidence:** {confidence}"
    if intent in {"fixture", "dgw_bgw"}:
        schedule = payload.get("schedule", {})
        if schedule.get("status") == "unavailable":
            return "Confirmed official fixture data is unavailable. I will not speculate.\n\n**Confidence:** LOW"
        doubles, blanks = schedule.get("double_gameweeks", []), schedule.get("blank_gameweeks", [])
        if intent == "fixture":
            outlook = payload.get("fixture_outlook", {})
            fixtures, runs = outlook.get("fixtures", []), outlook.get("best_runs", [])
            if outlook.get("matched_teams") and fixtures:
                lines = []
                for item in fixtures:
                    opponents = ", ".join(item.get("opponents") or []) or "no fixture"
                    venues = "/".join(item.get("home_away") or [])
                    lines.append(f"- **GW{item.get('gw')}:** {opponents} {f'({venues})' if venues else ''} · "
                                 f"FDR {_number(item.get('average_fdr'), 1)} · {item.get('schedule_label')}")
                return (f"### Confirmed fixtures: {', '.join(outlook['matched_teams'])}\n\n" + "\n".join(lines) +
                        f"\n\nOnly official FPL fixtures are included.\n\n**Confidence:** {confidence}")
            if runs:
                lines = [f"{index}. **{item.get('team')}** — {item.get('fixture_signal')} · average FDR "
                         f"{_number(item.get('average_fdr_5'), 1)} · {item.get('fixtures_next_5')} fixtures"
                         for index, item in enumerate(runs, 1)]
                return "### Best confirmed fixture runs\n\n" + "\n".join(lines) + f"\n\n**Confidence:** {confidence}"
        if not doubles and not blanks:
            return ("### Confirmed schedule\n\nNo confirmed Double or Blank Gameweek is currently present in the "
                    f"planning window.\n\n**Confidence:** {confidence}")
        return (f"### Confirmed schedule\n\n**DGWs:** {doubles or 'None'}  \n**BGWs:** {blanks or 'None'}  \n"
                f"Only official FPL fixtures are included.\n\n**Confidence:** {confidence}")
    if intent == "risk":
        risks = payload.get("risks", [])
        if not risks:
            return "Squad risk data is unavailable.\n\n**Confidence:** LOW"
        lines = [f"{index}. **{player['name']}** — {player.get('overall_signal')} · {player.get('risk_reason')}"
                 for index, player in enumerate(risks, 1)]
        return "### Biggest structured squad risks\n\n" + "\n".join(lines) + f"\n\n**Confidence:** {confidence}"
    if intent == "chip":
        states, plan = payload.get("chip_states", {}), payload.get("plan", [])
        if not plan:
            return "Chip evaluation data is unavailable.\n\n**Confidence:** LOW"
        requested = payload.get("requested_chip")
        if requested:
            scored = [row for row in plan if row.get(f"{requested}_score") is not None]
            row = max(scored, key=lambda item: item[f"{requested}_score"]) if scored else plan[0]
            label = requested.replace("_", " ").title()
            state = states.get(requested, "unknown")
            return (f"### {label}: {row.get(requested + '_signal', 'GREY')}\n\n"
                    f"**Gameweek:** GW{row.get('gw')}  \n"
                    f"**Score:** {_number(row.get(requested + '_score'))}  \n"
                    f"**Main reason:** {row.get(requested + '_reason', 'No reason available')}  \n"
                    f"**Main risk:** Chip state is {state}; projections are estimates, not guarantees.  \n"
                    f"**Confidence:** {confidence}")
        row = plan[0]
        lines = []
        for chip in ("wildcard", "free_hit", "bench_boost", "triple_captain"):
            label = chip.replace("_", " ").title()
            lines.append(f"- **{label}: {row.get(chip + '_signal', 'GREY')}** — {row.get(chip + '_reason', 'No reason available')}")
        note = ("Your chip availability is not verified, so opportunity signals remain GREY."
                if not any(value == "available" for value in states.values()) else
                "Signals apply only to chips marked available.")
        return "### Chip outlook\n\n" + "\n".join(lines) + f"\n\n{note}\n\n**Confidence:** {confidence}"
    return ("I can explain your current squad, transfers, captaincy, player comparisons, rankings, confirmed "
            "fixtures, risks, and chip plan. Ask a question using current FPL players or one of those topics.\n\n"
            f"**Confidence:** {confidence}")
