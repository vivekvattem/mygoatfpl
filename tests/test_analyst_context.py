from datetime import datetime, timezone

import pandas as pd

from fpl_predictor.analyst.context import build_analyst_context
from fpl_predictor.analyst.deterministic import deterministic_answer
from fpl_predictor.ui.data import DashboardBundle, DataStatus
from fpl_predictor.ui.state import AppSettings


def analyst_bundle(legal_squad):
    players = legal_squad.copy()
    names = ["Erling Haaland", "João Pedro", "Cole Palmer"] + [f"Player {i}" for i in range(4, 16)]
    players["player"] = names
    players["web_name"] = ["Haaland", "João Pedro", "Palmer"] + [f"P{i}" for i in range(4, 16)]
    players["owned"] = True
    players["total_xpts_3"] = players.availability_adjusted_xpts * 3
    players["total_xpts_5"] = players.availability_adjusted_xpts * 5
    players["ceiling_score"] = players.availability_adjusted_xpts * 8
    players["uncertainty_width"] = 4.0
    players["minutes_confidence"] = "high"
    players["fixture_signal"] = "YELLOW"; players["overall_signal"] = "GREEN"
    players["availability_signal"] = "GREEN"; players["minutes_signal"] = "GREEN"
    players["action"] = "HOLD"; players["risk_reason"] = "No major structured red flag"
    players["signal_reason"] = "+ minutes"; players["average_fdr_5"] = 3.0
    players["fixtures_next_5"] = 5
    for offset in range(1, 6):
        players[f"xpts_gw{offset}"] = players.availability_adjusted_xpts
        players[f"fixture_count_gw{offset}"] = 1
    squad = players.copy(); squad["multiplier"] = [1] * 11 + [0] * 4
    xi = players.head(11).copy()
    calendar = pd.DataFrame([
        {"team": team, "gw": gw, "fixture_count": 1, "is_blank": False, "is_double": False,
         "is_triple": False, "schedule_label": "NORMAL"}
        for team in sorted(players.team.unique()) for gw in range(2, 12)
    ])
    replacement = pd.DataFrame({"out_id": [1], "out": [names[0]], "in_id": [2], "in": [names[1]],
                                "net_gain_5gw": [1.4], "new_bank": [0.1], "hit_cost": [0]})
    decision = {"target_gw": 2, "entry_id": 1, "squad_source": "manual_file", "optimized_xi_xpts": 60.0,
                "transfer_decision": "ROLL TRANSFER", "best_transfer": {"out": names[0], "in": names[1],
                "out_id": 1, "in_id": 2, "net_gain_5gw": 1.4}}
    return DashboardBundle(players, squad, xi, replacement, pd.DataFrame(), replacement,
                           pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), decision,
                           {"target_gw": 2, "entry_history_chips": []}, {},
                           DataStatus(True, False, datetime.now(timezone.utc), "ok"), calendar, pd.DataFrame())


def test_transfer_and_comparison_contexts_are_compact_and_structured(legal_squad):
    bundle = analyst_bundle(legal_squad); settings = AppSettings(horizon=5)
    transfer = build_analyst_context(bundle, settings, "Should I sell Erling Haaland?")
    assert transfer.intent == "transfer" and transfer.payload["transfer"]["legality_verified"]
    assert transfer.payload["transfer"]["players"][0]["name"] == "Erling Haaland"
    comparison = build_analyst_context(bundle, settings, "Compare Cole Palmer and Haaland")
    assert comparison.intent == "player_comparison" and len(comparison.payload["players"]) == 2
    assert "ml_projection" in comparison.evidence


def test_captain_and_chip_contexts_reuse_existing_engines(legal_squad):
    bundle = analyst_bundle(legal_squad); settings = AppSettings()
    captain = build_analyst_context(bundle, settings, "Who should I captain?")
    assert captain.payload["captaincy"]["selected"]["captain"]["name"]
    chip = build_analyst_context(bundle, settings, "Should I Bench Boost?")
    assert len(chip.payload["plan"]) == 8
    assert chip.payload["chip_states"]["bench_boost"] == "unknown"


def test_weekly_brief_contains_all_primary_decisions(legal_squad):
    context = build_analyst_context(analyst_bundle(legal_squad), AppSettings(), "What should I do this week?")
    answer = deterministic_answer(context.intent, context.payload, context.confidence)
    for label in ("Projected XI", "Captain", "Vice", "Transfer", "Main risk", "Chip", "Schedule"):
        assert label in answer

