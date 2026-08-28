from fpl_predictor.analyst.deterministic import deterministic_answer


def test_transfer_fallback_explains_threshold_and_roll():
    payload = {"transfer": {"legality_verified": True, "decision": "ROLL TRANSFER", "horizon": 5,
                            "threshold": 1.5, "selling_price_scenario": False,
                            "best_transfer": {"out": "A", "in": "B", "net_gain_5gw": 1.41}}}
    answer = deterministic_answer("transfer", payload, "HIGH")
    assert "ROLL TRANSFER" in answer and "1.41" in answer and "1.50" in answer


def test_captain_fallback_uses_structured_candidate():
    player = {"name": "Haaland", "xpts_next": 5.2, "xpts_5gw": 24.0, "overall_signal": "GREEN",
              "action": "HOLD", "expected_minutes": 85, "ceiling": 72, "uncertainty_width": 4}
    payload = {"captaincy": {"selected": {"captain": player, "vice": {"name": "Palmer"}}}}
    answer = deterministic_answer("captaincy", payload, "MODERATE")
    assert "Captain: Haaland" in answer and "Ceiling score" in answer


def test_chip_fallback_keeps_unknown_state_explicit():
    row = {f"{chip}_signal": "GREY" for chip in ("wildcard", "free_hit", "bench_boost", "triple_captain")}
    for chip in ("wildcard", "free_hit", "bench_boost", "triple_captain"):
        row[f"{chip}_reason"] = "State unknown"
    payload = {"chip_states": {"wildcard": "unknown", "free_hit": "unknown",
                               "bench_boost": "unknown", "triple_captain": "unknown"}, "plan": [row]}
    answer = deterministic_answer("chip", payload, "LOW")
    assert "availability is not verified" in answer and "GREY" in answer


def test_no_data_fallback_does_not_invent_player():
    answer = deterministic_answer("player_lookup", {"players": []}, "LOW")
    assert "don't currently have verified data" in answer

