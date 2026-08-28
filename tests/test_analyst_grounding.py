from fpl_predictor.analyst.grounding import (
    validate_answer, validate_chip_claims, validate_numeric_claims,
    validate_player_mentions, validate_schedule_claims,
)


def test_grounding_rejects_player_not_supplied_to_context():
    context = {"player": {"name": "Cole Palmer"}}
    universe = {"Cole Palmer", "Erling Haaland"}
    assert validate_player_mentions("Cole Palmer is the pick.", context, universe)
    assert not validate_player_mentions("Erling Haaland is the pick.", context, universe)


def test_grounding_rejects_unsupported_material_number():
    context = {"gain": 1.41, "threshold": 1.5}
    assert validate_numeric_claims("The gain is 1.41 projected points.", context)
    assert not validate_numeric_claims("The gain is 9.99 projected points.", context)


def test_grounding_rejects_false_schedule_claim():
    context = {"schedule": {"double_gameweeks": [], "blank_gameweeks": []}}
    assert validate_schedule_claims("No confirmed Double Gameweek exists.", context)
    assert not validate_schedule_claims("A DGW is confirmed next week.", context)


def test_grounding_rejects_false_chip_state_and_combines_checks():
    context = {"chip_states": {"wildcard": "unknown"},
               "schedule": {"double_gameweeks": [], "blank_gameweeks": []}}
    assert not validate_chip_claims("Your Wildcard is available.", context)
    result = validate_answer("Your Wildcard is available.", context, set())
    assert not result.passed and "unsupported chip-state claim" in result.failures

