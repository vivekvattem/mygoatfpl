import pandas as pd

from fpl_predictor.analyst.intents import detect_intent, resolve_player_name, resolve_question_players


def _players():
    return pd.DataFrame({
        "player_id": [1, 2, 3, 4],
        "player": ["Erling Haaland", "João Pedro Junqueira de Jesus", "Cole Palmer", "Alex Palmer"],
        "web_name": ["Haaland", "João Pedro", "Palmer", "Palmer"],
    })


def test_entity_resolution_exact_accent_and_shorthand():
    players = _players()
    assert resolve_player_name("Erling Haaland", players).players == (1,)
    assert resolve_player_name("Joao Pedro", players).players == (2,)
    assert resolve_player_name("Haaland", players).players == (1,)


def test_entity_resolution_surfaces_ambiguity():
    result = resolve_player_name("Palmer", _players())
    assert result.status == "ambiguous"
    assert set(result.candidates) == {"Alex Palmer", "Cole Palmer"}


def test_question_resolution_finds_two_players_without_guessing():
    result = resolve_question_players("Compare Cole Palmer and Haaland", _players())
    assert result.status == "resolved" and set(result.players) == {1, 3}
    assert resolve_question_players("Is Palmer injured?", _players()).status == "ambiguous"


def test_intent_routing_is_deterministic_and_budget_precedes_transfer():
    assert detect_intent("Who should I captain?") == "captaincy"
    assert detect_intent("Any confirmed DGWs?") == "dgw_bgw"
    assert detect_intent("Best midfielder under 7.0") == "budget"
    assert detect_intent("Who should I buy under 7.0?") == "budget"
    assert detect_intent("Should I sell Calafiori?") == "transfer"

