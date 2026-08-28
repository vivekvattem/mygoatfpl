"""Deterministic intent routing and conservative player-name resolution."""

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
import unicodedata

import pandas as pd


SUPPORTED_INTENTS = {
    "squad_summary", "transfer", "captaincy", "player_comparison", "player_lookup",
    "fixture", "dgw_bgw", "chip", "risk", "ranking", "budget", "general_fpl",
}


@dataclass(frozen=True)
class EntityResolution:
    status: str
    players: tuple[int, ...] = ()
    candidates: tuple[str, ...] = ()
    query: str | None = None


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.casefold()).split())


def detect_intent(question: str) -> str:
    text = normalize_text(question)
    if any(term in text for term in ("double gameweek", "blank gameweek", "dgw", "bgw")):
        return "dgw_bgw"
    if any(term in text for term in ("wildcard", "free hit", "bench boost", "triple captain", "chip")):
        return "chip"
    if any(term in text for term in ("captain", "vice captain")):
        return "captaincy"
    if any(term in text for term in ("biggest risk", "biggest risks", "who is risky", "problems", "weakest")):
        return "risk"
    if any(term in text for term in ("what should i do", "this week", "weekly brief", "squad summary")):
        return "squad_summary"
    if re.search(r"\b(compare|versus|vs)\b", text) or " or " in f" {text} ":
        return "player_comparison"
    if re.search(r"\b(under|budget|afford|million|m)\b", text) and any(
            term in text for term in ("player", "midfielder", "defender", "forward", "goalkeeper", "buy", "best")):
        return "budget"
    if any(term in text for term in ("sell", "buy", "replace", "replacement", "roll", "transfer")):
        return "transfer"
    if any(term in text for term in ("best player", "top player", "rank", "ranking", "best midfielder",
                                      "best defender", "best forward", "best goalkeeper")):
        return "ranking"
    if any(term in text for term in ("fixture", "fixtures", "play next", "best schedule", "best fixtures")):
        return "fixture"
    if any(term in text for term in ("injured", "injury", "available", "cost", "price", "xpts", "expected minutes")):
        return "player_lookup"
    return "general_fpl"


def _aliases(players: pd.DataFrame) -> dict[str, list[int]]:
    aliases: dict[str, list[int]] = {}
    for row in players.itertuples(index=False):
        player_id = int(row.player_id)
        names = {str(getattr(row, "player", "")), str(getattr(row, "web_name", ""))}
        full = normalize_text(getattr(row, "player", ""))
        tokens = full.split()
        if tokens:
            names.add(tokens[-1])
        for name in names:
            alias = normalize_text(name)
            if alias:
                aliases.setdefault(alias, []).append(player_id)
    return aliases


def resolve_player_name(query: str, players: pd.DataFrame, fuzzy_cutoff: float = 0.88) -> EntityResolution:
    """Resolve one name; ambiguity is surfaced rather than guessed."""
    if players.empty or not {"player_id", "player"}.issubset(players.columns):
        return EntityResolution("not_found", query=query)
    normalized = normalize_text(query)
    aliases = _aliases(players)
    ids = aliases.get(normalized, [])
    if not ids:
        scored = sorted(((SequenceMatcher(None, normalized, alias).ratio(), alias)
                         for alias in aliases if len(alias) >= 4), reverse=True)
        if scored and scored[0][0] >= fuzzy_cutoff:
            top_score = scored[0][0]
            close = [alias for score, alias in scored if score >= top_score - 0.02]
            ids = sorted({player_id for alias in close for player_id in aliases[alias]})
    names = tuple(players[players.player_id.isin(ids)].player.astype(str).sort_values().unique())
    if len(ids) == 1:
        return EntityResolution("resolved", (int(ids[0]),), names, query)
    if len(ids) > 1:
        return EntityResolution("ambiguous", tuple(sorted(set(ids))), names, query)
    return EntityResolution("not_found", query=query)


def resolve_question_players(question: str, players: pd.DataFrame) -> EntityResolution:
    """Find up to two mentioned players, preferring longest explicit aliases."""
    text = normalize_text(question)
    aliases = _aliases(players)
    matches = []
    for alias, ids in aliases.items():
        if re.search(rf"(?:^|\s){re.escape(alias)}(?:$|\s)", text):
            matches.append((len(alias.split()), len(alias), alias, ids))
    matches.sort(reverse=True)
    selected: list[int] = []
    occupied_aliases: list[str] = []
    for _, _, alias, ids in matches:
        if any(alias in longer for longer in occupied_aliases):
            continue
        if len(set(ids)) > 1:
            names = tuple(players[players.player_id.isin(ids)].player.astype(str).sort_values().unique())
            return EntityResolution("ambiguous", tuple(sorted(set(ids))), names, alias)
        player_id = int(ids[0])
        if player_id not in selected:
            selected.append(player_id)
            occupied_aliases.append(alias)
        if len(selected) == 2:
            break
    if selected:
        names = tuple(players[players.player_id.isin(selected)].player.astype(str).tolist())
        return EntityResolution("resolved", tuple(selected), names)
    return EntityResolution("not_found")


def extract_budget(question: str) -> float | None:
    text = normalize_text(question)
    match = re.search(r"(?:under|below|budget|less than)\s*(?:ps|gbp)?\s*(\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else None


def extract_position(question: str) -> str | None:
    text = normalize_text(question)
    for terms, position in ((('goalkeeper', 'keeper', 'gk'), 'GK'), (('defender', 'defence', 'def'), 'DEF'),
                            (('midfielder', 'midfield', 'mid'), 'MID'), (('forward', 'striker', 'fwd'), 'FWD')):
        if any(re.search(rf"\b{term}\b", text) for term in terms):
            return position
    return None
