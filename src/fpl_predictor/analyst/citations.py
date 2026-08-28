"""Internal evidence labels used by the explainable analyst."""

from collections.abc import Iterable


EVIDENCE_LABELS = {
    "ml_projection": "ML Projection",
    "transfer_engine": "Transfer Engine",
    "fixture_calendar": "Fixture Calendar",
    "signal_engine": "Signal Engine",
    "captaincy_engine": "Captaincy Engine",
    "chip_planner": "Chip Planner",
    "availability": "Availability",
    "squad_state": "Squad State",
}


def evidence_badges(sources: Iterable[str]) -> list[str]:
    """Return stable, human-readable badges without duplicates."""
    seen: set[str] = set()
    labels = []
    for source in sources:
        if source in seen:
            continue
        seen.add(source)
        labels.append(EVIDENCE_LABELS.get(source, source.replace("_", " ").title()))
    return labels


def freshness_label(stale: bool) -> str:
    """Accessible freshness label for analyst surfaces."""
    return "STALE DATA" if stale else "LIVE DATA"

