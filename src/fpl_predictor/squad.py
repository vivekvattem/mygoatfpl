"""Future personalized-squad boundary (not implemented in Phase 1)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SquadPlayer:
    """Future representation of a player owned by an FPL entry."""

    entry_id: int
    gameweek: int
    player_id: int
    purchase_price: float
    selling_price: float
    captain: bool = False
    vice_captain: bool = False
    bench_order: int | None = None
