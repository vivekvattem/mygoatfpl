"""Contracts for future recommendation logic; no optimizer exists yet."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TransferEvaluation:
    """Transparent future transfer comparison result."""

    old_player_id: int
    new_player_id: int
    new_player_multi_gw_xpts: float
    old_player_multi_gw_xpts: float
    transfer_hit_cost: float = 0.0

    @property
    def expected_gain(self) -> float:
        return (
            self.new_player_multi_gw_xpts
            - self.old_player_multi_gw_xpts
            - self.transfer_hit_cost
        )
