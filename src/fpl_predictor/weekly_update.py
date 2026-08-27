"""Future weekly orchestration interfaces (deliberately unimplemented)."""

from pydantic import BaseModel, Field


class WeeklyUpdateRequest(BaseModel):
    """Validated input for the future personalized weekly workflow."""

    entry_id: int = Field(gt=0)
    horizon: int = Field(default=5, ge=1, le=8)


def run_weekly_update(request: WeeklyUpdateRequest) -> None:
    """Reserve the orchestration boundary for Phase 5 and later."""
    raise NotImplementedError(
        f"Personalized weekly updates for entry {request.entry_id} are not part of Phase 1."
    )
