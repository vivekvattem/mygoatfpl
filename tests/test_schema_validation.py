import pandas as pd
import pytest

from fpl_predictor.schema_validation import audit_live_schema, require_live_schema


def test_schema_parity_and_missing_required_failure():
    frame = pd.DataFrame({"price": [5.0], "position": ["MID"], "extra": [1]})
    report = audit_live_schema(frame, ["price", "position"])
    assert report.passed and report.unexpected == ["extra"]
    with pytest.raises(ValueError, match="missing"):
        require_live_schema(frame, ["price", "points_last_3"])
