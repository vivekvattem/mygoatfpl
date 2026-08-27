import json

import numpy as np
import pandas as pd

from fpl_predictor.live_inference import predict_live_players


class NegativeModel:
    def predict(self, frame): return np.full(len(frame), -1.0)


class FakeArtifacts:
    manifest = {"feature_names": ["price", "position"]}
    def load_position(self, position): return NegativeModel()


def test_synthetic_live_inference_preserves_raw_and_floors_display(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"uncertainty": {"bands": [{"position": p, "residual_lower": -2, "residual_upper": 3}
                                                                for p in ("GK", "DEF", "MID", "FWD")]}}))
    frame = pd.DataFrame({"price": [5.0], "position": ["MID"], "status": ["a"],
                          "chance_of_playing_next_round": [None], "games_last_5": [1]})
    output, report = predict_live_players(frame, FakeArtifacts(), summary)
    assert report.passed and output.raw_xpts.iloc[0] == -1
    assert output.display_xpts.iloc[0] == 0 and output.availability_adjusted_xpts.iloc[0] == 0
