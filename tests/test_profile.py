"""profile.py on a SYNTHETIC track with known gaps and object counts."""

import numpy as np

from possession import profile


def test_ball_and_player_profile_counts():
    n = 3000
    ball = np.full((n, 3), np.nan)
    ball[:, :2] = 0.5
    ball[1000:1100, :2] = np.nan            # one 100-frame gap
    ball[2000:2010, :2] = np.nan            # one 10-frame gap
    px = np.full((n, 4), 0.5)
    py = np.full((n, 4), 0.5)
    px[:1500, 3] = np.nan                   # a slot that only starts half way
    track = {"ball": ball, "px": px, "py": py, "ps": np.ones((n, 4), dtype=np.float32),
             "team": np.array([0, 0, 1, 1], dtype=np.int8)}
    mask = np.ones(n, dtype=bool)
    out = profile.flatten(profile.tracking_profile(track, mask, (105.0, 68.0)))
    assert out["ball.missing_share"] == round(110 / n, 4)
    assert out["ball.gaps_per_min"] == round(2 / (n / 25 / 60), 2)
    assert out["ball.outside_pitch_share"] == 0.0
    assert out["players.objects_mean"] == 3.5
    assert out["players.tracklet_starts_per_min"] == round(4 / (n / 25 / 60), 2)
    assert profile.run_lengths(np.array([1, 1, 0, 1, 0, 0, 1], dtype=bool)).tolist() == [2, 1, 1]
