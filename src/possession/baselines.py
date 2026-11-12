"""The official reference baseline, as reusable functions: nearest player to the ball + vote.

official/starter/baseline.py reads the raw ATD text line by line. Here the same computation runs
on our parsed arrays, so it works on ANY tracking dict in cache format -- our match, or a
degraded external game. scripts/02_reproduce_baseline.py proves it is bit-identical to the
official files on our match; the smoothing is the official function itself, imported.
"""

import numpy as np

from . import config

NOBODY = -1


def nearest_player_team(data, pitch=(config.PITCH_LENGTH_M, config.PITCH_WIDTH_M)):
    """Per frame: team (0 home / 1 away) of the slot closest to the ball, NOBODY if no ball.

    Vectorised version of official/starter/baseline.py, written to perform the same
    floating-point operations in the same order so the result is bit-identical:
    distances in metres, squared, first minimum wins on ties.
    pitch defaults to our match; external games pass their own size.
    """
    length, width = pitch
    ball_x = data["ball"][:, 0] * length
    ball_y = data["ball"][:, 1] * width

    dx = data["px"] * length - ball_x[:, None]
    dy = data["py"] * width - ball_y[:, None]
    dist2 = dx * dx + dy * dy                      # NaN where slot or ball is missing
    dist2 = np.where(np.isnan(dist2), np.inf, dist2)

    best_slot = np.argmin(dist2, axis=1)            # first minimum, like the official loop
    has_player = np.isfinite(dist2.min(axis=1))
    has_ball = ~np.isnan(data["ball"][:, 0])       # official: ball missing if x is NaN

    out = np.full(len(best_slot), NOBODY, dtype=np.int8)
    ok = has_ball & has_player
    out[ok] = data["team"][best_slot[ok]]
    return out


def official_smooth(codes, window):
    """The official majority-vote smoothing (official/starter/baseline.py), as label codes."""
    config.add_official_to_path()
    import baseline as official_baseline
    import possession_metrics as pm
    return pm.label_codes(official_baseline.smooth(codes, window))


def nearest_player_predictor(window):
    """A predictor (see harness.py) for the official baseline with a +/- window vote."""
    def predict(track, info):
        return official_smooth(nearest_player_team(track, info["pitch"]), window)
    predict.__name__ = "baseline_raw" if window == 0 else "baseline_smooth{}".format(window)
    return predict
