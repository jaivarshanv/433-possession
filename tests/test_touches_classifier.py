"""touches.py and classifier.py on SYNTHETIC data."""

import numpy as np

from possession import config, classifier, touches

H, A, D, X = config.H, config.A, config.D, config.X


def touch_features(n=100):
    f = {"ball_present": np.ones(n, dtype=bool), "ball_accel": np.zeros(n),
         "d_home": np.full(n, 5.0), "d_away": np.full(n, 5.0)}
    f["ball_accel"][[20, 21, 23]] = [30.0, 50.0, 20.0]    # one cluster, strongest at 21
    f["d_home"][18:26] = 0.5                               # a home player at the ball
    f["ball_accel"][60] = 40.0
    f["d_away"][60] = 1.0                                  # an away touch
    f["ball_accel"][80] = 40.0                             # nobody near: not a touch
    return f


def test_detect_one_touch_per_cluster_with_its_team():
    frames, teams = touches.detect(touch_features())
    assert frames.tolist() == [21, 60] and teams.tolist() == [0, 1]


def test_snap_after_last_moves_handover_to_the_new_team_touch():
    codes = np.array([H] * 50 + [A] * 50, dtype=np.int8)   # predicted hand-over at 50
    frames = np.array([45, 70], dtype=np.int64)            # home touch, then away touch
    teams = np.array([0, 1], dtype=np.int8)
    out = touches.snap_handovers(codes, frames, teams, rule="after_last")
    assert (out[:70] == H).all() and (out[70:] == A).all()
    # nothing to snap to -> unchanged; D boundaries are never moved
    assert np.array_equal(touches.snap_handovers(codes, frames[:0], teams[:0]), codes)
    dead = np.array([H] * 50 + [D] * 50, dtype=np.int8)
    assert np.array_equal(touches.snap_handovers(dead, frames, teams), dead)


def test_abstain_only_inside_long_gaps_of_one_possession():
    codes = np.array([H] * 300 + [A] * 100, dtype=np.int8)
    f = {"gap_len": np.zeros(400, dtype=np.int32), "ball_present": np.ones(400, dtype=bool)}
    f["gap_len"][100:200] = 100
    f["ball_present"][100:200] = False
    out = touches.abstain_in_gaps(codes, f)
    assert (out[100:200] == X).all() and (out[:100] == H).all() and (out[300:] == A).all()


def test_centre_per_half():
    x = np.r_[np.full(10, 5.0), np.full(10, 1.0)]
    half = np.r_[np.ones(10), np.full(10, 2)].astype(np.int8)
    np.testing.assert_allclose(classifier.centre_per_half(x, half), 0.0)


def synthetic_feats(n=600, seed=0):
    """A feature dict with every column the classifier reads, random but finite."""
    rng = np.random.default_rng(seed)
    f = {key: rng.uniform(0, 5, n) for key in (
        "gap_len", "gap_since", "gap_left", "ball_speed", "ball_accel", "static_len",
        "kick_since", "ball_out_m", "last_out_m", "next_out_m", "heading_out_m",
        "speed_mean_1s", "speed_mean_3s", "decel_1s", "speed_home_1s", "speed_away_1s",
        "sx_home", "sx_away", "sy_home", "sy_away", "len_home", "len_away", "wid_home",
        "wid_away", "cx_home", "cx_away", "vx_home", "vx_away", "d_home", "d_away")}
    f.update(last_bx=rng.uniform(0, 105, n), last_by=rng.uniform(0, 68, n),
             bx=rng.uniform(0, 105, n), ball_present=rng.random(n) < 0.7,
             n_home=rng.integers(5, 12, n), n_away=rng.integers(5, 12, n),
             n_near5=rng.integers(0, 5, n), n_near10=rng.integers(0, 9, n),
             pre_kickoff=np.zeros(n, dtype=bool), home_dir=np.ones(n, dtype=np.int8),
             half=np.ones(n, dtype=np.int8), pitch=(105.0, 68.0), fps=25)
    return f


def test_team_matrices_mirror_is_an_involution():
    f = synthetic_feats()
    seen, mirror = classifier.team_matrices(f)
    assert seen.shape == mirror.shape and seen.shape[0] == 600
    # mirroring the mirrored features gives the original columns back (pairs swap, signs flip)
    n_pairs = 10
    back = np.column_stack([mirror[:, n_pairs:2 * n_pairs], mirror[:, :n_pairs],
                            -mirror[:, 2 * n_pairs:2 * n_pairs + 4], mirror[:, 2 * n_pairs + 4:]])
    np.testing.assert_array_equal(np.nan_to_num(back), np.nan_to_num(seen))
    assert classifier.dead_matrix(f).shape[0] == 600
