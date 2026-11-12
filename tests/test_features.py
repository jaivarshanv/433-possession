"""features.py on SYNTHETIC tracks: helpers, ball kinematics, players, attacking direction."""

import numpy as np

from possession import features


def test_run_position_and_indices():
    mask = np.array([0, 1, 1, 1, 0, 1], dtype=bool)
    total, since, left = features.run_position(mask)
    assert total.tolist() == [0, 3, 3, 3, 0, 1]
    assert since.tolist() == [0, 0, 1, 2, 0, 0]
    assert left.tolist() == [0, 2, 1, 0, 0, 0]
    present = np.array([0, 1, 0, 0, 1, 0], dtype=bool)
    assert features.last_index(present).tolist() == [-1, 1, 1, 1, 4, 4]
    assert features.next_index(present).tolist() == [1, 1, 4, 4, 4, 6]


def test_rolling_nanmean_ignores_nan():
    x = np.array([1.0, np.nan, 3.0, 5.0, np.nan])
    out = features.rolling_nanmean(x, 3)
    np.testing.assert_allclose(out, [1.0, 2.0, 4.0, 4.0, 5.0])
    assert np.isnan(features.rolling_nanmean(np.full(4, np.nan), 3)).all()


def test_outside_distance():
    x = np.array([-2.0, 50.0, 107.0, np.nan])
    y = np.array([30.0, 70.0, 10.0, 5.0])
    out = features.outside_distance(x, y, 105.0, 68.0)
    np.testing.assert_allclose(out[:3], [2.0, 2.0, 2.0])
    assert np.isnan(out[3])


def synthetic_track(n=400):
    """Ball rolling at 10 m/s along x, then gone; 2 home + 2 away players standing still."""
    ball = np.full((n, 3), np.nan)
    t = np.arange(200)
    ball[:200, 0] = (10.0 + 0.4 * t) / 105.0          # 0.4 m per frame = 10 m/s
    ball[:200, 1] = 0.5
    px = np.tile(np.array([[0.2, 0.3, 0.7, 0.8]]), (n, 1))
    py = np.full((n, 4), 0.5)
    track = {"ball": ball, "px": px, "py": py, "ps": np.full((n, 4), 2.0, dtype=np.float32),
             "team": np.array([0, 0, 1, 1], dtype=np.int8)}
    info = {"pitch": (105.0, 68.0), "fps": 25, "half": np.r_[np.zeros(10), np.ones(n - 10)],
            "kickoff": {1: 50}}
    return track, info


def test_compute_ball_and_players():
    track, info = synthetic_track()
    f = features.compute(track, info)
    assert set(["ball_speed", "gap_len", "d_home", "home_dir", "pre_kickoff"]) <= set(f)
    assert np.allclose(f["ball_speed"][5:195], 10.0)
    assert np.nanmax(f["ball_accel"][5:195]) < 1e-6
    assert (f["gap_len"][200:] == 200).all() and (f["gap_len"][:200] == 0).all()
    assert f["gap_since"][200] == 0 and f["gap_left"][200] == 199
    assert np.isclose(f["last_bx"][300], 10.0 + 0.4 * 199)
    assert np.isnan(f["d_home"][300])                    # no ball, no distance
    assert (f["n_home"] == 2).all() and (f["n_away"] == 2).all()
    np.testing.assert_allclose(f["speed_mean"], 2.0)
    assert f["pre_kickoff"][10:50].all() and not f["pre_kickoff"][50:].any()
    # home stands near x = 0 at kick-off -> attacks towards x = length
    assert (f["home_dir"][10:] == 1).all() and (f["home_dir"][:10] == 0).all()


def test_attack_direction_flips_with_the_teams():
    cx_home = np.r_[np.full(100, 80.0), np.full(100, 20.0)]
    cx_away = np.r_[np.full(100, 20.0), np.full(100, 80.0)]
    half = np.r_[np.ones(100), np.full(100, 2)].astype(np.int8)
    out = features.home_attack_direction(cx_home, cx_away, half, {1: 10, 2: 110})
    assert (out[:100] == -1).all() and (out[100:] == 1).all()
