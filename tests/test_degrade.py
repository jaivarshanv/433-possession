"""degrade.py on a SYNTHETIC clean game: output contract, determinism, and each step's effect."""

import numpy as np
import pytest

from possession import config, degrade

H, A, D = config.H, config.A, config.D
CACHE_KEYS = {"ball", "px", "py", "ps", "team", "seen", "home_name", "away_name", "n_ball_fields"}


def synthetic_clean_game(n=6000, seed=0):
    """22 players random-walking on a 105 x 68 pitch, ball at a random player's feet.

    Labels: live everywhere except one stoppage (frames 2990-3500) where, like corrected
    tracking, the ball rolls over the touchline and is then NaN.
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.002, size=(n, 22, 2))
    pos = np.clip(0.5 + np.cumsum(steps, axis=0) * 0.3 + rng.uniform(-0.4, 0.4, (1, 22, 2)),
                  0.02, 0.98)
    carrier = np.repeat(rng.integers(0, 22, n // 250 + 1), 250)[:n]
    ball = pos[np.arange(n), carrier].copy()
    ball[2990:3000] = np.linspace([0.5, 0.99], [0.5, 1.03], 10)   # rolls over the line
    ball[3000:3500] = np.nan
    labels = np.full(n, H, dtype=np.int8)
    labels[n // 2 + 1000:] = A
    labels[2990:3500] = D
    game = {"px": pos[:, :, 0], "py": pos[:, :, 1], "ball": ball,
            "team": np.array([0] * 11 + [1] * 11, dtype=np.int8),
            "period": np.ones(n, dtype=np.int8), "pitch": (105.0, 68.0), "fps": 25,
            "name": "synthetic"}
    return game, labels


def test_output_has_the_cache_contract():
    game, labels = synthetic_clean_game()
    out = degrade.degrade(game, labels)
    assert set(out) == CACHE_KEYS
    n = len(labels)
    assert out["ball"].shape == (n, 3) and out["ball"].dtype == np.float64
    assert np.isnan(out["ball"][:, 2]).all()                     # no ball height
    slots = out["px"].shape[1]
    assert out["py"].shape == (n, slots) and out["ps"].shape == (n, slots)
    assert out["px"].dtype == np.float64 and out["ps"].dtype == np.float32
    assert out["team"].dtype == np.int8 and out["team"].shape == (slots,)
    assert (out["team"] == 0).sum() >= 24 and (out["team"] == 1).sum() >= 22
    assert out["seen"].all()
    # the clean reference has the same contract
    assert set(degrade.clean_tracking(game)) == CACHE_KEYS


def test_deterministic_per_replicate():
    game, labels = synthetic_clean_game()
    first, again = degrade.degrade(game, labels), degrade.degrade(game, labels)
    for key in ("ball", "px", "py", "ps", "team"):
        np.testing.assert_array_equal(first[key], again[key])
    other = degrade.degrade(game, labels, replicate=1)
    assert not np.array_equal(np.isnan(first["px"]), np.isnan(other["px"]))


def test_slots_are_not_people():
    game, labels = synthetic_clean_game()
    out = degrade.degrade(game, labels)
    # Every tracked object sits on (or, for a duplicate, next to) a real player of its team.
    length, width = game["pitch"]
    for slot in range(out["px"].shape[1]):
        rows = np.flatnonzero(~np.isnan(out["px"][:, slot]))
        same_team = game["team"] == out["team"][slot]
        d = np.hypot((game["px"][rows][:, same_team] - out["px"][rows, slot][:, None]) * length,
                     (game["py"][rows][:, same_team] - out["py"][rows, slot][:, None]) * width)
        assert (d.min(axis=1) <= degrade.DUP_OFFSET_M[1] + 1e-6).all()
    # ... but a slot does not follow one person: the busiest slot holds several players.
    owners = []
    for slot in range(out["px"].shape[1]):
        rows = np.flatnonzero(~np.isnan(out["px"][:, slot]))
        if len(rows):
            d = np.abs(game["px"][rows] - out["px"][rows, slot][:, None])
            owners.append(len(np.unique(np.argmin(d, axis=1))))
    assert max(owners) > 1
    # fewer objects than the 22 clean players on average, like our tracker
    assert (~np.isnan(out["px"])).sum(axis=1).mean() < 22


def test_ball_is_lost_in_live_play_but_never_moved_there():
    game, labels = synthetic_clean_game()
    out = degrade.degrade(game, labels)
    live = labels != D
    shown = live & ~np.isnan(out["ball"][:, 0]) & ~np.isnan(game["ball"][:, 0])
    np.testing.assert_array_equal(out["ball"][shown, :2], game["ball"][shown])
    assert np.isnan(out["ball"][live, 0]).sum() > 0


def test_ball_rolls_on_after_leaving_play():
    game, _ = synthetic_clean_game()
    ball = game["ball"].copy()
    degrade.roll_after_stoppage(ball, 2990, 3500, game["pitch"], np.random.default_rng(0))
    assert not np.isnan(ball[3000, 0])               # still followed after the clean NaN
    assert ball[3001, 1] >= ball[3000, 1]            # moving on in the same direction
    assert np.isnan(ball[3499, 0])                   # but lost long before the restart


def test_fill_short_gaps_and_bursts():
    ball = np.tile(np.arange(40, dtype=float)[:, None], (1, 2))
    ball[5:8] = np.nan        # 3 frames: filled
    ball[15:35] = np.nan      # 20 frames: kept
    degrade.fill_short_gaps(ball, 10)
    np.testing.assert_allclose(ball[5:8, 0], [5, 6, 7])
    assert np.isnan(ball[15:35]).all()

    rng = np.random.default_rng(0)
    assert not degrade.bursts(rng, np.zeros(1000), 10, 0.5).any()
    assert degrade.bursts(rng, np.ones(1000), 10, 0.5).all()
    assert degrade.lognormal_frames(rng, 1e9, 0.1, 5).max() == degrade.MAX_BURST


@pytest.mark.parametrize("half", [1, 2])
def test_speed_from_positions(half):
    x = np.arange(100, dtype=float)[:, None] / 105.0      # 1 m per frame along x
    speed = degrade.speed_from_positions(x, np.zeros_like(x), (105.0, 68.0), half)
    assert np.allclose(speed[half:-half], 25.0)           # 1 m/frame = 25 m/s
    assert np.isnan(speed[:half]).all() and np.isnan(speed[-half:]).all()
