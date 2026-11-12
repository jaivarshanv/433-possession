"""M2 on SYNTHETIC data: restart-like kicks, restart rules, bonuses, v2 cells, diagnostics."""

import numpy as np

from possession import config, deadball, diagnostics, evidence, features, restarts

H, A, D, X = config.H, config.A, config.D, config.X
L, W = 105.0, 68.0


def test_restart_kicks_from_rest_and_after_a_gap():
    n = 200
    speed = np.full(n, 6.0)
    present = np.ones(n, dtype=bool)
    speed[20:40] = 0.0                 # still for 20 frames, then kicked at frame 40
    present[100:160] = False           # 60-frame gap, reappears moving at frame 160
    speed[100:160] = np.nan
    events = features.restart_kicks(present, speed)
    assert events.tolist() == [40, 160]
    since, last = features.since_event(events, n)
    assert since[39] == features.NEVER and since[40] == 0 and since[45] == 5 and last[170] == 160


def test_restart_kind_from_position():
    assert restarts.restart_kind(1.0, 1.0, L, W, False) == "corner"
    assert restarts.restart_kind(104.0, 30.0, L, W, False) == "goal_kick"
    assert restarts.restart_kind(L / 2, W / 2, L, W, True) == "kick_off"
    assert restarts.restart_kind(L / 2, W / 2, L, W, False) == "other"     # no goal before it
    assert restarts.restart_kind(40.0, 67.5, L, W, False) == "throw_in"
    assert restarts.restart_kind(60.0, 20.0, L, W, False) == "other"


def test_attacker_of_end():
    assert restarts.attacker_of_end(+1, True) == 0      # home attacks towards x = length
    assert restarts.attacker_of_end(+1, False) == 1
    assert restarts.attacker_of_end(-1, False) == 0


def restart_features(n, kick_frame, kick_xy, home_dir=1, d_home=0.5, d_away=8.0, last_xy=(50, 30)):
    """Minimal feature dict around one restart kick."""
    since = np.full(n, features.NEVER, dtype=np.int32)
    since[kick_frame:] = np.arange(n - kick_frame)
    return {"kick_since": since,
            "kick_x": np.full(n, kick_xy[0]), "kick_y": np.full(n, kick_xy[1]),
            "ball_present": np.ones(n, dtype=bool), "bx": np.full(n, kick_xy[0]),
            "by": np.full(n, kick_xy[1]), "last_bx": np.full(n, float(last_xy[0])),
            "last_by": np.full(n, float(last_xy[1])),
            "home_dir": np.full(n, home_dir, dtype=np.int8),
            "d_home": np.full(n, d_home), "d_away": np.full(n, d_away)}


def test_find_restarts_applies_the_laws():
    info = {"pitch": (L, W), "fps": 25}
    codes = np.full(300, H, dtype=np.int8)
    codes[0] = X
    codes[100:150] = D                       # one stoppage, restart at 150
    # corner at the far end: home attacks the far end -> home takes it
    f = restart_features(300, 152, (104.5, 0.5), home_dir=1, d_home=9.0, d_away=0.5)
    assert restarts.find_restarts(f, codes, info) == [("corner", 0, 152)]
    # goal kick at the far end: the defenders of that goal (away) take it
    f = restart_features(300, 152, (101.0, 34.0), home_dir=1)
    assert restarts.find_restarts(f, codes, info) == [("goal_kick", 1, 152)]
    # kick-off after the ball vanished into the far goal: away conceded -> away kicks off
    f = restart_features(300, 152, (L / 2, W / 2), home_dir=1, last_xy=(104.0, 34.0))
    assert restarts.find_restarts(f, codes, info) == [("kick_off", 1, 152)]
    # free kick in midfield: the team of the player at the ball
    f = restart_features(300, 152, (60.0, 30.0), d_home=0.5, d_away=9.5)
    assert restarts.find_restarts(f, codes, info) == [("other", 0, 152)]


def test_bonuses_spread_their_total():
    table = deadball.CellTable(len(restarts.KINDS))
    table.p = np.full(len(restarts.KINDS), 0.9)
    bonus = restarts.restart_bonus([("throw_in", 1, 10)], table, 100, w_restart=1.0)
    assert np.isclose(bonus[:, 1].sum(), np.log(0.9 / 0.1)) and bonus[:, 0].sum() == 0
    tags = restarts.tagging_bonus([(20, 70, 0)], 100)
    assert np.isclose(tags[:, 0].sum(), restarts.TAG_TOTAL_LOG) and tags[:20].sum() == 0


def test_v2_cells_cover_their_ranges():
    n = 50
    f = {"speed_mean_1s": np.full(n, 2.0), "ball_present": np.r_[np.ones(25), np.zeros(25)] > 0,
         "ball_out_m": np.zeros(n), "last_out_m": np.zeros(n), "next_out_m": np.zeros(n),
         "heading_out_m": np.r_[np.full(25, np.nan), np.full(25, 3.0)],
         "static_len": np.zeros(n), "kick_since": np.r_[np.zeros(10), np.full(40, 500)],
         "gap_len": np.r_[np.zeros(25), np.full(25, 25)], "pre_kickoff": np.zeros(n, dtype=bool),
         "d_home": np.full(n, 0.5), "d_away": np.full(n, 6.0), "ball_speed": np.full(n, 1.0)}
    cells = deadball.dead_cells_v2(f)
    assert cells.min() >= 1 and cells.max() < deadball.N_CELLS_V2
    assert cells[0] != cells[15]                         # just kicked vs long ago differ
    team_cell, team = evidence.team_cells_v2(f)
    assert team_cell.max() < evidence.N_TEAM_CELLS_V2
    assert team_cell[0] == team_cell[15] + evidence.N_DIRECT   # restart context shifts cells


def test_diagnostics_on_known_errors():
    gt = np.array([X] + [H] * 50 + [D] * 50 + [A] * 50 + [X], dtype=np.int8)
    pred = gt.copy()
    pred[51:56] = H                     # stoppage starts 5 frames late
    pred[101:111] = D                   # and ends 10 frames late
    report = diagnostics.report([(gt, pred)])
    assert report["gt_stoppages"] == report["pred_stoppages"] == report["found"] == 1
    assert report["start_err_median"] == 5 and report["end_err_median"] == 10
    assert report["restart_team_right"] == 1.0          # right team 1 s after the restart
