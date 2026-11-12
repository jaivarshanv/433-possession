"""deadball.py, evidence.py and training.py on SYNTHETIC feature tables."""

import numpy as np

from possession import config, deadball, evidence, training

H, A, D, X = config.H, config.A, config.D, config.X


def synthetic_match(n_segments=30, length=150, seed=0):
    """Feature dict + labels: H, A and D segments with the obvious evidence for each.

    H: ball seen, a home player 0.5 m from it, away player 6 m away, players running.
    A: the same for away. D: ball missing, players walking. One X frame at each end.
    """
    rng = np.random.default_rng(seed)
    pattern = [H, A, D] * (n_segments // 3)
    gt = np.array([X] + [s for s in pattern for _ in range(length)] + [X], dtype=np.int8)
    n = len(gt)
    alive = (gt == H) | (gt == A)
    f = {
        "ball_present": alive.copy(),
        "ball_out_m": np.where(alive, 0.0, np.nan),
        "last_out_m": np.zeros(n), "next_out_m": np.zeros(n),
        "static_len": np.zeros(n, dtype=np.int32),
        "gap_len": np.where(alive, 0, length).astype(np.int32),
        "speed_mean_1s": np.where(alive, 3.0, 0.6) + rng.normal(0, 0.05, n),
        "pre_kickoff": np.zeros(n, dtype=bool),
        "d_home": np.where(gt == H, 0.5, np.where(gt == A, 6.0, np.nan)),
        "d_away": np.where(gt == A, 0.5, np.where(gt == H, 6.0, np.nan)),
        "ball_speed": np.where(alive, 1.0, np.nan),
        "half": np.where(gt == X, 0, 1).astype(np.int8),
    }
    return f, {"game": "synthetic", "half": f["half"]}, gt


def test_dead_cells_rules():
    f, _, gt = synthetic_match()
    cells = deadball.dead_cells(f)
    assert cells.min() >= 0 and cells.max() < deadball.N_CELLS
    assert len(np.intersect1d(cells[gt == D], cells[gt == H])) == 0   # dead and live separate
    f["pre_kickoff"][5:10] = True
    assert (deadball.dead_cells(f)[5:10] == 0).all()


def test_cell_table_shrinks_rare_cells():
    cells = np.array([0] * 1000 + [1] * 2)
    hits = np.array([True] * 900 + [False] * 100 + [True, True])
    table = deadball.CellTable(3).fit(cells, hits)
    assert abs(table.p[0] - 0.9) < 0.01
    assert table.p[1] < 0.95                     # 2 of 2 hits does not mean certainty
    assert np.isclose(table.p[2], table.rate)    # unseen cell = overall rate


def test_segments_hysteresis():
    p = np.r_[np.full(10, 0.1), np.full(40, 0.9), np.full(5, 0.5), np.full(10, 0.1),
              np.full(10, 0.9), np.full(10, 0.1)]
    half = np.ones(len(p), dtype=np.int8)
    # first stoppage: opens at 0.9, survives the 0.5 dip (above exit), closes at 0.1;
    # second one is only 10 frames long -> dropped
    assert deadball.segments(p, half, min_frames=25) == [(10, 55)]


def test_team_cells_direct_and_carry():
    f, _, gt = synthetic_match()
    cell, team = evidence.team_cells(f)
    assert (team[gt == H] == 0).all() and (team[gt == A] == 1).all()
    assert (cell[gt == H] < evidence.N_DIRECT).all()
    dead = np.flatnonzero(gt == D)
    assert (cell[dead] >= evidence.N_DIRECT).all()                  # carried, not direct
    assert (team[dead] == 1).all()                                  # A came before each D


def test_rules_model_recovers_a_clear_match():
    f, info, gt = synthetic_match()
    model = evidence.RulesModel().fit([(f, info, gt)])
    codes = model.predict(f, info)
    graded = gt != X
    assert np.mean(codes[graded] == gt[graded]) > 0.95
    scores = model.scores(f)
    assert scores.shape == (len(gt), 3) and np.isfinite(scores).all()


def test_logo_predictor_leaves_the_game_out(monkeypatch):
    seen = []

    class Recorder:
        def fit(self, train):
            seen.append(train)
            return self

        def predict(self, feats, info):
            return np.zeros(len(info["half"]), dtype=np.int8)

    monkeypatch.setattr(training, "training_set", lambda exclude, replicates: exclude)
    monkeypatch.setattr(training.features, "compute", lambda track, info: {})
    predictor = training.LogoPredictor(Recorder)
    info = {"game": "metrica2", "half": np.ones(5)}
    predictor({}, info)
    predictor({}, info)                      # fitted once per held-out game
    assert seen == ["metrica2"]
