"""The harness on SYNTHETIC inputs: pooling, predictor contract, official scoring, registry."""

import numpy as np
import pytest

from possession import baselines, config, harness, predictors, scoring

H, A, D, X = config.H, config.A, config.D, config.X
official_present = (config.OFFICIAL_DIR / "games.py").exists()
pytestmark = pytest.mark.skipif(not official_present, reason="official/ submodule missing")

CYCLE = (H, A, H, D, A, D)


def synthetic_gt(seed):
    """Two graded 'halves' of random-length segments cycling through CYCLE, X elsewhere."""
    rng = np.random.default_rng(seed)
    gt = np.full(6000, X, dtype=np.int8)
    for first, last in ((100, 2900), (3100, 5900)):
        frame, step = first, 0
        while frame < last:
            length = int(rng.integers(50, 400))
            gt[frame:min(frame + length, last)] = CYCLE[step % len(CYCLE)]
            frame, step = frame + length, step + 1
    return gt


def synthetic_inputs():
    """[(track, info, gt)] for two fake games; the track is irrelevant to the cheat below."""
    out = []
    for seed in (0, 1):
        gt = synthetic_gt(seed)
        info = {"name": "fake{}".format(seed), "replicate": 0, "pitch": (105.0, 68.0),
                "fps": 25, "half": (gt != X).astype(np.int8), "kickoff": {1: 100, 2: 3100}}
        out.append(({"ball": np.full((len(gt), 3), np.nan)}, info, gt))
    return out


def test_oracle_scores_one_and_constant_scores_zero():
    inputs = synthetic_inputs()
    truth = {info["name"]: gt for _, info, gt in inputs}
    oracle = harness.evaluate(lambda track, info: truth[info["name"]], inputs, bootstrap_n=50)
    assert oracle["pooled"]["spell_f1_mean"] == pytest.approx(1.0)
    assert all(r["spell_f1_mean"] == pytest.approx(1.0) for r in oracle["per_game"].values())
    assert oracle["interval"][0] == pytest.approx(1.0)
    constant = harness.evaluate(lambda track, info: np.full(len(info["half"]), H), inputs)
    assert constant["pooled"]["spell_f1_mean"] == pytest.approx(0.0, abs=0.05)


def test_pooling_keeps_games_apart():
    inputs = synthetic_inputs()
    truth = {info["name"]: gt for _, info, gt in inputs}
    out = harness.evaluate(lambda track, info: truth[info["name"]], inputs)
    assert out["pooled"]["n_gt_spells"] == sum(r["n_gt_spells"] for r in out["per_game"].values())
    # even when one game ends and the next starts in the same possession
    gt = np.full(100, H, dtype=np.int8)
    pooled, _ = harness.pooled_arrays([(gt, gt), (gt, gt)])
    assert scoring.score_codes(pooled, pooled)["n_gt_spells"] == 2


def test_prediction_contract_is_checked():
    with pytest.raises(ValueError):
        harness.check_prediction(np.zeros(10), 11, "short")
    with pytest.raises(ValueError):
        harness.check_prediction(np.zeros(10, dtype=float), 10, "floats")
    with pytest.raises(ValueError):
        harness.check_prediction(np.full(10, 7), 10, "bad code")
    assert harness.check_prediction(np.zeros(10, dtype=np.int64), 10, "ok").dtype == np.int8


def test_registered_baselines_follow_the_contract():
    assert {"baseline_raw", "baseline_smooth25", "baseline_smooth50",
            "baseline_smooth125"} <= set(predictors.REGISTRY)
    with pytest.raises(KeyError):
        predictors.get("no such predictor")
    n = 300
    track = {"ball": np.full((n, 3), np.nan), "px": np.full((n, 2), np.nan),
             "py": np.full((n, 2), np.nan), "team": np.array([0, 1], dtype=np.int8)}
    track["px"][:, 0], track["py"][:, 0] = 0.30, 0.5     # home player
    track["px"][:, 1], track["py"][:, 1] = 0.70, 0.5     # away player
    track["ball"][:100, :2] = [0.32, 0.5]                # at the home player
    track["ball"][200:, :2] = [0.69, 0.5]                # at the away player
    info = {"pitch": (105.0, 68.0)}
    raw = predictors.get("baseline_raw")(track, info)
    assert raw.shape == (n,) and set(np.unique(raw).tolist()) == {H, A, D}
    assert (raw[:100] == H).all() and (raw[100:200] == D).all() and (raw[200:] == A).all()
    assert (baselines.nearest_player_team(track, (105.0, 68.0))[100:200] == baselines.NOBODY).all()
