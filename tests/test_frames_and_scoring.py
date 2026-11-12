"""Frame helpers, and the scoring wrapper against the public ground truth (which is public)."""

import numpy as np
import pytest

from possession import config, frames


def test_runs():
    assert frames.runs(np.array([0, 0, 1, 1, 1, 0])) == [(0, 0, 2), (1, 2, 5), (0, 5, 6)]
    assert frames.runs(np.array([])) == []


def test_graded_count():
    # DATA.md: 158,857 graded frames
    assert frames.graded_mask().sum() == 158_857


def test_possession_runs_ignore_dead_and_halftime():
    codes = np.full(config.N_FRAMES, config.D, dtype=np.int8)
    first = config.HALVES[1][0]
    codes[first:first + 100] = config.H          # possession 1
    codes[first + 150:first + 200] = config.H    # D in between -> possession 2
    codes[first + 200:first + 300] = config.A    # hand-over -> possession 3
    assert len(frames.possession_runs(codes)) == 3


official_present = (config.OFFICIAL_DIR / "games.py").exists()


@pytest.mark.skipif(not official_present, reason="official/ submodule not checked out")
def test_perfect_prediction_scores_one():
    from possession import scoring
    gt = scoring.load_public_gt_codes()
    result = scoring.score_codes(gt, gt.copy())
    assert result["spell_f1_mean"] == pytest.approx(1.0)
    assert result["n_gt_spells"] == 17


@pytest.mark.skipif(not official_present, reason="official/ submodule not checked out")
def test_constant_prediction_scores_zero():
    from possession import scoring
    gt = scoring.load_public_gt_codes()
    constant = np.full(len(gt), config.H, dtype=gt.dtype)
    assert scoring.score_codes(gt, constant)["spell_f1_mean"] == pytest.approx(0.0, abs=0.1)
