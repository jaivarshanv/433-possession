"""Our vectorised nearest-player code must match the official baseline loop exactly.

Uses a random SYNTHETIC ATD file (with missing players and missing balls), runs both the
official `nearest_player_labels` (reads the raw text) and our parser + `nearest_player_team`,
and requires identical output on every frame.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from possession import atd, baselines, config

ROOT = Path(__file__).resolve().parents[1]
official_present = (config.OFFICIAL_DIR / "games.py").exists()


def load_script(name):
    """Import a script whose file name starts with a digit."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_atd(tmp_path, n_frames=400, n_slots=10, seed=0):
    """Write a random ATD pair with gaps. Returns (raw_path, meta_path)."""
    rng = np.random.default_rng(seed)
    players = "\n".join('<Player id="P{}" teamId="{}"/>'.format(i, "TH" if i < 5 else "TA")
                        for i in range(n_slots))
    meta = tmp_path / "s_atd.xml"
    meta.write_text('<Root><Team id="TH"><Name>H</Name></Team>'
                    '<Team id="TA"><Name>A</Name></Team>{}</Root>'.format(players))
    lines = []
    for frame_id in range(1, n_frames + 1):
        groups = []
        for _ in range(n_slots):
            if rng.random() < 0.15:
                groups.append("NaN,NaN,NaN")
            else:
                groups.append("{:.5f},{:.5f},{:.2f}".format(rng.random(), rng.random(),
                                                           rng.random() * 8))
        ball = "NaN,NaN,NaN" if rng.random() < 0.4 else "{:.5f},{:.5f},0.0".format(
            rng.random(), rng.random())
        lines.append("{}:{}:{}".format(frame_id, ";".join(groups), ball))
    raw = tmp_path / "s_atd.txt"
    raw.write_text("\n".join(lines) + "\n")
    return raw, meta


@pytest.mark.skipif(not official_present, reason="official/ submodule not checked out")
def test_matches_official_baseline(tmp_path):
    config.add_official_to_path()
    import baseline as official

    raw, meta = synthetic_atd(tmp_path)
    n_frames = 400
    expected = official.nearest_player_labels(
        str(raw), official.team_of_each_track(str(meta)), n_frames,
        (config.PITCH_LENGTH_M, config.PITCH_WIDTH_M))

    arrays, _ = atd.parse_atd(raw, meta, n_frames)
    ours = baselines.nearest_player_team(arrays)
    # the reproduction script must use exactly this function
    assert load_script("02_reproduce_baseline").nearest_player_team is baselines.nearest_player_team

    assert np.array_equal(ours, expected)
    # and the smoothing applied on top is the official function, so smoothed outputs agree too
    assert np.array_equal(official.smooth(ours, 50), official.smooth(expected, 50))
