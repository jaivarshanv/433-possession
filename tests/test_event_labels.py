"""Label derivation rules on a hand-built SYNTHETIC game, one scenario per rule."""

import numpy as np
import pandas as pd

from possession import config, event_labels

H, A, D, X = config.H, config.A, config.D, config.X


def event(team, type_, start, end=None, subtype="", player="", to="", period=1):
    return {"team": team, "type": type_, "subtype": subtype, "period": period,
            "start": start, "end": start if end is None else end, "player": player, "to": to}


def synthetic_game():
    """1000 frames, two periods of 500. Ball in mid-pitch unless a scenario moves it."""
    n = 1000
    ball = np.full((n, 2), 0.5)
    ball[200:210] = [0.5, 1.02]          # ball crosses the touchline ...
    ball[210:260] = np.nan               # ... and the corrected tracking drops it until the throw
    ball[300:305] = np.nan               # a 5-frame tracking hole during live play
    ball[950:] = np.nan                  # after the final whistle
    events = pd.DataFrame([
        # period 1
        event("H", "SET PIECE", 10, subtype="KICK OFF", player="h1"),
        event("H", "PASS", 10, 30, player="h1", to="h2"),
        event("A", "RECOVERY", 60, player="a1"),            # one-touch: same frame as ...
        event("A", "BALL LOST", 60, 75, player="a1"),       # ... the loss -> no change
        event("H", "PASS", 80, 100, player="h2", to="h3"),
        event("A", "RECOVERY", 120, player="a2"),           # kept 30 frames -> change at 120
        event("A", "PASS", 150, 170, player="a2", to="a3"),
        event("A", "BALL OUT", 190, 205, player="a3"),
        event("H", "SET PIECE", 260, subtype="THROW IN", player="h4"),
        event("H", "PASS", 260, 280, player="h4", to="h5"),
        event("H", "CHALLENGE", 330, subtype="TACKLE-FAULT-WON", player="h5"),
        event("H", "FAULT RECEIVED", 330, player="h5"),
        event("H", "SET PIECE", 380, subtype="FREE KICK", player="h5"),   # ball never lost
        event("H", "PASS", 380, 400, player="h5", to="h6"),
        # period 2
        event("A", "SET PIECE", 520, subtype="KICK OFF", player="a1", period=2),
        event("A", "PASS", 520, 540, player="a1", to="a2", period=2),
        event("H", "PASS", 600, 620, player="h1", to="h2", period=2),  # first-time pass: H
        event("H", "PASS", 640, 660, player="h2", to="h3", period=2),  # controls at reception
    ])
    period = np.repeat(np.array([1, 2], dtype=np.int8), 500)
    return {"ball": ball, "events": events, "period": period, "name": "synthetic"}


def labels():
    return event_labels.derive_labels(synthetic_game())


def test_pre_kickoff_is_dead_and_kickoff_team_starts():
    codes, _ = labels()
    assert (codes[0:10] == D).all() and codes[10] == H
    assert (codes[500:520] == D).all() and codes[520] == A


def test_one_touch_recovery_does_not_change_possession():
    codes, _ = labels()
    assert (codes[10:120] == H).all()


def test_controlled_recovery_changes_possession_at_the_touch():
    codes, _ = labels()
    assert codes[119] == H and (codes[120:200] == A).all()


def test_ball_out_is_dead_from_the_line_to_the_restart():
    codes, report = labels()
    assert codes[199] == A                 # still alive up to the crossing
    assert (codes[200:260] == D).all()     # from the first frame outside, not from the NaN
    assert codes[260] == H                 # the throw-in team restarts
    assert report["restart_event"] >= 1


def test_short_tracking_hole_stays_alive():
    codes, report = labels()
    assert (codes[295:310] == H).all()
    assert report["live_tracking_hole"] == 1


def test_restart_without_ball_gap_is_dead_from_the_foul():
    codes, report = labels()
    assert codes[329] == H and (codes[330:380] == D).all() and codes[380] == H
    assert report["restart_without_gap"] == 1


def test_possession_changes_at_controlled_reception_not_first_touch():
    codes, _ = labels()
    assert codes[619] == A and codes[620] == H


def test_end_of_period_and_every_frame_labelled():
    codes, _ = labels()
    assert (codes[950:1000] == D).all()
    assert set(np.unique(codes).tolist()) <= {H, A, D}   # no X inside the two periods


def test_stoppage_and_restart_subtypes():
    events = pd.DataFrame([
        event("H", "SHOT", 0, subtype="OFF TARGET-OUT"),
        event("H", "SHOT", 0, subtype="ON TARGET-SAVED"),
        event("H", "PASS", 0, subtype="GOAL KICK"),
        event("H", "CHALLENGE", 0, subtype="TACKLE-ADVANTAGE-WON"),
        event("H", "BALL LOST", 0, subtype="OFFSIDE"),
    ])
    assert event_labels.is_stoppage(events).tolist() == [True, False, False, False, True]
    assert event_labels.is_restart(events).tolist() == [False, False, True, False, False]
