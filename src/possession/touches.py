"""Ball touches, and snapping possession hand-overs to them (M3 / M4).

A TOUCH is a sharp change of the ball's velocity -- a kick, a reception, a deflection -- with a
player close enough to have caused it:

    ball seen, |change of ball velocity| > ACCEL_MS2 (m/s per second), nearest player <= TOUCH_M

High-acceleration frames come in short clusters; each cluster gives one touch, at its strongest
frame, credited to the team of the nearest player there. Slots are tracklets, so a touch has a
team, never a player identity.

Why it matters: the truth changes possession AT the new team's controlled touch (a recovery,
the reception of an interception). The decoder's hand-overs are badly timed -- only ~20% within
0.5 s on the external games, and typically ~1 s EARLY, because a defender closing down the ball
is already "the nearest player" before he wins it. Touches, by contrast, sit on the true
hand-over (median +1 frame, 79% within 0.5 s). snap_handovers() therefore moves each predicted
H<->A hand-over to the definition of a change of possession: the first touch of the new team
after the old team's last touch (rule "after_last"), searched mostly forward.
"""

import numpy as np

from . import config, decoder, features, restarts

H, A, D, X = config.H, config.A, config.D, config.X

TOUCH_M = 1.5          # a player within 1.5 m of the ball may have touched it
ACCEL_MS2 = 10.0       # ball velocity changed faster than 10 m/s per second
MERGE_FRAMES = 3       # high-acceleration frames closer than this are one touch
SNAP_S = 1.0           # a hand-over moves to a touch at most 1 s away


def detect(f, touch_m=TOUCH_M, accel_ms2=ACCEL_MS2):
    """(frames, teams) of the touches, sorted by frame. teams: 0 home / 1 away."""
    d_home = np.nan_to_num(f["d_home"], nan=np.inf)
    d_away = np.nan_to_num(f["d_away"], nan=np.inf)
    near = np.minimum(d_home, d_away) <= touch_m
    accel = np.nan_to_num(f["ball_accel"], nan=0.0)
    candidate = f["ball_present"] & near & (accel > accel_ms2)

    # merge clusters: close the small holes, then take the strongest frame of each run
    marks = candidate.copy()
    for start, stop in features.runs_of(~candidate):
        if 0 < start and stop < len(candidate) and stop - start < MERGE_FRAMES:
            marks[start:stop] = True
    frames, teams = [], []
    for start, stop in features.runs_of(marks):
        rows = np.arange(start, stop)[candidate[start:stop]]
        best = int(rows[np.argmax(accel[rows])])
        frames.append(best)
        teams.append(0 if d_home[best] <= d_away[best] else 1)
    return np.array(frames, dtype=np.int64), np.array(teams, dtype=np.int8)


# "first touch after the last" window around a predicted hand-over. Chosen on the external
# harness (D17): predicted hand-overs come EARLY (median -1 s: the new team's player is nearest
# the ball while still challenging for it), so the search looks mostly forward; 6-10 s were
# equally good, -0.5/+6 s kept.
SNAP_BEFORE_S, SNAP_AFTER_S = 0.5, 6.0


def handover_frame(f, old, new, lo, hi, touch_frames, touch_teams, rule):
    """Where the hand-over near f should be, or None.

    rule "nearest": the touch of the new team nearest to f.
    rule "after_last": the definition of a change of possession -- the first touch of the new
        team after the old team's LAST touch in [lo, hi] (or the first new-team touch at all if
        the old team has none there).
    """
    in_window = (touch_frames >= lo) & (touch_frames <= hi)
    new_touch = touch_frames[in_window & (touch_teams == new)]
    if not len(new_touch):
        return None
    if rule == "nearest":
        return int(new_touch[np.argmin(np.abs(new_touch - f))])
    old_touch = touch_frames[in_window & (touch_teams == old)]
    if len(old_touch):
        after = new_touch[new_touch > old_touch.max()]
        return int(after[0]) if len(after) else None
    return int(new_touch[0])


def snap_handovers(codes, touch_frames, touch_teams, fps=config.FPS, snap_s=SNAP_S,
                   rule="after_last", before_s=SNAP_BEFORE_S, after_s=SNAP_AFTER_S):
    """Move each H<->A hand-over to a touch of the NEW team (see handover_frame for the rules).

    "nearest" searches +/- snap_s around the hand-over; "after_last" searches before_s before to
    after_s after it. The boundary never moves past the neighbouring boundaries (each segment
    keeps >= 1 frame), so no segment disappears and no new one is created: only timing changes.
    """
    out = codes.copy()
    back = int((snap_s if rule == "nearest" else before_s) * fps)
    ahead = int((snap_s if rule == "nearest" else after_s) * fps)
    change = np.flatnonzero(out[1:] != out[:-1]) + 1            # first frame of a new label
    bounds = np.concatenate(([0], change, [len(out)]))
    for i in range(1, len(bounds) - 1):
        f = int(bounds[i])
        old, new = out[f - 1], out[f]
        if old not in (H, A) or new not in (H, A) or old == new:
            continue
        lo = max(int(bounds[i - 1]) + 1, f - back)              # keep >= 1 frame of `old`
        hi = min(int(bounds[i + 1]) - 1, f + ahead)             # keep >= 1 frame of `new`
        g = handover_frame(f, old, new, lo, hi, touch_frames, touch_teams, rule)
        if g is None:
            continue
        if g < f:
            out[g:f] = new
        elif g > f:
            out[f:g] = old
        bounds[i] = g                                           # the next boundary sees it
    return out


# --- X policy (M4 ablation) -------------------------------------------------------------------

X_GAP_S = 3.0          # abstain only inside ball gaps at least this long ...
X_MARGIN_S = 1.0       # ... and at least this far from either end of the possession


def abstain_in_gaps(codes, f, fps=config.FPS):
    """Write X inside long ball gaps that lie wholly inside one predicted possession.

    On spellF1 an X is replaced by the previous label of the half, i.e. by this same
    possession, so the possession count and boundaries cannot change; macroF1 counts every X
    as a miss. Kept only as a measured ablation (D18).
    """
    out = codes.copy()
    margin = int(X_MARGIN_S * fps)
    gap = (f["gap_len"] >= X_GAP_S * fps) & ~f["ball_present"]
    change = np.flatnonzero(out[1:] != out[:-1]) + 1
    bounds = np.concatenate(([0], change, [len(out)]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        if out[a] not in (H, A) or b - a <= 2 * margin:
            continue
        inner = np.arange(a + margin, b - margin)
        out[inner[gap[inner]]] = X
    return out


class SnapModel(restarts.DeadballModel):
    """v3: v2, then every H<->A hand-over moved to the first new-team touch after the old
    team's last touch (snap_handovers). Optional X policy for the ablation."""

    def __init__(self, x_policy=False, w_dur=1.0, snap=True, **kwargs):
        super().__init__(**kwargs)
        self.x_policy = x_policy
        self.w_dur = w_dur
        self.snap = snap

    def predict(self, feats, info):
        codes = super().predict(feats, info)
        if self.snap:
            frames, teams = detect(feats)
            codes = snap_handovers(codes, frames, teams)
        if self.x_policy:
            codes = abstain_in_gaps(codes, feats)
        return codes
