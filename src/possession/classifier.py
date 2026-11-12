"""Gradient-boosted per-frame probabilities: P(dead) and P(home | alive) (M3, optional).

The rule cells of v1-v3 look at a handful of features at a time. Here two
HistGradientBoostingClassifier models (scikit-learn, pinned) see everything at once: ball
kinematics, gaps, restart-like kicks, player speeds, touches, and team shape (spread, extent,
how far forward each team sits and moves along its attacking axis -- signal when the ball is
not seen). Everything else of v3 is kept: HSMM decoding, restart evidence, hand-over snapping.

Rules that keep this honest:
  * trained ONLY on degraded external games, and leave-one-game-out (training.LogoPredictor);
  * the possession model is exactly SIDE-SYMMETRIC: it is trained on every frame twice, as
    seen and mirrored (home <-> away swapped, label flipped), and predicts the average of the
    two views, so it cannot learn "home is more often in possession";
  * every 5th frame only (neighbouring frames are near copies), fixed seed, no early stopping
    (deterministic).
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from . import config, evidence, features, touches

H, A, D, X = config.H, config.A, config.D, config.X

SUBSAMPLE = 5
CAP_S = 30.0                    # "seconds since" features are capped here
TOUCH_WINDOW_S = 3.0            # count touches per team over the last 3 s
GBM_PARAMS = dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=200,
                  l2_regularization=1.0, early_stopping=False, random_state=0)
EPS = 1e-3


def seconds(frames, fps):
    """Frame counts -> seconds, capped at CAP_S."""
    return np.minimum(np.asarray(frames, dtype=np.float64) / fps, CAP_S)


def line_distance(x, y, length, width):
    """Distance (m) from a position to the nearest pitch line (negative outside)."""
    return np.minimum.reduce([x, length - x, y, width - y])


def touch_history(f):
    """Per frame: sign of the last touch's team (+1 home, -1 away, 0 none), seconds since it,
    and the number of home / away touches in the last TOUCH_WINDOW_S."""
    fps = f["fps"]
    n = len(f["ball_present"])
    frames, teams = touches.detect(f)
    marks = np.zeros(n, dtype=bool)
    marks[frames] = True
    last = features.last_index(marks)
    team_at = np.zeros(n, dtype=np.int8)
    team_at[frames] = np.where(teams == 0, 1, -1)
    sign = np.where(last >= 0, team_at[np.maximum(last, 0)], 0)
    since = np.where(last >= 0, np.arange(n) - last, features.NEVER)
    window = int(TOUCH_WINDOW_S * fps)
    idx = np.arange(n)
    counts = []
    for team in (0, 1):
        hits = np.zeros(n)
        hits[frames[teams == team]] = 1
        cum = np.concatenate(([0.0], np.cumsum(hits)))
        counts.append(cum[idx + 1] - cum[np.maximum(idx + 1 - window, 0)])
    return sign, seconds(since, fps), counts[0], counts[1]


def dead_matrix(f):
    """(N, k) features for P(dead). No team-specific column, so it is side-neutral by design."""
    fps = f["fps"]
    length, width = f["pitch"]
    last_line = line_distance(f["last_bx"], f["last_by"], length, width)
    cols = [
        f["ball_present"].astype(float), seconds(f["gap_len"], fps), seconds(f["gap_since"], fps),
        seconds(f["gap_left"], fps), f["ball_speed"], f["ball_accel"], seconds(f["static_len"], fps),
        seconds(f["kick_since"], fps), f["ball_out_m"], f["last_out_m"], f["next_out_m"],
        f["heading_out_m"], last_line, f["speed_mean_1s"], f["speed_mean_3s"], f["decel_1s"],
        np.fmax(f["speed_home_1s"], f["speed_away_1s"]),
        np.fmin(f["speed_home_1s"], f["speed_away_1s"]),
        (f["n_home"] + f["n_away"]).astype(float), f["n_near5"].astype(float),
        f["n_near10"].astype(float), f["sx_home"] + f["sx_away"], f["sy_home"] + f["sy_away"],
        f["pre_kickoff"].astype(float),
    ]
    return np.column_stack(cols)


def centre_per_half(x, half):
    """x minus its own median over each half (NaN-aware): "unusual for this team right now".

    v5 (D20): team shape carries STYLE as well as possession -- on our match the home side is
    on average 2.7 m wider and 5 m higher than the away side over the whole game, far more than
    in any external game. Centring each team's shape on its own half-median keeps the
    possession signal (a team spreads out when it has the ball) and removes the style offset.
    Uses only the recording's own tracking: no labels.
    """
    out = np.array(x, dtype=np.float64)
    for number in (1, 2):
        rows = half == number
        if rows.any():
            out[rows] -= np.nanmedian(out[rows])
    return out


def team_matrices(f, centre=False):
    """(as-seen, mirrored) (N, m) feature matrices for P(home | alive).

    Columns come in home/away PAIRS (swapped by the mirror) and SIGNED columns (negated by it),
    plus neutral ones, so mirroring = "the same frame with the teams' roles exchanged".
    centre=True (v5) centres every team-shape / team-speed column on its half-median.
    """
    fps = f["fps"]
    length, _ = f["pitch"]
    home_dir = f["home_dir"].astype(float)
    cell, team = evidence.team_cells(f, fps)
    direct = cell < evidence.N_DIRECT
    last = features.last_index(direct)
    carry_sign = np.where(team == 0, 1.0, np.where(team == 1, -1.0, 0.0))
    carry_since = seconds(np.where(last >= 0, np.arange(len(cell)) - last, features.NEVER), fps)
    t_sign, t_since, t_home, t_away = touch_history(f)

    shape_pairs = [
        (f["sx_home"], f["sx_away"]), (f["sy_home"], f["sy_away"]),
        (f["len_home"], f["len_away"]), (f["wid_home"], f["wid_away"]),
        ((f["cx_home"] - length / 2) * home_dir, (f["cx_away"] - length / 2) * -home_dir),
        (f["vx_home"] * home_dir, f["vx_away"] * -home_dir),
        (f["speed_home_1s"], f["speed_away_1s"]),
    ]
    if centre:
        shape_pairs = [(centre_per_half(a, f["half"]), centre_per_half(b, f["half"]))
                       for a, b in shape_pairs]
    pairs = [(f["d_home"], f["d_away"]), (t_home, t_away)] + shape_pairs + [
        (f["n_home"].astype(float), f["n_away"].astype(float))]
    signed = [carry_sign, t_sign.astype(float), (f["bx"] - length / 2) * home_dir,
              (f["last_bx"] - length / 2) * home_dir]
    neutral = [f["ball_present"].astype(float), f["ball_speed"], seconds(f["gap_len"], fps),
               seconds(f["kick_since"], fps), carry_since, t_since, f["speed_mean_1s"]]
    seen = np.column_stack([p[0] for p in pairs] + [p[1] for p in pairs] + signed + neutral)
    mirror = np.column_stack([p[1] for p in pairs] + [p[0] for p in pairs]
                             + [-s for s in signed] + neutral)
    return seen, mirror


class GbmModel(touches.SnapModel):
    """v4: v3 with the rule-cell probabilities replaced by two gradient-boosted models.

    gbm_dead / gbm_team choose, separately, whether P(dead) and P(home | alive) come from the
    gradient-boosted models or from the v2 rule cells (the hybrids are measured ablations).
    centre (v5): team-shape columns centred on each team's own half-median (centre_per_half).
    """

    def __init__(self, gbm_dead=True, gbm_team=True, centre=False, **kwargs):
        super().__init__(**kwargs)
        self.gbm_dead, self.gbm_team = gbm_dead, gbm_team
        self.centre = centre

    def fit(self, train):
        dead_x, dead_y, team_x, team_y = [], [], [], []
        for feats, _, gt in train:
            rows = np.arange(0, len(gt), SUBSAMPLE)
            rows = rows[gt[rows] != X]
            dead_x.append(dead_matrix(feats)[rows])
            dead_y.append(gt[rows] == D)
            alive = rows[(gt[rows] == H) | (gt[rows] == A)]
            seen, mirror = team_matrices(feats, self.centre)
            team_x += [seen[alive], mirror[alive]]
            team_y += [gt[alive] == H, gt[alive] == A]
        self.dead_gbm = HistGradientBoostingClassifier(**GBM_PARAMS).fit(
            np.vstack(dead_x), np.concatenate(dead_y))
        self.team_gbm = HistGradientBoostingClassifier(**GBM_PARAMS).fit(
            np.vstack(team_x), np.concatenate(team_y))
        super().fit(train)          # cells (unused), durations, transitions, restart table
        return self

    def probabilities(self, feats):
        """(P(dead), P(home | alive)) per frame from the two models (or the rule cells)."""
        rule_dead, rule_home = super().probabilities(feats)
        if self.gbm_dead:
            p_dead = np.clip(self.dead_gbm.predict_proba(dead_matrix(feats))[:, 1], EPS, 1 - EPS)
            p_dead[feats["pre_kickoff"]] = 1 - EPS               # certain before kick-off
        else:
            p_dead = rule_dead
        if self.gbm_team:
            seen, mirror = team_matrices(feats, self.centre)
            p_home = 0.5 * (self.team_gbm.predict_proba(seen)[:, 1]
                            + self.team_gbm.predict_proba(mirror)[:, 0])
            p_home = np.clip(p_home, EPS, 1 - EPS)
        else:
            p_home = rule_home
        return p_dead, p_home
