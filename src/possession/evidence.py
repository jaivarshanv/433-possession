"""Possession evidence, per-frame state scores, and the v1 rule-based model (M1).

Possession evidence v1, in football terms:
  * DIRECT: the ball is seen and a player is within GATE_M of it -> his team is probably the
    one playing it. How sure depends on how close he is, whether an opponent is close too
    (a contested ball) and how fast the ball moves (a fast ball is passing by, not controlled).
  * CARRY: no direct evidence -> the team of the last direct evidence, less and less surely as
    the seconds pass ("a loose ball stays with the last team").
Each frame falls into one cell; CellTable (deadball.py) counts on the EXTERNAL labels how often
the team of that cell really has the ball. Cells are written relative to "the team the evidence
points at", so the model cannot learn a home/away preference.

State scores for the decoder: with P(dead) from deadball.py and P(home | alive) from here,
    score[D] = w_dead * (log P(dead) - log prior_dead)
    score[H] = w_dead * (log P(alive) - log prior_alive) + w_team * (log P(home|alive) - log 0.5)
    score[A] = same with P(away|alive)
These are log-likelihood ratios; the two weights (< 1, because neighbouring frames are far from
independent) are chosen on the external harness.
"""

import numpy as np

from . import config, deadball, decoder, features

H, A, D, X = config.H, config.A, config.D, config.X

GATE_M = 2.5                    # a player within 2.5 m of the ball may be playing it
NEAR_EDGES_M = (0.75, 1.5)      # how close the nearest player is -> 3 bins
CONTEST_M = 2.0                 # an opponent this close too = contested ball
BALL_SPEED_EDGES = (3.0, 8.0)   # m/s -> ball slow / medium / fast
CARRY_EDGES_S = (1.0, 3.0, 8.0)  # seconds since the last direct evidence -> 4 bins

N_DIRECT = (len(NEAR_EDGES_M) + 1) * 2 * (len(BALL_SPEED_EDGES) + 1)
N_CARRY = len(CARRY_EDGES_S) + 1
NO_EVIDENCE = N_DIRECT + N_CARRY
N_TEAM_CELLS = NO_EVIDENCE + 1

W_DEAD, W_TEAM = 0.1, 0.1       # default weights; v1_rules uses the harness-chosen ones


def team_cells(f, fps=config.FPS):
    """(cell, team) per frame: the evidence cell and the team (0 home / 1 away / -1 none)."""
    d_home = np.nan_to_num(f["d_home"], nan=np.inf)
    d_away = np.nan_to_num(f["d_away"], nan=np.inf)
    nearest_team = np.where(d_home <= d_away, 0, 1).astype(np.int8)
    d_near = np.minimum(d_home, d_away)
    d_other = np.maximum(d_home, d_away)
    direct = f["ball_present"] & (d_near <= GATE_M)

    near_bin = np.digitize(d_near, NEAR_EDGES_M)
    contested = (d_other <= CONTEST_M).astype(int)
    speed_bin = np.digitize(np.nan_to_num(f["ball_speed"], nan=0.0), BALL_SPEED_EDGES)
    direct_cell = (near_bin * 2 + contested) * (len(BALL_SPEED_EDGES) + 1) + speed_bin

    last = features.last_index(direct)
    seconds_since = (np.arange(len(direct)) - last) / fps
    carry_cell = N_DIRECT + np.digitize(seconds_since, CARRY_EDGES_S)
    has_last = last >= 0
    team_last = np.where(has_last, nearest_team[np.maximum(last, 0)], -1)

    cell = np.where(direct, direct_cell, np.where(has_last, carry_cell, NO_EVIDENCE))
    team = np.where(direct, nearest_team, team_last).astype(np.int8)
    return cell.astype(np.int32), team


def home_given_alive(p_match, team):
    """P(home has the ball | alive) from P(evidence team has it) and which team that is."""
    return np.where(team == 0, p_match, np.where(team == 1, 1 - p_match, 0.5))


def state_scores(p_dead, prior_dead, p_home, w_dead, w_team):
    """(N, 3) decoder scores from per-frame probabilities (see module docstring)."""
    alive_ratio = np.log1p(-p_dead) - np.log1p(-prior_dead)
    scores = np.empty((len(p_dead), 3))
    scores[:, D] = w_dead * (np.log(p_dead) - np.log(prior_dead))
    scores[:, H] = w_dead * alive_ratio + w_team * (np.log(p_home) - np.log(0.5))
    scores[:, A] = w_dead * alive_ratio + w_team * (np.log1p(-p_home) - np.log(0.5))
    return scores


class RulesModel:
    """v1: rule cells (dead ball, possession) + HSMM, every probability counted on external labels.

    fit(train) takes [(features, info, gt)] of external games; predict(features, info) -> codes.
    """

    def __init__(self, w_dead=W_DEAD, w_team=W_TEAM):
        self.w_dead, self.w_team = w_dead, w_team

    def fit(self, train):
        dead_c, dead_hit, team_c, team_hit = [], [], [], []
        for feats, _, gt in train:
            graded = gt != X
            dead_c.append(deadball.dead_cells(feats)[graded])
            dead_hit.append(gt[graded] == D)
            cell, team = team_cells(feats)
            usable = ((gt == H) | (gt == A)) & (team >= 0)
            team_c.append(cell[usable])
            team_hit.append(gt[usable] == team[usable])     # H/A codes equal team codes 0/1
        self.dead = deadball.CellTable(deadball.N_CELLS).fit(np.concatenate(dead_c),
                                                             np.concatenate(dead_hit))
        self.team = deadball.CellTable(N_TEAM_CELLS).fit(np.concatenate(team_c),
                                                          np.concatenate(team_hit))
        labels = [gt for _, _, gt in train]
        self.durations = decoder.fit_durations(labels)
        self.log_trans = decoder.fit_transitions(labels)
        return self

    def probabilities(self, feats):
        """(P(dead), P(home | alive)) per frame."""
        p_dead = self.dead.predict(deadball.dead_cells(feats))
        cell, team = team_cells(feats)
        return p_dead, home_given_alive(self.team.predict(cell), team)

    def scores(self, feats):
        p_dead, p_home = self.probabilities(feats)
        return state_scores(p_dead, self.dead.rate, p_home, self.w_dead, self.w_team)

    def predict(self, feats, info):
        return decoder.decode(self.scores(feats), feats["half"], self.durations, self.log_trans)
